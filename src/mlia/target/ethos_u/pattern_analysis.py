# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U pattern analysis module for detecting optimization opportunities."""

from __future__ import annotations

from dataclasses import dataclass, field

from mlia.core.data_analysis import Fact, PatternAnalyzer, register_fact_type
from mlia.target.ethos_u.data_analysis import (
    EthosULayerHighNetworkShare,
    EthosULayerHighOpCycles,
    EthosULayerLowMacUtil,
    EthosULayerHighMemoryPressure,
    EthosULayerSuboptimalActivation,
)


@register_fact_type(
    "ethos_u_ineffective_activation_pattern",
    "pattern",
    "Multiple layers using ineffective activations that could be optimized",
)
@dataclass
class IneffectiveActivationPattern(Fact):
    """Pattern indicating multiple layers with inefficient activations.

    This composite fact is generated when multiple layers use
    suboptimal activation functions that are hard to quantize effectively.
    """

    affected_layers: list[str] = field(default_factory=list)
    layer_count: int = 0
    activation_types: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = super().to_dict()
        result.update(
            {
                "affected_layers": self.affected_layers,
                "layer_count": self.layer_count,
                "activation_types": self.activation_types,
                "recommendation": self.recommendation,
            }
        )
        return result


@dataclass
class LayerCollectionPattern(Fact):
    """Base pattern for collecting layers by shared advice.

    This composite fact is generated to group all layers identified
    as resulting in the same piece of advice.
    """

    facts: list[Fact]
    layer_count: int = 0
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = super().to_dict()
        result.update(
            {
                "facts": [fact.to_dict() for fact in self.facts],
            }
        )
        return result


@dataclass
class HighImpactLayersPattern(LayerCollectionPattern):
    """Pattern containing all layers high network share or op cycles.

    This composite fact is generated to group all layers identified
    as having a high share of the total op cycles (if using corstone), or
    highest x op cycles (if using vela).
    """


@dataclass
class LowMacUtilLayersPattern(LayerCollectionPattern):
    """Pattern containing all layers with a low mac util.

    This composite fact is generated to group all layers identified
    as having a low mac util and a high share of total op cycles into one fact.
    """


@dataclass
class MemoryBoundLayersPattern(LayerCollectionPattern):
    """Pattern containing all layers considered memory bound.

    This composite fact is generated to group all layers identified
    as having a high memory use, low mac util and a high share of
    total op cycles.
    """


class ActivationFunctionPatternAnalyzer(PatternAnalyzer):
    """Analyzes activation function patterns across layers.

    Detects when multiple layers use suboptimal activation functions that could
    be replaced with more efficient alternatives or quantized differently.
    """

    def has_already_generated_patterns(self, facts: list[Fact]) -> bool:
        """Check if we've already generated ineffective activation patterns.

        :param facts: List of facts to check
        :return: True if patterns already exist, False otherwise
        """
        # Check if any IneffectiveActivationPattern facts already exist
        return any(isinstance(f, IneffectiveActivationPattern) for f in facts)

    def analyze_patterns(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to detect inefficient activation patterns.

        :param facts: List of all facts from analysis
        :return: List of newly detected pattern facts
        """
        if self.has_already_generated_patterns(facts):
            return []

        bad_activation_facts = [
            f for f in facts if isinstance(f, EthosULayerSuboptimalActivation)
        ]

        if not bad_activation_facts:
            return []

        affected_layers = []
        activation_types_set = set()

        for fact in bad_activation_facts:
            if hasattr(fact, "operator_name"):
                affected_layers.append(f"{fact.operator_name} ({fact.location})")
            if hasattr(fact, "activation_type"):
                activation_types_set.add(fact.activation_type)

        # Create single pattern for all suboptimal activations
        activation_types = sorted(activation_types_set)

        recommendation = (
            "Consider replacing suboptimal activation functions with "
            "NPU-friendly alternatives."
        )

        # Identify activation functions that use unsupported operations
        unsupported_activations = {"MISH", "SELU", "SOFTPLUS"}
        used_unsupported_activations = unsupported_activations & set(activation_types)

        if used_unsupported_activations:
            recommendation += (
                "\nThe following activation functions use unsupported operations: "
                + ", ".join(sorted(used_unsupported_activations))
            )

        pattern = IneffectiveActivationPattern(
            affected_layers=affected_layers,
            layer_count=len(bad_activation_facts),
            activation_types=activation_types,
            recommendation=recommendation,
        )

        return [pattern]


class LayerHotSpotPatternAnalyzer(PatternAnalyzer):
    """Analyzes layers to identify performance issues."""

    def has_already_generated_patterns(self, facts: list[Fact]) -> bool:
        """Check if we've already generated patterns.

        :param facts: List of facts to check
        :return: True if patterns already exist, False otherwise
        """
        return any(
            isinstance(
                f,
                (
                    HighImpactLayersPattern,
                    LowMacUtilLayersPattern,
                    MemoryBoundLayersPattern,
                ),
            )
            for f in facts
        )

    def analyze_patterns(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to detect inefficient layer patterns.

        :param facts: List of all facts from analysis
        :return: List of newly detected pattern facts
        """
        # Skip analysis if we've already generated patterns
        if self.has_already_generated_patterns(facts):
            return []

        patterns = []
        available_facts = list(facts)

        network_patterns = self.analyze_layer_impact(available_facts)
        patterns.extend(network_patterns)
        available_facts.extend(network_patterns)

        mac_patterns = self.analyze_mac_util(available_facts)
        patterns.extend(mac_patterns)
        available_facts.extend(mac_patterns)

        memory_patterns = self.analyze_memory_bound(available_facts)
        patterns.extend(memory_patterns)
        return patterns

    def analyze_layer_impact(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to combine layers with high network share or high op cycles.

        :param facts: List of all facts from analysis
        :return: Detected pattern fact
        """
        high_impact_layer_facts = [
            f for f in facts if isinstance(f, EthosULayerHighNetworkShare)
        ]
        recommendation = (
            "This ranking is based on the percentage of total cycles per layer."
        )

        if len(high_impact_layer_facts) == 0:
            high_impact_layer_facts = [
                f for f in facts if isinstance(f, EthosULayerHighOpCycles)
            ]
            recommendation = (
                "This ranking is not based on the percentage of total cycles per layer."
                " Try running on Corstone backend for"
                " more accurate impact results."
            )

        # Create single grouped pattern fact for all detected high network layers
        pattern = HighImpactLayersPattern(
            facts=high_impact_layer_facts,
            layer_count=len(high_impact_layer_facts),
            recommendation=recommendation,
        )

        return [pattern]

    def analyze_mac_util(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to combine layers with low mac util.

        :param facts: List of all facts from analysis
        :return: Detected pattern fact
        """
        # Filter low MAC utilization facts to those already selected by the
        # high-impact layer pattern.
        low_mac_util_facts = [f for f in facts if isinstance(f, EthosULayerLowMacUtil)]

        high_impact_patterns = [
            f for f in facts if isinstance(f, HighImpactLayersPattern)
        ]

        high_impact_facts: list[Fact] = []
        for pattern in high_impact_patterns:
            high_impact_facts.extend(pattern.facts)

        high_keys = {f.key for f in high_impact_facts if hasattr(f, "key")}
        filtered_low_mac_facts = [
            f for f in low_mac_util_facts if hasattr(f, "key") and f.key in high_keys
        ]

        if not filtered_low_mac_facts:
            return []

        # Create single grouped pattern fact for all detected high network layers
        pattern = LowMacUtilLayersPattern(
            facts=filtered_low_mac_facts,
            layer_count=len(filtered_low_mac_facts),
        )

        return [pattern]

    def analyze_memory_bound(self, facts: list[Fact]) -> list[Fact]:
        """Analyze facts to combine layers with high memory use into one pattern.

        :param facts: List of all facts from analysis
        :return: Detected pattern fact
        """
        # Filter memory-high facts to those already selected by the low-MAC
        # utilization pattern.
        high_memory_facts = [
            f for f in facts if isinstance(f, EthosULayerHighMemoryPressure)
        ]

        low_mac_util_patterns = [
            f for f in facts if isinstance(f, LowMacUtilLayersPattern)
        ]

        low_mac_util_pattern_facts: list[Fact] = []
        for pattern in low_mac_util_patterns:
            low_mac_util_pattern_facts.extend(pattern.facts)

        low_mac_keys = {f.key for f in low_mac_util_pattern_facts if hasattr(f, "key")}
        filtered_mem_fact = [
            f for f in high_memory_facts if hasattr(f, "key") and f.key in low_mac_keys
        ]

        if not filtered_mem_fact:
            return []

        # Create single grouped pattern fact for all detected high network layers
        pattern = MemoryBoundLayersPattern(
            facts=filtered_mem_fact,
            layer_count=len(filtered_mem_fact),
        )

        return [pattern]
