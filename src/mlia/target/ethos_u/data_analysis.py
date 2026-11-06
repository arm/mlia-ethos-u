# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U data analysis module."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import singledispatchmethod

from mlia.backend.vela._supported_ops_types import Constraint
from mlia.backend.vela.compat import Operators
from mlia.core.common import DataItem
from mlia.core.data_analysis import (
    Fact,
    FactExtractor,
    LayerCompatibilityIssue,
    LayerPerformanceIssue,
    register_fact_type,
)
from mlia.target.ethos_u.common_reporters import analyze_tflite_compatibility_common
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.utils.tflite_shims import TFLiteCompatibilityInfo
from mlia.target.ethos_u.performance import (
    CombinedPerformanceResult,
    CorstonePerformanceResult,
    VelaPerformanceResult,
    OptimizationPerformanceMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class HasCPUOnlyOperators(Fact):
    """Model has CPU only operators."""

    cpu_only_ops: list[str]
    affected_entity_ids: list[str] = field(default_factory=list)


@dataclass
class HasUnsupportedOnNPUOperators(Fact):
    """Model has unsupported on NPU operators."""

    npu_unsupported_ratio: float


@dataclass
class AllOperatorsSupportedOnNPU(Fact):
    """All model's operators supported on NPU."""


@register_fact_type(
    "ethos_u_operator_advice",
    "layer",
    "Observed Vela advice for one operator",
)
@dataclass
class VelaOperatorAdvice(Fact):
    """Advice associated with one operator rejected by Vela."""

    entity_id: str
    message: str


@register_fact_type(
    "ethos_u_catalog_advice",
    "layer",
    "Vela catalog support constraints for rejected operators",
)
@dataclass
class VelaCatalogAdvice(Fact):
    """Reference constraints from the matching Vela support catalog."""

    message: str
    affected_entity_ids: list[str]


def _format_catalog_constraints(
    constraints: tuple[Constraint, ...], depth: int = 0
) -> list[str]:
    """Format catalog constraints and their nested qualifications."""
    lines: list[str] = []
    for constraint in constraints:
        lines.append(f"{'  ' * depth}- {constraint.text}")
        lines.extend(_format_catalog_constraints(constraint.children, depth + 1))
    return lines


@register_fact_type(
    "ethos_u_layer_compatibility",
    "layer",
    "Ethos-U specific layer compatibility information",
)
@dataclass
class EthosULayerCompatibilityIssue(LayerCompatibilityIssue):
    """Ethos-U specific layer compatibility fact.

    Extends base LayerCompatibilityIssue with NPU placement information.
    """

    npu_placement: str = "unknown"  # 'npu', 'cpu', or 'unknown'


@register_fact_type(
    "ethos_u_layer_suboptimal_activation",
    "layer",
    "Layer uses suboptimal activation function on Ethos-U",
)
@dataclass
class EthosULayerSuboptimalActivation(LayerCompatibilityIssue):
    """Fact indicating a layer uses suboptimal activation function."""

    activation_type: str = "unknown"


@register_fact_type(
    "ethos_u_layer_high_network_share",
    "layer",
    "Layer has a high number of op cycles compared to others",
)
@dataclass
class EthosULayerHighNetworkShare(LayerPerformanceIssue):
    """Fact indicating a layer has high network share."""


@register_fact_type(
    "ethos_u_layer_high_op_cycles",
    "layer",
    "Layer has a high number of op cycles",
)
@dataclass
class EthosULayerHighOpCycles(LayerPerformanceIssue):
    """Fact indicating a layer has high op cycles."""


@register_fact_type(
    "ethos_u_layer_high_memory_pressure",
    "layer",
    "Layer op cycles dominated by memory access cycles",
)
@dataclass
class EthosULayerHighMemoryPressure(LayerPerformanceIssue):
    """Fact indicating memory access dominates layer execution time."""

    mem_to_npu_ratio: float


@register_fact_type(
    "ethos_u_layer_low_mac_util",
    "layer",
    "Layer has lower mac util than expected",
)
@dataclass
class EthosULayerLowMacUtil(LayerPerformanceIssue):
    """Fact indicating mac util is lower than expected for that op."""

    severity: str


@dataclass
class PerfMetricDiff:
    """Performance metric difference."""

    original_value: int | float
    optimized_value: int | float

    @property
    def diff(self) -> float:
        """Difference between metrics."""
        if self.original_value == 0:
            return 0

        return 100 - ((self.optimized_value / self.original_value) * 100)

    @property
    def improved(self) -> bool:
        """Return true if metric improved."""
        return self.diff > 0

    @property
    def degraded(self) -> bool:
        """Return true if metric degraded."""
        return self.diff < 0

    @property
    def same(self) -> bool:
        """Return true if metric stays the same."""
        return self.diff == 0


@dataclass
class OptimizationDiff:
    """Optimization performance impact."""

    opt_type: list[OptimizationSettings]
    opt_diffs: dict[str, PerfMetricDiff]


@dataclass
class OptimizationResults(Fact):
    """Optimization results."""

    diffs: list[OptimizationDiff]


@dataclass
class LutPatternRule:
    """Activation funciton operator patterns."""

    op_seq: list[str]
    op_label: str


PATTERN_RULES = [
    LutPatternRule(
        ["greater", "fully_connected", "exp", "sub", "mul", "select"], "SELU"
    ),
    LutPatternRule(["greater", "mul", "exp", "sub", "mul", "select"], "SELU"),
    LutPatternRule(["exp", "add", "log", "tanh", "mul"], "MISH"),
    LutPatternRule(["exp", "add", "log"], "SOFTPLUS"),
    LutPatternRule(["logistic", "mul"], "LOGISTIC BASED OPERATION"),
    LutPatternRule(["exp"], "EXPONENTIAL"),
    LutPatternRule(["gelu"], "GELU"),
    LutPatternRule(["elu"], "ELU"),
    LutPatternRule(["tanh"], "TANH"),
    LutPatternRule(["logistic"], "SIGMOID"),
    LutPatternRule(["softmax"], "SOFTMAX"),
]


class EthosUDataAnalyzer(FactExtractor):
    """Ethos-U data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:
        """Analyze the data."""
        logger.debug(
            "Unhandled data_item type: %s - %s.%s",
            type(data_item),
            data_item.__class__.__module__,
            data_item.__class__.__name__,
        )

    def _sequence_matches(self, ops: list, idx: int, function_ops: list[str]) -> bool:
        """Check operators sequence matches activation function operator pattern."""
        if idx + len(function_ops) > len(ops):
            return False
        for i, function_op in enumerate(function_ops):
            # Check compatibility operators sequentially against activation function
            if not ops[idx + i].op_type.lower() == function_op:
                return False
        return True

    def _analyze_activation_function(self, operators: Operators) -> None:
        """Analyze operators from compatibility check for activation functions."""
        ops = operators.ops
        idx = 0
        while idx < len(ops):
            # Check if activation function pattern occers
            for rule in PATTERN_RULES:
                if self._sequence_matches(ops, idx, rule.op_seq):
                    if any(
                        op.run_on_npu.supported is None
                        for op in ops[idx : idx + len(rule.op_seq)]
                    ):
                        idx += 1
                        break
                    operator = ops[idx]
                    lut_fact = EthosULayerSuboptimalActivation(
                        operator_name=operator.name,
                        location=operator.identity.entity_id,
                        operator_type=operator.op_type,
                        is_supported=operator.run_on_npu.supported,
                        reasons=operator.run_on_npu.reasons,
                        activation_type=rule.op_label,
                    )
                    self.add_fact(lut_fact)
                    # Skip to next operation not in detected pattern
                    idx += len(rule.op_seq)
                    break
            else:
                idx += 1

    @analyze_data.register
    def analyze_operator_compatibility(self, operators: Operators) -> None:
        """Analyse operator compatibility information."""
        # With no analyzed operators, there are no compatibility facts to produce.
        # Return before the empty list's unsupported ratio of 1.0 is mistaken for
        # evidence that every operator is unsupported.
        if operators.total_number == 0:
            return

        self._analyze_activation_function(operators)
        operator_advice: list[VelaOperatorAdvice] = []
        for operator in operators.ops:
            support = operator.run_on_npu.supported
            if support is None:
                continue
            # Determine NPU placement
            if operator.cpu_only:
                npu_placement = "cpu"
            elif support:
                npu_placement = "npu"
            else:
                npu_placement = "unknown"

            # Create layer compatibility fact
            layer_fact = EthosULayerCompatibilityIssue(
                operator_name=operator.name,
                location=operator.identity.entity_id,
                operator_type=operator.op_type,
                is_supported=support,
                reasons=operator.run_on_npu.reasons,
                npu_placement=npu_placement,
            )
            self.add_fact(layer_fact)

            if support:
                continue

            messages = dict.fromkeys(
                " — ".join(
                    part for part in (description.strip(), detail.strip()) if part
                )
                for description, detail in operator.run_on_npu.confirmed_reasons
                if description.strip() or detail.strip()
            )
            for message in messages:
                operator_advice.append(
                    VelaOperatorAdvice(operator.identity.entity_id, message)
                )

        # Keep network-level facts for backward compatibility
        cpu_only_operators = [
            operator for operator in operators.ops if operator.cpu_only
        ]
        cpu_only = [operator.op_type for operator in cpu_only_operators]
        if cpu_only:
            self.add_fact(
                HasCPUOnlyOperators(
                    cpu_only,
                    [operator.identity.entity_id for operator in cpu_only_operators],
                )
            )

        unsupported_ratio = operators.npu_unsupported_ratio
        if unsupported_ratio is not None and unsupported_ratio != 0:
            self.add_fact(HasUnsupportedOnNPUOperators(unsupported_ratio))

        if unsupported_ratio == 0:
            self.add_fact(AllOperatorsSupportedOnNPU())

        for advice in operator_advice:
            self.add_fact(advice)

        self._add_catalog_advice(operators)

    def _add_catalog_advice(self, operators: Operators) -> None:
        """Add possible catalog constraints for rejected, listed operators."""
        profile = operators.support_profile
        version = operators.support_catalog_version
        if profile is None or version is None:
            return

        rejected = [
            (operator, constraints)
            for operator in operators.ops
            if operator.run_on_npu.supported is False
            if (constraints := operators.catalog_constraints_for(operator.op_type))
            is not None
        ]
        if not rejected:
            return

        if profile.generic_constraints:
            constraints_text = "\n".join(
                _format_catalog_constraints(profile.generic_constraints)
            )
            self.add_fact(
                VelaCatalogAdvice(
                    f"Vela {version} profile-wide support conditions for rejected "
                    "operators (possible constraints, not confirmed causes):\n"
                    f"{constraints_text}",
                    [operator.identity.entity_id for operator, _ in rejected],
                )
            )

        by_type: dict[str, tuple[tuple[Constraint, ...], list[str]]] = {}
        for operator, constraints in rejected:
            if not constraints:
                continue
            if operator.op_type not in by_type:
                by_type[operator.op_type] = (constraints, [])
            by_type[operator.op_type][1].append(operator.identity.entity_id)

        for op_type, (constraints, entity_ids) in by_type.items():
            constraints_text = "\n".join(_format_catalog_constraints(constraints))
            self.add_fact(
                VelaCatalogAdvice(
                    f"Vela {version} support conditions for {op_type} "
                    "(possible constraints, not confirmed causes):\n"
                    f"{constraints_text}",
                    entity_ids,
                )
            )

    @analyze_data.register
    def analyze_optimization_results(
        self, optimization_results: OptimizationPerformanceMetrics
    ) -> None:
        """Analyse optimization performance metrics."""
        optimizations = optimization_results.optimizations_perf_metrics
        if not optimizations:
            return

        orig = optimization_results.original_perf_metrics
        orig_memory = orig.memory_usage
        orig_cycles = orig.npu_cycles

        diffs: list[OptimizationDiff] = []
        for opt_type, opt_perf_metrics in optimizations:
            opt = opt_perf_metrics
            opt_memory = opt.memory_usage
            opt_cycles = opt.npu_cycles

            opt_diffs: dict[str, PerfMetricDiff] = {}

            if orig_memory and opt_memory:
                opt_diffs.update(
                    {
                        "sram": PerfMetricDiff(
                            orig_memory.sram_memory_area_size,
                            opt_memory.sram_memory_area_size,
                        ),
                        "dram": PerfMetricDiff(
                            orig_memory.dram_memory_area_size,
                            opt_memory.dram_memory_area_size,
                        ),
                        "on_chip_flash": PerfMetricDiff(
                            orig_memory.on_chip_flash_memory_area_size,
                            opt_memory.on_chip_flash_memory_area_size,
                        ),
                        "off_chip_flash": PerfMetricDiff(
                            orig_memory.off_chip_flash_memory_area_size,
                            opt_memory.off_chip_flash_memory_area_size,
                        ),
                    }
                )
            if orig_cycles and opt_cycles:
                opt_diffs["npu_total_cycles"] = PerfMetricDiff(
                    orig_cycles.npu_total_cycles,
                    opt_cycles.npu_total_cycles,
                )

            diff = OptimizationDiff(opt_type=opt_type, opt_diffs=opt_diffs)
            diffs.append(diff)

        self.add_fact(OptimizationResults(diffs))

    @analyze_data.register
    def analyze_tflite_compatibility(self, data_item: TFLiteCompatibilityInfo) -> None:
        """Analyze TensorFlow Lite compatibility information."""
        analyze_tflite_compatibility_common(self, data_item)

    @analyze_data.register
    def analyze_combined_performance_result(
        self, data_item: CombinedPerformanceResult
    ) -> None:
        """Analyze combined performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            logger.debug(
                "No standardized output for: %s - %s.%s",
                type(data_item),
                data_item.__class__.__module__,
                data_item.__class__.__name__,
            )
            return
        corstone_result = 1
        result = standard_out["results"][corstone_result]
        breakdowns = self.resolve_breakdown_entities(result)
        self.analyze_network_share(breakdowns)
        self.analyze_memory_pressure(breakdowns)
        self.analyze_mac_util(breakdowns)

    @analyze_data.register
    def analyze_corstone_performance_result(
        self, data_item: CorstonePerformanceResult
    ) -> None:
        """Analyze Corstone-only performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            logger.debug(
                "No standardized output for: %s - %s.%s",
                type(data_item),
                data_item.__class__.__module__,
                data_item.__class__.__name__,
            )
            return
        corstone_result = 0
        result = standard_out["results"][corstone_result]
        breakdowns = self.resolve_breakdown_entities(result)
        if breakdowns:
            self.analyze_network_share(breakdowns)
            self.analyze_memory_pressure(breakdowns)
            self.analyze_mac_util(breakdowns)

    @analyze_data.register
    def analyze_vela_performance_result(self, data_item: VelaPerformanceResult) -> None:
        """Analyze combined performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            logger.debug(
                "No standardized output for: %s - %s.%s",
                type(data_item),
                data_item.__class__.__module__,
                data_item.__class__.__name__,
            )
            return
        vela_result = 0
        result = standard_out["results"][vela_result]
        breakdowns = self.resolve_breakdown_entities(result)
        self.analyze_op_cycles(breakdowns)
        self.analyze_memory_pressure(breakdowns)
        self.analyze_mac_util(breakdowns)

    def resolve_breakdown_entities(self, result: dict) -> list[dict]:
        """Return breakdowns enriched with their source entity data."""
        entities_by_id = {
            entity["id"]: entity
            for entity in result.get("entities", [])
            if isinstance(entity, dict) and "id" in entity
        }
        resolved = []
        for breakdown in result.get("breakdowns", []):
            entity_id = breakdown.get("entity_id", "")
            entity = entities_by_id.get(entity_id)
            if entity is None:
                logger.debug(
                    "Ignoring performance breakdown for unknown entity: %s",
                    entity_id,
                )
                continue
            resolved.append(
                {
                    **breakdown,
                    "name": entity.get("name", entity_id),
                    "entity_id": entity_id,
                }
            )
        return resolved

    def layer_locations_text(self, layer: dict) -> str:
        """Return the complete standardized entity ID for display/reference."""
        return str(layer.get("entity_id", ""))

    def get_metric_tup(
        self, layer: dict, metric_names: list[str]
    ) -> tuple[str, float, str] | None:
        """Return the first matching metric as (name, float(value), unit)."""
        metrics = layer["metrics"]
        for metric in metrics:
            name = metric["name"]
            if name in metric_names:
                try:
                    return (metric["name"], float(metric["value"]), metric["unit"])
                except (TypeError, ValueError):
                    return None
        return None

    METRIC_ALIASES = {
        "network_share": ["network_share", "network_cycles", "network"],
        "op_cycles": ["op_cycles"],
        "npu_cycles": ["npu_cycles", "npu"],
        "sram_cycles": ["sram_access_cycles", "sram_ac"],
        "dram_cycles": ["dram_access_cycles", "dram_ac"],
        "on_flash_cycles": ["on_chip_flash_access_cycles", "onflash_ac"],
        "off_flash_cycles": ["off_chip_flash_access_cycles", "offflash_ac"],
        "mac_count": ["mac_count"],
        "mac_util": ["util_mac_percentage", "util_mac"],
        "sram_usage": ["sram_usage", "staging_usage"],
    }
    RELATIVE_FLOOR = 0.10
    MAX_ITEMS = 10

    def analyze_network_share(self, breakdowns: list) -> None:
        """Detect layers with largest network share."""
        relative_floor = self.RELATIVE_FLOOR
        max_items = self.MAX_ITEMS

        # Collect (network_share_pct, layer) pairs for easy sorting
        layers_scored = []
        for layer in breakdowns:
            metric = self.get_metric_tup(layer, self.METRIC_ALIASES["network_share"])
            if metric is not None:
                layers_scored.append((metric[1], layer))

        if not layers_scored:
            return

        value = 0
        layers_scored.sort(key=lambda x: x[value], reverse=True)

        # Determine threshold relative to highest share
        highest_op = 0
        max_pct = layers_scored[highest_op][value]
        threshold = max_pct * relative_floor

        selected = [(pct, layer) for pct, layer in layers_scored if pct >= threshold]
        selected = selected[:max_items]

        for pct, layer in selected:
            layer_fact = EthosULayerHighNetworkShare(
                operator_name=layer["name"],
                location=self.layer_locations_text(layer),
                metric="network_share",
                metric_value=pct,
                metric_unit="%",
            )
            self.add_fact(layer_fact)

    def analyze_op_cycles(self, breakdowns: list) -> None:
        """Detect the top 10 layers with the highest op cycles."""
        max_items = self.MAX_ITEMS

        layers_scored = []
        for layer in breakdowns:
            metric = self.get_metric_tup(layer, self.METRIC_ALIASES["op_cycles"])
            if metric is None:
                continue

            metric_name, op_cycles, metric_unit = metric
            layers_scored.append((op_cycles, metric_name, metric_unit, layer))

        if not layers_scored:
            return

        # sort by op_cycles, high to low
        value = 0
        layers_scored.sort(key=lambda item: item[value], reverse=True)

        # only emit fact for the the top MAX_ITEMS
        for op_cycles, metric_name, metric_unit, layer in layers_scored[:max_items]:
            layer_fact = EthosULayerHighOpCycles(
                operator_name=layer["name"],
                location=self.layer_locations_text(layer),
                metric=metric_name,
                metric_value=op_cycles,
                metric_unit=metric_unit,
            )
            self.add_fact(layer_fact)

    def analyze_memory_pressure(self, breakdowns: list) -> None:
        """Detect if op cycles is memory dominated."""
        for layer in breakdowns:
            try:
                npu_metric = self.get_metric_tup(
                    layer, self.METRIC_ALIASES["npu_cycles"]
                )
                if npu_metric is None:
                    continue

                npu_cycles = npu_metric[1]
                # get sram_ac, dram_ac, onfalsh_ac, offflash_ac
                candidates = [
                    self.get_metric_tup(layer, self.METRIC_ALIASES["sram_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["dram_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["on_flash_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["off_flash_cycles"]),
                ]
                valid_candidates: list[tuple[str, float, str]] = [
                    candidate for candidate in candidates if candidate is not None
                ]
                if not valid_candidates:
                    continue

                max_name, max_val, max_unit = max(valid_candidates, key=lambda t: t[1])

                if npu_cycles and (max_val / npu_cycles) > 1:
                    layer_fact = EthosULayerHighMemoryPressure(
                        operator_name=layer["name"],
                        location=self.layer_locations_text(layer),
                        metric=max_name,
                        metric_value=max_val,
                        metric_unit=max_unit,
                        mem_to_npu_ratio=max_val / npu_cycles,
                    )
                    self.add_fact(layer_fact)

            except (KeyError, IndexError, TypeError, ValueError):
                continue

    def analyze_mac_util(self, breakdowns: list) -> None:
        """Detect if mac util is lower then expected."""
        for layer in breakdowns:
            mac_util_metric = self.get_metric_tup(
                layer, self.METRIC_ALIASES["mac_util"]
            )
            if mac_util_metric is None:
                continue

            severity = None

            if layer["name"] == "Conv2D" and mac_util_metric[1] < 30:
                severity = "very low"
            elif layer["name"] == "Conv2D" and mac_util_metric[1] < 55:
                severity = "low"
            elif layer["name"] == "DepthwiseConv2D" and mac_util_metric[1] < 3:
                severity = "low"

            metric, value, unit = mac_util_metric

            if severity is not None:
                layer_fact = EthosULayerLowMacUtil(
                    operator_name=layer["name"],
                    location=self.layer_locations_text(layer),
                    metric=metric,
                    metric_value=value,
                    metric_unit=unit,
                    severity=severity,
                )
                self.add_fact(layer_fact)
