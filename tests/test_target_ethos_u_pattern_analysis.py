# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U pattern analysis tests."""

from dataclasses import dataclass

import pytest
from mlia.core.data_analysis import Fact
from mlia.target.ethos_u.data_analysis import (
    EthosULayerHighMemoryPressure,
    EthosULayerHighNetworkShare,
    EthosULayerLowMacUtil,
    EthosULayerSuboptimalActivation,
)
from mlia.target.ethos_u.pattern_analysis import (
    ActivationFunctionPatternAnalyzer,
    IneffectiveActivationPattern,
    LayerHotSpotPatternAnalyzer,
    LowMacUtilLayersPattern,
    MemoryBoundLayersPattern,
)


class BrokenSuboptimalActivation(EthosULayerSuboptimalActivation):
    """Suboptimal activation fact that raises when accessing location."""

    def __getattribute__(self, name):  # type: ignore[override]
        if name == "location":
            raise AttributeError("location attribute unavailable")
        return super().__getattribute__(name)


@dataclass(frozen=True)
class AnalyzePatternsCase:
    """Test case for ActivationFunctionPatternAnalyzer.analyze_patterns."""

    facts: list[object]
    expected_patterns: list[IneffectiveActivationPattern]


@pytest.mark.parametrize(
    "affected_layers, activation_types",
    [
        pytest.param(
            [],
            [],
            id="correct with both lists empty",
        ),
        pytest.param(
            ["thing1", "thing2"],
            [],
            id="correct with one lists empty",
        ),
        pytest.param(
            ["thing1", "thing2"],
            ["thing1", "thing2"],
            id="correct with both lists",
        ),
    ],
)
def test_ethos_u_ineffective_activation_pattern_to_dict(
    affected_layers: list[str],
    activation_types: list[str],
) -> None:
    activation_pattern = IneffectiveActivationPattern(
        affected_layers=affected_layers,
        activation_types=activation_types,
    )

    got = activation_pattern.to_dict()

    assert got["affected_layers"] == affected_layers
    assert got["activation_types"] == activation_types


def test_has_already_generated_patterns_returns_false_without_pattern() -> None:
    analyzer = ActivationFunctionPatternAnalyzer()
    facts: list[object] = [object(), object()]

    assert analyzer.has_already_generated_patterns(facts) is False


def test_has_already_generated_patterns_returns_true_with_pattern() -> None:
    analyzer = ActivationFunctionPatternAnalyzer()
    facts = [object(), IneffectiveActivationPattern()]

    assert analyzer.has_already_generated_patterns(facts) is True


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            AnalyzePatternsCase(
                facts=[
                    IneffectiveActivationPattern(
                        affected_layers=["layer1 (operator/0)"],
                        layer_count=1,
                        activation_types=["TANH"],
                        recommendation=(
                            "Consider replacing suboptimal activation functions with "
                            "NPU-friendly alternatives."
                        ),
                    )
                ],
                expected_patterns=[],
            ),
            id="already generated patterns",
        ),
        pytest.param(
            AnalyzePatternsCase(
                facts=[object(), object()],
                expected_patterns=[],
            ),
            id="no bad facts",
        ),
        pytest.param(
            AnalyzePatternsCase(
                facts=[
                    EthosULayerSuboptimalActivation(
                        operator_name="conv1",
                        location="operator/0",
                        operator_type="CONV_2D",
                        is_supported=True,
                        reasons=[],
                        activation_type="TANH",
                    ),
                    EthosULayerSuboptimalActivation(
                        operator_name="conv2",
                        location="operator/1",
                        operator_type="CONV_2D",
                        is_supported=True,
                        reasons=[],
                        activation_type="ELU",
                    ),
                ],
                expected_patterns=[
                    IneffectiveActivationPattern(
                        affected_layers=[
                            "conv1 (operator/0)",
                            "conv2 (operator/1)",
                        ],
                        layer_count=2,
                        activation_types=["ELU", "TANH"],
                        recommendation=(
                            "Consider replacing suboptimal activation functions with "
                            "NPU-friendly alternatives."
                        ),
                    )
                ],
            ),
            id="pattern created-no unsupported",
        ),
        pytest.param(
            AnalyzePatternsCase(
                facts=[
                    EthosULayerSuboptimalActivation(
                        operator_name="mish_layer",
                        location="operator/0",
                        operator_type="CONV_2D",
                        is_supported=True,
                        reasons=[],
                        activation_type="MISH",
                    ),
                    EthosULayerSuboptimalActivation(
                        operator_name="tanh_layer",
                        location="operator/1",
                        operator_type="CONV_2D",
                        is_supported=True,
                        reasons=[],
                        activation_type="TANH",
                    ),
                    EthosULayerSuboptimalActivation(
                        operator_name="selu_layer",
                        location="operator/2",
                        operator_type="CONV_2D",
                        is_supported=True,
                        reasons=[],
                        activation_type="SELU",
                    ),
                ],
                expected_patterns=[
                    IneffectiveActivationPattern(
                        affected_layers=[
                            "mish_layer (operator/0)",
                            "tanh_layer (operator/1)",
                            "selu_layer (operator/2)",
                        ],
                        layer_count=3,
                        activation_types=["MISH", "SELU", "TANH"],
                        recommendation=(
                            "Consider replacing suboptimal activation functions with "
                            "NPU-friendly alternatives.\n"
                            "The following activation functions use unsupported "
                            "operations: MISH, SELU"
                        ),
                    )
                ],
            ),
            id="unsupported present",
        ),
    ],
)
def test_analyze_patterns_returns_expected_patterns(
    case: AnalyzePatternsCase,
) -> None:
    analyzer = ActivationFunctionPatternAnalyzer()

    assert analyzer.analyze_patterns(case.facts) == case.expected_patterns


def test_analyze_patterns_expected_exception() -> None:
    fact = [
        BrokenSuboptimalActivation(
            operator_name="broken_layer",
            location="operator/0",
            operator_type="BROKEN",
            is_supported=True,
            reasons=[],
            activation_type="MISH",
        )
    ]
    analyzer = ActivationFunctionPatternAnalyzer()

    with pytest.raises(AttributeError):
        analyzer.analyze_patterns(fact)


@pytest.mark.parametrize(
    "facts, pattern_type, expected_facts",
    [
        (
            [
                EthosULayerHighNetworkShare(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="network_share",
                    metric_value=60,
                    metric_unit="%",
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                ),
                EthosULayerLowMacUtil(
                    operator_name="DepthwiseConv2D",
                    location="operator/1",
                    metric="util_mac_percentage",
                    metric_value=2,
                    metric_unit="%",
                    severity="low",
                ),
                Fact(),
            ],
            LowMacUtilLayersPattern,
            [
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                )
            ],
        ),
        (
            [
                EthosULayerHighNetworkShare(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="network_share",
                    metric_value=60,
                    metric_unit="%",
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=120,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.2,
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="DepthwiseConv2D",
                    location="operator/1",
                    metric="dram_access_cycles",
                    metric_value=130,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.3,
                ),
                Fact(),
            ],
            MemoryBoundLayersPattern,
            [
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=120,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.2,
                )
            ],
        ),
    ],
    ids=["low-mac-hotspots", "memory-bound-hotspots"],
)
def test_layer_hotspot_pattern_analyzer_groups_only_hotspots(
    facts: list[Fact],
    pattern_type: type[LowMacUtilLayersPattern | MemoryBoundLayersPattern],
    expected_facts: list[Fact],
) -> None:
    """Test hotspot grouping keeps only facts that satisfy the prerequisite chain."""

    analyzer = LayerHotSpotPatternAnalyzer()
    patterns = analyzer.analyze_patterns(facts)

    matching_patterns = [
        pattern for pattern in patterns if isinstance(pattern, pattern_type)
    ]

    assert len(matching_patterns) == 1
    assert matching_patterns[0].layer_count == len(expected_facts)
    assert matching_patterns[0].facts == expected_facts
