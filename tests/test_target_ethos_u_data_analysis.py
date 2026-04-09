# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U data analysis module."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compat import (
    NpuSupported,
    Operator,
    Operators,
    VelaCompatibilityResult,
)
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.target.ethos_u.common_reporters import (
    ModelHasCustomOperators,
    ModelIsNotTFLiteCompatible,
    TFLiteCompatibilityCheckFailed,
)
from mlia.target.ethos_u.utils.tflite_shims import (
    TFLiteCompatibilityInfo,
    TFLiteCompatibilityStatus,
    TFLiteConversionError,
    TFLiteConversionErrorCode,
)
from mlia.target.ethos_u.data_analysis import (
    AllOperatorsSupportedOnNPU,
    EthosUDataAnalyzer,
    EthosULayerCompatibilityIssue,
    EthosULayerSuboptimalActivation,
    EthosULayerHighMemoryPressure,
    EthosULayerHighNetworkShare,
    EthosULayerHighOpCycles,
    EthosULayerLowMacUtil,
    HasCPUOnlyOperators,
    HasUnsupportedOnNPUOperators,
    PerfMetricDiff,
    OptimizationResults,
)
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.performance import (
    MemoryUsage,
    NPUCycles,
    OptimizationPerformanceMetrics,
    PerformanceMetrics,
)
from mlia.target.ethos_u.performance import (
    CombinedPerformanceResult,
    CorstonePerformanceResult,
    VelaPerformanceResult,
)


def _fact_payload(fact: Fact) -> object:
    if is_dataclass(fact):
        return asdict(fact)
    return fact


@pytest.mark.parametrize(
    "input_data, expected_facts",
    [
        [
            Operators(
                [
                    Operator(
                        "CPU operator",
                        "CPU operator type",
                        NpuSupported(False, [("CPU only operator", "")]),
                    )
                ]
            ),
            [
                EthosULayerCompatibilityIssue(
                    operator_name="CPU operator",
                    location="operator/0",
                    operator_type="CPU operator type",
                    is_supported=False,
                    reasons=[("CPU only operator", "")],
                    npu_placement="cpu",
                ),
                HasCPUOnlyOperators(["CPU operator type"]),
                HasUnsupportedOnNPUOperators(1.0),
            ],
        ],
        [
            Operators(
                [
                    Operator(
                        "NPU operator",
                        "NPU operator type",
                        NpuSupported(True, []),
                    )
                ]
            ),
            [
                EthosULayerCompatibilityIssue(
                    operator_name="NPU operator",
                    location="operator/0",
                    operator_type="NPU operator type",
                    is_supported=True,
                    reasons=[],
                    npu_placement="npu",
                ),
                AllOperatorsSupportedOnNPU(),
            ],
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.COMPATIBLE),
            [],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR
            ),
            [ModelHasCustomOperators()],
        ],
        [
            TFLiteCompatibilityInfo(status=TFLiteCompatibilityStatus.UNKNOWN_ERROR),
            [TFLiteCompatibilityCheckFailed()],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR
            ),
            [ModelIsNotTFLiteCompatible(custom_ops=[], flex_ops=[])],
        ],
        [
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_errors=[
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        "custom_op1",
                        [],
                    ),
                    TFLiteConversionError(
                        "error",
                        TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                        "flex_op1",
                        [],
                    ),
                ],
            ),
            [
                ModelIsNotTFLiteCompatible(
                    custom_ops=["custom_op1"],
                    flex_ops=["flex_op1"],
                )
            ],
        ],
    ],
)
def test_ethos_u_data_analyzer(
    input_data: DataItem, expected_facts: list[Fact]
) -> None:
    """Test Ethos-U data analyzer."""

    analyzer = EthosUDataAnalyzer()
    analyzer.analyze_data(input_data)
    assert [_fact_payload(fact) for fact in analyzer.get_analyzed_data()] == [
        _fact_payload(fact) for fact in expected_facts
    ]


def test_perf_metric_diff_non_zero_original() -> None:
    """Test PerfMetricDiff.diff with non-zero original value."""

    diff = PerfMetricDiff(original_value=100, optimized_value=80)

    assert diff.diff == pytest.approx(20.0)


def test_perf_metric_diff_zero_original() -> None:
    """Test PerfMetricDiff.diff when original value is zero."""

    diff = PerfMetricDiff(original_value=0, optimized_value=42)

    assert diff.diff == 0


def _make_performance_metrics(
    memory: MemoryUsage | None, cycles: NPUCycles | None
) -> PerformanceMetrics:
    """Create PerformanceMetrics instance for tests."""
    _TEST_TARGET_CONFIG = MagicMock(spec=EthosUConfiguration)
    return PerformanceMetrics(
        target_config=_TEST_TARGET_CONFIG,
        npu_cycles=cycles,
        memory_usage=memory,
        layerwise_perf_info=None,
    )


def test_analyze_optimization_results_no_optimizations() -> None:
    """Test that no facts are added when there are no optimizations."""

    analyzer = EthosUDataAnalyzer()

    original_metrics = _make_performance_metrics(
        memory=MemoryUsage(1, 2, 3, 4),
        cycles=NPUCycles(1, 2, 3, 4, 5, 6),
    )

    optimization_results = OptimizationPerformanceMetrics(
        original_perf_metrics=original_metrics,
        optimizations_perf_metrics=[],
    )

    analyzer.analyze_data(optimization_results)

    assert analyzer.get_analyzed_data() == []


def test_analyze_optimization_results_with_memory_and_cycles() -> None:
    """Test optimization analysis when both memory and cycles are present."""

    analyzer = EthosUDataAnalyzer()

    original_memory = MemoryUsage(100, 200, 300, 400)
    original_cycles = NPUCycles(1, 2, 3, 4, 5, 6)
    original_metrics = _make_performance_metrics(
        memory=original_memory,
        cycles=original_cycles,
    )

    optimized_memory = MemoryUsage(50, 100, 150, 200)
    optimized_cycles = NPUCycles(1, 2, 2, 4, 5, 6)
    optimized_metrics = _make_performance_metrics(
        memory=optimized_memory,
        cycles=optimized_cycles,
    )

    opt_settings = [OptimizationSettings("pruning", 0.5, [])]

    optimization_results = OptimizationPerformanceMetrics(
        original_perf_metrics=original_metrics,
        optimizations_perf_metrics=[(opt_settings, optimized_metrics)],
    )

    analyzer.analyze_data(optimization_results)

    facts = analyzer.get_analyzed_data()
    assert len(facts) == 1
    assert isinstance(facts[0], OptimizationResults)

    result = facts[0]
    assert len(result.diffs) == 1

    diff = result.diffs[0]
    assert diff.opt_type == opt_settings

    expected_keys = {
        "sram",
        "dram",
        "on_chip_flash",
        "off_chip_flash",
        "npu_total_cycles",
    }
    assert set(diff.opt_diffs.keys()) == expected_keys

    sram_diff = diff.opt_diffs["sram"]
    assert sram_diff.original_value == original_memory.sram_memory_area_size
    assert sram_diff.optimized_value == optimized_memory.sram_memory_area_size

    cycles_diff = diff.opt_diffs["npu_total_cycles"]
    assert cycles_diff.original_value == original_cycles.npu_total_cycles
    assert cycles_diff.optimized_value == optimized_cycles.npu_total_cycles


def _make_npu_supported_operator(name: str, op_type: str) -> Operator:
    """Helper to create an NPU-supported operator."""

    return Operator(
        name=name,
        op_type=op_type,
        run_on_npu=NpuSupported(True, []),
    )


def test_analyze_activation_function_detects_suboptimal_activation() -> None:
    """Test that suboptimal activation patterns are detected."""

    analyzer = EthosUDataAnalyzer()

    # First operator does not start a pattern; second starts a MISH pattern.
    ops = [
        _make_npu_supported_operator("conv", "CONV_2D"),
        _make_npu_supported_operator("exp_op", "EXP"),
        _make_npu_supported_operator("add_op", "ADD"),
        _make_npu_supported_operator("log_op", "LOG"),
        _make_npu_supported_operator("tanh_op", "TANH"),
        _make_npu_supported_operator("mul_op", "MUL"),
    ]

    vela_result = VelaCompatibilityResult(legacy_info=Operators(ops=ops))

    analyzer.analyze_data(vela_result)

    facts = analyzer.get_analyzed_data()
    suboptimal_facts = [
        fact for fact in facts if isinstance(fact, EthosULayerSuboptimalActivation)
    ]

    assert len(suboptimal_facts) == 1
    fact = suboptimal_facts[0]

    assert fact.location == "operator/1"
    assert fact.operator_type == "EXP"
    assert fact.activation_type == "MISH"
    assert fact.is_supported is True
    assert fact.reasons == []


def test_analyze_activation_function_no_matching_pattern() -> None:
    """Test that no facts are added when no pattern matches."""

    analyzer = EthosUDataAnalyzer()

    ops = [
        _make_npu_supported_operator("conv1", "CONV_2D"),
        _make_npu_supported_operator("pool1", "MAX_POOL"),
    ]

    vela_result = VelaCompatibilityResult(legacy_info=Operators(ops=ops))

    analyzer.analyze_data(vela_result)

    facts = analyzer.get_analyzed_data()
    assert not any(isinstance(fact, EthosULayerSuboptimalActivation) for fact in facts)


@pytest.mark.parametrize(
    "input_data, expected_facts, excluded_fact_types",
    [
        (
            CorstonePerformanceResult(
                legacy_info=MagicMock(),
                standardized_output={
                    "results": [
                        {
                            "breakdowns": [
                                {
                                    "name": "Conv2D",
                                    "location": "operator/0",
                                    "metrics": [
                                        {
                                            "name": "npu_cycles",
                                            "value": 100,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "dram_access_cycles",
                                            "value": 150,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "util_mac_percentage",
                                            "value": 20,
                                            "unit": "%",
                                        },
                                        {
                                            "name": "network_share",
                                            "value": 45,
                                            "unit": "%",
                                        },
                                    ],
                                },
                                {
                                    "name": "Add",
                                    "location": "operator/1",
                                    "metrics": [
                                        {
                                            "name": "npu_cycles",
                                            "value": 120,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "dram_access_cycles",
                                            "value": 80,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "util_mac_percentage",
                                            "value": 90,
                                            "unit": "%",
                                        },
                                        {
                                            "name": "network_share",
                                            "value": 4,
                                            "unit": "%",
                                        },
                                    ],
                                },
                            ]
                        }
                    ]
                },
            ),
            [
                EthosULayerHighNetworkShare(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="network_share",
                    metric_value=45,
                    metric_unit="%",
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=150,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.5,
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                ),
            ],
            (),
        ),
        (
            CombinedPerformanceResult(
                legacy_info=MagicMock(),
                standardized_output={
                    "results": [
                        {
                            "breakdowns": [
                                {
                                    "name": "VelaConv2D",
                                    "location": "operator/0",
                                    "metrics": [
                                        {
                                            "name": "op_cycles",
                                            "value": 999,
                                            "unit": "cycles",
                                        }
                                    ],
                                }
                            ]
                        },
                        {
                            "breakdowns": [
                                {
                                    "name": "CorstoneConv2D",
                                    "location": "operator/1",
                                    "metrics": [
                                        {"name": "unused_0", "value": 0, "unit": ""},
                                        {"name": "unused_1", "value": 0, "unit": ""},
                                        {"name": "unused_2", "value": 0, "unit": ""},
                                        {
                                            "name": "network_share",
                                            "value": 60,
                                            "unit": "%",
                                        },
                                    ],
                                }
                            ]
                        },
                    ]
                },
            ),
            [
                EthosULayerHighNetworkShare(
                    operator_name="CorstoneConv2D",
                    location="operator/1",
                    metric="network_share",
                    metric_value=60,
                    metric_unit="%",
                )
            ],
            (EthosULayerHighOpCycles,),
        ),
        (
            VelaPerformanceResult(
                legacy_info=MagicMock(),
                standardized_output={
                    "results": [
                        {
                            "breakdowns": [
                                {
                                    "name": "Conv2D",
                                    "location": "operator/0",
                                    "metrics": [
                                        {
                                            "name": "op_cycles",
                                            "value": 500,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "npu_cycles",
                                            "value": 100,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "dram_access_cycles",
                                            "value": 120,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "util_mac_percentage",
                                            "value": 25,
                                            "unit": "%",
                                        },
                                    ],
                                },
                                {
                                    "name": "DepthwiseConv2D",
                                    "location": "operator/1",
                                    "metrics": [
                                        {
                                            "name": "op_cycles",
                                            "value": 100,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "npu_cycles",
                                            "value": 100,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "dram_access_cycles",
                                            "value": 50,
                                            "unit": "cycles",
                                        },
                                        {
                                            "name": "util_mac_percentage",
                                            "value": 5,
                                            "unit": "%",
                                        },
                                    ],
                                },
                            ]
                        }
                    ]
                },
            ),
            [
                EthosULayerHighOpCycles(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="op_cycles",
                    metric_value=500,
                    metric_unit="cycles",
                ),
                EthosULayerHighOpCycles(
                    operator_name="DepthwiseConv2D",
                    location="operator/1",
                    metric="op_cycles",
                    metric_value=100,
                    metric_unit="cycles",
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=120,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.2,
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=25,
                    metric_unit="%",
                    severity="very low",
                ),
            ],
            (),
        ),
    ],
    ids=["corstone", "combined-uses-corstone", "vela"],
)
def test_ethos_u_data_analyzer_performance_results(
    input_data: DataItem,
    expected_facts: list[Fact],
    excluded_fact_types: tuple[type[Fact], ...],
) -> None:
    """Test performance analysis facts across supported result types."""

    analyzer = EthosUDataAnalyzer()
    analyzer.analyze_data(input_data)

    facts = analyzer.get_analyzed_data()
    assert [_fact_payload(fact) for fact in facts] == [
        _fact_payload(fact) for fact in expected_facts
    ]
    for excluded_fact_type in excluded_fact_types:
        assert not any(isinstance(fact, excluded_fact_type) for fact in facts)
