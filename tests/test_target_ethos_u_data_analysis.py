# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U data analysis module."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compat import NpuSupported, Operator, Operators
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
    EthosULayerHighMemoryPressure,
    EthosULayerHighNetworkShare,
    EthosULayerHighOpCycles,
    EthosULayerLowMacUtil,
    HasCPUOnlyOperators,
    HasUnsupportedOnNPUOperators,
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
