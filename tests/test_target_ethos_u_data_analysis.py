# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U data analysis module."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pytest

from mlia.backend.vela.compat import NpuSupported, Operator, Operators
from mlia.core.common import DataItem
from mlia.core.data_analysis import Fact
from mlia.target.ethos_u.common_reporters import (
    ModelHasCustomOperators,
    ModelIsNotTFLiteCompatible,
    TFLiteCompatibilityCheckFailed,
)
from mlia.target.ethos_u.tflite_shims import (
    TFLiteCompatibilityInfo,
    TFLiteCompatibilityStatus,
    TFLiteConversionError,
    TFLiteConversionErrorCode,
)
from mlia.target.ethos_u.data_analysis import (
    AllOperatorsSupportedOnNPU,
    EthosUDataAnalyzer,
    EthosULayerCompatibilityIssue,
    HasCPUOnlyOperators,
    HasUnsupportedOnNPUOperators,
)


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

    def _fact_payload(fact: Fact) -> object:
        if is_dataclass(fact):
            return asdict(fact)
        return fact

    analyzer = EthosUDataAnalyzer()
    analyzer.analyze_data(input_data)
    assert [_fact_payload(fact) for fact in analyzer.get_analyzed_data()] == [
        _fact_payload(fact) for fact in expected_facts
    ]
