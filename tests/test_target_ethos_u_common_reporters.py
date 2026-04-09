# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for common Ethos-U reporters."""

from __future__ import annotations

from typing import Any

import pytest
from mlia.core.output_schema import AdviceCategory as SchemaAdviceCategory
from mlia.core.output_schema import AdviceSeverity
from mlia.core.reporting import Report, Table
from mlia.utils.console import remove_ascii_codes

from mlia.target.ethos_u.common_reporters import (
    ModelIsNotTFLiteCompatible,
    TFLiteCompatibilityCheckFailed,
    handle_model_is_not_tflite_compatible_common,
    handle_tflite_check_failed_common,
    report_tflite_compatibility,
)
from mlia.target.ethos_u.utils.tflite_shims import (
    TFLiteCompatibilityInfo,
    TFLiteCompatibilityStatus,
    TFLiteConversionError,
    TFLiteConversionErrorCode,
)


class DummyAdviceCollector:
    """Simple helper for capturing advice calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_advice(
        self,
        message: str,
        category: SchemaAdviceCategory,
        severity: AdviceSeverity,
        **_: Any,
    ) -> None:
        """Collect advice call arguments."""
        self.calls.append(
            {
                "message": message,
                "category": category,
                "severity": severity,
            }
        )


def test_report_tflite_compatibility_with_conversion_errors() -> None:
    """Report should list individual TFLite conversion errors."""
    compat_info = TFLiteCompatibilityInfo(
        status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
        conversion_errors=[
            TFLiteConversionError(
                "First error message",
                TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                "custom_op",
                ["node_1", "node_2"],
            ),
            TFLiteConversionError(
                "Second error message",
                TFLiteConversionErrorCode.NEEDS_FLEX_OPS,
                "flex_op",
                ["node_3"],
            ),
        ],
    )

    report = report_tflite_compatibility(compat_info)

    assert isinstance(report, Report)
    assert isinstance(report, Table)

    plain_text = remove_ascii_codes(report.to_plain_text())
    assert "TensorFlow Lite conversion errors" in plain_text
    assert "custom_op" in plain_text
    assert "flex_op" in plain_text

    json_dict = report.to_json()
    assert "tensorflow_lite_conversion_errors" in json_dict
    assert json_dict["tensorflow_lite_conversion_errors"] == [
        {
            "operator": "custom_op",
            "operator_location": "node_1, node_2",
            "error_code": "NEEDS_CUSTOM_OPS",
            "error_message": "First error message",
        },
        {
            "operator": "flex_op",
            "operator_location": "node_3",
            "error_code": "NEEDS_FLEX_OPS",
            "error_message": "Second error message",
        },
    ]


def test_report_tflite_compatibility_with_exception() -> None:
    """Report should show exception details when no conversion errors."""
    compat_info = TFLiteCompatibilityInfo(
        status=TFLiteCompatibilityStatus.UNKNOWN_ERROR,
        conversion_exception=RuntimeError("compatibility check failed"),
    )

    report = report_tflite_compatibility(compat_info)

    assert isinstance(report, Report)
    assert isinstance(report, Table)

    plain_text = remove_ascii_codes(report.to_plain_text())
    assert "TensorFlow Lite compatibility errors" in plain_text
    assert "compatibility check failed" in plain_text

    json_dict = report.to_json()
    assert json_dict == {
        "tflite_compatibility": [
            {
                "reason": "TensorFlow Lite compatibility check failed with exception",
                "exception_details": "compatibility check failed",
            }
        ]
    }


@pytest.mark.parametrize(
    "fact, expected_messages",
    [
        pytest.param(
            ModelIsNotTFLiteCompatible(flex_ops=["flex_op1", "flex_op2"]),
            ["flex_op1, flex_op2"],
            id="flex_ops_only",
        ),
        pytest.param(
            ModelIsNotTFLiteCompatible(custom_ops=["custom_op1"]),
            ["custom_op1"],
            id="custom_ops_only",
        ),
        pytest.param(
            ModelIsNotTFLiteCompatible(
                custom_ops=["custom_op1"],
                flex_ops=["flex_op1"],
            ),
            ["flex_op1", "custom_op1"],
            id="flex_and_custom_ops",
        ),
        pytest.param(
            ModelIsNotTFLiteCompatible(custom_ops=[], flex_ops=[]),
            [
                "Model could not be converted into TensorFlow Lite format. "
                "Please refer to the table for more details."
            ],
            id="no_ops_generic_message",
        ),
    ],
)
def test_handle_model_is_not_tflite_compatible_common(
    fact: ModelIsNotTFLiteCompatible,
    expected_messages: list[str],
) -> None:
    """Common handler should emit appropriate advice messages."""
    collector = DummyAdviceCollector()

    handle_model_is_not_tflite_compatible_common(collector, fact)

    assert len(collector.calls) == len(expected_messages)
    for call, expected in zip(collector.calls, expected_messages):
        assert call["category"] == SchemaAdviceCategory.COMPATIBILITY
        assert call["severity"] == AdviceSeverity.WARNING
        assert expected in call["message"]


def test_handle_tflite_check_failed_common() -> None:
    """Common handler should emit generic failure advice."""
    collector = DummyAdviceCollector()
    fact = TFLiteCompatibilityCheckFailed()

    handle_tflite_check_failed_common(collector, fact)

    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["category"] == SchemaAdviceCategory.COMPATIBILITY
    assert call["severity"] == AdviceSeverity.WARNING
    assert (
        call["message"] == "Model could not be converted into TensorFlow Lite format. "
        "Please refer to the table for more details."
    )
