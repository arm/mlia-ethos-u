# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for attaching Ethos-U advice to complete standardized results."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compat import (
    NpuSupported,
    Operator,
    OperatorIdentity,
    Operators,
)
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.performance import VelaPerformanceResult
from mlia.target.ethos_u.result_advice import attach_result_advice


def _compatibility_output(tmp_path: Path, operators: Operators) -> dict[str, Any]:
    """Build a canonical Ethos-U65 compatibility result for advice tests."""
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"model")
    return operators.to_standardized_output(
        model_path=model_path,
        target_config={
            "profile_name": "ethos-u65-256",
            "target": "ethos-u65",
            "mac": 256,
        },
    )


def test_zero_support_advice_names_target_and_gives_next_steps(
    tmp_path: Path,
) -> None:
    """A measured zero-support result should give a target-aware verdict."""
    operators = Operators(
        [
            Operator(
                name="unsupported",
                op_type="CUSTOM",
                run_on_npu=NpuSupported(
                    supported=False,
                    reasons=[("Unsupported operator", "CUSTOM")],
                ),
                identity=OperatorIdentity.tflite(0, 0),
            )
        ]
    )
    output = _compatibility_output(tmp_path, operators)
    context = ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY})

    attach_result_advice(output, operators, context)

    advice = output["results"][0]["advice"]
    assert advice
    assert advice[0]["category"] == "compatibility"

    assert advice[0]["message"] == (
        "This model cannot be accelerated on Ethos-U65 in its current form: none of "
        "the analyzed operators can run on the NPU. Either choose a different model "
        "designed for Ethos-U65 and check it or, if your deployment hardware offers "
        "another suitable target, check the model against that target's profile "
        "before choosing it."
    )


def test_empty_operator_analysis_does_not_produce_zero_support_verdict(
    tmp_path: Path,
) -> None:
    """Missing operator evidence should not be reported as zero support."""
    operators = Operators([])
    output = _compatibility_output(tmp_path, operators)
    context = ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY})

    attach_result_advice(output, operators, context)

    messages = " ".join(
        advice["message"] for advice in output["results"][0].get("advice", [])
    )
    assert "cannot be accelerated" not in messages
    assert "none of the analyzed operators" not in messages
    assert "100% of operators" not in messages


def test_performance_advice_is_attached_to_its_result() -> None:
    """Performance advice should remain local to its backend result."""
    output: dict[str, Any] = {
        "results": [
            {
                "kind": "performance",
                "status": "ok",
                "producer": "vela",
                "entities": [],
                "breakdowns": [],
            }
        ]
    }
    performance = VelaPerformanceResult(output)
    action_resolver = MagicMock()
    action_resolver.check_operator_compatibility.return_value = []
    context = ExecutionContext(
        advice_category={AdviceCategory.PERFORMANCE},
        action_resolver=action_resolver,
    )

    attach_result_advice(output, performance, context)

    advice = output["results"][0]["advice"]
    assert advice
    assert {item["category"] for item in advice} == {"performance"}


def test_advice_attachment_rejects_merged_results() -> None:
    """Advice must be generated before independent backend results are merged."""
    output: dict[str, Any] = {"results": [{}, {}]}

    with pytest.raises(
        ValueError,
        match="requires one complete standardized result",
    ):
        attach_result_advice(
            output,
            MagicMock(),
            ExecutionContext(advice_category={AdviceCategory.PERFORMANCE}),
        )
