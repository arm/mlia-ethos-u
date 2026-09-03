# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for attaching Ethos-U advice to complete standardized results."""

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


def test_compatibility_advice_is_attached_to_its_result() -> None:
    """Compatibility analysis should complete its own standardized result."""
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
    output: dict[str, Any] = {
        "results": [
            {
                "kind": "compatibility",
                "status": "incompatible",
                "producer": "vela",
                "entities": [],
            }
        ]
    }
    context = ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY})

    attach_result_advice(output, operators, context)

    advice = output["results"][0]["advice"]
    assert advice
    assert {item["category"] for item in advice} == {"compatibility"}


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
