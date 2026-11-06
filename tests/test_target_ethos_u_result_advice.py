# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for attaching Ethos-U advice to complete standardized results."""

from copy import deepcopy
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
from mlia.backend.vela.supported_ops import parse_report
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.output_rendering import standardized_output_to_text
from mlia.target.ethos_u.performance import VelaPerformanceResult
from mlia.target.ethos_u.result_advice import attach_result_advice

CATALOG = parse_report(
    (
        Path(__file__).parents[1] / "tests/test_resources/vela/SUPPORTED_OPS.md"
    ).read_text(encoding="utf-8")
)


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
    assert advice[1] == {
        "id": "1",
        "category": "compatibility",
        "severity": "warning",
        "message": "Unsupported operator — CUSTOM",
        "affected_entity_ids": ["source_operator/operator/0"],
    }


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


def test_rejected_operators_get_visible_catalog_constraints(
    tmp_path: Path,
) -> None:
    """Catalog conditions appear once per type without changing observed reasons."""
    operators = Operators(
        [
            Operator(
                "pool_0",
                "AVERAGE_POOL_2D",
                NpuSupported(False, [("Observed rejection", "invalid kernel")]),
                OperatorIdentity.tflite(0, 0),
            ),
            Operator(
                "pool_1",
                "AVERAGE_POOL_2D",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 1),
            ),
            Operator(
                "abs",
                "ABS",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 2),
            ),
            Operator(
                "unchecked",
                "ADD",
                NpuSupported(None, []),
                OperatorIdentity.tflite(0, 3),
            ),
            Operator(
                "custom",
                "CUSTOM",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 4),
            ),
        ],
        support_profile=CATALOG.profiles["ethos-u65", "tflite"],
        support_catalog_version=CATALOG.vela_version,
    )
    output = _compatibility_output(tmp_path, operators)
    checks_before = deepcopy(output["results"][0]["checks"])

    attach_result_advice(
        output,
        operators,
        ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY}),
    )

    result = output["results"][0]
    catalog_advice = [item for item in result["advice"] if item["severity"] == "info"]
    assert len(catalog_advice) == 3
    assert catalog_advice[0]["affected_entity_ids"] == [
        "source_operator/operator/0",
        "source_operator/operator/1",
        "source_operator/operator/2",
    ]
    assert "profile-wide support conditions" in catalog_advice[0]["message"]
    assert (
        "Only the first dimension of tensors can be dynamic."
        in catalog_advice[0]["message"]
    )
    pool_advice, abs_advice = catalog_advice[1:]
    assert pool_advice["affected_entity_ids"] == [
        "source_operator/operator/0",
        "source_operator/operator/1",
    ]
    assert "support conditions for AVERAGE_POOL_2D" in pool_advice["message"]
    assert "- Kernel constraints:\n  - IF padding=VALID:" in pool_advice["message"]
    assert abs_advice["affected_entity_ids"] == ["source_operator/operator/2"]
    assert "support conditions for ABS" in abs_advice["message"]
    assert "32-bit feature-maps are not supported." in abs_advice["message"]
    for advice in (pool_advice, abs_advice):
        assert advice["category"] == "compatibility"
        assert "possible constraints, not confirmed causes" in advice["message"]
    rendered_output = standardized_output_to_text(output)
    assert "Kernel constraints:" in rendered_output
    assert "pool_0, pool_1" in rendered_output
    assert result["checks"] == checks_before
    assert result["checks"][0]["details"]["reasons"] == [
        {"description": "Observed rejection", "detail": "invalid kernel"}
    ]


def test_catalog_advice_requires_matching_profile(tmp_path: Path) -> None:
    """A version-mismatched Vela run has no catalog reference advice."""
    operators = Operators(
        [
            Operator(
                "pool",
                "AVERAGE_POOL_2D",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 0),
            )
        ]
    )
    output = _compatibility_output(tmp_path, operators)

    attach_result_advice(
        output,
        operators,
        ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY}),
    )

    assert all(
        "support conditions" not in advice["message"]
        for advice in output["results"][0]["advice"]
    )


def test_catalog_advice_shows_shared_constraints_without_operator_constraints(
    tmp_path: Path,
) -> None:
    """A listed operator with no specific constraints still gets shared guidance."""
    operators = Operators(
        [
            Operator(
                "add",
                "ADD",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 0),
            )
        ],
        support_profile=CATALOG.profiles["ethos-u65", "tflite"],
        support_catalog_version=CATALOG.vela_version,
    )
    output = _compatibility_output(tmp_path, operators)

    attach_result_advice(
        output,
        operators,
        ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY}),
    )

    catalog_advice = [
        item for item in output["results"][0]["advice"] if item["severity"] == "info"
    ]
    assert len(catalog_advice) == 1
    assert catalog_advice[0]["affected_entity_ids"] == ["source_operator/operator/0"]
    assert "profile-wide support conditions" in catalog_advice[0]["message"]


@pytest.mark.parametrize("include_unknown", [False, True])
def test_incomplete_cpu_rejection_keeps_operator_advice(
    tmp_path: Path, include_unknown: bool
) -> None:
    """Confirmed CPU fallback details survive without a false zero-support claim."""
    ops = [
        Operator(
            "rejected",
            "CONV_2D",
            NpuSupported(
                False,
                [
                    ("CPU only operator", ""),
                    ("Feature maps must be quantized.", "input is Float32"),
                ],
            ),
            OperatorIdentity.tflite(0, 0),
        )
    ]
    if include_unknown:
        ops.append(
            Operator(
                "unchecked",
                "ADD",
                NpuSupported(None, []),
                OperatorIdentity.tflite(0, 1),
            )
        )
    operators = Operators(
        ops,
        incomplete_reason="Vela stopped before checking all operators.",
    )
    output = _compatibility_output(tmp_path, operators)
    context = ExecutionContext(advice_category={AdviceCategory.COMPATIBILITY})

    attach_result_advice(output, operators, context)

    advice = output["results"][0]["advice"]
    operator_advice = next(
        item
        for item in advice
        if item.get("affected_entity_ids") == ["source_operator/operator/0"]
        and item["message"] == "Feature maps must be quantized. — input is Float32"
    )
    assert operator_advice["category"] == "compatibility"
    assert all(
        "none of the analyzed operators" not in item["message"] for item in advice
    )


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
