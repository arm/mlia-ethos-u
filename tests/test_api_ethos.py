# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ethos-specific API behavior."""

from pathlib import Path

import pytest

import mlia.api as mlia_api
from mlia import ValidationMode, run_advisor
from mlia.api import get_advisor
from mlia.backend.vela.compat import Operators, VelaCompatibilityResult
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.advisor import EthosUInferenceAdvisor
from mlia.target.ethos_u.data_collection import EthosUOperatorCompatibility


def test_get_advisor_ethos(test_keras_model: Path) -> None:
    """Test function for getting the Ethos-U advisor."""
    ethos_u55_advisor = get_advisor(
        ExecutionContext(), "ethos-u55-256", str(test_keras_model)
    )
    assert isinstance(ethos_u55_advisor, EthosUInferenceAdvisor)


def test_run_advisor_returns_standardized_output_without_target_event_handler(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
) -> None:
    """Core should return collector output without target presentation events."""
    standardized_output = Operators([]).to_standardized_output(
        model_path=test_tflite_model,
        target_config={"target": "ethos-u55", "mac": 256},
        backend_config={},
        cli_arguments=[],
    )
    result_item = VelaCompatibilityResult(
        legacy_info=Operators([]),
        standardized_output=standardized_output,
    )

    monkeypatch.setattr(
        EthosUOperatorCompatibility,
        "collect_data",
        lambda self: result_item,
    )
    monkeypatch.setattr(
        EthosUInferenceAdvisor,
        "get_analyzers",
        lambda self, context: [],
    )
    monkeypatch.setattr(
        EthosUInferenceAdvisor,
        "get_pattern_analyzers",
        lambda self, context: [],
    )
    monkeypatch.setattr(mlia_api, "validate_backend", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        mlia_api,
        "ensure_backends_installed",
        lambda *args, **kwargs: None,
    )

    result = run_advisor(
        advice_category="compatibility",
        target_profile="ethos-u55-256",
        model=test_tflite_model,
        validation=ValidationMode.OFF,
    )

    assert result == standardized_output
