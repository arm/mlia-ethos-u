# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U operators."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlia.backend.errors import BackendUnavailableError
from mlia.target.ethos_u import operators


def test_report_calls_generate_supported_operators_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure report delegates to Vela compat helper."""
    mock_generate = MagicMock()
    monkeypatch.setattr(
        "mlia.target.ethos_u.operators.generate_supported_operators_report",
        mock_generate,
    )

    operators.report()

    mock_generate.assert_called_once_with()


def test_report_propagates_backend_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure report does not swallow BackendUnavailableError."""
    mock_generate = MagicMock(
        side_effect=BackendUnavailableError("Backend vela is not available", "vela")
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.operators.generate_supported_operators_report",
        mock_generate,
    )

    with pytest.raises(BackendUnavailableError):
        operators.report()
