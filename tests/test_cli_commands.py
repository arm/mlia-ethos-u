# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.commands module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mlia.api
import mlia.cli.commands as cli_commands
import mlia.cli.command_validators as cli_validators


@pytest.mark.parametrize(
    "i_agree_to_the_contained_eula, noninteractive, expected_accept_eula",
    [
        pytest.param(False, False, None, id="interactive-default"),
        pytest.param(True, False, True, id="accepted"),
        pytest.param(False, True, False, id="noninteractive-without-eula"),
    ],
)
def test_check_passes_eula_selection_to_get_advice(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    i_agree_to_the_contained_eula: bool,
    noninteractive: bool,
    expected_accept_eula: bool | None,
) -> None:
    """Test check() passes the expected accept_eula value."""
    execution_context = MagicMock()
    get_advice = MagicMock()

    monkeypatch.setattr(
        cli_commands,
        "_create_check_context",
        MagicMock(return_value=execution_context),
    )
    monkeypatch.setattr(
        cli_validators,
        "validate_check_target_profile",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(mlia.api, "get_advice", get_advice)

    cli_commands.check(
        model=str(test_tflite_model),
        target_profile="ethos-u55-256",
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        noninteractive=noninteractive,
    )

    assert get_advice.call_args.kwargs["accept_eula"] == expected_accept_eula
