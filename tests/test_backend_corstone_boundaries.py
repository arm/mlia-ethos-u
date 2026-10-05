# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Corstone runtime boundary gateways."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
from mlia.utils.proc import Command

from mlia.backend.corstone.boundaries.process import (
    process_backend_output_with_notice,
    run_installer_with_notice,
)


def test_installer_notice_precedes_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer gateway announces the action before starting it."""
    events = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.corstone.boundaries.process.log_boundary_action",
        events.notice,
    )
    monkeypatch.setattr(
        "mlia.backend.corstone.boundaries.process.subprocess.check_call",
        events.execute,
    )
    logger = logging.getLogger("test")
    command = ["./installer", "--quiet"]

    run_installer_with_notice(logger, "corstone-300", tmp_path, command)

    assert events.mock_calls == [
        call.notice(
            logger,
            "Running downloaded %s installer in '%s'; staged files will be written "
            "to '%s'.",
            "corstone-300",
            tmp_path,
            tmp_path / "corstone-300",
        ),
        call.execute(command, cwd=tmp_path),
    ]


def test_backend_notice_precedes_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The backend gateway announces the action before starting it."""
    events = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.corstone.boundaries.process.log_boundary_action",
        events.notice,
    )
    monkeypatch.setattr(
        "mlia.backend.corstone.boundaries.process.process_command_output",
        events.execute,
    )
    logger = logging.getLogger("test")
    command = Command(["fvp"])
    consumers = [MagicMock()]

    process_backend_output_with_notice(
        logger, "corstone-300", tmp_path, command, consumers
    )

    assert events.mock_calls == [
        call.notice(
            logger,
            "Running Corstone backend '%s'; derived output will be written under '%s'.",
            "corstone-300",
            tmp_path,
        ),
        call.execute(command, consumers),
    ]
