# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified process boundary gateways for the Corstone backend."""

from __future__ import annotations

import logging
import subprocess  # nosec
from pathlib import Path

from mlia.utils.proc import Command, OutputConsumer, process_command_output

try:
    from mlia.utils.logging import log_boundary_action
except ImportError:  # pragma: no cover - compatibility with older MLIA cores

    def log_boundary_action(
        logger: logging.Logger, message: str, *args: object
    ) -> None:
        """Log a boundary action when the core helper is unavailable."""
        logger.info(message, *args, extra={"boundary_action": True})


def run_installer_with_notice(
    logger: logging.Logger,
    corstone_name: str,
    dist_dir: Path,
    command: list[str],
) -> None:
    """Announce and execute a downloaded Corstone installer."""
    log_boundary_action(
        logger,
        "Running downloaded %s installer in '%s'; staged files will be written "
        "to '%s'.",
        corstone_name,
        dist_dir,
        dist_dir / corstone_name,
    )
    subprocess.check_call(command, cwd=dist_dir)  # nosec


def process_backend_output_with_notice(
    logger: logging.Logger,
    corstone_name: str,
    output_dir: Path,
    command: Command,
    consumers: list[OutputConsumer],
) -> None:
    """Announce and execute a Corstone backend command."""
    log_boundary_action(
        logger,
        "Running Corstone backend '%s'; derived output will be written under '%s'.",
        corstone_name,
        output_dir,
    )
    process_command_output(command, consumers)
