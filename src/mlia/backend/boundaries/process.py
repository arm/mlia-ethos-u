# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Classified process boundary gateways for Ethos-U backends."""

from __future__ import annotations

import logging
import subprocess  # nosec
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from mlia.utils.proc import Command, OutputConsumer, process_command_output

if TYPE_CHECKING:
    from mlia.backend.vela.compiler import VelaCompiler, VelaSummary

try:
    from mlia.utils.boundaries.process import command_output
except ImportError:  # pragma: no cover - compatibility with MLIA Core 0.12.2
    from mlia.utils.proc import command_output

try:
    from mlia.utils.logging import log_boundary_action
except ImportError:  # pragma: no cover - compatibility with MLIA Core 0.12.2

    def log_boundary_action(
        logger: logging.Logger, message: str, *args: object
    ) -> None:
        """Log a boundary action when the Core helper is unavailable."""
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


def compile_vela_model_with_notice(
    logger: logging.Logger,
    compiler: VelaCompiler,
    model_path: Path,
    output_dir: Path,
    diagnostic_consumer: Callable[[str], None],
) -> tuple[VelaSummary, Path]:
    """Announce Vela compatibility compilation before it writes artifacts."""
    log_boundary_action(
        logger,
        "Running Vela compatibility analysis for '%s'; derived output will be "
        "written under '%s'.",
        model_path,
        output_dir,
    )
    return compiler.compile_model(model_path, diagnostic_consumer=diagnostic_consumer)


def run_supported_ops_report_with_notice(
    logger: logging.Logger, directory: Path
) -> None:
    """Announce and run Vela to generate a supported-operators report."""
    log_boundary_action(
        logger,
        "Running Vela to generate a supported-operators report under '%s'.",
        directory,
    )
    command = Command(
        [sys.executable, "-m", "ethosu.vela", "--supported-ops-report"],
        cwd=directory,
    )
    output: list[str] = []
    try:
        output.extend(command_output(command))
    except subprocess.CalledProcessError as error:
        detail = "".join(output).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            "Vela supported-ops report generation failed with exit status "
            f"{error.returncode}{suffix}"
        ) from error
