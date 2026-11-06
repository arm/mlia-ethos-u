# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parsing and caching installed Vela supported-operator reports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, call

import pytest
from mlia.utils.proc import Command

from mlia.backend.vela import supported_ops
from mlia.backend.boundaries import filesystem as backend_filesystem
from mlia.backend.boundaries import process as backend_process

REPORT = (
    Path(__file__).parents[1] / "tests/test_resources/vela/SUPPORTED_OPS.md"
).read_text(encoding="utf-8")
CATALOG = supported_ops.parse_report(REPORT)


def test_supported_ops_catalog_profiles() -> None:
    """The parser retains every target and format in the fixture report."""
    assert set(CATALOG.profiles) == {
        ("ethos-u55", "tflite"),
        ("ethos-u65", "tflite"),
        ("ethos-u85", "tflite"),
        ("ethos-u55", "tosa"),
        ("ethos-u65", "tosa"),
        ("ethos-u85", "tosa"),
    }
    assert CATALOG.vela_version == "5.2.0"
    assert (
        CATALOG.profiles["ethos-u55", "tflite"]
        is CATALOG.profiles["ethos-u65", "tflite"]
    )


def test_tflite_allowlist_and_constraints() -> None:
    """TFLite profiles retain shared and operator-specific constraints."""
    profile = CATALOG.profiles["ethos-u55", "tflite"]
    assert profile.policy == "allowlist"
    assert profile.fallback == "cpu"
    assert len(profile.operators) == 55
    assert profile.operators["ADD"].constraints == ()
    assert profile.operators["ABS"].constraints[0].text == (
        "32-bit feature-maps are not supported."
    )
    quantization = next(
        item
        for item in profile.generic_constraints
        if item.text.startswith("Input(s), Output and Weight")
    )
    assert quantization.children[0].text.startswith("Exceptions: [ARGMAX")


def test_tosa_support_policies() -> None:
    """TOSA allowlist and baseline policies retain distinct semantics."""
    u55 = CATALOG.profiles["ethos-u55", "tosa"]
    assert u55.policy == "allowlist"
    assert u55.fallback == "compiler_error"
    assert "CONV2D" in u55.operators
    assert "BITWISE_AND" in u55.unsupported_operators
    u85 = CATALOG.profiles["ethos-u85", "tosa"]
    assert u85.policy == "baseline_with_exclusions"
    assert u85.baseline == (
        ("version", "v1.0"),
        ("profile", "PRO-INT"),
        ("level", "8k"),
    )
    assert "EXT-CONTROLFLOW (experimental)" in u85.extensions


def test_catalog_is_immutable() -> None:
    """Callers cannot alter parsed profiles."""
    with pytest.raises(TypeError):
        CATALOG.profiles[("ethos-u55", "tflite")] = CATALOG.profiles[
            "ethos-u85", "tflite"
        ]


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ("# Supported Ops", "Missing Vela version"),
        ("Vela version: `5.2.0`", "Missing report section"),
    ],
)
def test_parser_rejects_incomplete_reports(report: str, message: str) -> None:
    """Missing version or all support sections is invalid."""
    with pytest.raises(ValueError, match=message):
        supported_ops.parse_report(report)


def test_parser_rejects_missing_specific_constraints() -> None:
    """A linked operator cannot silently lose its constraint section."""
    report = REPORT.replace(
        "### Ethos-U55 and Ethos-U65 TFLite ABS Constraints",
        "### Damaged ABS Constraints",
        1,
    )
    with pytest.raises(ValueError, match="Specific constraint sections are missing"):
        supported_ops.parse_report(report)


def test_parser_accepts_missing_profile_section() -> None:
    """Older reports can omit profiles introduced later."""
    catalog = supported_ops.parse_report(
        REPORT.split("## Ethos-U85 TOSA Operator Support")[0]
    )
    assert ("ethos-u85", "tosa") not in catalog.profiles
    assert ("ethos-u55", "tflite") in catalog.profiles


def test_generate_report_uses_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The announced Core command uses an isolated temporary directory."""
    working_directory: Path | None = None
    events = MagicMock()

    def run_vela(command: Command) -> Iterator[str]:
        nonlocal working_directory
        events.execute(command)
        working_directory = command.cwd
        (working_directory / "SUPPORTED_OPS.md").write_text(REPORT, encoding="utf-8")
        return iter(())

    monkeypatch.setattr(backend_process, "log_boundary_action", events.notice)
    monkeypatch.setattr(backend_process, "command_output", run_vela)
    assert supported_ops._generate_report() == REPORT
    assert working_directory is not None
    assert not working_directory.exists()
    assert events.mock_calls == [
        call.notice(
            supported_ops.logger,
            "Running Vela to generate a supported-operators report under '%s'.",
            working_directory,
        ),
        call.execute(
            Command(
                [sys.executable, "-m", "ethosu.vela", "--supported-ops-report"],
                cwd=working_directory,
            )
        ),
    ]


def test_generate_report_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed Core command retains its exit status and diagnostic output."""

    def fail_vela(_command: Command) -> Iterator[str]:
        yield "Vela error\n"
        raise subprocess.CalledProcessError(7, ["vela"])

    monkeypatch.setattr(backend_process, "command_output", fail_vela)
    with pytest.raises(RuntimeError, match="exit status 7: Vela error"):
        supported_ops._generate_report()


def test_catalog_cache_cold_warm_and_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reports are reused across processes and regenerated if invalid."""
    path = tmp_path / "cache" / "report.md"
    generate = MagicMock(return_value=REPORT)
    monkeypatch.setattr(supported_ops, "_cache_path", lambda _version: path)
    monkeypatch.setattr(supported_ops, "_generate_report", generate)
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is not None
    assert path.read_text(encoding="utf-8") == REPORT
    assert supported_ops.load_catalog("5.2.0") is not None
    generate.assert_called_once_with()
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is not None
    generate.assert_called_once_with()
    path.write_text("corrupt", encoding="utf-8")
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is not None
    assert generate.call_count == 2
    caplog.clear()
    path.write_bytes(b"\xff")
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is not None
    assert generate.call_count == 3
    assert path.read_text(encoding="utf-8") == REPORT
    assert "Ignoring invalid Vela supported-ops cache" in caplog.text


def test_cache_failure_does_not_lose_fresh_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wrong-version cache and unwritable directory still use fresh data."""
    path = tmp_path / "report.md"
    path.write_text(REPORT.replace("`5.2.0`", "`5.3.0`", 1), encoding="utf-8")
    monkeypatch.setattr(supported_ops, "_cache_path", lambda _version: path)
    monkeypatch.setattr(supported_ops, "_generate_report", lambda: REPORT)
    monkeypatch.setattr(
        supported_ops,
        "publish_supported_ops_cache",
        MagicMock(side_effect=OSError("read-only")),
    )
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is not None
    assert "`5.3.0`" in path.read_text(encoding="utf-8")


def test_cache_replacement_failure_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed atomic replacement keeps the old cache and removes staging."""
    path = tmp_path / "report.md"
    path.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(
        backend_filesystem.os,
        "replace",
        MagicMock(side_effect=OSError("replacement failed")),
    )

    with pytest.raises(OSError, match="replacement failed"):
        backend_filesystem.publish_supported_ops_cache(path, REPORT)

    assert path.read_text(encoding="utf-8") == "previous report"
    assert list(tmp_path.iterdir()) == [path]


def test_report_failure_uses_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable report leaves catalog advice disabled."""
    monkeypatch.setattr(
        supported_ops, "_cache_path", lambda _version: tmp_path / "none"
    )
    monkeypatch.setattr(
        supported_ops, "_generate_report", MagicMock(side_effect=RuntimeError("failed"))
    )
    supported_ops._CATALOG_CACHE.clear()
    assert supported_ops.load_catalog("5.2.0") is None
