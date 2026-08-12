# SPDX-FileCopyrightText: Copyright 2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.command_validators module."""

from __future__ import annotations


import pytest

from mlia.cli import command_validators


def test_validate_backend_returns_canonical_backend_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend validation should normalize user input to registry keys."""
    monkeypatch.setattr(
        command_validators,
        "get_target",
        lambda target_profile: "target",
    )
    monkeypatch.setattr(
        command_validators,
        "supported_backends",
        lambda target: ["corstone-300", "vela"],
    )

    assert command_validators.validate_backend(
        "target-profile",
        ["Corstone300", "Vela"],
    ) == ["corstone-300", "vela"]


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("", ""),
        ("lowercase", "lowercase"),
        ("UPPERCASE", "uppercase"),
        ("VELA", "vela"),
        ("check-no-hyphens", "checknohyphens"),
        ("MixedCase-With-Hyphens", "mixedcasewithhyphens"),
        ("corstone-310", "corstone310"),
        ("---multiple---hyphens---", "multiplehyphens"),
    ],
)
def test_normalize_string(input_string: str, expected_output: str) -> None:
    """Test normalize_string function with various inputs."""
    assert command_validators.normalize_string(input_string) == expected_output
