# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module options."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from mlia.cli.options import (
    add_backend_install_options,
    add_backend_options,
    add_backend_uninstall_options,
    add_check_category_options,
    add_debug_options,
    add_model_options,
    add_output_directory,
    add_output_options,
    add_target_options,
    get_output_format,
    get_target_profile_opts,
)
from mlia.core.common import AdviceCategory
from mlia.core.typing import OutputFormat


@pytest.mark.parametrize(
    "args, expected_opts",
    [
        [
            {},
            [],
        ],
        [
            {"target_profile": "profile"},
            ["--target-profile", "profile"],
        ],
        [
            # for the default profile empty list should be returned
            {"target": "ethos-u55-256"},
            [],
        ],
        [
            # Test list handling in construct_param
            {"target_profile": ["profile1", "profile2"]},
            ["--target-profile", "profile1", "--target-profile", "profile2"],
        ],
    ],
)
def test_get_target_opts(args: dict | None, expected_opts: list[str]) -> None:
    """Test getting target options."""
    assert get_target_profile_opts(args) == expected_opts


@pytest.mark.parametrize(
    "args, expected_output_format",
    [
        [
            {},
            "plain_text",
        ],
        [
            {"json": True},
            "json",
        ],
        [
            {"json": False},
            "plain_text",
        ],
    ],
)
def test_get_output_format(args: dict, expected_output_format: OutputFormat) -> None:
    """Test get_output_format function."""
    arguments = argparse.Namespace(**args)
    output_format = get_output_format(arguments)
    assert output_format == expected_output_format


def test_add_check_category_options() -> None:
    """Test add_check_category_options adds correct arguments."""
    parser = argparse.ArgumentParser()
    add_check_category_options(parser)

    args = parser.parse_args(["--performance", "--compatibility"])
    assert args.performance is True
    assert args.compatibility is True

    args = parser.parse_args([])
    assert args.performance is False
    assert args.compatibility is False


def test_add_target_options() -> None:
    """Test add_target_options adds target-profile argument."""
    parser = argparse.ArgumentParser()
    add_target_options(parser, required=False)

    args = parser.parse_args(["--target-profile", "ethos-u55-256"])
    assert args.target_profile == "ethos-u55-256"

    args = parser.parse_args(["-t", "tosa"])
    assert args.target_profile == "tosa"


@pytest.mark.parametrize(
    "supported_advice",
    [
        [AdviceCategory.PERFORMANCE, AdviceCategory.COMPATIBILITY],
    ],
    ids=["performance_and_compatibility"],
)
def test_add_target_options_with_supported_advice(
    supported_advice: list[AdviceCategory],
) -> None:
    """Test add_target_options filters profiles based on supported advice."""
    parser = argparse.ArgumentParser()
    add_target_options(parser, supported_advice=supported_advice, required=False)

    # Should accept target profiles that support the specified advice categories
    args = parser.parse_args(["--target-profile", "ethos-u55-256"])
    assert args.target_profile == "ethos-u55-256"


def test_add_model_options() -> None:
    """Test add_model_options adds model argument."""
    parser = argparse.ArgumentParser()
    add_model_options(parser)

    args = parser.parse_args(["model.tflite"])
    assert args.model == "model.tflite"


def test_add_output_options() -> None:
    """Test add_output_options adds json flag."""
    parser = argparse.ArgumentParser()
    add_output_options(parser)

    args = parser.parse_args(["--json"])
    assert args.json is True

    args = parser.parse_args([])
    assert args.json is False


def test_add_debug_options() -> None:
    """Test add_debug_options adds debug flag."""
    parser = argparse.ArgumentParser()
    add_debug_options(parser)

    args = parser.parse_args(["--debug"])
    assert args.debug is True

    args = parser.parse_args(["-d"])
    assert args.debug is True

    args = parser.parse_args([])
    assert args.debug is False


def test_add_backend_install_options(tmp_path: Path) -> None:
    """Test add_backend_install_options adds all install arguments."""
    parser = argparse.ArgumentParser()
    add_backend_install_options(parser)

    # Test with valid directory
    valid_dir = tmp_path / "install"
    valid_dir.mkdir()

    args = parser.parse_args(
        [
            "--path",
            str(valid_dir),
            "--i-agree-to-the-contained-eula",
            "--force",
            "--noninteractive",
            "backend1",
            "backend2",
        ]
    )

    assert args.path == valid_dir
    assert args.i_agree_to_the_contained_eula is True
    assert args.force is True
    assert args.noninteractive is True
    assert args.names == ["backend1", "backend2"]


def test_add_backend_install_options_invalid_directory(tmp_path: Path) -> None:
    """Test add_backend_install_options rejects invalid directory."""
    parser = argparse.ArgumentParser()
    add_backend_install_options(parser)

    invalid_path = tmp_path / "does_not_exist"

    with pytest.raises(SystemExit):
        parser.parse_args(["--path", str(invalid_path), "backend"])


def test_add_backend_uninstall_options() -> None:
    """Test add_backend_uninstall_options adds names argument."""
    parser = argparse.ArgumentParser()
    add_backend_uninstall_options(parser)

    args = parser.parse_args(["backend1"])
    assert args.names == ["backend1"]

    args = parser.parse_args(["backend1", "backend2", "backend3"])
    assert args.names == ["backend1", "backend2", "backend3"]


def test_add_output_directory() -> None:
    """Test add_output_directory adds output-dir argument."""
    parser = argparse.ArgumentParser()
    add_output_directory(parser)

    args = parser.parse_args(["--output-dir", "/path/to/output"])
    assert args.output_dir == Path("/path/to/output")


def test_add_backend_options() -> None:
    """Test add_backend_options adds backend argument."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        mock_backends.return_value = ["vela", "tosa-checker", "corstone-300"]
        add_backend_options(parser)

        args = parser.parse_args(["-b", "vela"])
        assert args.backend == ["vela"]

        args = parser.parse_args(["--backend", "vela", "--backend", "tosa-checker"])
        assert args.backend == ["vela", "tosa-checker"]


def test_add_backend_options_multiple_corstone_error() -> None:
    """Test add_backend_options rejects multiple Corstone backends."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        mock_backends.return_value = ["corstone-300", "corstone-310"]
        add_backend_options(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--backend", "corstone-300", "--backend", "corstone-310"]
            )


def test_add_backend_options_with_skip_list() -> None:
    """Test add_backend_options with backends_to_skip."""
    parser = argparse.ArgumentParser()

    with patch("mlia.cli.options.get_available_backends") as mock_backends:
        mock_backends.return_value = ["vela", "tosa-checker", "corstone-300"]
        add_backend_options(parser, backends_to_skip=["tosa-checker"])

        # tosa-checker should not be in choices
        with pytest.raises(SystemExit):
            parser.parse_args(["--backend", "tosa-checker"])
