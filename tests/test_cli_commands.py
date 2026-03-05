# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cli.commands module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from mlia.backend.manager import DefaultInstallationManager
from mlia.cli.commands import (
    backend_install,
    backend_list,
    backend_uninstall,
    check,
    target_list,
)
from mlia.core.context import ExecutionContext


def test_operators_expected_parameters(sample_context: ExecutionContext) -> None:
    """Test operators command wrong parameters."""
    with pytest.raises(Exception, match="Model is not provided"):
        check(sample_context, "ethos-u55-256")


def test_performance_unknown_target(
    sample_context: ExecutionContext, test_tflite_model: Path
) -> None:
    """Test that command should fail if unknown target passed."""
    with pytest.raises(
        Exception,
        match=(
            r"Profile 'unknown' is neither a valid built-in target profile "
            r"name or a valid file path."
        ),
    ):
        check(
            sample_context,
            model=str(test_tflite_model),
            target_profile="unknown",
            performance=True,
        )


@pytest.fixture(name="installation_manager_mock")
def fixture_mock_installation_manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock installation manager."""
    install_manager_mock = MagicMock(spec=DefaultInstallationManager)
    monkeypatch.setattr(
        "mlia.cli.commands.get_installation_manager",
        MagicMock(return_value=install_manager_mock),
    )
    return install_manager_mock


def test_backend_command_action_list(installation_manager_mock: MagicMock) -> None:
    """Test mlia-backend command list."""
    backend_list()

    installation_manager_mock.show_env_details.assert_called_once()


def test_target_command_action_list(caplog: pytest.LogCaptureFixture) -> None:
    """Test mlia-target command list."""
    caplog.set_level(level=20)
    target_list()

    # Verify that the target profiles were logged
    assert "Available Target Profiles" in caplog.text


@pytest.mark.parametrize(
    "backend_name",
    [
        "backend_name",
        "BACKEND_NAME",
        "BaCkend_NAme",
    ],
)
def test_backend_command_action_uninstall(
    installation_manager_mock: MagicMock,
    backend_name: str,
) -> None:
    """Test mlia-backend command uninstall."""
    backend_uninstall([backend_name])

    installation_manager_mock.uninstall.assert_called_once()


@pytest.mark.parametrize(
    "i_agree_to_the_contained_eula, force, backend_name, expected_calls",
    [
        [False, False, "backend_name", [call(["backend_name"], False, False)]],
        [True, False, "backend_name", [call(["backend_name"], True, False)]],
        [True, True, "BACKEND_NAME", [call(["BACKEND_NAME"], True, True)]],
    ],
)
def test_backend_command_action_add_download(
    installation_manager_mock: MagicMock,
    i_agree_to_the_contained_eula: bool,
    force: bool,
    backend_name: str,
    expected_calls: Any,
) -> None:
    """Test mlia-backend command "install" with download option."""
    backend_install(
        names=[backend_name],
        i_agree_to_the_contained_eula=i_agree_to_the_contained_eula,
        force=force,
    )

    assert installation_manager_mock.download_and_install.mock_calls == expected_calls


@pytest.mark.parametrize(
    "backend_name, force",
    [
        ["backend_name", False],
        ["backend_name", True],
        ["BACKEND_NAME", True],
    ],
)
def test_backend_command_action_install_from_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
    backend_name: str,
    force: bool,
) -> None:
    """Test mlia-backend command "install" with backend path."""
    backend_install(path=tmp_path, names=[backend_name], force=force)
    installation_manager_mock.install_from.assert_called_once()


def test_backend_command_action_install_no_names_with_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Test backend_install raises ValueError with no names & path."""
    with pytest.raises(ValueError, match="backend name"):
        backend_install(path=tmp_path, names=[])
    installation_manager_mock.install_from.assert_not_called()


def test_backend_command_action_install_multiple_names_with_path(
    installation_manager_mock: MagicMock,
    tmp_path: Path,
) -> None:
    """Test backend_install raises ValueError when multiple names & path."""
    with pytest.raises(ValueError, match="backend name"):
        backend_install(path=tmp_path, names=["backend1", "backend2"])
    installation_manager_mock.install_from.assert_not_called()


@pytest.mark.parametrize(
    "compatibility, performance, expected_category",
    [
        [True, True, {"compatibility", "performance"}],
        [True, False, {"compatibility"}],
        [False, True, {"performance"}],
        [False, False, {"compatibility"}],
    ],
)
def test_check_category_combinations(
    sample_context: ExecutionContext,
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    compatibility: bool,
    performance: bool,
    expected_category: set[str],
) -> None:
    """Test check() with different category combinations."""
    # Mock get_advice to capture what category is passed
    get_advice_mock = MagicMock()
    monkeypatch.setattr("mlia.cli.commands.get_advice", get_advice_mock)

    # Mock validators
    monkeypatch.setattr("mlia.cli.commands.validate_check_target_profile", MagicMock())
    monkeypatch.setattr(
        "mlia.cli.commands.validate_backend", MagicMock(return_value=None)
    )

    check(
        sample_context,
        target_profile="ethos-u55-256",
        model=str(test_tflite_model),
        compatibility=compatibility,
        performance=performance,
    )

    # Verify get_advice was called with the expected category
    get_advice_mock.assert_called_once()
    call_args = get_advice_mock.call_args
    assert call_args[0][2] == expected_category
