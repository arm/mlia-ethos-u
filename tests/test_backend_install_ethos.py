# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for plugin backend installation helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.install import (
    DownloadAndInstall,
    InstallFromPath,
    PyPackageBackendInstallation,
)
from mlia.backend.vela.install import get_vela_installation


def test_get_vela_backend_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test the Vela backend installation helper."""
    mock_package_manager = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.install.get_package_manager",
        lambda: mock_package_manager,
    )

    installation_func = get_vela_installation()
    assert installation_func.name == "vela"
    assert isinstance(installation_func, PyPackageBackendInstallation)
    assert installation_func.supports(DownloadAndInstall())
    assert installation_func.supports(InstallFromPath(tmp_path))
