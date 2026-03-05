# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-specific filesystem tests."""

from mlia.utils.filesystem import get_vela_config


def test_get_vela_config() -> None:
    """Test Vela config files getter."""
    assert get_vela_config().is_file()
    assert get_vela_config().name == "vela.ini"
