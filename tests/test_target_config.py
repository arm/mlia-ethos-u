# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend config module."""

from __future__ import annotations

from typing import Callable
from unittest.mock import MagicMock

import pytest

from mlia.backend.config import BackendConfiguration, BackendType, System
from mlia.core.common import AdviceCategory
from mlia.target.config import (
    TargetInfo,
    TargetProfile,
    get_builtin_supported_profile_names,
    get_builtin_target_profile_path,
    is_builtin_target_profile,
    load_profile,
)
from mlia.utils.registry import Registry


def test_builtin_supported_profile_names() -> None:
    """Test built-in profile names."""
    builtin_supported_profile_names = get_builtin_supported_profile_names()
    expected = {
        "ethos-u55-128",
        "ethos-u55-256",
        "ethos-u65-256",
        "ethos-u65-512",
        "ethos-u85-128",
        "ethos-u85-512",
        "ethos-u85-256",
        "ethos-u85-1024",
        "ethos-u85-2048",
    }
    assert expected.issubset(set(builtin_supported_profile_names))
    for profile_name in builtin_supported_profile_names:
        assert is_builtin_target_profile(profile_name)
        profile_file = get_builtin_target_profile_path(profile_name)
        assert profile_file.is_file()


def test_builtin_profile_files() -> None:
    """Test function 'get_bulitin_profile_file'."""
    profile_file = get_builtin_target_profile_path("ethos-u55-256")
    assert profile_file.is_file()

    profile_file = get_builtin_target_profile_path("UNKNOWN_FILE_THAT_DOES_NOT_EXIST")
    assert not profile_file.exists()


def test_load_profile() -> None:
    """Test getting profile data."""
    profile_file = get_builtin_target_profile_path("ethos-u55-256")
    result = load_profile(profile_file)
    assert result["profile_name"] == "ethos-u55-256"
    assert result["target_type"] == "ethos-u55"
    assert result["config"]["mac"] == 256
    assert result["config"]["memory_mode"] == "Shared_Sram"
    assert result["config"]["system_config"] == "Ethos_U55_High_End_Embedded"

    with pytest.raises(Exception, match=r"No such file or directory: 'unknown'"):
        load_profile("unknown")


class MyTargetProfile(TargetProfile):
    """Test class deriving from TargetProfile."""

    def verify(self) -> None:
        """Verify the target profile."""
        super().verify()
        assert self.target


@pytest.mark.parametrize(
    "profile_class, fn_init, target, super_target, target_override",
    [
        (
            MyTargetProfile,
            MyTargetProfile,
            "AnyTarget",
            "MySuperTarget",
            "",
        ),
    ],
)
def test_target_profile(
    profile_class: type[MyTargetProfile],
    fn_init: Callable[..., MyTargetProfile],
    target: str,
    super_target: str,
    target_override: str,
) -> None:
    """Test the class 'TargetProfile'."""
    profile = fn_init(target=target)
    assert profile.target == target

    profile = profile_class.load_json_data({"target": super_target})
    assert profile.target == super_target

    profile = fn_init(target="")
    profile.target = target_override
    with pytest.raises(ValueError):
        profile.verify()


@pytest.mark.parametrize(
    ("advice", "check_system", "supported"),
    (
        (None, False, True),
        (None, True, True),
        (AdviceCategory.COMPATIBILITY, True, True),
        (AdviceCategory.OPTIMIZATION, True, False),
    ),
)
def test_target_info(
    monkeypatch: pytest.MonkeyPatch,
    advice: AdviceCategory | None,
    check_system: bool,
    supported: bool,
) -> None:
    """Test the class 'TargetInfo'."""
    info = TargetInfo(
        ["backend"],
        ["backend"],
        MagicMock(),
        MagicMock(),
    )
    assert str(info) == "backend"

    backend_registry = Registry[BackendConfiguration]()
    backend_registry.register(
        "backend",
        BackendConfiguration(
            [AdviceCategory.COMPATIBILITY],
            [System.CURRENT],
            BackendType.BUILTIN,
            None,
        ),
    )
    monkeypatch.setattr("mlia.target.config.backend_registry", backend_registry)

    assert info.is_supported(advice, check_system) == supported
    assert bool(info.filter_supported_backends(advice, check_system)) == supported

    # Test with unknown backend
    info = TargetInfo(
        ["unknown_backend"],
        ["unknown_backend"],
        MagicMock(),
        MagicMock(),
    )
    assert not info.is_supported(advice, check_system)
    assert not info.filter_supported_backends(advice, check_system)
