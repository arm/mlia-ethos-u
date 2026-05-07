# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config module."""

from __future__ import annotations

from contextlib import ExitStack as does_not_raise
from pathlib import Path
from typing import Any

import pytest
from mlia.backend.errors import BackendUnavailableError

from mlia.backend.vela.compiler import VelaCompilerOptions
from mlia.target.ethos_u import config as config_mod
from mlia.target.ethos_u.config import EthosUConfiguration, get_default_ethos_u_backends


def test_compiler_options_default_init() -> None:
    """Test compiler options default init."""
    opts = VelaCompilerOptions()

    assert opts.config_file is None
    assert opts.system_config == "internal-default"
    assert opts.memory_mode == "internal-default"
    assert opts.accelerator_config is None
    assert opts.max_block_dependency == 3
    assert opts.arena_cache_size is None
    assert opts.tensor_allocator == "HillClimb"
    assert opts.cpu_tensor_alignment == 16
    assert opts.optimization_strategy == "Performance"
    assert opts.output_dir == Path("output")


def test_ethosu_target() -> None:
    """Test Ethos-U target configuration init."""
    default_config = EthosUConfiguration.load_profile("ethos-u55-256")

    assert default_config.target == "ethos-u55"
    assert default_config.mac == 256
    assert default_config.compiler_options is not None


def test_ethosu_configuration_repr() -> None:
    """Test Ethos-U configuration __repr__ implementation."""
    default_config = EthosUConfiguration.load_profile("ethos-u55-256")

    assert repr(default_config) == "<Ethos-U configuration target=ethos-u55>"


def test_ethosu_configuration_str() -> None:
    """Test Ethos-U configuration __str__ implementation."""
    default_config = EthosUConfiguration.load_profile("ethos-u55-256")

    result = str(default_config)

    assert "Ethos-U target=ethos-u55" in result
    assert "mac=256" in result
    assert "compiler_options=" in result


ETHOSU_CONFIGURATION_PARAMS = [
    pytest.param(
        {},
        pytest.raises(
            KeyError,
            match=r"'target'",
        ),
        id="missing-target",
    ),
    pytest.param(
        {"target": "ethos-u65", "mac": 512},
        pytest.raises(
            KeyError,
            match=r"'system_config'",
        ),
        id="missing-system-config",
    ),
    pytest.param(
        {
            "target": "ethos-u65",
            "mac": 2,
            "system_config": "Ethos_U65_Embedded",
            "memory_mode": "Shared_Sram",
        },
        pytest.raises(
            Exception,
            match=r"Mac value for selected target should be in \[256, 512\]",
        ),
        id="invalid-mac-ethos-u65",
    ),
    pytest.param(
        {
            "target": "ethos-u55",
            "mac": 1,
            "system_config": "Ethos_U55_High_End_Embedded",
            "memory_mode": "Shared_Sram",
        },
        pytest.raises(
            Exception,
            match="Mac value for selected target should be "
            r"in \[32, 64, 128, 256\]",
        ),
        id="invalid-mac-ethos-u55",
    ),
    pytest.param(
        {
            "target": "ethos-u65",
            "mac": 512,
            "system_config": "Ethos_U65_Embedded",
            "memory_mode": "Shared_Sram",
        },
        does_not_raise(),
        id="valid-ethos-u65",
    ),
    pytest.param(
        {
            "target": "ethos-u85",
            "mac": 32,
            "system_config": "Ethos_U85_SYS_DRAM_High",
            "memory_mode": "Shared_Sram",
        },
        pytest.raises(
            Exception,
            match="Mac value for selected target should be "
            r"in \[128, 256, 512, 1024, 2048\]",
        ),
        id="invalid-mac-ethos-u85",
    ),
    pytest.param(
        {
            "target": "ethos-u99",
            "mac": 256,
            "system_config": "Ethos_U65_Embedded",
            "memory_mode": "Shared_Sram",
        },
        pytest.raises(
            ValueError,
            match="Unsupported target: ethos-u99",
        ),
        id="unsupported-target",
    ),
    pytest.param(
        {
            "target": "ethos-u85",
            "mac": 1024,
            "system_config": "Ethos_U85_SYS_DRAM_High",
            "memory_mode": "Shared_Sram",
        },
        does_not_raise(),
        id="valid-ethos-u85",
    ),
]


@pytest.mark.parametrize(
    "profile_data, expected_error",
    ETHOSU_CONFIGURATION_PARAMS,
)
def test_ethosu_configuration(
    profile_data: dict[str, Any], expected_error: Any
) -> None:
    """Test creating Ethos-U configuration."""
    with expected_error:
        cfg = EthosUConfiguration(**profile_data)
        cfg.verify()


@pytest.mark.parametrize(
    "profile_data, expected_error",
    ETHOSU_CONFIGURATION_PARAMS,
)
def test_ethosu_configuration_when_vela_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    profile_data: dict[str, Any],
    expected_error: Any,
) -> None:
    """Test creating Ethos-U configuration when vela unavailable."""
    from mlia.target.ethos_u import config as config_mod
    from mlia.backend.errors import BackendUnavailableError

    def _missing_vela():
        return None, None

    with expected_error:
        monkeypatch.setattr(config_mod, "_load_vela", _missing_vela)
        cfg = config_mod.EthosUConfiguration(**profile_data)
        cfg.verify()
        with pytest.raises(BackendUnavailableError):
            _ = cfg.resolved_compiler_config


def test_config_exposes_vela_symbols_when_backend_available() -> None:
    """Config module should resolve Vela symbols when backend is available."""
    from mlia.target.ethos_u import config as config_mod

    vela_compiler_options, resolve_compiler_config = config_mod._load_vela()
    if vela_compiler_options is None or resolve_compiler_config is None:
        pytest.skip("Vela backend is not available")

    assert vela_compiler_options is not None
    assert resolve_compiler_config is not None


def test_config_fallback_raises_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback path should raise BackendUnavailableError when Vela backend is missing."""
    from mlia.target.ethos_u import config as config_mod

    def _missing_vela():
        return None, None

    monkeypatch.setattr(config_mod, "_load_vela", _missing_vela)

    cfg = config_mod.EthosUConfiguration(
        target="ethos-u55",
        mac=256,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
    )
    with pytest.raises(BackendUnavailableError):
        _ = cfg.resolved_compiler_config


@pytest.mark.parametrize(
    "supported_backends, expected_backends",
    [
        pytest.param(
            [],
            [],
            id="returns-empty-list-when-none-supported",
        ),
        pytest.param(
            ["Unsupported-backend", "vela"],
            ["vela"],
            id="returns-vela-when-vela-and-unsupported",
        ),
        pytest.param(
            ["Unsupported-backend", "corstone-300"],
            ["corstone-300"],
            id="returns-corstone-when-corstone-and-unsupported",
        ),
        pytest.param(
            ["vela", "corstone-300"],
            ["vela", "corstone-300"],
            id="returns-corstone-and-vela-when-corstone-and-vela",
        ),
    ],
)
def test_get_default_ethosu_backends(
    monkeypatch: pytest.MonkeyPatch,
    supported_backends: list,
    expected_backends: list,
):
    monkeypatch.setattr(
        config_mod,
        "get_available_backends",
        lambda: ["vela", "corstone-300"],
    )

    resulting_backends = get_default_ethos_u_backends(supported_backends)
    assert sorted(resulting_backends) == sorted(expected_backends)
