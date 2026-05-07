# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U configuration."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from mlia.backend.corstone import is_corstone_backend
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.manager import get_available_backends
from mlia.target.config import TargetProfile
from mlia.utils.filesystem import get_vela_config

if TYPE_CHECKING:
    from mlia.backend.vela.compiler import (
        VelaCompilerOptions,
        VelaInitData,
    )

logger = logging.getLogger(__name__)


def _load_vela() -> tuple[type[Any] | None, Any | None]:
    """Load Vela compiler symbols lazily to allow auto-install at runtime."""
    try:
        from mlia.backend.vela.compiler import (
            VelaCompilerOptions,
            resolve_compiler_config,
        )
    except ImportError:
        return None, None
    return VelaCompilerOptions, resolve_compiler_config


class EthosUConfiguration(TargetProfile):
    """Ethos-U configuration."""

    def __init__(self, **kwargs: Any) -> None:
        """Init Ethos-U target configuration."""
        target = kwargs["target"]
        super().__init__(target)

        mac = kwargs["mac"]

        config_in = kwargs.get("config")
        if not config_in:
            config_in = str(get_vela_config())
        logger.debug("DEBUG Vela config file: %s", config_in)

        self.mac = mac

        vela_compiler_options, _ = _load_vela()
        if vela_compiler_options is None:
            # Store parameters for later use when Vela becomes available
            self._vela_options_kwargs = {
                "system_config": kwargs["system_config"],
                "memory_mode": kwargs["memory_mode"],
                "config_file": str(config_in),
                "accelerator_config": f"{self.target}-{mac}",
            }
            self.compiler_options: VelaCompilerOptions | None = None
        else:
            self.compiler_options = vela_compiler_options(
                system_config=kwargs["system_config"],
                memory_mode=kwargs["memory_mode"],
                config_file=str(config_in),
                accelerator_config=f"{self.target}-{mac}",
            )

    def verify(self) -> None:
        """Check the parameters."""
        super().verify()

        target_mac_ranges = {
            "ethos-u55": [32, 64, 128, 256],
            "ethos-u65": [256, 512],
            "ethos-u85": [128, 256, 512, 1024, 2048],
        }

        if self.target not in target_mac_ranges:
            raise ValueError(f"Unsupported target: {self.target}")

        target_mac_range = target_mac_ranges[self.target]
        if self.mac not in target_mac_range:
            raise ValueError(
                f"Mac value for selected target should be in {target_mac_range}."
            )

    @property
    def resolved_compiler_config(self) -> VelaInitData:
        """Resolve compiler configuration."""
        vela_compiler_options, resolve_compiler_config = _load_vela()
        if vela_compiler_options is None or resolve_compiler_config is None:
            raise BackendUnavailableError("Backend vela is not available", "vela")
        if self.compiler_options is None:
            raise BackendUnavailableError("Backend vela is not available", "vela")
        return resolve_compiler_config(self.compiler_options)

    def __str__(self) -> str:
        """Return string representation."""
        compiler_opts = getattr(self, "compiler_options", "N/A (Vela not available)")
        return (
            f"Ethos-U target={self.target} "
            f"mac={self.mac} "
            f"compiler_options={compiler_opts}"
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<Ethos-U configuration target={self.target}>"


def get_default_ethos_u_backends(
    supported_backends_priority_order: list[str],
) -> list[str]:
    """Return default backends for Ethos-U targets."""
    available_backends = get_available_backends()

    default_backends = []
    corstone_added = False
    for backend in supported_backends_priority_order:
        # Include vela as a conceptual default even if not installed
        # (CLI will warn user if it's missing)
        if backend == "vela":
            default_backends.append(backend)
            continue

        # For other backends, only include if available
        if backend not in available_backends:
            continue
        if is_corstone_backend(backend):
            if corstone_added:
                continue  # only add one Corstone backend
            corstone_added = True
        default_backends.append(backend)
    return default_backends
