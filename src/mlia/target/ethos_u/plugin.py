# SPDX-FileCopyrightText: Copyright 2023,2025-2026 Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U target module."""

from mlia.plugins.plugins import TargetPlugin
from mlia.target.ethos_u.advisor import configure_and_get_ethosu_advisor
from mlia.target.ethos_u.config import EthosUConfiguration, get_default_ethos_u_backends
from mlia.target.registry import TargetInfo, TargetRegistry

ETHOS_U85 = "Ethos-U85"
SUPPORTED_BACKENDS_PRIORITY_ETHOS_U85 = [
    "vela",
    "corstone-320",
]

ETHOS_U65 = "Ethos-U65"
SUPPORTED_BACKENDS_PRIORITY_ETHOS_U65 = [
    "vela",
    "corstone-310",
    "corstone-300",
]

ETHOS_U55 = "Ethos-U55"
SUPPORTED_BACKENDS_PRIORITY_ETHOS_U55 = ["vela", "corstone-310", "corstone-300"]


def _create_target_info(supported_backends: list[str]) -> TargetInfo:
    """Build TargetInfo while remaining compatible with older mlia cores."""
    kwargs = {
        "supported_backends": supported_backends,
        "default_backends": get_default_ethos_u_backends(supported_backends),
        "advisor_factory_func": configure_and_get_ethosu_advisor,
        "target_profile_cls": EthosUConfiguration,
    }

    return TargetInfo(**kwargs)


class EthosUTargetPlugin(TargetPlugin):
    """Ethos-U Target Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: TargetRegistry) -> None:
        """Register the target with the registry."""
        registry.register(
            ETHOS_U85.lower(),
            _create_target_info(SUPPORTED_BACKENDS_PRIORITY_ETHOS_U85),
            pretty_name=ETHOS_U85,
        )

        registry.register(
            ETHOS_U65.lower(),
            _create_target_info(SUPPORTED_BACKENDS_PRIORITY_ETHOS_U65),
            pretty_name=ETHOS_U65,
        )

        registry.register(
            ETHOS_U55.lower(),
            _create_target_info(SUPPORTED_BACKENDS_PRIORITY_ETHOS_U55),
            pretty_name=ETHOS_U55,
        )
