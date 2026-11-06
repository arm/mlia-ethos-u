# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Types for Vela supported-operators reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

SupportPolicy = Literal["allowlist", "baseline_with_exclusions"]
FallbackPolicy = Literal["cpu", "compiler_error"]


@dataclass(frozen=True)
class Constraint:
    """A constraint from the Vela report, including nested qualifications."""

    text: str
    children: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class OperatorSupport:
    """Operator-specific constraints in addition to profile-wide constraints."""

    constraints: tuple[Constraint, ...] = ()


@dataclass(frozen=True)
class SupportProfile:
    """Supported-operator information for one accelerator and model format."""

    policy: SupportPolicy
    fallback: FallbackPolicy
    baseline: tuple[tuple[str, str], ...] = ()
    extensions: tuple[str, ...] = ()
    generic_constraints: tuple[Constraint, ...] = ()
    operators: Mapping[str, OperatorSupport] = field(default_factory=dict)
    unsupported_operators: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Protect operator data from mutation by callers."""
        object.__setattr__(self, "operators", MappingProxyType(dict(self.operators)))


@dataclass(frozen=True)
class SupportedOpsCatalog:
    """Vela supported-operator information indexed by target and format."""

    vela_version: str
    profiles: Mapping[tuple[str, str], SupportProfile]

    def __post_init__(self) -> None:
        """Protect profile data from mutation by callers."""
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
