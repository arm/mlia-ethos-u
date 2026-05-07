# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Optimization-related shims for Ethos-U plugin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - exercised when legacy is installed
    from mlia.nn.select import OptimizationSettings
except ModuleNotFoundError:  # pragma: no cover - minimal fallback

    @dataclass
    class OptimizationSettings:  # type: ignore[no-redef]
        """Minimal OptimizationSettings fallback."""

        optimization_type: str
        optimization_target: int | float
        layers_to_optimize: list[str] | None
        dataset: Path | None = None

        def __str__(self) -> str:
            """Return string representation."""
            return f"{self.optimization_type}: {self.optimization_target}"

        def next_target(self) -> OptimizationSettings:
            """Return next optimization target."""
            if self.optimization_type == "pruning":
                next_target = round(min(self.optimization_target + 0.1, 0.9), 2)
                return OptimizationSettings(
                    self.optimization_type, next_target, self.layers_to_optimize
                )

            if self.optimization_type == "clustering":
                next_target = math.log(self.optimization_target, 2)
                if next_target.is_integer():
                    next_target -= 1
                next_target = max(int(2 ** int(next_target)), 4)
                return OptimizationSettings(
                    self.optimization_type, next_target, self.layers_to_optimize
                )

            if self.optimization_type == "rewrite":
                return OptimizationSettings(
                    self.optimization_type,
                    self.optimization_target,
                    self.layers_to_optimize,
                    self.dataset,
                )

            raise ValueError(f"Optimization type {self.optimization_type} is unknown.")
