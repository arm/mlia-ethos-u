# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Legacy optimization shims for Ethos-U plugin."""

from __future__ import annotations

from typing import Any

from mlia.core.data_collection import ContextAwareDataCollector

try:  # pragma: no cover - exercised when legacy is installed
    from mlia.target.common.optimization import (  # pylint: disable=import-error
        OptimizingPerformaceDataCollector,
        add_common_optimization_params,
    )

    LEGACY_OPTIMIZATION_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - minimal fallback
    LEGACY_OPTIMIZATION_AVAILABLE = False

    class OptimizingPerformaceDataCollector(  # type: ignore[no-redef]
        ContextAwareDataCollector
    ):
        """Fallback optimization collector when legacy is not installed."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Accept any initialization arguments for compatibility."""
            super().__init__()

        def collect_data(self) -> Any:  # type: ignore[override]
            """Collect data or raise when legacy optimization is unavailable."""
            raise RuntimeError("Optimization requires the legacy plugin (mlia-legacy).")

    def add_common_optimization_params(
        _advisor_parameters: dict, _extra_args: dict
    ) -> None:
        """No-op when legacy optimization support is unavailable."""
        return
