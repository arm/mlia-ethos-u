# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Legacy module shims for Ethos-U target plugin."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


try:  # pragma: no cover - exercised when legacy is installed
    from mlia.target.common.optimization import (  # pylint: disable=import-error
        OptimizingDataCollector as OptimizingPerformaceDataCollector,
        add_common_optimization_params,
    )

    LEGACY_OPTIMIZATION_AVAILABLE = True

except ModuleNotFoundError:  # pragma: no cover - minimal fallback
    # disable optimization if module is not present
    LEGACY_OPTIMIZATION_AVAILABLE = False
    OptimizingPerformaceDataCollector = object  # type: ignore[assignment]

    def add_common_optimization_params(  # type: ignore[override]
        _advisor_parameters: dict[str, Any], _extra_args: dict[str, Any]
    ) -> None:
        """Fallback helper for common optimization parameters."""
        logger.debug(
            "Legacy optimization helpers not available."
            " install mlia-legacy to enable optimization."
        )
