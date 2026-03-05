# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TFLite compatibility shims for Ethos-U plugin.

If legacy TensorFlow helpers are available, re-export them. Otherwise provide
minimal fallbacks that support .tflite inputs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any


# Compatibility helpers
try:  # pragma: no cover - exercised when legacy is installed
    from mlia.nn.tensorflow.tflite_compat import (  # pylint: disable=import-error
        TFLiteChecker,
        TFLiteCompatibilityInfo,
        TFLiteCompatibilityStatus,
        TFLiteConversionError,
        TFLiteConversionErrorCode,
    )
except ModuleNotFoundError:  # pragma: no cover - minimal fallback

    class TFLiteCompatibilityStatus(Enum):  # type: ignore[no-redef]
        """Minimal compatibility status."""

        COMPATIBLE = auto()
        MODEL_WITH_CUSTOM_OP_ERROR = auto()
        TFLITE_CONVERSION_ERROR = auto()
        UNKNOWN_ERROR = auto()

    class TFLiteConversionErrorCode(Enum):  # type: ignore[no-redef]
        """Minimal conversion error codes."""

        NEEDS_FLEX_OPS = auto()
        NEEDS_CUSTOM_OPS = auto()

    @dataclass
    class TFLiteConversionError:  # type: ignore[no-redef, override]
        """Minimal conversion error info."""

        message: str
        code: TFLiteConversionErrorCode
        operator: str
        location: list[str]

    @dataclass
    class TFLiteCompatibilityInfo:  # type: ignore[no-redef, override]
        """Minimal compatibility info stub."""

        status: TFLiteCompatibilityStatus
        conversion_exception: Exception | None = None
        conversion_errors: list[Any] | None = None

        @property
        def compatible(self) -> bool:
            """Return True when the model is compatible."""
            return self.status == TFLiteCompatibilityStatus.COMPATIBLE

        @property
        def conversion_failed_with_errors(self) -> bool:
            """Return True when conversion failed with errors."""
            return self.status == TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR

        @property
        def conversion_failed_for_model_with_custom_ops(self) -> bool:
            """Return True when conversion failed due to custom ops."""
            return self.status == TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR

        @property
        def check_failed_with_unknown_error(self) -> bool:
            """Return True when check failed with an unknown error."""
            return self.status == TFLiteCompatibilityStatus.UNKNOWN_ERROR

        @property
        def required_custom_ops(self) -> list[str]:
            """Return required custom ops, if any."""
            if not self.conversion_errors:
                return []
            return [
                err.operator
                for err in self.conversion_errors
                if err.code == TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS
            ]

        @property
        def required_flex_ops(self) -> list[str]:
            """Return required flex ops, if any."""
            if not self.conversion_errors:
                return []
            return [
                err.operator
                for err in self.conversion_errors
                if err.code == TFLiteConversionErrorCode.NEEDS_FLEX_OPS
            ]

    class TFLiteChecker:  # type: ignore[no-redef, override]
        """Stub checker that requires legacy TensorFlow helpers."""

        def __init__(self, quantized: bool = False) -> None:
            """Initialize the checker."""
            self.quantized = quantized

        def check_compatibility(self, model: Any) -> TFLiteCompatibilityInfo:
            """Check compatibility or raise when legacy is unavailable."""
            raise RuntimeError(
                "TensorFlow Lite compatibility checks require the legacy plugin "
                "(mlia-legacy)."
            )


# Model conversion helpers
try:  # pragma: no cover - exercised when legacy is installed
    from mlia.nn.tensorflow.config import (  # pylint: disable=import-error
        ModelConfiguration,
        get_tflite_model,
    )
except ModuleNotFoundError:  # pragma: no cover - minimal fallback

    class ModelConfiguration:  # type: ignore[no-redef, override]
        """Minimal model configuration wrapper."""

        def __init__(self, model_path: str | Path) -> None:
            """Initialize with a model path."""
            self.model_path = str(model_path)

    class TFLiteModel(ModelConfiguration):
        """Minimal TFLite model wrapper."""

    def get_tflite_model(model: str | Path, _ctx: Any) -> TFLiteModel:
        """Return a TFLite model wrapper for .tflite inputs only."""
        model_path = Path(model)
        if model_path.suffix != ".tflite":
            raise RuntimeError(
                "Non-TFLite inputs require the legacy plugin (mlia-legacy)."
            )
        return TFLiteModel(model_path)


# Simple format check
try:  # pragma: no cover - exercised when legacy is installed
    from mlia.nn.tensorflow.utils import is_tflite_model  # pylint: disable=import-error
except ModuleNotFoundError:  # pragma: no cover - minimal fallback

    def is_tflite_model(model: str | Path) -> bool:
        """Check if path contains a TensorFlow Lite model."""
        return Path(model).suffix == ".tflite"
