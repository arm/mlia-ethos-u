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
    from mlia.nn.tensorflow.tflite_compat import (
        TFLiteChecker as LegacyChecker,
        TFLiteCompatibilityInfo,
        TFLiteCompatibilityStatus,
        TFLiteConversionError,
        TFLiteConversionErrorCode,
    )
    from mlia.nn.tensorflow.utils import (
        is_keras_model,
        is_saved_model,
    )
    import tf_keras as keras
except ModuleNotFoundError:

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
    class TFLiteConversionError:  # type: ignore[no-redef]
        """Minimal conversion error info."""

        message: str
        code: TFLiteConversionErrorCode
        operator: str
        location: list[str]

    @dataclass
    class TFLiteCompatibilityInfo:  # type: ignore[no-redef]
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

    class LegacyChecker:  # type: ignore[no-redef, override]
        """Stub checker that requires legacy TensorFlow helpers."""

        def __init__(self, quantized: bool = False) -> None:
            """Initialize the checker."""
            self.quantized = quantized

        def check_compatibility(self, model: Any) -> TFLiteCompatibilityInfo:
            """Check compatibility or raise when legacy is unavailable."""
            raise RuntimeError(
                "This requires mlia-legacy which needs to be installed separately."
                " Please install mlia-legacy or ensure your model is already in compatible format."
            )


# Backwards-compatible alias for existing imports.
TFLiteChecker = LegacyChecker


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
                "Non-TFLite inputs require specialized converter plugins."
            )
        return TFLiteModel(model_path)


# Simple format check
try:  # pragma: no cover - exercised when legacy is installed
    from mlia.nn.tensorflow.utils import is_tflite_model  # pylint: disable=import-error
except ModuleNotFoundError:  # pragma: no cover - minimal fallback

    def is_tflite_model(model: str | Path) -> bool:
        """Check if path contains a TensorFlow Lite model."""
        return Path(model).suffix == ".tflite"


def is_legacy_model(model: Any) -> bool:
    """Return True when the input requires legacy TensorFlow helpers."""
    if isinstance(model, (str, Path)):
        # Keras .h5/.hdf5 or SavedModel directory.
        try:
            return is_keras_model(model) or is_saved_model(model)
        except NameError:
            return _is_legacy_model_path(model)
    try:
        return isinstance(model, keras.Model)  # type: ignore[name-defined]
    except NameError:
        return False


def _is_legacy_model_path(model: str | Path) -> bool:
    """Return True if the path points to a legacy Keras or SavedModel artifact."""
    model_path = Path(model)
    if model_path.is_dir():
        return (
            model_path.joinpath("keras_metadata.pb").exists()
            or model_path.joinpath("saved_model.pb").exists()
        )
    return model_path.suffix in {".h5", ".hdf5"}
