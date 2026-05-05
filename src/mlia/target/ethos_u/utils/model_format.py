# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Model file format helpers for Ethos-U plugin."""

from __future__ import annotations

from pathlib import Path


def is_tosa_file(path: str | Path) -> bool:
    """Check whether model is a TOSA graph file."""
    return Path(path).suffix.lower() in {".tosa", ".tosamlir"}


_SUPPORTED_PYTORCH_EXTENSIONS = {".pt2"}


def is_tflite_model(path: str | Path) -> bool:
    """Check whether model is a TensorFlow Lite file."""
    return Path(path).suffix.lower() == ".tflite"


def is_pte_file(path: str | Path) -> bool:
    """Check whether model is an ExecuTorch program file."""
    return Path(path).suffix.lower() == ".pte"


def is_supported_pytorch_extension(path: str | Path) -> bool:
    """Check whether model is a PyTorch exported program (.pt2)."""
    return Path(path).suffix.lower() in _SUPPORTED_PYTORCH_EXTENSIONS


def is_pytorch_file(path: str | Path) -> bool:
    """Check whether model is a PyTorch exported program file (.pt2)."""
    return is_supported_pytorch_extension(path)
