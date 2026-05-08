# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for model format helper predicates."""

from pathlib import Path

import pytest

from mlia.target.ethos_u.utils.model_format import (
    is_pte_file,
    is_pytorch_file,
    is_supported_pytorch_extension,
    is_tosa_file,
)


@pytest.mark.parametrize(
    "model_path, expected",
    [
        (Path("model.tosa"), True),
        (Path("model.tosamlir"), True),
        (Path("model.TOSA"), True),
        (Path("model.tflite"), False),
        (Path("model.pt2"), False),
        (Path("model.pte"), False),
    ],
)
def test_is_tosa_file(model_path: Path, expected: bool) -> None:
    """Test TOSA extension predicate."""
    assert is_tosa_file(model_path) is expected


@pytest.mark.parametrize(
    "model_path, expected",
    [
        (Path("model.pt2"), True),
        (Path("model.PT2"), True),
        (Path("model.pte"), False),
        (Path("model.PTE"), False),
        (Path("model.tosa"), False),
        (Path("model.tflite"), False),
    ],
)
def test_is_supported_pytorch_extension(model_path: Path, expected: bool) -> None:
    """Test PyTorch extension predicate."""
    assert is_supported_pytorch_extension(model_path) is expected


@pytest.mark.parametrize(
    "model_path, expected",
    [
        (Path("missing_model.pt2"), True),
        (Path("missing_model.pte"), False),
        (Path("missing_model.tosa"), False),
        (Path("missing_model.tflite"), False),
    ],
)
def test_is_pytorch_file_is_predicate(model_path: Path, expected: bool) -> None:
    """Test is_pytorch_file behaves as a suffix predicate and never validates path existence."""
    assert is_pytorch_file(model_path) is expected


@pytest.mark.parametrize(
    "model_path, expected",
    [
        (Path("model.pte"), True),
        (Path("model.PTE"), True),
        (Path("missing_model.pte"), True),
        (Path("model.pt2"), False),
        (Path("model.tosa"), False),
        (Path("model.tflite"), False),
    ],
)
def test_is_pte_file_is_predicate(model_path: Path, expected: bool) -> None:
    """Test ExecuTorch extension predicate."""
    assert is_pte_file(model_path) is expected
