# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Transformer-related tests for the Vela compiler wrapper."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compiler import VelaCompiler
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.transformers.error import TransformerNotFoundError


def test_vela_compiler_reports_missing_pt2_to_tosa_transformer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing PT2->TOSA transformer should report the plugin requirement."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert target_config.compiler_options is not None, (
        "Vela should be available in tests"
    )
    compiler = VelaCompiler(target_config.compiler_options)

    model_path = tmp_path / "model.pt2"
    model_path.write_text("mock pytorch export")

    monkeypatch.setattr(
        "mlia.backend.vela.compiler.transform_model",
        MagicMock(
            side_effect=TransformerNotFoundError(
                "Transformer for model is not available."
            )
        ),
    )

    with pytest.raises(
        TransformerNotFoundError,
        match=(
            "Transformer for model is not available\\.\n"
            "PyTorch to TOSA conversion could not be resolved\\. "
            "Try installing the 'mlia-converters-pytorch' plugin\\."
        ),
    ):
        compiler._preprocess_model(model_path)
