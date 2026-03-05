# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ethos-specific API behavior."""

from pathlib import Path

from mlia.api import get_advisor
from mlia.core.context import ExecutionContext
from mlia.target.ethos_u.advisor import EthosUInferenceAdvisor


def test_get_advisor_ethos(test_keras_model: Path) -> None:
    """Test function for getting the Ethos-U advisor."""
    ethos_u55_advisor = get_advisor(
        ExecutionContext(), "ethos-u55-256", str(test_keras_model)
    )
    assert isinstance(ethos_u55_advisor, EthosUInferenceAdvisor)
