# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U MLIA module."""

from __future__ import annotations


from mlia.target.ethos_u.advisor import (
    EthosUInferenceAdvisor,
)


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"
