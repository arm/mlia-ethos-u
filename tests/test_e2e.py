# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pytest-native MLIA e2e tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("mlia.testing")

from mlia.testing import e2e as mlia_e2e
from mlia.testing.e2e import COMMON_PATTERNS
from mlia.testing.e2e import COMPATIBILITY_PATTERNS
from mlia.testing.e2e import PERFORMANCE_PATTERNS

ETHOS_U_LAYER_WISE_PATTERNS = (
    r".*Layer-Wise Metrics:.*",
    r".*│.*Layer((.|\n)*)Name[ ]+│.*",
    r".*│.*TFLite((.|\n)*)Operator[ ]+│.*",
    r".*│.*SRAM((.|\n)*)Usage[ ]+│.*",
    r".*│.*OP((.|\n)*)Cycles[ ]+│.*",
    r".*│.*NPU((.|\n)*)Cycles[ ]+│.*",
    r".*│.*SRAM((.|\n)*)AC[ ]+│.*",
    r".*│.*DRAM((.|\n)*)AC[ ]+│.*",
    r".*│.*OnFlash((.|\n)*)AC[ ]+│.*",
    r".*│.*OffFlash((.|\n)*)AC[ ]+│.*",
    r".*│.*MAC((.|\n)*)Count[ ]+│.*",
    r".*│.*MAC((.|\n)*)Util((.|\n)*)\(%\)[ ]+│.*",
)


def assert_matches(pattern: str, output: str) -> None:
    assert re.search(pattern, output), f"Pattern: {pattern}\n\n{output}"


def expects_vela_layer_wise_metrics(case: mlia_e2e.E2ECase) -> bool:
    return "--backend" not in case.args or "vela" in case.args


@mlia_e2e.parametrize(mlia_e2e.E2E_COMPATIBILITY)
def test_e2e_compatibility(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *COMPATIBILITY_PATTERNS):
        assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)


@mlia_e2e.parametrize(mlia_e2e.E2E_PERFORMANCE)
def test_e2e_performance(
    case: mlia_e2e.E2ECase,
    tmp_path: Path,
) -> None:
    result = mlia_e2e.run_case(case, workdir=tmp_path)
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, f"{case}\n\n{output}"
    for pattern in (*COMMON_PATTERNS, *PERFORMANCE_PATTERNS):
        assert_matches(pattern, output)
    if expects_vela_layer_wise_metrics(case):
        for pattern in ETHOS_U_LAYER_WISE_PATTERNS:
            assert_matches(pattern, output)
    mlia_e2e.emit_e2e_results(result)
