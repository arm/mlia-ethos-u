# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U pattern analysis module."""

import pytest

from mlia.core.data_analysis import Fact
from mlia.target.ethos_u.data_analysis import (
    EthosULayerHighMemoryPressure,
    EthosULayerHighNetworkShare,
    EthosULayerLowMacUtil,
)
from mlia.target.ethos_u.pattern_analysis import (
    LayerHotSpotPatternAnalyzer,
    LowMacUtilLayersPattern,
    MemoryBoundLayersPattern,
)


@pytest.mark.parametrize(
    "facts, pattern_type, expected_facts",
    [
        (
            [
                EthosULayerHighNetworkShare(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="network_share",
                    metric_value=60,
                    metric_unit="%",
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                ),
                EthosULayerLowMacUtil(
                    operator_name="DepthwiseConv2D",
                    location="operator/1",
                    metric="util_mac_percentage",
                    metric_value=2,
                    metric_unit="%",
                    severity="low",
                ),
                Fact(),
            ],
            LowMacUtilLayersPattern,
            [
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                )
            ],
        ),
        (
            [
                EthosULayerHighNetworkShare(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="network_share",
                    metric_value=60,
                    metric_unit="%",
                ),
                EthosULayerLowMacUtil(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="util_mac_percentage",
                    metric_value=20,
                    metric_unit="%",
                    severity="very low",
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=120,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.2,
                ),
                EthosULayerHighMemoryPressure(
                    operator_name="DepthwiseConv2D",
                    location="operator/1",
                    metric="dram_access_cycles",
                    metric_value=130,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.3,
                ),
                Fact(),
            ],
            MemoryBoundLayersPattern,
            [
                EthosULayerHighMemoryPressure(
                    operator_name="Conv2D",
                    location="operator/0",
                    metric="dram_access_cycles",
                    metric_value=120,
                    metric_unit="cycles",
                    mem_to_npu_ratio=1.2,
                )
            ],
        ),
    ],
    ids=["low-mac-hotspots", "memory-bound-hotspots"],
)
def test_layer_hotspot_pattern_analyzer_groups_only_hotspots(
    facts: list[Fact],
    pattern_type: type[LowMacUtilLayersPattern | MemoryBoundLayersPattern],
    expected_facts: list[Fact],
) -> None:
    """Test hotspot grouping keeps only facts that satisfy the prerequisite chain."""

    analyzer = LayerHotSpotPatternAnalyzer()
    patterns = analyzer.analyze_patterns(facts)

    matching_patterns = [
        pattern for pattern in patterns if isinstance(pattern, pattern_type)
    ]

    assert len(matching_patterns) == 1
    assert matching_patterns[0].layer_count == len(expected_facts)
    assert matching_patterns[0].facts == expected_facts
