# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Attach Ethos-U advice to the complete result that produced it."""

from __future__ import annotations

from typing import Any

from mlia.core.advice_generation import Advice
from mlia.core.common import DataItem
from mlia.core.context import Context
from mlia.target.ethos_u.advice_generation import (
    EthosUAdviceProducer,
    EthosUStaticAdviceProducer,
)
from mlia.target.ethos_u.data_analysis import EthosUDataAnalyzer
from mlia.target.ethos_u.pattern_analysis import (
    ActivationFunctionPatternAnalyzer,
    LayerHotSpotPatternAnalyzer,
)


def _analyze_result(data_item: DataItem, max_pattern_passes: int = 5) -> list[DataItem]:
    """Return facts and composite patterns derived from one result."""
    analyzer = EthosUDataAnalyzer()
    analyzer.analyze_data(data_item)
    facts = list(analyzer.get_analyzed_data())
    pattern_analyzers = [
        ActivationFunctionPatternAnalyzer(),
        LayerHotSpotPatternAnalyzer(),
    ]

    for _ in range(max_pattern_passes):
        new_patterns: list[DataItem] = []
        for pattern_analyzer in pattern_analyzers:
            new_patterns.extend(pattern_analyzer.analyze_patterns(facts))
        if not new_patterns:
            break
        facts.extend(new_patterns)

    return facts


def _generate_advice(data_item: DataItem, context: Context) -> list[Advice]:
    """Generate dynamic and static advice for one complete result."""
    facts = _analyze_result(data_item)
    advice: list[Advice] = []

    for producer in (EthosUAdviceProducer(), EthosUStaticAdviceProducer()):
        producer.set_context(context)
        for fact in facts:
            producer.produce_advice(fact)
        produced = producer.get_advice()
        if isinstance(produced, Advice):
            advice.append(produced)
        else:
            advice.extend(produced)

    return advice


def attach_result_advice(
    output: dict[str, Any], data_item: DataItem, context: Context
) -> None:
    """Attach generated advice to the single result represented by data_item."""
    results = output.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise ValueError(
            "Ethos-U advice generation requires one complete standardized result."
        )

    advice = _generate_advice(data_item, context)
    if advice:
        results[0].setdefault("advice", []).extend(
            item.to_schema().to_dict() for item in advice
        )
