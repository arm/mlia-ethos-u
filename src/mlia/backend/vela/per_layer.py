# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Shared parsing helpers for Vela per-layer CSV output."""

from __future__ import annotations

import csv
from collections import Counter
from typing import TextIO


_NETWORK_PERCENTAGE_ALIASES = frozenset(
    {
        "Network%",
        "Network% (cycles)",
        "Network% (1)",
        "Network% (MAC)",
    }
)
_NETWORK_PERCENTAGE_HEADERS = ("Network% (cycles)", "Network% (MAC)")


def _canonicalize_network_percentage_headers(headers: list[str]) -> list[str]:
    """Map the two supported network percentage columns to semantic names."""
    indexes = [
        index
        for index, header in enumerate(headers)
        if header in _NETWORK_PERCENTAGE_ALIASES
    ]
    if len(indexes) != len(_NETWORK_PERCENTAGE_HEADERS):
        return headers

    canonical_headers = headers.copy()
    for index, header in zip(indexes, _NETWORK_PERCENTAGE_HEADERS):
        canonical_headers[index] = header
    return canonical_headers


def _normalize_headers(headers: list[str]) -> list[str]:
    """Disambiguate repeated CSV columns while preserving their semantics."""
    headers = _canonicalize_network_percentage_headers(headers)
    occurrences: Counter[str] = Counter()
    normalized_headers = []
    for header in headers:
        occurrence = occurrences[header]
        normalized_headers.append(
            header if occurrence == 0 else f"{header} ({occurrence})"
        )
        occurrences[header] += 1
    return normalized_headers


def read_per_layer_csv_rows(csv_file: TextIO) -> list[dict[str, str]]:
    """Read normalized Vela per-layer rows, excluding repeated header rows."""
    reader = csv.reader(csv_file, delimiter=",")
    try:
        raw_headers = list(next(reader))
    except StopIteration:
        return []

    headers = _normalize_headers(raw_headers)
    return [dict(zip(headers, row)) for row in reader if row != raw_headers]
