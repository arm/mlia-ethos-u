# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Load and cache the supported-operators report from the installed Vela."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_path

from mlia.backend.boundaries.filesystem import publish_supported_ops_cache
from mlia.backend.boundaries.filesystem import temporary_supported_ops_directory
from mlia.backend.boundaries.process import run_supported_ops_report_with_notice
from mlia.backend.vela._supported_ops_types import Constraint
from mlia.backend.vela._supported_ops_types import OperatorSupport
from mlia.backend.vela._supported_ops_types import SupportedOpsCatalog
from mlia.backend.vela._supported_ops_types import SupportProfile

TARGETS = {
    "Ethos-U55 and Ethos-U65": ("ethos-u55", "ethos-u65"),
    "Ethos-U85": ("ethos-u85",),
}
FORMATS = ("TFLite", "TOSA")
REPORT_FILENAME = "SUPPORTED_OPS.md"
_CATALOG_CACHE: dict[str, SupportedOpsCatalog] = {}
logger = logging.getLogger(__name__)


@dataclass
class _ConstraintNode:
    """Mutable constraint node used while parsing Markdown indentation."""

    text: str
    children: list[_ConstraintNode] = field(default_factory=list)

    def freeze(self) -> Constraint:
        """Convert this parser node into its immutable catalog type."""
        return Constraint(self.text, tuple(child.freeze() for child in self.children))


def _section(lines: list[str], heading: str) -> list[str]:
    """Return the body belonging to an exact Markdown heading."""
    marker = f"## {heading}"
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise ValueError(f"Missing report section: {heading}") from exc
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    return lines[start:end]


def _bullet_tree(lines: list[str]) -> tuple[Constraint, ...]:
    """Parse Markdown bullets into an ordered constraint tree."""
    roots: list[_ConstraintNode] = []
    stack: list[tuple[int, _ConstraintNode]] = []
    for line in lines:
        match = re.match(r"^(\s*)-\s+(.+?)\s*$", line)
        if not match:
            continue
        indent = len(match.group(1))
        node = _ConstraintNode(match.group(2))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stack:
            stack[-1][1].children.append(node)
        else:
            roots.append(node)
        stack.append((indent, node))
    return tuple(node.freeze() for node in roots)


def _summary_operators(lines: list[str], heading: str) -> dict[str, bool]:
    """Read operator names and specific-constraint links from a summary table."""
    operators = {}
    for line in _section(lines, heading):
        match = re.match(r"^\| ([A-Z][A-Z0-9_]*) \| (.+) \|$", line)
        if match:
            operators[match.group(1)] = "[Specific]" in match.group(2)
    if not operators:
        raise ValueError(f"No operators found in report section: {heading}")
    return operators


def _specific_constraints(
    lines: list[str], target: str, model_format: str
) -> dict[str, tuple[Constraint, ...]]:
    """Read every operator-specific constraint section for a profile."""
    prefix = f"### {target} {model_format} "
    suffix = " Constraints"
    constraints: dict[str, tuple[Constraint, ...]] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith(prefix) or not line.endswith(suffix):
            index += 1
            continue
        operator = line[len(prefix) : -len(suffix)]
        start = index + 1
        end = next(
            (
                item
                for item in range(start, len(lines))
                if lines[item].startswith("## ") or lines[item].startswith("### ")
            ),
            len(lines),
        )
        constraints[operator] = _bullet_tree(lines[start:end])
        index = end
    return constraints


def _allowlist_profile(
    lines: list[str], target: str, model_format: str
) -> SupportProfile:
    """Create a typed profile from an explicit summary table."""
    summary = f"{target} {model_format} Summary Table"
    generic = f"{target} {model_format} Generic Constraints"
    operators = _summary_operators(lines, summary)
    specifics = _specific_constraints(lines, target, model_format)
    expected_specifics = {
        operator for operator, has_specific in operators.items() if has_specific
    }
    unknown = set(specifics).difference(expected_specifics)
    unknown.discard("Not Supported Operators")
    if unknown:
        raise ValueError(
            f"Specific constraints have no matching link: {sorted(unknown)}"
        )
    missing = expected_specifics.difference(specifics)
    if missing:
        raise ValueError(f"Specific constraint sections are missing: {sorted(missing)}")

    unsupported: list[str] = []
    if model_format == "TOSA":
        heading = f"### {target} TOSA Not Supported Operators"
        if heading in lines:
            start = lines.index(heading) + 1
            end = next(
                (
                    item
                    for item in range(start, len(lines))
                    if lines[item].startswith("## ")
                ),
                len(lines),
            )
            unsupported = [item.text for item in _bullet_tree(lines[start:end])]

    return SupportProfile(
        policy="allowlist",
        fallback="cpu" if model_format == "TFLite" else "compiler_error",
        generic_constraints=_bullet_tree(_section(lines, generic)),
        operators={
            operator: OperatorSupport(specifics.get(operator, ()))
            for operator in operators
        },
        unsupported_operators=frozenset(unsupported),
    )


def _u85_tosa_profile(lines: list[str]) -> SupportProfile:
    """Create the typed U85 TOSA baseline-with-exclusions profile."""
    bullets = _bullet_tree(_section(lines, "Ethos-U85 TOSA Operator Support"))
    by_name = {item.text: item.children for item in bullets}
    required = {"Baseline:", "Supported extensions:", "Unsupported operators:"}
    if not required.issubset(by_name):
        raise ValueError("Ethos-U85 TOSA support section is incomplete")

    baseline = []
    for item in by_name["Baseline:"]:
        try:
            name, value = item.text.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid U85 TOSA baseline: {item.text}") from exc
        baseline.append((name.removeprefix("TOSA ").lower(), value.strip()))

    return SupportProfile(
        policy="baseline_with_exclusions",
        fallback="compiler_error",
        baseline=tuple(baseline),
        extensions=tuple(item.text for item in by_name["Supported extensions:"]),
        unsupported_operators=frozenset(
            item.text for item in by_name["Unsupported operators:"]
        ),
    )


def parse_report(report: str) -> SupportedOpsCatalog:
    """Parse the generated Markdown report into a typed catalog."""
    lines = report.splitlines()
    version_match = re.search(r"^Vela version: `([^`]+)`$", report, re.MULTILINE)
    if not version_match:
        raise ValueError("Missing Vela version")

    profiles: dict[tuple[str, str], SupportProfile] = {}
    for target, target_names in TARGETS.items():
        for model_format in FORMATS:
            if target == "Ethos-U85" and model_format == "TOSA":
                if "## Ethos-U85 TOSA Operator Support" not in lines:
                    continue
                profile = _u85_tosa_profile(lines)
            else:
                if f"## {target} {model_format} Summary Table" not in lines:
                    continue
                profile = _allowlist_profile(lines, target, model_format)
            for target_name in target_names:
                profiles[(target_name, model_format.lower())] = profile

    if not profiles:
        raise ValueError("Missing report section: no recognized support profiles")
    return SupportedOpsCatalog(version_match.group(1), profiles)


def _cache_path(vela_version: str) -> Path:
    """Return a safe per-user path for a versioned Vela report."""
    version_key = hashlib.sha256(vela_version.encode("utf-8")).hexdigest()
    return user_cache_path("mlia") / "vela" / "supported-ops-v1" / f"{version_key}.md"


def _catalog_for(report: str, vela_version: str) -> SupportedOpsCatalog:
    """Parse a report and reject data from a different Vela version."""
    catalog = parse_report(report)
    if catalog.vela_version != vela_version:
        raise ValueError(
            f"Report Vela version {catalog.vela_version} does not match "
            f"installed Vela {vela_version}"
        )
    return catalog


def _generate_report() -> str:
    """Run the installed Vela in an isolated working directory."""
    with temporary_supported_ops_directory() as directory:
        run_supported_ops_report_with_notice(logger, directory)
        report = Path(directory) / REPORT_FILENAME
        if not report.is_file():
            raise RuntimeError(f"Vela did not create {REPORT_FILENAME}")
        return report.read_text(encoding="utf-8")


def load_catalog(vela_version: str) -> SupportedOpsCatalog | None:
    """Return version-matched support data, generating it when needed."""
    if cached := _CATALOG_CACHE.get(vela_version):
        return cached

    path = _cache_path(vela_version)
    try:
        report = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        pass
    except UnicodeDecodeError as error:
        logger.warning("Ignoring invalid Vela supported-ops cache: %s", error)
    except OSError as error:
        logger.warning("Cannot read Vela supported-ops cache: %s", error)
    else:
        try:
            catalog = _catalog_for(report, vela_version)
        except ValueError as error:
            logger.warning("Ignoring invalid Vela supported-ops cache: %s", error)
        else:
            _CATALOG_CACHE[vela_version] = catalog
            return catalog

    try:
        report = _generate_report()
        catalog = _catalog_for(report, vela_version)
    except (OSError, RuntimeError, ValueError) as error:
        logger.warning(
            "Vela supported-ops report is unavailable; using runtime diagnostics "
            "without catalog-based advice: %s",
            error,
        )
        return None

    _CATALOG_CACHE[vela_version] = catalog
    try:
        publish_supported_ops_cache(path, report)
    except OSError as error:
        logger.warning("Cannot cache Vela supported-ops report: %s", error)
    return catalog
