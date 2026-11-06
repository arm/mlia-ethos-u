# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela operator compatibility module."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mlia
import mlia.core.output_schema as schema
from mlia.backend.boundaries.process import compile_vela_model_with_notice
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela._supported_ops_types import Constraint, SupportProfile
from mlia.backend.vela.supported_ops import load_catalog
from mlia.target.ethos_u.utils.model_format import is_pytorch_file, is_tosa_file
from mlia.utils.filesystem import sha256
from mlia.utils.logging import redirect_output

if TYPE_CHECKING:
    from mlia.backend.vela.compiler import VelaCompiler


@dataclass(frozen=True)
class VelaDeps:
    """Resolved Vela dependencies."""

    ethosu_vela_version: str
    Op: Any
    optype_to_builtintype: Any
    generate_supported_ops: Any
    VelaCompiler: Any
    layer_metrics: Any
    parse_layerwise_perf_csv: Any


_VELA_DEPS_CACHE: VelaDeps | None = None


def _load_vela_deps() -> VelaDeps:
    """Load Vela modules on demand."""
    try:
        import importlib

        importlib.invalidate_caches()
        from ethosu.vela import __version__ as ethosu_vela_version
        from ethosu.vela.operation import Op
        from ethosu.vela.tflite_mapping import optype_to_builtintype
        from ethosu.vela.vela import generate_supported_ops

        from mlia.backend.vela.compiler import VelaCompiler
        from mlia.backend.vela.performance import layer_metrics
        from mlia.backend.vela.performance import parse_layerwise_perf_csv
    except ImportError as exc:
        raise BackendUnavailableError("Backend vela is not available", "vela") from exc

    return VelaDeps(
        ethosu_vela_version=ethosu_vela_version,
        Op=Op,
        optype_to_builtintype=optype_to_builtintype,
        generate_supported_ops=generate_supported_ops,
        VelaCompiler=VelaCompiler,
        layer_metrics=layer_metrics,
        parse_layerwise_perf_csv=parse_layerwise_perf_csv,
    )


def _get_vela_deps() -> VelaDeps:
    """Return cached Vela deps or load them."""
    global _VELA_DEPS_CACHE

    if _VELA_DEPS_CACHE is None:
        _VELA_DEPS_CACHE = _load_vela_deps()
    return _VELA_DEPS_CACHE


logger = logging.getLogger(__name__)

# Glob pattern for Vela layerwise CSV files
_VELA_LAYERWISE_CSV_GLOB_PATTERN = "*{model_name}*per-layer.csv"

# TFLite operator names to filter from layerwise data
_TFLITE_LAYERWISE_FILTERED_OP_NAMES = ["Placeholder", "Const"]

_ENTITY_KIND_PERFORMANCE_LAYER = "performance_layer"
_ENTITY_KIND_COMPILED_OPERATOR = "compiled_operator"
_ENTITY_KIND_VELA_DIAGNOSTIC = "vela_diagnostic"
_CPU_ONLY_REASON = ("CPU only operator", "")
_PROFILE_EXCLUSION_DESCRIPTION = (
    "Operator is not supported by the selected Vela profile."
)
_INCOMPLETE_COMPATIBILITY_REASON = "Vela stopped after an operator-support error; compatibility analysis is incomplete."
_SRAM_CAPACITY_MESSAGE = "Model is too large and uses too much RAM"
_SRAM_CAPACITY_WARNING = "Warning: SRAM target for arena memory area exceeded."
_SRAM_COMPATIBILITY_WARNING = (
    "Vela reported that the model exceeds the SRAM target; "
    "operator compatibility checks completed."
)
_UNVALIDATED_OPERATOR_REASON = {
    "description": "Operator support was not determined",
    "detail": "Vela stopped before validating this operator.",
}

_DIAGNOSTIC_HEADER = re.compile(
    r"^Warning \(supported operators\) operator: (?P<op_type>[^,]+), "
    r"ofm: '(?P<name>.*)'$"
)
_TFLITE_SEMANTIC_ERROR = re.compile(
    r"Error:\s+(?P<op_type>[A-Z][A-Z0-9_]*)\s+'(?P<name>[^']+)'\s+"
    r"Does not have valid TFLite Semantics\.\s*\n"
    r"\s*-\s*(?P<constraint>[^\n]+)\n"
    r"\s*(?P<detail>[^\n]+)\n"
    r"Failed to parse TFLite file",
)
_OPERATOR_SUPPORT_ERROR_MARKERS = (
    "does not have valid tflite semantics",
    "not supported",
    "unsupported operator",
    "constraint",
)


@dataclass(frozen=True)
class OperatorDiagnostic:
    """A structured operator rejection emitted by Regor."""

    op_type: str
    name: str
    reason: str
    constraint: str


def _parse_operator_diagnostics(lines: list[str]) -> list[OperatorDiagnostic]:
    """Parse Regor supported-operator warning blocks."""
    diagnostics: list[OperatorDiagnostic] = []
    index = 0
    while index < len(lines):
        header = _DIAGNOSTIC_HEADER.match(lines[index].strip())
        if header is None:
            index += 1
            continue

        index += 1
        reason = ""
        constraints: list[str] = []
        while index < len(lines):
            line = lines[index].strip()
            if _DIAGNOSTIC_HEADER.match(line):
                break
            index += 1
            if not line:
                if reason or constraints:
                    break
                continue
            if line.startswith("Reason:"):
                reason = line.removeprefix("Reason:").strip()
            elif reason:
                constraints.append(line)

        diagnostics.append(
            OperatorDiagnostic(
                op_type=header.group("op_type").strip(),
                name=header.group("name"),
                reason=reason,
                constraint=" ".join(constraints),
            )
        )
    return diagnostics


def _exception_messages(error: BaseException) -> list[str]:
    """Return messages from an exception and its explicit cause chain."""
    messages: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current))
        current = current.__cause__
    return messages


def _parse_tflite_semantic_errors(messages: list[str]) -> list[OperatorDiagnostic]:
    """Parse Regor TFLite semantic failures from compiler exception messages."""
    diagnostics: list[OperatorDiagnostic] = []
    for message in messages:
        for match in _TFLITE_SEMANTIC_ERROR.finditer(message):
            diagnostics.append(
                OperatorDiagnostic(
                    op_type=match.group("op_type"),
                    name=match.group("name"),
                    reason=match.group("detail").strip().rstrip(","),
                    constraint=match.group("constraint").strip(),
                )
            )
    return diagnostics


def _diagnostic_matches_support_error(
    diagnostic: OperatorDiagnostic, message: str
) -> bool:
    """Return whether an error positively identifies a captured rejection."""
    normalized_message = " ".join(message.split()).casefold()
    if not any(
        marker in normalized_message for marker in _OPERATOR_SUPPORT_ERROR_MARKERS
    ):
        return False

    op_type = " ".join(diagnostic.op_type.split()).casefold()
    if op_type not in normalized_message:
        return False

    identifying_details = (
        diagnostic.name,
        diagnostic.reason,
        diagnostic.constraint,
    )
    return any(
        normalized and normalized in normalized_message
        for detail in identifying_details
        if (normalized := " ".join(detail.split()).casefold())
    )


def _operator_support_failure_diagnostics(
    error: RuntimeError, lines: list[str]
) -> list[OperatorDiagnostic]:
    """Return diagnostics only for positively identified support failures."""
    messages = _exception_messages(error)
    semantic_diagnostics = _parse_tflite_semantic_errors(messages)
    emitted_diagnostics = _parse_operator_diagnostics(lines)
    if semantic_diagnostics:
        return list(dict.fromkeys([*emitted_diagnostics, *semantic_diagnostics]))

    if emitted_diagnostics and any(
        _diagnostic_matches_support_error(diagnostic, message)
        for diagnostic in emitted_diagnostics
        for message in messages
    ):
        return emitted_diagnostics
    return []


def _operator_key(op_type: str, name: str) -> tuple[str, str]:
    """Return a spelling-insensitive key for an operator diagnostic."""
    return (re.sub(r"[^A-Z0-9]", "", op_type.upper()), name)


def _profile_for(
    compiler_options: Any, model_format: str, vela_version: str
) -> SupportProfile | None:
    """Resolve installed-Vela support information for a compiler configuration."""
    accelerator = str(getattr(compiler_options, "accelerator_config", ""))
    match = re.search(r"ethos[-_]u(55|65|85)", accelerator, re.IGNORECASE)
    if match is None:
        return None
    catalog = load_catalog(vela_version)
    if catalog is None:
        return None
    key = (f"ethos-u{match.group(1)}", model_format)
    profile = catalog.profiles.get(key)
    if profile is None:
        logger.warning(
            "Vela %s has no parsed support profile for %s %s; using runtime "
            "diagnostics without catalog-based advice.",
            vela_version,
            *key,
        )
    return profile


def _profile_operator_name(profile: SupportProfile, op_type: str) -> str | None:
    """Find the report spelling of an operator name."""
    normalized = _operator_key(op_type, "")[0]
    return next(
        (
            name
            for name in profile.operators
            if _operator_key(name, "")[0] == normalized
        ),
        None,
    )


def _rejection_info(
    op_type: str,
    diagnostic: OperatorDiagnostic | None,
    profile: SupportProfile | None,
    rejected: bool = False,
    compilation_completed: bool = True,
) -> NpuSupported:
    """Build placement information from Regor and generated profile data."""
    cpu_fallback_confirmed = rejected or (
        compilation_completed and profile is not None and profile.fallback == "cpu"
    )
    cpu_fallback_reasons = [_CPU_ONLY_REASON] if cpu_fallback_confirmed else []
    if diagnostic is not None:
        description = diagnostic.constraint.strip()
        detail = diagnostic.reason.strip()
        reasons = [(description, detail)] if description or detail else []
        return NpuSupported(False, [*cpu_fallback_reasons, *reasons])

    if profile is None:
        return NpuSupported(not rejected, cpu_fallback_reasons)

    report_name = _profile_operator_name(profile, op_type)
    normalized_unsupported = {
        _operator_key(name, "")[0] for name in profile.unsupported_operators
    }
    explicitly_unsupported = _operator_key(op_type, "")[0] in normalized_unsupported
    absent_from_allowlist = profile.policy == "allowlist" and report_name is None
    if not (rejected or explicitly_unsupported or absent_from_allowlist):
        return NpuSupported(True, [])

    if explicitly_unsupported or absent_from_allowlist:
        if cpu_fallback_confirmed:
            detail = f"{op_type} uses CPU fallback for this target and model format."
        elif profile.fallback == "cpu":
            detail = (
                f"{op_type} is unsupported for this target and model format; "
                "CPU fallback was not confirmed."
            )
        else:
            detail = (
                f"{op_type} uses compilation failure for this target and model format."
            )
        return NpuSupported(
            False,
            [
                *cpu_fallback_reasons,
                (
                    _PROFILE_EXCLUSION_DESCRIPTION,
                    detail,
                ),
            ],
        )

    # The report lists possible constraints, but cannot identify which one failed.
    return NpuSupported(False, cpu_fallback_reasons)


def _get_layerwise_csv_pattern(model_name: str) -> str:
    """Format the layerwise CSV glob pattern for a model.

    Args:
        model_name: The model name to use in the pattern

    Returns:
        Formatted glob pattern string
    """
    return _VELA_LAYERWISE_CSV_GLOB_PATTERN.format(model_name=model_name)


@dataclass
class NpuSupported:
    """Operator's npu supported attribute."""

    supported: bool | None
    reasons: list[tuple[str, str]]

    @property
    def confirmed_reasons(self) -> list[tuple[str, str]]:
        """Return Vela diagnostics without placement or catalog-only markers."""
        return [
            reason
            for reason in self.reasons
            if reason != _CPU_ONLY_REASON
            and reason[0] != _PROFILE_EXCLUSION_DESCRIPTION
        ]


@dataclass(frozen=True)
class OperatorIdentity:
    """Result-local identity for a checked operator."""

    entity_id: str
    entity_kind: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _validate_index(index: int, description: str) -> None:
        """Validate an identity coordinate."""
        if type(index) is not int or index < 0:  # pylint: disable=unidiomatic-typecheck
            raise ValueError(f"{description} must be a non-negative integer.")

    @classmethod
    def tflite(cls, subgraph_index: int, operator_index: int) -> OperatorIdentity:
        """Create a canonical identity for a source TFLite operator."""
        return cls(
            entity_id=schema.tflite_source_operator_id(operator_index, subgraph_index),
            entity_kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
            attributes={
                "subgraph_index": subgraph_index,
                "operator_index": operator_index,
            },
        )

    @classmethod
    def performance_layer(cls, layer_index: int) -> OperatorIdentity:
        """Create an identity scoped to a Vela per-layer performance row."""
        cls._validate_index(layer_index, "Vela performance layer index")
        return cls(
            entity_id=f"{_ENTITY_KIND_PERFORMANCE_LAYER}/{layer_index}",
            entity_kind=_ENTITY_KIND_PERFORMANCE_LAYER,
            attributes={
                "identity_scope": "vela_per_layer_csv",
                "layer_index": layer_index,
            },
        )

    @classmethod
    def compiled_tflite(
        cls, subgraph_index: int, operator_index: int
    ) -> OperatorIdentity:
        """Create an identity scoped to a Vela-generated TFLite artifact."""
        cls._validate_index(subgraph_index, "Compiled TFLite subgraph index")
        cls._validate_index(operator_index, "Compiled TFLite operator index")
        if subgraph_index == 0:
            entity_id = f"{_ENTITY_KIND_COMPILED_OPERATOR}/operator/{operator_index}"
        else:
            entity_id = (
                f"{_ENTITY_KIND_COMPILED_OPERATOR}/subgraph/{subgraph_index}"
                f"/operator/{operator_index}"
            )
        return cls(
            entity_id=entity_id,
            entity_kind=_ENTITY_KIND_COMPILED_OPERATOR,
            attributes={
                "identity_scope": "vela_compiled_tflite",
                "subgraph_index": subgraph_index,
                "operator_index": operator_index,
            },
        )

    @classmethod
    def diagnostic(cls, diagnostic_index: int) -> OperatorIdentity:
        """Create an identity for an unmatched compiler diagnostic."""
        cls._validate_index(diagnostic_index, "Vela diagnostic index")
        return cls(
            entity_id=f"{_ENTITY_KIND_VELA_DIAGNOSTIC}/{diagnostic_index}",
            entity_kind=_ENTITY_KIND_VELA_DIAGNOSTIC,
            attributes={
                "identity_scope": "vela_compiler_diagnostic",
                "diagnostic_index": diagnostic_index,
            },
        )


@dataclass
class Operator:
    """Model operator."""

    name: str
    op_type: str
    run_on_npu: NpuSupported
    identity: OperatorIdentity

    @property
    def cpu_only(self) -> bool:
        """Return true if operator is CPU only."""
        return (
            self.run_on_npu.supported is False
            and _CPU_ONLY_REASON in self.run_on_npu.reasons
        )


@dataclass
class Operators:
    """Model's operators."""

    ops: list[Operator]
    incomplete_reason: str | None = field(default=None, repr=False, compare=False)
    compiler_warnings: list[str] = field(
        default_factory=list, repr=False, compare=False
    )
    support_profile: SupportProfile | None = field(
        default=None, repr=False, compare=False
    )
    support_catalog_version: str | None = field(default=None, repr=False, compare=False)

    def catalog_constraints_for(self, op_type: str) -> tuple[Constraint, ...] | None:
        """Return catalog constraints for a named operator, if it is listed."""
        if self.support_profile is None:
            return None
        name = _profile_operator_name(self.support_profile, op_type)
        return self.support_profile.operators[name].constraints if name else None

    @property
    def is_complete(self) -> bool:
        """Return whether every operator's compatibility was determined."""
        return self.incomplete_reason is None and all(
            op.run_on_npu.supported is not None for op in self.ops
        )

    @property
    def npu_supported_ratio(self) -> float | None:
        """Return NPU supported ratio."""
        if not self.is_complete:
            return None
        total = self.total_number
        npu_supported = self.npu_supported_number

        if total == 0 or npu_supported == 0:
            return 0

        return npu_supported / total

    @property
    def npu_unsupported_ratio(self) -> float | None:
        """Return NPU unsupported ratio."""
        supported_ratio = self.npu_supported_ratio
        return None if supported_ratio is None else 1 - supported_ratio

    @property
    def total_number(self) -> int:
        """Return total number of operators."""
        return len(self.ops)

    @property
    def npu_supported_number(self) -> int:
        """Return number of npu supported operators."""
        return sum(op.run_on_npu.supported is True for op in self.ops)

    def _accelerator_operator_percentage_metrics(self) -> list[schema.Metric]:
        """Build compatibility operator percentage metrics."""
        supported_ratio = self.npu_supported_ratio
        if self.total_number and supported_ratio is not None:
            return [
                schema.Metric(
                    name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                    value=supported_ratio * 100,
                    unit=schema.UNIT_PERCENT,
                )
            ]

        reason = "Operator placement data is not available."
        if not self.is_complete:
            reason = self.incomplete_reason or _INCOMPLETE_COMPATIBILITY_REASON
        return [
            schema.Metric(
                name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason=reason,
            )
        ]

    def to_standardized_output(
        self,
        model_path: Path,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            target_config: Optional target configuration parameters
            backend_config: Optional backend configuration parameters

        Returns:
            Standardized output dictionary
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version
        try:
            deps = _get_vela_deps()
            backend_version = deps.ethosu_vela_version
        except Exception as exc:
            logger.warning("Failed to get vela version: %s", exc)
            backend_version = "unknown"

        backend = schema.Backend(
            id="vela",
            name="Vela Compiler",
            version=backend_version,
            configuration=backend_config or {},
        )

        # Create target with NPU component
        target_type = (target_config or {}).get("target", "ethos-u")
        mac = (target_config or {}).get("mac", "unknown")
        profile_name = (target_config or {}).get("profile_name")
        if profile_name is None:
            profile_name = f"{target_type}-{mac}" if mac != "unknown" else target_type

        npu_component = schema.Component(
            type=schema.ComponentType.NPU,
            family=target_type,
            model=None,
            variant=str(mac) if mac != "unknown" else None,
        )

        target = schema.Target(
            profile_name=profile_name,
            target_type="npu",
            components=[npu_component],
            configuration=target_config or {},
        )

        # Create model
        model_hash = sha256(model_path)
        model_format = model_path.suffix.lstrip(".") if model_path.suffix else "unknown"
        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
        )

        # Create checks for each operator
        checks: list[schema.Check] = []
        entities: list[schema.Entity] = []

        for operator in self.ops:
            entity_id = operator.identity.entity_id
            supported = operator.run_on_npu.supported
            if supported is True:
                placement = schema.PlacementType.NPU.value
            elif operator.cpu_only:
                placement = schema.PlacementType.CPU.value
            else:
                placement = schema.PlacementType.UNKNOWN.value

            # Create entity for this operator
            entity = schema.Entity(
                id=entity_id,
                kind=operator.identity.entity_kind,
                name=operator.name,
                placement=placement,
                attributes={
                    "op_type": operator.op_type,
                    **operator.identity.attributes,
                },
            )
            entities.append(entity)

            # Create check for NPU placement
            if supported is True:
                status = schema.CheckStatus.PASS
                details = {}
            elif supported is False:
                status = schema.CheckStatus.FAIL
                details = {
                    "reasons": [
                        {"description": desc, "detail": detail}
                        for desc, detail in operator.run_on_npu.reasons
                        if (desc, detail) != _CPU_ONLY_REASON
                    ]
                }
            else:
                status = schema.CheckStatus.PARTIAL
                details = {"reasons": [_UNVALIDATED_OPERATOR_REASON.copy()]}

            check = schema.Check(
                id=f"npu_support_{entity_id}",
                status=status,
                entity_id=entity_id,
                details=details,
            )
            checks.append(check)

        # Determine overall result status
        if not self.is_complete:
            result_status = schema.ResultStatus.PARTIAL
        elif all(operator.run_on_npu.supported for operator in self.ops):
            result_status = schema.ResultStatus.OK
        elif any(operator.run_on_npu.supported for operator in self.ops):
            result_status = schema.ResultStatus.PARTIAL
        else:
            result_status = schema.ResultStatus.INCOMPATIBLE

        # Create result
        custom_entity_kinds = sorted(
            {
                operator.identity.entity_kind
                for operator in self.ops
                if operator.identity.entity_kind != schema.ENTITY_KIND_SOURCE_OPERATOR
            }
        )
        warnings = (
            [self.incomplete_reason or _INCOMPLETE_COMPATIBILITY_REASON]
            if not self.is_complete
            else []
        )
        warnings.extend(self.compiler_warnings)
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=backend.id,
            warnings=warnings,
            errors=[],
            metrics=self._accelerator_operator_percentage_metrics(),
            checks=checks,
            entities=entities,
            entity_kinds=[
                schema.EntityKind(id=entity_kind) for entity_kind in custom_entity_kinds
            ],
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        ).to_dict()


@dataclass
class VelaCompatibilityResult:
    """Canonical Vela compatibility result."""

    standardized_output: dict[str, Any]


def _operators_from_graph(
    graph: Any,
    vela_internal_ops: tuple,
    deps: VelaDeps,
    identity_factory: Callable[[int, int], OperatorIdentity],
    diagnostics: list[OperatorDiagnostic] | None = None,
    profile: SupportProfile | None = None,
    incomplete_reason: str | None = None,
    compiler_warnings: list[str] | None = None,
) -> Operators:
    """Extract checked operators while preserving graph-local coordinates."""
    pending_diagnostics: dict[tuple[str, str], list[OperatorDiagnostic]] = {}
    for diagnostic in diagnostics or []:
        pending_diagnostics.setdefault(
            _operator_key(diagnostic.op_type, diagnostic.name), []
        ).append(diagnostic)

    operators = []
    for subgraph_index, subgraph in enumerate(graph.subgraphs):
        for op in subgraph.get_all_ops():
            if op.type in vela_internal_ops:
                continue
            op_type = deps.optype_to_builtintype(op.type)
            matches = pending_diagnostics.get(_operator_key(op_type, op.name), [])
            matched_diagnostic = matches.pop(0) if matches else None
            support = _rejection_info(
                op_type,
                matched_diagnostic,
                profile,
                compilation_completed=incomplete_reason is None,
            )
            if incomplete_reason is not None and support.supported is True:
                support = NpuSupported(None, [])
            operators.append(
                Operator(
                    name=op.name,
                    op_type=op_type,
                    run_on_npu=support,
                    identity=identity_factory(subgraph_index, op.op_index),
                )
            )

    diagnostic_index = 0
    for unmatched in pending_diagnostics.values():
        for diagnostic in unmatched:
            operators.append(
                Operator(
                    name=diagnostic.name,
                    op_type=diagnostic.op_type,
                    run_on_npu=_rejection_info(
                        diagnostic.op_type,
                        diagnostic,
                        profile,
                        compilation_completed=incomplete_reason is None,
                    ),
                    identity=OperatorIdentity.diagnostic(diagnostic_index),
                )
            )
            diagnostic_index += 1
    return Operators(
        operators,
        incomplete_reason=incomplete_reason,
        compiler_warnings=compiler_warnings or [],
        support_profile=profile,
        support_catalog_version=deps.ethosu_vela_version if profile else None,
    )


def _supported_compiled_model_operators(
    model_path: Path,
    compiler_options: Any,
    vela_compiler: VelaCompiler,
    vela_internal_ops: tuple,
    deps: VelaDeps,
) -> Operators:
    """Extract PyTorch/TOSA-derived operators without claiming TFLite identity.

    Vela's per-layer CSV does not expose verified source operation coordinates
    for these input formats, so rows receive result-local performance identities.
    If the CSV is unavailable, operators read from Vela's generated TFLite
    artifact receive identities explicitly scoped to that compiled artifact.
    """
    diagnostic_lines: list[str] = []
    try:
        _, compiled_model_path = compile_vela_model_with_notice(
            logger,
            vela_compiler,
            model_path,
            compiler_options.output_dir,
            diagnostic_lines.append,
        )
    except RuntimeError as error:
        diagnostics = _operator_support_failure_diagnostics(error, diagnostic_lines)
        if not diagnostics:
            raise
        profile = _profile_for(compiler_options, "tosa", deps.ethosu_vela_version)
        return Operators(
            [
                Operator(
                    diagnostic.name,
                    diagnostic.op_type,
                    _rejection_info(
                        diagnostic.op_type,
                        diagnostic,
                        profile,
                        compilation_completed=False,
                    ),
                    OperatorIdentity.diagnostic(index),
                )
                for index, diagnostic in enumerate(diagnostics)
            ],
            incomplete_reason=_INCOMPLETE_COMPATIBILITY_REASON,
            support_profile=profile,
            support_catalog_version=deps.ethosu_vela_version if profile else None,
        )

    diagnostics = _parse_operator_diagnostics(diagnostic_lines)
    pending_diagnostics: dict[tuple[str, str], list[OperatorDiagnostic]] = {}
    for diagnostic in diagnostics:
        pending_diagnostics.setdefault(
            _operator_key(diagnostic.op_type, diagnostic.name), []
        ).append(diagnostic)
    profile = _profile_for(compiler_options, "tosa", deps.ethosu_vela_version)

    output_dir = compiler_options.output_dir
    model_name = model_path.stem
    csv_pattern = _get_layerwise_csv_pattern(model_name)
    csv_paths = list(Path(output_dir).glob(csv_pattern))

    if not csv_paths:
        logger.warning(
            "Layerwise CSV not found for %s, reading compiled model directly",
            compiled_model_path,
        )
        if compiled_model_path.suffix.lower() != ".tflite":
            logger.warning(
                "Compiled model is not TFLite (%s); skipping direct model read.",
                compiled_model_path,
            )
            return Operators([])
        graph, _ = vela_compiler._read_model(compiled_model_path)
        return _operators_from_graph(
            graph,
            vela_internal_ops,
            deps,
            OperatorIdentity.compiled_tflite,
            diagnostics,
            profile,
        )

    csv_path = csv_paths[0]
    original_layerwise_info = deps.parse_layerwise_perf_csv(
        vela_csv_file=csv_path, metrics=deps.layer_metrics
    )

    operators = []
    for idx, layer in enumerate(original_layerwise_info.layerwise_info):
        if (
            not layer.tflite_operator
            or layer.tflite_operator in _TFLITE_LAYERWISE_FILTERED_OP_NAMES
        ):
            continue
        name = layer.name or f"op_{idx}"
        matches = pending_diagnostics.get(
            _operator_key(layer.tflite_operator, name), []
        )
        matched_diagnostic = matches.pop(0) if matches else None
        placement = str(getattr(layer, "placement", "")).upper()
        support = (
            NpuSupported(True, [])
            if placement == "NPU"
            else _rejection_info(
                layer.tflite_operator,
                matched_diagnostic,
                profile,
                rejected=placement == "CPU",
            )
        )
        operators.append(
            Operator(
                name=name,
                op_type=layer.tflite_operator,
                run_on_npu=support,
                identity=OperatorIdentity.performance_layer(idx),
            )
        )

    diagnostic_index = 0
    for unmatched in pending_diagnostics.values():
        for diagnostic in unmatched:
            operators.append(
                Operator(
                    diagnostic.name,
                    diagnostic.op_type,
                    _rejection_info(diagnostic.op_type, diagnostic, profile),
                    OperatorIdentity.diagnostic(diagnostic_index),
                )
            )
            diagnostic_index += 1

    return Operators(
        operators,
        support_profile=profile,
        support_catalog_version=deps.ethosu_vela_version if profile else None,
    )


def supported_operators(model_path: Path, compiler_options: Any) -> Operators:
    """Return list of model's operators.

    For PyTorch and TOSA files, extracts operator information from Vela's
    layerwise performance CSV which preserves original operator details.
    For TFLite files, analyzes the model directly using Vela's Python API.
    """
    deps = _get_vela_deps()

    logger.debug("Check supported operators for the model %s", model_path)

    vela_internal_ops = (
        deps.Op.Placeholder,
        deps.Op.SubgraphInput,
        deps.Op.Const,
    )
    vela_compiler = deps.VelaCompiler(compiler_options)

    if is_pytorch_file(model_path) or is_tosa_file(model_path):
        return _supported_compiled_model_operators(
            model_path, compiler_options, vela_compiler, vela_internal_ops, deps
        )

    graph, _ = vela_compiler._read_model(model_path)
    diagnostic_lines: list[str] = []
    compiler_warnings: list[str] = []
    try:
        compile_vela_model_with_notice(
            logger,
            vela_compiler,
            model_path,
            getattr(compiler_options, "output_dir", Path("output")),
            diagnostic_lines.append,
        )
        diagnostics = _parse_operator_diagnostics(diagnostic_lines)
        incomplete_reason = None
    except MemoryError as error:
        if str(error) != _SRAM_CAPACITY_MESSAGE or not any(
            _SRAM_CAPACITY_WARNING in line for line in diagnostic_lines
        ):
            raise
        diagnostics = _parse_operator_diagnostics(diagnostic_lines)
        incomplete_reason = None
        compiler_warnings.append(_SRAM_COMPATIBILITY_WARNING)
    except RuntimeError as error:
        diagnostics = _operator_support_failure_diagnostics(error, diagnostic_lines)
        if not diagnostics:
            raise
        incomplete_reason = _INCOMPLETE_COMPATIBILITY_REASON

    return _operators_from_graph(
        graph,
        vela_internal_ops,
        deps,
        OperatorIdentity.tflite,
        diagnostics,
        _profile_for(compiler_options, "tflite", deps.ethosu_vela_version),
        incomplete_reason,
        compiler_warnings,
    )


def generate_supported_operators_report() -> None:
    """Generate supported operators report in current working directory."""
    deps = _get_vela_deps()

    with redirect_output(logger):
        deps.generate_supported_ops()


def get_vela() -> bool:
    """Check if vela backend is available."""
    try:
        _get_vela_deps()
    except BackendUnavailableError:
        return False
    return True
