# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela operator compatibility module."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mlia
import mlia.core.output_schema as schema
from mlia.backend.errors import BackendUnavailableError
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
    TFLiteSemantic: Any
    TFLiteSupportedOperators: Any
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
        from ethosu.vela.tflite_model_semantic import TFLiteSemantic
        from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators
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
        TFLiteSemantic=TFLiteSemantic,
        TFLiteSupportedOperators=TFLiteSupportedOperators,
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

    supported: bool
    reasons: list[tuple[str, str]]


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
        cpu_only_reasons = [("CPU only operator", "")]
        return (
            not self.run_on_npu.supported
            and self.run_on_npu.reasons == cpu_only_reasons
        )


@dataclass
class Operators:
    """Model's operators."""

    ops: list[Operator]

    @property
    def npu_supported_ratio(self) -> float:
        """Return NPU supported ratio."""
        total = self.total_number
        npu_supported = self.npu_supported_number

        if total == 0 or npu_supported == 0:
            return 0

        return npu_supported / total

    @property
    def npu_unsupported_ratio(self) -> float:
        """Return NPU unsupported ratio."""
        return 1 - self.npu_supported_ratio

    @property
    def total_number(self) -> int:
        """Return total number of operators."""
        return len(self.ops)

    @property
    def npu_supported_number(self) -> int:
        """Return number of npu supported operators."""
        return sum(op.run_on_npu.supported for op in self.ops)

    def _accelerator_operator_percentage_metrics(self) -> list[schema.Metric]:
        """Build compatibility operator percentage metrics."""
        if self.total_number:
            return [
                schema.Metric(
                    name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                    value=self.npu_supported_ratio * 100,
                    unit=schema.UNIT_PERCENT,
                )
            ]

        return [
            schema.Metric(
                name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason="Operator placement data is not available.",
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

        npu_component = schema.Component(
            type=schema.ComponentType.NPU,
            family=target_type,
            model=None,
            variant=mac if mac != "unknown" else None,
        )

        target = schema.Target(
            profile_name=target_type,
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

            # Create entity for this operator
            entity = schema.Entity(
                id=entity_id,
                kind=operator.identity.entity_kind,
                name=operator.name,
                placement=(
                    schema.PlacementType.NPU.value
                    if operator.run_on_npu.supported
                    else schema.PlacementType.CPU.value
                ),
                attributes={
                    "op_type": operator.op_type,
                    **operator.identity.attributes,
                },
            )
            entities.append(entity)

            # Create check for NPU placement
            if operator.run_on_npu.supported:
                status = schema.CheckStatus.PASS
                details = {}
            else:
                status = schema.CheckStatus.FAIL
                details = {
                    "reasons": [
                        {"description": desc, "detail": detail}
                        for desc, detail in operator.run_on_npu.reasons
                    ]
                }

            check = schema.Check(
                id=f"npu_support_{entity_id}",
                status=status,
                entity_id=entity_id,
                details=details,
            )
            checks.append(check)

        # Determine overall result status
        if all(operator.run_on_npu.supported for operator in self.ops):
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
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=backend.id,
            warnings=[],
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
    """Wrapper for Vela compatibility with both legacy and standardized output."""

    legacy_info: Operators
    standardized_output: dict[str, Any] | None = None


def _operators_from_graph(
    graph: Any,
    vela_internal_ops: tuple,
    deps: VelaDeps,
    identity_factory: Callable[[int, int], OperatorIdentity],
) -> Operators:
    """Extract checked operators while preserving graph-local coordinates."""
    operators = []
    for subgraph_index, subgraph in enumerate(graph.subgraphs):
        for op in subgraph.get_all_ops():
            if op.type in vela_internal_ops:
                continue
            operators.append(
                Operator(
                    name=op.name,
                    op_type=deps.optype_to_builtintype(op.type),
                    run_on_npu=_run_on_npu(op, deps),
                    identity=identity_factory(subgraph_index, op.op_index),
                )
            )
    return Operators(operators)


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
    _, compiled_model_path = vela_compiler.compile_model(model_path)

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
        )

    csv_path = csv_paths[0]
    original_layerwise_info = deps.parse_layerwise_perf_csv(
        vela_csv_file=csv_path, metrics=deps.layer_metrics
    )

    operators = [
        Operator(
            name=layer.name or f"op_{idx}",
            op_type=layer.tflite_operator,
            run_on_npu=NpuSupported(True, []),
            identity=OperatorIdentity.performance_layer(idx),
        )
        for idx, layer in enumerate(original_layerwise_info.layerwise_info)
        if layer.tflite_operator
        and layer.tflite_operator not in _TFLITE_LAYERWISE_FILTERED_OP_NAMES
    ]

    return Operators(operators)


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

    return _operators_from_graph(
        graph,
        vela_internal_ops,
        deps,
        OperatorIdentity.tflite,
    )


def _run_on_npu(operator, deps: VelaDeps) -> NpuSupported:
    """Return information if operator can run on NPU.

    Vela does a number of checks that can help establish whether
    a particular operator is supported to run on NPU.

    There are two groups of checks:
      - general TensorFlow Lite constraints
      - operator specific constraints

    If an operator is not supported on NPU then this function
    will return the reason of that.

    The reason is split in two parts:
      - general description of why the operator cannot be placed on NPU
      - details on the particular operator
    """
    vela_internal_ops = (
        deps.Op.Placeholder,
        deps.Op.SubgraphInput,
        deps.Op.Const,
    )
    semantic_checker = deps.TFLiteSemantic()
    semantic_constraints = itertools.chain(
        semantic_checker.generic_constraints,
        semantic_checker.specific_constraints[operator.type],
    )

    for constraint in semantic_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    if operator.type not in deps.TFLiteSupportedOperators.supported_operators:
        reasons = (
            [("CPU only operator", "")]
            if operator.type not in vela_internal_ops
            else []
        )

        return NpuSupported(False, reasons)

    tflite_supported_operators = deps.TFLiteSupportedOperators()
    operation_constraints = itertools.chain(
        tflite_supported_operators.generic_constraints,
        tflite_supported_operators.specific_constraints[operator.type],
    )
    for constraint in operation_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    return NpuSupported(True, [])


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
