# SPDX-FileCopyrightText: Copyright 2022, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela operator compatibility module."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.errors import BackendUnavailableError
from mlia.target.ethos_u.utils.model_format import is_pytorch_file, is_tosa_file
from mlia.utils.filesystem import sha256
from mlia.utils.logging import redirect_output

try:
    from ethosu.vela import __version__ as ethosu_vela_version
    from ethosu.vela.operation import Op
    from ethosu.vela.tflite_mapping import optype_to_builtintype
    from ethosu.vela.tflite_model_semantic import TFLiteSemantic
    from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators
    from ethosu.vela.vela import generate_supported_ops

    from mlia.backend.vela.compiler import VelaCompiler  # pylint: disable=C0412
    from mlia.backend.vela.performance import layer_metrics
    from mlia.backend.vela.performance import parse_layerwise_perf_csv

    _VELA_INSTALLED = True

except ImportError:
    if TYPE_CHECKING:
        from ethosu.vela import __version__ as ethosu_vela_version
        from ethosu.vela.operation import Op
        from ethosu.vela.tflite_mapping import optype_to_builtintype
        from ethosu.vela.tflite_model_semantic import TFLiteSemantic
        from ethosu.vela.tflite_supported_operators import TFLiteSupportedOperators
        from ethosu.vela.vela import generate_supported_ops

        from mlia.backend.vela.compiler import VelaCompiler
        from mlia.backend.vela.performance import layer_metrics
        from mlia.backend.vela.performance import parse_layerwise_perf_csv
    else:

        def __getattr__(name: str) -> Any:
            """Raise BackendUnavailableError for Vela-related attributes."""
            if name in {
                "Op",
                "optype_to_builtintype",
                "TFLiteSemantic",
                "TFLiteSupportedOperators",
                "generate_supported_ops",
                "VelaCompiler",
                "ethosu_vela_version",
                "layer_metrics",
                "parse_layerwise_perf_csv",
            }:
                raise BackendUnavailableError("Backend vela is not available", "vela")
            raise AttributeError(name)

    _VELA_INSTALLED = False

logger = logging.getLogger(__name__)

# Glob pattern for Vela layerwise CSV files
_VELA_LAYERWISE_CSV_GLOB_PATTERN = "*{model_name}*per-layer.csv"

# TFLite operator names to filter from layerwise data
_TFLITE_LAYERWISE_FILTERED_OP_NAMES = ["Placeholder", "Const"]


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


@dataclass
class Operator:
    """Model operator."""

    name: str
    op_type: str
    run_on_npu: NpuSupported

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

    def to_standardized_output(  # pylint: disable=too-many-locals
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
        # pylint: disable=duplicate-code
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version
        try:
            backend_version = ethosu_vela_version
        except Exception as exc:  # pylint: disable=broad-exception-caught
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

        for idx, operator in enumerate(self.ops):
            entity_id = f"op_{idx}"

            # Create entity for this operator
            entity = schema.Entity(
                scope=schema.OperatorScope.OPERATOR,
                name=operator.name,
                location=f"operator/{idx}",
                placement="npu" if operator.run_on_npu.supported else "cpu",
                id=entity_id,
                attributes={
                    "op_type": operator.op_type,
                    "index": idx,
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
        result = schema.Result(
            kind=schema.ResultKind.COMPATIBILITY,
            status=result_status,
            producer=backend.id,
            warnings=[],
            errors=[],
            checks=checks,
            entities=entities,
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


def _supported_pytorch_operators(
    model_path: Path,
    compiler_options: Any,
    vela_compiler: VelaCompiler,
    vela_internal_ops: tuple,
) -> Operators:
    """Extract operators from PyTorch/TOSA models via layerwise CSV.

    For PyTorch and TOSA files, we need to compile first and extract operators
    from the layerwise CSV, since direct model reading after compilation shows
    only fused ops.

    Args:
        model_path: Path to the PyTorch/TOSA model file
        compiler_options: Vela compiler options
        vela_compiler: VelaCompiler instance
        vela_internal_ops: Tuple of internal Vela operations to filter out

    Returns:
        Operators object containing the model's operators
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
        # pylint: disable=protected-access
        graph, _ = vela_compiler._read_model(compiled_model_path)
        return Operators(
            [
                Operator(op.name, optype_to_builtintype(op.type), _run_on_npu(op))
                for sg in graph.subgraphs
                for op in sg.get_all_ops()
                if op.type not in vela_internal_ops
            ]
        )

    csv_path = csv_paths[0]
    original_layerwise_info = parse_layerwise_perf_csv(
        vela_csv_file=csv_path, metrics=layer_metrics
    )

    operators = [
        Operator(
            layer.name or f"op_{idx}",
            layer.tflite_operator,
            NpuSupported(True, []),
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
    # pylint: disable=too-many-locals,protected-access
    if not get_vela():
        raise BackendUnavailableError("Backend vela is not available", "vela")

    logger.debug("Check supported operators for the model %s", model_path)

    vela_internal_ops = (Op.Placeholder, Op.SubgraphInput, Op.Const)
    vela_compiler = VelaCompiler(compiler_options)

    if is_pytorch_file(model_path) or is_tosa_file(model_path):
        return _supported_pytorch_operators(
            model_path, compiler_options, vela_compiler, vela_internal_ops
        )

    graph, _ = vela_compiler._read_model(model_path)

    return Operators(
        [
            Operator(op.name, optype_to_builtintype(op.type), _run_on_npu(op))
            for sg in graph.subgraphs
            for op in sg.get_all_ops()
            if op.type not in vela_internal_ops
        ]
    )


def _run_on_npu(operator) -> NpuSupported:  # type: ignore
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
    if not get_vela():
        raise BackendUnavailableError("Backend vela is not available", "vela")

    vela_internal_ops = (Op.Placeholder, Op.SubgraphInput, Op.Const)
    semantic_checker = TFLiteSemantic()
    semantic_constraints = itertools.chain(
        semantic_checker.generic_constraints,
        semantic_checker.specific_constraints[operator.type],
    )

    for constraint in semantic_constraints:
        op_valid, op_reason = constraint(operator)
        if not op_valid:
            return NpuSupported(False, [(constraint.__doc__, op_reason)])

    if operator.type not in TFLiteSupportedOperators.supported_operators:
        reasons = (
            [("CPU only operator", "")]
            if operator.type not in vela_internal_ops
            else []
        )

        return NpuSupported(False, reasons)

    tflite_supported_operators = TFLiteSupportedOperators()
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
    if not get_vela():
        raise BackendUnavailableError("Backend vela is not available", "vela")

    with redirect_output(logger):
        generate_supported_ops()


def get_vela() -> bool:
    """Check if vela backend is available."""
    return _VELA_INSTALLED
