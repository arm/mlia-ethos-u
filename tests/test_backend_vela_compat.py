# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compat."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import ANY, MagicMock, call

from packaging.version import Version
import pytest
import tensorflow as tf
import tf_keras as keras

import mlia.backend.vela.compat as vela_compat
import mlia.core.output_schema as schema
from mlia.backend.boundaries import process as backend_process
from mlia.backend.vela._supported_ops_types import SupportedOpsCatalog
from mlia.backend.vela.supported_ops import parse_report
from mlia.core.output_rendering import standardized_output_to_json
from mlia.core.output_rendering import standardized_output_to_text
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela.compat import (
    NpuSupported,
    Operator,
    OperatorIdentity,
    Operators,
    generate_supported_operators_report,
    get_vela,
    supported_operators,
)
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.utils.filesystem import working_directory

TEST_MODEL_TFLITE_INT8_FILE = "test_model_int8.tflite"
CATALOG = parse_report(
    (
        Path(__file__).parents[1] / "tests/test_resources/vela/SUPPORTED_OPS.md"
    ).read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def use_fixture_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep compatibility tests independent of the persistent user cache."""
    monkeypatch.setattr(
        vela_compat,
        "load_catalog",
        lambda version: CATALOG if version == CATALOG.vela_version else None,
    )


def fail_load_vela_deps() -> None:
    """Raise the same error as a failed Vela dependency load."""
    raise BackendUnavailableError("Backend vela is not available", "vela")


@pytest.mark.parametrize(
    "name, op_type, npu_supported",
    [
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(False, [("CPU only operator", "")]),
        ),
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(True, []),
        ),
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(False, [("Other reason", "")]),
        ),
        (
            "sequential/conv1/Relu;sequential/conv1/BiasAdd;",
            "CONV_2D",
            NpuSupported(
                False,
                [
                    ("CPU only operator", ""),
                    ("Detailed reason", "Catalog explanation"),
                ],
            ),
        ),
    ],
)
def test_operator(name: str, op_type: str, npu_supported: NpuSupported) -> None:
    """Test Operator class."""
    operator = Operator(
        name,
        op_type,
        npu_supported,
        OperatorIdentity.tflite(subgraph_index=0, operator_index=0),
    )
    cpu_only = (
        not npu_supported.supported
        and (
            "CPU only operator",
            "",
        )
        in npu_supported.reasons
    )
    assert operator.cpu_only == cpu_only


@pytest.mark.parametrize(
    "ops",
    [
        [
            Operator(
                name="sequential/conv1/Relu;sequential/conv1/BiasAdd;"
                "sequential/conv2/Conv2D;sequential/conv1/Conv2D",
                op_type="CONV_2D",
                run_on_npu=NpuSupported(supported=True, reasons=[]),
                identity=OperatorIdentity.tflite(0, 0),
            ),
            Operator(
                name="sequential/conv2/Relu;sequential/conv2/BiasAdd;"
                "sequential/conv2/Conv2D",
                op_type="CONV_2D",
                run_on_npu=NpuSupported(supported=True, reasons=[]),
                identity=OperatorIdentity.tflite(0, 1),
            ),
            Operator(
                name="sequential/max_pooling2d/MaxPool",
                op_type="MAX_POOL_2D",
                run_on_npu=NpuSupported(supported=False, reasons=[]),
                identity=OperatorIdentity.tflite(0, 2),
            ),
        ],
        [],
    ],
)
def test_operators(ops: list[Operator]) -> None:
    """Test operators function."""
    operators = Operators(ops)

    total_ops = len(ops)
    npu_supported_ops = sum(op.run_on_npu.supported for op in ops)

    assert operators.total_number == total_ops
    assert operators.npu_supported_number == npu_supported_ops

    if total_ops > 0:
        assert operators.npu_supported_ratio == npu_supported_ops / total_ops

    assert operators.npu_unsupported_ratio == 1 - operators.npu_supported_ratio


@pytest.mark.parametrize(
    "model, expected_ops",
    [
        (
            TEST_MODEL_TFLITE_INT8_FILE,
            Operators(
                ops=[
                    Operator(
                        name="sequential/conv1/Relu;sequential/conv1/BiasAdd;"
                        "sequential/conv2/Conv2D;sequential/conv1/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                        identity=OperatorIdentity.tflite(0, 0),
                    ),
                    Operator(
                        name="sequential/conv2/Relu;sequential/conv2/BiasAdd;"
                        "sequential/conv2/Conv2D",
                        op_type="CONV_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                        identity=OperatorIdentity.tflite(0, 1),
                    ),
                    Operator(
                        name="sequential/max_pooling2d/MaxPool",
                        op_type="MAX_POOL_2D",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                        identity=OperatorIdentity.tflite(0, 2),
                    ),
                    Operator(
                        name="sequential/flatten/Reshape",
                        op_type="RESHAPE",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                        identity=OperatorIdentity.tflite(0, 3),
                    ),
                    Operator(
                        name="Identity",
                        op_type="FULLY_CONNECTED",
                        run_on_npu=NpuSupported(supported=True, reasons=[]),
                        identity=OperatorIdentity.tflite(0, 4),
                    ),
                ]
            ),
        )
    ],
)
def test_supported_operators(
    test_models_path: Path, tmp_path: Path, model: str, expected_ops: Operators
) -> None:
    """Test operators function."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    if target_config.compiler_options is not None:
        target_config.compiler_options.output_dir = tmp_path

    try:
        operators = supported_operators(
            test_models_path / model, target_config.compiler_options
        )
        assert len(operators.ops) == len(expected_ops.ops)
        for expected, actual in zip(expected_ops.ops, operators.ops):
            # do not compare names as they could be different on each model generation
            assert expected.op_type == actual.op_type
            assert expected.identity == actual.identity
            assert isinstance(actual.run_on_npu.supported, bool)
            if actual.run_on_npu.supported:
                assert actual.run_on_npu.reasons == []
            else:
                assert isinstance(actual.run_on_npu.reasons, list)
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip("Vela backend not available, skipping operators test")


def test_supported_operators_preserves_tflite_subgraph_and_op_indexes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Direct TFLite identities should use Vela's original graph coordinates."""
    placeholder_type = object()
    subgraph_input_type = object()
    const_type = object()
    checked_type = object()
    main_op = SimpleNamespace(name="main", type=checked_type, op_index=7)
    nested_op = SimpleNamespace(name="nested", type=checked_type, op_index=7)
    nested_other_op = SimpleNamespace(
        name="nested_other", type=checked_type, op_index=2
    )
    graph = SimpleNamespace(
        subgraphs=[
            SimpleNamespace(
                get_all_ops=MagicMock(
                    return_value=[
                        SimpleNamespace(
                            name="input", type=placeholder_type, op_index=None
                        ),
                        main_op,
                    ]
                )
            ),
            SimpleNamespace(
                get_all_ops=MagicMock(return_value=[nested_op, nested_other_op])
            ),
        ]
    )
    compiler = MagicMock()
    compiler._read_model.return_value = (graph, None)
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=placeholder_type,
            SubgraphInput=subgraph_input_type,
            Const=const_type,
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=MagicMock(return_value="CONV_2D"),
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    operators = supported_operators(
        model_file, compiler_options=SimpleNamespace(accelerator_config="ethos-u55-256")
    )

    assert operators.support_profile is CATALOG.profiles["ethos-u55", "tflite"]
    assert operators.support_catalog_version == "5.2.0"
    assert [operator.identity.entity_id for operator in operators.ops] == [
        "source_operator/operator/7",
        "source_operator/subgraph/1/operator/7",
        "source_operator/subgraph/1/operator/2",
    ]
    output = operators.to_standardized_output(model_path=model_file)
    result = output["results"][0]
    assert [entity["id"] for entity in result["entities"]] == [
        "source_operator/operator/7",
        "source_operator/subgraph/1/operator/7",
        "source_operator/subgraph/1/operator/2",
    ]
    assert [check["entity_id"] for check in result["checks"]] == [
        "source_operator/operator/7",
        "source_operator/subgraph/1/operator/7",
        "source_operator/subgraph/1/operator/2",
    ]
    assert result.get("entity_kinds", []) == []


def test_parse_operator_diagnostics() -> None:
    """Regor warning blocks retain the concrete reason and failed constraint."""
    diagnostics = vela_compat._parse_operator_diagnostics(
        [
            "unrelated output",
            "Warning (supported operators) operator: CONV_2D, ofm: 'output'",
            "Reason: Operation has tensor with unsupported DataType Float32 (input)",
            'Feature-map dataTypes must be one of {"INT8", "UINT8"}',
            "",
        ]
    )

    assert diagnostics == [
        vela_compat.OperatorDiagnostic(
            op_type="CONV_2D",
            name="output",
            reason="Operation has tensor with unsupported DataType Float32 (input)",
            constraint='Feature-map dataTypes must be one of {"INT8", "UINT8"}',
        )
    ]


def test_parse_tflite_semantic_error() -> None:
    """Regor semantic failures retain operator identity and concrete details."""
    diagnostics = vela_compat._parse_tflite_semantic_errors(
        [
            "Compilation failed: Error: HARD_SWISH 'PartitionedCall:0' Does not "
            "have valid TFLite Semantics.\n"
            " - IFM has to be 8bit\n"
            "   IFM type=FLOAT32, \n"
            "Failed to parse TFLite file\n"
        ]
    )

    assert diagnostics == [
        vela_compat.OperatorDiagnostic(
            op_type="HARD_SWISH",
            name="PartitionedCall:0",
            reason="IFM type=FLOAT32",
            constraint="IFM has to be 8bit",
        )
    ]


def test_supported_operators_uses_regor_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A Regor warning is associated with the matching parsed source operator."""
    checked_type = object()
    graph = SimpleNamespace(
        subgraphs=[
            SimpleNamespace(
                get_all_ops=MagicMock(
                    return_value=[
                        SimpleNamespace(name="output", type=checked_type, op_index=3)
                    ]
                )
            )
        ]
    )
    compiler = MagicMock()
    compiler._read_model.return_value = (graph, None)

    def compile_model(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> tuple[None, Path]:
        for line in [
            "Warning (supported operators) operator: CONV_2D, ofm: 'output'",
            "Reason: Float32 input tensor",
            "Feature-map dataTypes must be quantized",
            "",
        ]:
            diagnostic_consumer(line)
        return None, tmp_path / "model_vela.tflite"

    compiler.compile_model.side_effect = compile_model
    deps = SimpleNamespace(
        ethosu_vela_version="5.3.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=MagicMock(return_value="CONV_2D"),
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    operators = supported_operators(
        model_file,
        SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
    )

    assert operators.ops[0].identity == OperatorIdentity.tflite(0, 3)
    assert operators.ops[0].run_on_npu == NpuSupported(
        False,
        [("Feature-map dataTypes must be quantized", "Float32 input tensor")],
    )
    output = operators.to_standardized_output(model_file)
    assert output["results"][0]["checks"][0]["details"]["reasons"] == [
        {
            "description": "Feature-map dataTypes must be quantized",
            "detail": "Float32 input tensor",
        }
    ]


def test_supported_operators_announces_compilation_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TFLite compatibility compilation announces its derived output."""
    compiler = MagicMock()
    compiler._read_model.return_value = (SimpleNamespace(subgraphs=[]), None)
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
    )
    events = MagicMock()
    compiler.compile_model = events.compile
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    monkeypatch.setattr(backend_process, "log_boundary_action", events.notice)
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    supported_operators(
        model_file,
        SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
    )

    assert events.mock_calls == [
        call.notice(
            vela_compat.logger,
            "Running Vela compatibility analysis for '%s'; derived output will be "
            "written under '%s'.",
            model_file,
            tmp_path,
        ),
        call.compile(model_file, diagnostic_consumer=ANY),
    ]


def test_supported_operators_propagates_failure_after_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A diagnostic must not disguise a later fatal TFLite compiler error."""
    compiler = MagicMock()
    compiler._read_model.return_value = (SimpleNamespace(subgraphs=[]), None)

    def fail_after_diagnostic(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        diagnostic_consumer(
            "Warning (supported operators) operator: CONV_2D, ofm: 'output'"
        )
        raise RuntimeError("fatal compiler error")

    compiler.compile_model.side_effect = fail_after_diagnostic
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="fatal compiler error"):
        supported_operators(
            model_file,
            SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
        )


def test_tflite_support_error_marks_unvalidated_operators_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stopped compiler cannot confirm placement for remaining graph operators."""
    rejected_type, unchecked_type, excluded_type = object(), object(), object()
    graph = SimpleNamespace(
        subgraphs=[
            SimpleNamespace(
                get_all_ops=MagicMock(
                    return_value=[
                        SimpleNamespace(
                            name="rejected", type=rejected_type, op_index=0
                        ),
                        SimpleNamespace(
                            name="unchecked", type=unchecked_type, op_index=1
                        ),
                        SimpleNamespace(
                            name="excluded", type=excluded_type, op_index=2
                        ),
                    ]
                )
            )
        ]
    )
    compiler = MagicMock()
    compiler._read_model.return_value = (graph, None)

    def fail_on_support(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        for line in (
            "Warning (supported operators) operator: CONV_2D, ofm: 'rejected'",
            "Reason: Input rank is five",
            "Input rank must be four",
            "",
        ):
            diagnostic_consumer(line)
        raise RuntimeError(
            "Compilation failed: Unsupported operator CONV_2D 'rejected': "
            "Input rank must be four"
        )

    compiler.compile_model.side_effect = fail_on_support
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=lambda op_type: {
            rejected_type: "CONV_2D",
            unchecked_type: "ADD",
            excluded_type: "CUSTOM",
        }[op_type],
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", lambda: deps)
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    operators = supported_operators(
        model_file,
        SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
    )
    result = operators.to_standardized_output(model_file)["results"][0]

    assert [op.run_on_npu.supported for op in operators.ops] == [False, None, False]
    assert not operators.ops[0].cpu_only
    assert not operators.ops[2].cpu_only
    assert operators.npu_supported_ratio is None
    assert result["status"] == "partial"
    assert result["warnings"]
    assert [entity["placement"] for entity in result["entities"]] == [
        "Unknown",
        "Unknown",
        "Unknown",
    ]
    assert [check["status"] for check in result["checks"]] == [
        "fail",
        "partial",
        "fail",
    ]
    assert result["checks"][1]["details"]["reasons"] == [
        {
            "description": "Operator support was not determined",
            "detail": "Vela stopped before validating this operator.",
        }
    ]
    assert (
        "CPU fallback was not confirmed"
        in result["checks"][2]["details"]["reasons"][0]["detail"]
    )
    assert result["metrics"] == [
        {
            "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
            "unit": schema.UNIT_PERCENT,
            "availability": "unavailable",
            "reason": vela_compat._INCOMPLETE_COMPATIBILITY_REASON,
        }
    ]


def test_tflite_semantic_failure_keeps_earlier_operator_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A semantic abort retains earlier failures without duplicating its own."""
    prior_type, semantic_type, unchecked_type = object(), object(), object()
    graph = SimpleNamespace(
        subgraphs=[
            SimpleNamespace(
                get_all_ops=lambda: [
                    SimpleNamespace(name="prior", type=prior_type, op_index=0),
                    SimpleNamespace(name="semantic", type=semantic_type, op_index=1),
                    SimpleNamespace(name="unchecked", type=unchecked_type, op_index=2),
                ]
            )
        ]
    )
    compiler = MagicMock()
    compiler._read_model.return_value = (graph, None)

    def fail_on_semantics(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        for line in (
            "Warning (supported operators) operator: CONV_2D, ofm: 'prior'",
            "Reason: Invalid input rank",
            "Input rank must be four",
            "",
            "Warning (supported operators) operator: HARD_SWISH, ofm: 'semantic'",
            "Reason: IFM type=FLOAT32",
            "IFM has to be 8bit",
            "",
        ):
            diagnostic_consumer(line)
        raise RuntimeError(
            "Compilation failed: Error: HARD_SWISH 'semantic' Does not have "
            "valid TFLite Semantics.\n"
            " - IFM has to be 8bit\n"
            "   IFM type=FLOAT32, \n"
            "Failed to parse TFLite file\n"
        )

    compiler.compile_model.side_effect = fail_on_semantics
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=lambda op_type: {
            prior_type: "CONV_2D",
            semantic_type: "HARD_SWISH",
            unchecked_type: "ADD",
        }[op_type],
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", lambda: deps)
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    operators = supported_operators(
        model_file,
        SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
    )
    result = operators.to_standardized_output(model_file)["results"][0]

    assert len(operators.ops) == 3
    assert [op.run_on_npu.supported for op in operators.ops] == [False, False, None]
    assert [check["status"] for check in result["checks"]] == [
        "fail",
        "fail",
        "partial",
    ]
    assert result["checks"][0]["details"]["reasons"][0] == {
        "description": "Input rank must be four",
        "detail": "Invalid input rank",
    }
    assert result["checks"][1]["details"]["reasons"][0] == {
        "description": "IFM has to be 8bit",
        "detail": "IFM type=FLOAT32",
    }
    assert result["metrics"][0]["availability"] == "unavailable"


def test_tflite_sram_capacity_warning_preserves_operator_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The wrapper's post-compile SRAM error still leaves support data usable."""
    rejected_type, supported_type = object(), object()
    graph = SimpleNamespace(
        subgraphs=[
            SimpleNamespace(
                get_all_ops=lambda: [
                    SimpleNamespace(name="rejected", type=rejected_type, op_index=0),
                    SimpleNamespace(name="supported", type=supported_type, op_index=1),
                ]
            )
        ]
    )
    compiler = MagicMock()
    compiler._read_model.return_value = (graph, None)

    def exceed_sram(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        for line in (
            "Warning (supported operators) operator: CONV_2D, ofm: 'rejected'",
            "Reason: Invalid input rank",
            "Input rank must be four",
            "",
            vela_compat._SRAM_CAPACITY_WARNING,
        ):
            diagnostic_consumer(line)
        raise MemoryError(vela_compat._SRAM_CAPACITY_MESSAGE)

    compiler.compile_model.side_effect = exceed_sram
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=lambda op_type: {
            rejected_type: "CONV_2D",
            supported_type: "ADD",
        }[op_type],
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", lambda: deps)
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    operators = supported_operators(
        model_file,
        SimpleNamespace(accelerator_config="ethos-u55-256", output_dir=tmp_path),
    )
    result = operators.to_standardized_output(model_file)["results"][0]

    assert operators.is_complete
    assert [check["status"] for check in result["checks"]] == ["fail", "pass"]
    assert [entity["placement"] for entity in result["entities"]] == ["CPU", "NPU"]
    assert result["metrics"][0]["value"] == 50
    assert result["warnings"] == [vela_compat._SRAM_COMPATIBILITY_WARNING]
    assert result["checks"][0]["details"]["reasons"][0] == {
        "description": "Input rank must be four",
        "detail": "Invalid input rank",
    }


@pytest.mark.parametrize(
    "message, emit_sram_warning",
    [
        ("unrelated allocation failure", True),
        (vela_compat._SRAM_CAPACITY_MESSAGE, False),
    ],
)
def test_tflite_unconfirmed_memory_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    message: str,
    emit_sram_warning: bool,
) -> None:
    """Only the wrapper's identified post-compile SRAM condition is recoverable."""
    compiler = MagicMock()
    compiler._read_model.return_value = (SimpleNamespace(subgraphs=[]), None)

    def fail_compilation(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        if emit_sram_warning:
            diagnostic_consumer(vela_compat._SRAM_CAPACITY_WARNING)
        raise MemoryError(message)

    compiler.compile_model.side_effect = fail_compilation
    deps = SimpleNamespace(
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        VelaCompiler=MagicMock(return_value=compiler),
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", lambda: deps)
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")

    with pytest.raises(MemoryError, match=message):
        supported_operators(model_file, SimpleNamespace(output_dir=tmp_path))


def test_supported_operators_recovers_float32_hard_swish_rejection(
    tmp_path: Path,
) -> None:
    """A Regor semantic rejection remains an actionable compatibility result."""
    input_tensor = keras.Input(shape=(8, 8, 1), batch_size=1, name="input")
    output_tensor = keras.layers.Lambda(
        lambda value: value * tf.nn.relu6(value + 3.0) / 6.0,
        name="hard_swish",
    )(input_tensor)
    model = keras.Model(input_tensor, output_tensor)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_file = tmp_path / "hard_swish.tflite"
    model_file.write_bytes(converter.convert())
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    target_config.compiler_options.output_dir = tmp_path

    try:
        operators = supported_operators(model_file, target_config.compiler_options)
    except BackendUnavailableError:
        pytest.skip("Vela backend not available")

    hard_swish = next(op for op in operators.ops if op.op_type == "HARD_SWISH")
    assert hard_swish.name == "PartitionedCall:0"
    assert not hard_swish.cpu_only
    assert ("IFM has to be 8bit", "IFM type=FLOAT32") in hard_swish.run_on_npu.reasons
    result = operators.to_standardized_output(model_file)["results"][0]
    assert (
        next(
            entity["placement"]
            for entity in result["entities"]
            if entity["id"] == hard_swish.identity.entity_id
        )
        == "Unknown"
    )
    checks = result["checks"]
    hard_swish_check = next(
        check for check in checks if check["entity_id"] == hard_swish.identity.entity_id
    )
    assert hard_swish_check["details"]["reasons"][0] == {
        "description": "IFM has to be 8bit",
        "detail": "IFM type=FLOAT32",
    }
    assert len(hard_swish_check["details"]["reasons"]) == 1


def test_profile_for_version_mismatch_uses_fallback() -> None:
    """An unavailable runtime report leaves catalog advice disabled."""
    profile = vela_compat._profile_for(
        SimpleNamespace(accelerator_config="ethos-u55-256"),
        "tflite",
        "5.3.0",
    )

    assert profile is None


def test_profile_for_matching_version_uses_catalog() -> None:
    """A matching runtime enables catalog-backed compatibility advice."""
    profile = vela_compat._profile_for(
        SimpleNamespace(accelerator_config="ethos-u55-256"),
        "tflite",
        "5.2.0",
    )

    assert profile is CATALOG.profiles["ethos-u55", "tflite"]


def test_profile_for_missing_report_section_uses_diagnostics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing target profile does not stop operator diagnostics."""
    monkeypatch.setattr(
        vela_compat,
        "load_catalog",
        lambda _version: SupportedOpsCatalog("5.2.0", {}),
    )

    assert (
        vela_compat._profile_for(
            SimpleNamespace(accelerator_config="ethos-u55-256"), "tflite", "5.2.0"
        )
        is None
    )
    assert "no parsed support profile" in caplog.text


def test_rejection_info_preserves_catalog_cpu_fallback_placement(
    tmp_path: Path,
) -> None:
    """Catalog fallback retains CPU placement and its detailed explanation."""
    profile = CATALOG.profiles["ethos-u55", "tflite"]

    rejection = vela_compat._rejection_info("UNKNOWN_OP", None, profile)
    operator = Operator(
        "output",
        "UNKNOWN_OP",
        rejection,
        OperatorIdentity.tflite(0, 0),
    )

    assert operator.cpu_only
    assert rejection.reasons == [
        ("CPU only operator", ""),
        (
            "Operator is not supported by the selected Vela profile.",
            "UNKNOWN_OP uses CPU fallback for this target and model format.",
        ),
    ]
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")
    output = Operators([operator]).to_standardized_output(model_file)
    assert output["results"][0]["checks"][0]["details"]["reasons"] == [
        {
            "description": "Operator is not supported by the selected Vela profile.",
            "detail": "UNKNOWN_OP uses CPU fallback for this target and model format.",
        }
    ]


def test_rejection_info_does_not_infer_cpu_fallback_after_failed_compile() -> None:
    """A stopped compile cannot confirm where an excluded operator would run."""
    profile = CATALOG.profiles["ethos-u55", "tflite"]

    rejection = vela_compat._rejection_info(
        "UNKNOWN_OP", None, profile, compilation_completed=False
    )

    assert rejection.supported is False
    assert ("CPU only operator", "") not in rejection.reasons
    assert rejection.reasons == [
        (
            "Operator is not supported by the selected Vela profile.",
            "UNKNOWN_OP is unsupported for this target and model format; "
            "CPU fallback was not confirmed.",
        )
    ]


@pytest.mark.parametrize(
    "profile_key, expected_reasons",
    [
        (("ethos-u55", "tflite"), [("CPU only operator", "")]),
        (("ethos-u55", "tosa"), [("CPU only operator", "")]),
    ],
)
def test_rejection_info_does_not_guess_failed_constraint(
    profile_key: tuple[str, str], expected_reasons: list[tuple[str, str]]
) -> None:
    """Observed CPU placement cannot identify a failed catalog constraint."""
    profile = CATALOG.profiles[profile_key]

    rejection = vela_compat._rejection_info("ADD", None, profile, rejected=True)

    assert rejection == NpuSupported(False, expected_reasons)


def test_rejection_info_without_catalog_has_no_backend_reason() -> None:
    """Version mismatch keeps only the internal CPU placement marker."""
    assert vela_compat._rejection_info("ADD", None, None, rejected=True) == (
        NpuSupported(False, [("CPU only operator", "")])
    )


def test_rejection_info_preserves_explicit_catalog_exclusion() -> None:
    """A named unsupported TOSA operator has a confirmed catalog reason."""
    profile = CATALOG.profiles["ethos-u55", "tosa"]

    rejection = vela_compat._rejection_info("BITWISE_AND", None, profile)

    assert rejection == NpuSupported(
        False,
        [
            (
                "Operator is not supported by the selected Vela profile.",
                "BITWISE_AND uses compilation failure for this target and model format.",
            )
        ],
    )


@pytest.mark.parametrize(
    "constraint, reason, expected_reasons",
    [
        ("", "", [("CPU only operator", "")]),
        (
            "",
            "Float32 input tensor",
            [("CPU only operator", ""), ("", "Float32 input tensor")],
        ),
    ],
)
def test_rejection_info_does_not_invent_diagnostic_text(
    constraint: str, reason: str, expected_reasons: list[tuple[str, str]]
) -> None:
    """Incomplete Regor warnings contribute only the text Regor supplied."""
    profile = CATALOG.profiles["ethos-u55", "tflite"]
    diagnostic = vela_compat.OperatorDiagnostic("ADD", "output", reason, constraint)

    rejection = vela_compat._rejection_info("ADD", diagnostic, profile)

    assert rejection == NpuSupported(False, expected_reasons)


def test_rejection_info_preserves_diagnostic_for_cpu_fallback() -> None:
    """Runtime diagnostics remain visible for operators placed on the CPU."""
    profile = CATALOG.profiles["ethos-u55", "tflite"]
    diagnostic = vela_compat.OperatorDiagnostic(
        op_type="CONV_2D",
        name="output",
        reason="Float32 input tensor",
        constraint="Feature-map dataTypes must be quantized",
    )

    rejection = vela_compat._rejection_info("CONV_2D", diagnostic, profile)

    assert rejection.reasons == [
        ("CPU only operator", ""),
        ("Feature-map dataTypes must be quantized", "Float32 input tensor"),
    ]


@pytest.mark.parametrize("model_suffix", [".pt2", ".tosa"])
def test_compiled_source_formats_use_performance_layer_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_suffix: str
) -> None:
    """CSV-derived rows must not claim TFLite source-operator identity."""
    model_file = tmp_path / f"model{model_suffix}"
    model_file.write_bytes(b"test model content")
    (tmp_path / "model_per-layer.csv").write_text("unused", encoding="utf-8")
    compiler = MagicMock()
    events = MagicMock()
    compiler.compile_model = events.compile
    events.compile.return_value = (None, tmp_path / "compiled.tflite")
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        parse_layerwise_perf_csv=MagicMock(
            return_value=SimpleNamespace(
                layerwise_info=[
                    SimpleNamespace(name="constant", tflite_operator="Const"),
                    SimpleNamespace(
                        name="layer", tflite_operator="CONV_2D", placement="CPU"
                    ),
                ]
            )
        ),
        layer_metrics=[],
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    monkeypatch.setattr(backend_process, "log_boundary_action", events.notice)
    compiler_options = SimpleNamespace(
        output_dir=tmp_path, accelerator_config="ethos-u55-256"
    )

    operators = vela_compat._supported_compiled_model_operators(
        model_file,
        compiler_options,
        compiler,
        vela_internal_ops=(),
        deps=deps,
    )

    assert events.mock_calls == [
        call.notice(
            vela_compat.logger,
            "Running Vela compatibility analysis for '%s'; derived output will be "
            "written under '%s'.",
            model_file,
            tmp_path,
        ),
        call.compile(model_file, diagnostic_consumer=ANY),
    ]
    assert [operator.identity for operator in operators.ops] == [
        OperatorIdentity.performance_layer(1)
    ]
    assert operators.ops[0].cpu_only
    output = operators.to_standardized_output(model_path=model_file)
    result = output["results"][0]
    assert result["entities"] == [
        {
            "id": "performance_layer/1",
            "kind": "performance_layer",
            "name": "layer",
            "placement": "CPU",
            "attributes": {
                "op_type": "CONV_2D",
                "identity_scope": "vela_per_layer_csv",
                "layer_index": 1,
            },
        }
    ]
    assert result["entity_kinds"] == [{"id": "performance_layer"}]
    assert result["checks"][0]["status"] == "fail"
    assert result["checks"][0]["details"]["reasons"] == []


@pytest.mark.parametrize(
    ("placement", "expected_placement", "expected_supported"),
    [("NPU", "NPU", True), (None, "Unknown", False)],
)
def test_compiled_source_explicit_npu_placement_overrides_catalog(
    tmp_path: Path,
    placement: str | None,
    expected_placement: str,
    expected_supported: bool,
) -> None:
    """A Vela NPU row is authoritative even when the catalog omits its type."""
    model_file = tmp_path / "model.tosa"
    model_file.write_bytes(b"test model content")
    (tmp_path / "model_per-layer.csv").write_text("unused", encoding="utf-8")
    compiler = MagicMock()
    compiler.compile_model.return_value = (None, tmp_path / "compiled.tflite")
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        parse_layerwise_perf_csv=MagicMock(
            return_value=SimpleNamespace(
                layerwise_info=[
                    SimpleNamespace(
                        name="new operator",
                        tflite_operator="UNLISTED_OP",
                        placement=placement,
                    )
                ]
            )
        ),
        layer_metrics=[],
    )

    operators = vela_compat._supported_compiled_model_operators(
        model_file,
        SimpleNamespace(output_dir=tmp_path, accelerator_config="ethos-u55-256"),
        compiler,
        vela_internal_ops=(),
        deps=deps,
    )

    assert operators.ops[0].run_on_npu.supported is expected_supported
    result = operators.to_standardized_output(model_path=model_file)["results"][0]
    assert result["entities"][0]["placement"] == expected_placement


def test_compiled_model_fallback_uses_compiled_artifact_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fallback graph reads should remain scoped to the compiled artifact."""
    checked_type = object()
    compiler = MagicMock()
    compiler.compile_model.return_value = (None, tmp_path / "compiled.tflite")
    compiler._read_model.return_value = (
        SimpleNamespace(
            subgraphs=[
                SimpleNamespace(get_all_ops=MagicMock(return_value=[])),
                SimpleNamespace(
                    get_all_ops=MagicMock(
                        return_value=[
                            SimpleNamespace(
                                name="compiled", type=checked_type, op_index=5
                            )
                        ]
                    )
                ),
            ]
        ),
        None,
    )
    deps = SimpleNamespace(
        ethosu_vela_version="5.2.0",
        optype_to_builtintype=MagicMock(return_value="CONV_2D"),
    )
    model_file = tmp_path / "model.pt2"
    model_file.write_bytes(b"test model content")

    operators = vela_compat._supported_compiled_model_operators(
        model_file,
        SimpleNamespace(output_dir=tmp_path),
        compiler,
        vela_internal_ops=(),
        deps=deps,
    )

    assert [operator.identity for operator in operators.ops] == [
        OperatorIdentity.compiled_tflite(1, 5)
    ]
    assert not operators.ops[0].identity.entity_id.startswith("source_operator/")


def test_compiled_model_returns_tosa_diagnostic_on_compiler_error(
    tmp_path: Path,
) -> None:
    """A TOSA rejection remains reportable when its no-fallback compile fails."""
    compiler = MagicMock()

    def compile_model(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        for line in [
            "Warning (supported operators) operator: CONV2D, ofm: 'output'",
            "Reason: Input rank is five",
            "Input rank must be four",
            "",
        ]:
            diagnostic_consumer(line)
        support_error = RuntimeError(
            "Compilation failed: Unsupported operator CONV2D 'output': "
            "Input rank must be four"
        )
        raise RuntimeError("TOSA compilation failed") from support_error

    compiler.compile_model.side_effect = compile_model
    model_file = tmp_path / "model.tosa"
    model_file.write_bytes(b"model")

    operators = vela_compat._supported_compiled_model_operators(
        model_file,
        SimpleNamespace(output_dir=tmp_path, accelerator_config="ethos-u55-256"),
        compiler,
        vela_internal_ops=(),
        deps=SimpleNamespace(ethosu_vela_version="5.2.0"),
    )

    assert operators.ops == [
        Operator(
            name="output",
            op_type="CONV2D",
            run_on_npu=NpuSupported(
                False, [("Input rank must be four", "Input rank is five")]
            ),
            identity=OperatorIdentity.diagnostic(0),
        )
    ]
    assert not operators.ops[0].cpu_only
    assert operators.npu_unsupported_ratio is None
    result = operators.to_standardized_output(model_file)["results"][0]
    assert result["status"] == "partial"
    assert result["warnings"]
    assert result["entities"][0]["placement"] == "Unknown"
    assert result["metrics"][0]["availability"] == "unavailable"
    assert "value" not in result["metrics"][0]
    reasons = result["checks"][0]["details"]["reasons"]
    assert reasons[0] == {
        "description": "Input rank must be four",
        "detail": "Input rank is five",
    }
    assert len(reasons) == 1


def test_compiled_model_propagates_unrelated_error_after_diagnostic(
    tmp_path: Path,
) -> None:
    """An earlier operator warning must not conceal a later compiler failure."""
    compiler = MagicMock()

    def compile_model(
        _model: Path, *, diagnostic_consumer: Callable[[str], None]
    ) -> None:
        for line in [
            "Warning (supported operators) operator: CONV2D, ofm: 'output'",
            "Reason: Input rank is five",
            "Input rank must be four",
            "",
        ]:
            diagnostic_consumer(line)
        unrelated_error = RuntimeError("Compilation failed: output write failure")
        raise RuntimeError("TOSA compilation failed") from unrelated_error

    compiler.compile_model.side_effect = compile_model
    model_file = tmp_path / "model.tosa"
    model_file.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="TOSA compilation failed"):
        vela_compat._supported_compiled_model_operators(
            model_file,
            SimpleNamespace(output_dir=tmp_path, accelerator_config="ethos-u55-256"),
            compiler,
            vela_internal_ops=(),
            deps=SimpleNamespace(ethosu_vela_version="5.2.0"),
        )


def test_generate_supported_operators_report(tmp_path: Path) -> None:
    """Test generating supported operators report."""
    try:
        with working_directory(tmp_path):
            generate_supported_operators_report()

            md_file = tmp_path / "SUPPORTED_OPS.md"
            assert md_file.is_file()
            assert md_file.stat().st_size > 0
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip(
            "Vela backend not available, skipping supported operators report test"
        )


def test_compatibility_check_should_fail_if_checker_not_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path | Path
) -> None:
    """Test that compatibility check should fail if Vela is not available."""
    monkeypatch.setattr(vela_compat, "_VELA_DEPS_CACHE", None)
    monkeypatch.setattr(
        vela_compat,
        "_load_vela_deps",
        fail_load_vela_deps,
    )

    with working_directory(tmp_path):
        with pytest.raises(
            BackendUnavailableError, match="Backend vela is not available"
        ):
            generate_supported_operators_report()


def test_compatibility_check_should_fail_if_checker_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path | Path
) -> None:
    """Test that compatibility check should fail if Vela checker returns False."""
    monkeypatch.setattr(vela_compat, "_VELA_DEPS_CACHE", None)
    monkeypatch.setattr(
        vela_compat,
        "_load_vela_deps",
        fail_load_vela_deps,
    )

    with working_directory(tmp_path):
        with pytest.raises(
            BackendUnavailableError, match="Backend vela is not available"
        ):
            generate_supported_operators_report()


def test_get_vela_returns_availability_status() -> None:
    """Test that get_vela returns the correct availability status."""
    # The function should return True if ethosu.vela is available, False otherwise
    result = get_vela()
    # The result should be a boolean indicating vela availability
    assert isinstance(result, bool)


def test_operators_to_standardized_output(tmp_path: Path) -> None:
    """Test conversion of Operators to standardized output."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    ops = [
        Operator(
            name="conv1",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
            identity=OperatorIdentity.tflite(0, 0),
        ),
        Operator(
            name="conv2",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(
                supported=False, reasons=[("CPU only operator", "")]
            ),
            identity=OperatorIdentity.tflite(0, 1),
        ),
        Operator(
            name="pool1",
            op_type="MAX_POOL_2D",
            run_on_npu=NpuSupported(
                supported=False,
                reasons=[("Constraint failed", "Invalid tensor shape")],
            ),
            identity=OperatorIdentity.tflite(0, 2),
        ),
    ]

    operators = Operators(ops)
    output: dict = operators.to_standardized_output(
        model_path=model_file,
        target_config={"target": "ethos-u55", "mac": 256},
    )

    # Verify structure
    assert "schema_version" in output
    assert output["schema_version"] == schema.SCHEMA_VERSION
    assert "backends" in output
    assert "target" in output
    assert "model" in output
    assert "context" in output
    assert "results" in output

    # Verify backend
    backends = output["backends"]
    assert len(backends) == 1
    backend = backends[0]
    assert backend["name"] == "Vela Compiler"
    assert "version" in backend

    # Verify result
    results = output["results"]
    assert len(results) == 1
    result = results[0]
    assert result["kind"] == "compatibility"
    assert result["status"] == "partial"  # Some supported, some not
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": pytest.approx(100 / 3),
        "unit": schema.UNIT_PERCENT,
    }

    # Verify checks and entities
    assert "checks" in result
    assert "entities" in result
    checks = result["checks"]
    entities = result["entities"]

    assert len(checks) == 3  # One check per operator
    assert len(entities) == 3  # One entity per operator
    assert [entity["id"] for entity in entities] == [
        f"source_operator/operator/{index}" for index in range(3)
    ]

    # Verify first operator (supported)
    assert entities[0]["name"] == "conv1"
    assert entities[0]["kind"] == "source_operator"
    assert entities[0]["placement"] == "NPU"
    assert entities[0]["attributes"] == {
        "op_type": "CONV_2D",
        "subgraph_index": 0,
        "operator_index": 0,
    }
    assert checks[0]["status"] == "pass"
    assert checks[0]["entity_id"] == entities[0]["id"]
    assert "reasons" not in checks[0].get("details", {})

    # Verify second operator (CPU only)
    assert entities[1]["name"] == "conv2"
    assert entities[1]["placement"] == "CPU"
    assert checks[1]["status"] == "fail"
    assert checks[1]["entity_id"] == entities[1]["id"]
    assert checks[1]["details"]["reasons"] == []

    # Verify third operator (constraint failed)
    assert entities[2]["name"] == "pool1"
    assert entities[2]["placement"] == "Unknown"
    assert checks[2]["status"] == "fail"
    assert checks[2]["entity_id"] == entities[2]["id"]
    assert checks[2]["details"]["reasons"] == [
        {"description": "Constraint failed", "detail": "Invalid tensor shape"}
    ]
    assert all("reasons" not in entity["attributes"] for entity in entities)


def test_failed_operator_checks_include_only_observed_reasons(tmp_path: Path) -> None:
    """Catalog possibilities do not appear alongside observed Vela failures."""
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"model")
    failed_reasons = [("CPU only operator", ""), ("Runtime check", "Actual failure")]
    operators = Operators(
        [
            Operator(
                "conv",
                "CONV_2D",
                NpuSupported(False, failed_reasons.copy()),
                OperatorIdentity.tflite(0, 0),
            ),
            Operator(
                "add",
                "ADD",
                NpuSupported(False, [("CPU only operator", "")]),
                OperatorIdentity.tflite(0, 1),
            ),
            Operator(
                "unknown",
                "UNKNOWN",
                NpuSupported(False, []),
                OperatorIdentity.tflite(0, 2),
            ),
            Operator(
                "passing",
                "CONV_2D",
                NpuSupported(True, []),
                OperatorIdentity.tflite(0, 3),
            ),
        ],
    )

    checks = operators.to_standardized_output(model_file)["results"][0]["checks"]
    assert checks[0]["details"]["reasons"] == [
        {"description": "Runtime check", "detail": "Actual failure"},
    ]
    assert checks[1]["details"]["reasons"] == []
    assert checks[2]["details"]["reasons"] == []
    assert checks[3].get("details", {}) == {}
    assert operators.ops[0].run_on_npu.reasons == failed_reasons
    assert Operators(operators.ops).to_standardized_output(model_file)["results"][0][
        "checks"
    ][0]["details"]["reasons"] == [
        {"description": "Runtime check", "detail": "Actual failure"}
    ]


def test_unexplained_failure_has_no_reasons(tmp_path: Path) -> None:
    """Unexplained failures have an empty reasons list."""
    model_file = tmp_path / "model.tosa"
    model_file.write_bytes(b"model")
    operators = Operators(
        [
            Operator(
                "op",
                "ADD",
                NpuSupported(False, []),
                OperatorIdentity.diagnostic(0),
            )
        ],
    )

    checks = operators.to_standardized_output(model_file)["results"][0]["checks"]
    assert checks[0]["details"]["reasons"] == []


def test_operator_reasons_render_with_development_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entity-linked Vela checks supply the core terminal reason column."""
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")
    operators = Operators(
        [
            Operator(
                "supported",
                "CONV_2D",
                NpuSupported(True, []),
                OperatorIdentity.tflite(0, 0),
            ),
            Operator(
                "fallback",
                "HARD_SWISH",
                NpuSupported(
                    False,
                    [
                        ("CPU only operator", ""),
                        ("IFM must be 8 bit", "IFM type=FLOAT32"),
                        ("OFM must be 8 bit", "OFM type=FLOAT32"),
                    ],
                ),
                OperatorIdentity.tflite(0, 1),
            ),
            Operator(
                "unexplained",
                "CUSTOM",
                NpuSupported(False, [("CPU only operator", "")]),
                OperatorIdentity.tflite(0, 2),
            ),
        ],
    )
    output = operators.to_standardized_output(model_file)
    result = output["results"][0]
    entities = result["entities"]
    checks = result["checks"]

    assert [check["entity_id"] for check in checks] == [
        entity["id"] for entity in entities
    ]
    assert [entity["kind"] for entity in entities] == ["source_operator"] * 3
    assert [entity["placement"] for entity in entities] == ["NPU", "CPU", "CPU"]
    assert all("reasons" not in entity["attributes"] for entity in entities)
    assert checks[0].get("details", {}) == {}
    assert checks[1]["details"]["reasons"] == [
        {"description": "IFM must be 8 bit", "detail": "IFM type=FLOAT32"},
        {"description": "OFM must be 8 bit", "detail": "OFM type=FLOAT32"},
    ]
    assert checks[2]["status"] == "fail"
    assert checks[2]["details"]["reasons"] == []
    serialized = json.loads(standardized_output_to_json(output))
    assert serialized["results"][0]["checks"] == checks

    if Version(version("mlia")) < Version("0.12.4.dev4"):
        pytest.skip("Installed mlia-core predates the entity reason column")

    monkeypatch.setattr("mlia.utils.console._output_width", lambda: 400)
    rendered = standardized_output_to_text(output)
    assert "Reason" in rendered
    assert "Name" in rendered
    assert "Kind" in rendered
    assert "ID" in rendered
    assert "Placement" in rendered
    assert (
        "IFM must be 8 bit: IFM type=FLOAT32; OFM must be 8 bit: OFM type=FLOAT32"
    ) in rendered
    assert "CPU only operator" not in rendered
    assert "-" in next(line for line in rendered.splitlines() if "supported" in line)
    assert "-" in next(line for line in rendered.splitlines() if "unexplained" in line)


def test_operators_to_standardized_output_all_supported(tmp_path: Path) -> None:
    """Test conversion when all operators are supported."""
    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    ops = [
        Operator(
            name="conv1",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
            identity=OperatorIdentity.tflite(0, 0),
        ),
        Operator(
            name="conv2",
            op_type="CONV_2D",
            run_on_npu=NpuSupported(supported=True, reasons=[]),
            identity=OperatorIdentity.tflite(0, 1),
        ),
    ]

    operators = Operators(ops)
    output: dict = operators.to_standardized_output(
        model_path=model_file,
    )

    results = output["results"]
    assert len(results) == 1
    result = results[0]
    assert result["status"] == "ok"  # All supported
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE] == {
        "name": schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        "value": 100,
        "unit": schema.UNIT_PERCENT,
    }


def test_operators_to_standardized_output_reports_unavailable_percentage_for_no_ops(
    tmp_path: Path,
) -> None:
    """Operator percentage should be explicit when no operators are available."""
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = Operators([]).to_standardized_output(model_path=model_file)

    result = output["results"][0]
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    metric = metrics[schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE]
    assert metric["unit"] == schema.UNIT_PERCENT
    assert metric["availability"] == "unavailable"
    assert "value" not in metric
    assert metric["reason"]


def test_operators_to_standardized_output_handles_broken_vela(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Standardized output should tolerate Vela version lookup failures."""
    monkeypatch.setattr(
        "mlia.backend.vela.compat._get_vela_deps",
        MagicMock(side_effect=RuntimeError("broken vela")),
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = Operators([]).to_standardized_output(model_path=model_file)

    assert output["backends"][0]["version"] == "unknown"
