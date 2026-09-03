# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/compat."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import mlia.backend.vela.compat as vela_compat
import mlia.core.output_schema as schema
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
    cpu_only = not npu_supported.supported and npu_supported.reasons == [
        ("CPU only operator", "")
    ]
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
    test_models_path: Path, model: str, expected_ops: Operators
) -> None:
    """Test operators function."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")

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
        ethosu_vela_version="test",
        Op=SimpleNamespace(
            Placeholder=placeholder_type,
            SubgraphInput=subgraph_input_type,
            Const=const_type,
        ),
        VelaCompiler=MagicMock(return_value=compiler),
        optype_to_builtintype=MagicMock(return_value="CONV_2D"),
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    monkeypatch.setattr(
        vela_compat,
        "_run_on_npu",
        MagicMock(return_value=NpuSupported(supported=True, reasons=[])),
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    operators = supported_operators(model_file, compiler_options=object())

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


@pytest.mark.parametrize("model_suffix", [".pt2", ".tosa"])
def test_compiled_source_formats_use_performance_layer_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_suffix: str
) -> None:
    """CSV-derived rows must not claim TFLite source-operator identity."""
    model_file = tmp_path / f"model{model_suffix}"
    model_file.write_bytes(b"test model content")
    (tmp_path / "model_per-layer.csv").write_text("unused", encoding="utf-8")
    compiler = MagicMock()
    compiler.compile_model.return_value = (None, tmp_path / "compiled.tflite")
    deps = SimpleNamespace(
        ethosu_vela_version="test",
        parse_layerwise_perf_csv=MagicMock(
            return_value=SimpleNamespace(
                layerwise_info=[
                    SimpleNamespace(name="constant", tflite_operator="Const"),
                    SimpleNamespace(name="layer", tflite_operator="CONV_2D"),
                ]
            )
        ),
        layer_metrics=[],
    )
    monkeypatch.setattr(vela_compat, "_get_vela_deps", MagicMock(return_value=deps))
    compiler_options = SimpleNamespace(output_dir=tmp_path)

    operators = vela_compat._supported_compiled_model_operators(
        model_file,
        compiler_options,
        compiler,
        vela_internal_ops=(),
        deps=deps,
    )

    assert [operator.identity for operator in operators.ops] == [
        OperatorIdentity.performance_layer(1)
    ]
    output = operators.to_standardized_output(model_path=model_file)
    result = output["results"][0]
    assert result["entities"] == [
        {
            "id": "performance_layer/1",
            "kind": "performance_layer",
            "name": "layer",
            "placement": "NPU",
            "attributes": {
                "op_type": "CONV_2D",
                "identity_scope": "vela_per_layer_csv",
                "layer_index": 1,
            },
        }
    ]
    assert result["entity_kinds"] == [{"id": "performance_layer"}]


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
        optype_to_builtintype=MagicMock(return_value="CONV_2D"),
    )
    monkeypatch.setattr(
        vela_compat,
        "_run_on_npu",
        MagicMock(return_value=NpuSupported(supported=True, reasons=[])),
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

    # Verify second operator (CPU only)
    assert entities[1]["name"] == "conv2"
    assert entities[1]["placement"] == "CPU"
    assert checks[1]["status"] == "fail"
    assert checks[1]["entity_id"] == entities[1]["id"]
    assert "reasons" in checks[1]["details"]

    # Verify third operator (constraint failed)
    assert entities[2]["name"] == "pool1"
    assert entities[2]["placement"] == "CPU"
    assert checks[2]["status"] == "fail"
    assert checks[2]["entity_id"] == entities[2]["id"]
    assert "reasons" in checks[2]["details"]


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
