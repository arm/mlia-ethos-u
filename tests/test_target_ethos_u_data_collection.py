# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the data collection module for Ethos-U."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.corstone.performance import (
    CorstoneModelPerformanceMetrics,
    CorstonePerformanceMetrics as CorstonePerf,
)
from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela.compat import Operators, VelaCompatibilityResult
from mlia.backend.vela.performance import LayerwisePerfInfo
from mlia.backend.vela.performance import PerformanceMetrics as VelaPerf
from mlia.core.common import AdviceCategory
from mlia.core.context import Context, ExecutionContext
from mlia.core.data_collection import DataCollector
from mlia.core.output_validation import validate_standardized_output
from mlia.core.errors import ConfigurationError
from mlia.core.errors import FunctionalityNotSupportedError
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.utils.legacy_shims import (
    LEGACY_OPTIMIZATION_AVAILABLE,
    add_common_optimization_params,
)
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.data_collection import (
    EthosUOperatorCompatibility,
    EthosUOptimizationPerformance,
    EthosUPerformance,
)
from mlia.target.ethos_u.performance import (
    CombinedPerformanceResult,
    CorstonePerformanceResult,
    MemoryUsage,
    NPUCycles,
    OptimizationPerformanceResult,
    PerformanceMetrics,
    VelaPerformanceResult,
)
from mlia.target.ethos_u.utils.tflite_shims import (
    ModelConfiguration,
    TFLiteCompatibilityInfo,
    TFLiteCompatibilityResult,
    TFLiteCompatibilityStatus,
    TFLiteConversionError,
    TFLiteConversionErrorCode,
)


@pytest.mark.parametrize(
    "collector, expected_name",
    [
        (
            EthosUOperatorCompatibility,
            "ethos_u_operator_compatibility",
        ),
        (
            EthosUPerformance,
            "ethos_u_performance",
        ),
        (
            EthosUOptimizationPerformance,
            "ethos_u_model_optimizations",
        ),
    ],
)
def test_collectors_metadata(
    collector: DataCollector,
    expected_name: str,
) -> None:
    """Test collectors metadata."""
    assert collector.name() == expected_name


def setup_optimization(optimizations: list) -> Context:
    """Set up optimization params for the context."""
    params: dict = {}
    add_common_optimization_params(
        params,
        {
            "optimization_targets": optimizations,
        },
    )

    context = ExecutionContext(config_parameters=params)
    return context


def mock_corstone_backend_configuration(
    monkeypatch: pytest.MonkeyPatch, profile: str = "default"
) -> MagicMock:
    """Mock the effective Corstone backend configuration."""
    backend_repository = MagicMock()
    backend_repository.get_backend_settings.return_value = (
        Path("backend"),
        {"profile": profile},
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_backend_repository",
        MagicMock(return_value=backend_repository),
    )
    return backend_repository


def test_operator_compatibility_collector(
    sample_context: Context, test_tflite_model: Path
) -> None:
    """Test operator compatibility data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    collector = EthosUOperatorCompatibility(test_tflite_model, target)
    collector.set_context(sample_context)

    try:
        result = collector.collect_data()
        assert isinstance(result, VelaCompatibilityResult)
        validate_standardized_output(result.standardized_output)
        assert set(vars(result)) == {"standardized_output"}
    except BackendUnavailableError:
        # If Vela is not available, the test should pass (expected behavior)
        pytest.skip("Vela backend not available, skipping operator compatibility test")


@pytest.mark.parametrize(
    "compatibility, expected_status",
    [
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.TFLITE_CONVERSION_ERROR,
                conversion_errors=[
                    TFLiteConversionError(
                        "custom operator",
                        TFLiteConversionErrorCode.NEEDS_CUSTOM_OPS,
                        "CustomOp",
                        ["model"],
                    )
                ],
            ),
            "incompatible",
        ),
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.MODEL_WITH_CUSTOM_OP_ERROR
            ),
            "incompatible",
        ),
        (
            TFLiteCompatibilityInfo(
                status=TFLiteCompatibilityStatus.UNKNOWN_ERROR,
                conversion_exception=RuntimeError("conversion failed"),
            ),
            "failed",
        ),
    ],
)
def test_legacy_tflite_failure_returns_complete_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compatibility: TFLiteCompatibilityInfo,
    expected_status: str,
) -> None:
    """A failed TensorFlow Lite precheck should still return canonical output."""
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"keras model")
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    checker = MagicMock()
    checker.check_compatibility.return_value = compatibility
    supported_operators_mock = MagicMock()
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_legacy_model",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.LegacyChecker",
        MagicMock(return_value=checker),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.supported_operators",
        supported_operators_mock,
    )
    context = ExecutionContext(
        advice_category={AdviceCategory.COMPATIBILITY},
        output_dir=tmp_path,
    )
    collector = EthosUOperatorCompatibility(model_path, target)
    collector.set_context(context)

    result = collector.collect_data()

    assert isinstance(result, TFLiteCompatibilityResult)
    assert set(vars(result)) == {"standardized_output"}
    validate_standardized_output(result.standardized_output)
    canonical_result = result.standardized_output["results"][0]
    assert canonical_result["kind"] == "compatibility"
    assert canonical_result["status"] == expected_status
    assert canonical_result["checks"][0]["status"] == "fail"
    assert canonical_result["metrics"][0]["availability"] == "unavailable"
    assert canonical_result["advice"]
    assert {item["category"] for item in canonical_result["advice"]} == {
        "compatibility"
    }
    supported_operators_mock.assert_not_called()


def test_legacy_tflite_failure_supports_saved_model_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SavedModel conversion failures should include deterministic model metadata."""
    model_path = tmp_path / "saved_model"
    (model_path / "variables").mkdir(parents=True)
    (model_path / "saved_model.pb").write_bytes(b"model")
    (model_path / "variables" / "variables.data").write_bytes(b"weights")
    compatibility = TFLiteCompatibilityInfo(
        status=TFLiteCompatibilityStatus.UNKNOWN_ERROR,
        conversion_exception=RuntimeError("conversion failed"),
    )
    checker = MagicMock()
    checker.check_compatibility.return_value = compatibility
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_legacy_model",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.LegacyChecker",
        MagicMock(return_value=checker),
    )
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    context = ExecutionContext(
        advice_category={AdviceCategory.COMPATIBILITY}, output_dir=tmp_path
    )
    collector = EthosUOperatorCompatibility(model_path, target)
    collector.set_context(context)

    result = collector.collect_data()

    assert isinstance(result, TFLiteCompatibilityResult)
    validate_standardized_output(result.standardized_output)
    model = result.standardized_output["model"]
    assert model["format"] == "saved_model"
    assert len(model["hash"]) == 64
    assert model["size_bytes"] == 12


def test_legacy_compatibility_uses_converted_artifact_and_original_metadata(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    tmp_path: Path,
) -> None:
    """Identify the submitted model while checking its converted TFLite artifact."""
    saved_model = tmp_path / "saved_model"
    (saved_model / "variables").mkdir(parents=True)
    (saved_model / "saved_model.pb").write_bytes(b"model")
    (saved_model / "variables" / "variables.data").write_bytes(b"weights")
    converted_model = tmp_path / "converted.tflite"
    converted_model.write_bytes(b"converted model")
    model_configuration = MagicMock(spec=ModelConfiguration)
    model_configuration.model_path = str(converted_model)
    checker = MagicMock()
    checker.check_compatibility.return_value.compatible = True
    operators = MagicMock(spec=Operators)
    operators.to_standardized_output.return_value = {
        "model": {"name": converted_model.name}
    }
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_legacy_model",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.LegacyChecker",
        MagicMock(return_value=checker),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_tflite_model",
        MagicMock(return_value=model_configuration),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.supported_operators",
        MagicMock(return_value=operators),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.attach_result_advice", MagicMock()
    )
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    collector = EthosUOperatorCompatibility(saved_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()

    assert isinstance(result, VelaCompatibilityResult)
    assert (
        operators.to_standardized_output.call_args.kwargs["model_path"]
        == converted_model
    )
    assert result.standardized_output["model"]["name"] == "saved_model"
    assert result.standardized_output["model"]["format"] == "saved_model"
    assert result.standardized_output["model"]["size_bytes"] == 12


def test_performance_collector(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation(monkeypatch, target)

    collector = EthosUPerformance(test_tflite_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()

    assert isinstance(result, VelaPerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    validate_standardized_output(result.standardized_output)


def test_performance_collector_with_vela(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with Vela backend."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation_with_vela(monkeypatch, target)
    attach_advice = MagicMock()
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.attach_result_advice", attach_advice
    )

    collector = EthosUPerformance(test_tflite_model, target, backends=["vela"])
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With only Vela backend, collector returns VelaPerformanceResult
    assert isinstance(result, VelaPerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    assert isinstance(result.standardized_output, dict)
    attach_advice.assert_called_once_with(
        result.standardized_output, result, sample_context
    )


def test_performance_collector_with_corstone(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with Corstone backend."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    serializer = mock_performance_estimation_with_corstone(
        monkeypatch, target, profile="AVH"
    )

    collector = EthosUPerformance(test_tflite_model, target, backends=["corstone-310"])
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With only Corstone backend, collector returns CorstonePerformanceResult
    assert isinstance(result, CorstonePerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    assert isinstance(result.standardized_output, dict)
    assert serializer.call_args.kwargs["backend_config"] == {
        "fvp": "corstone-310",
        "target": "ethos-u55",
        "mac": 256,
        "profile": "AVH",
    }


def test_performance_collector_with_both_backends(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, test_tflite_model: Path
) -> None:
    """Test performance data collector with both Vela and Corstone backends."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation_with_both(monkeypatch, target)

    collector = EthosUPerformance(
        test_tflite_model, target, backends=["vela", "corstone-310"]
    )
    collector.set_context(sample_context)

    result = collector.collect_data()
    # With both backends, collector returns CombinedPerformanceResult
    assert isinstance(result, CombinedPerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    assert isinstance(result.standardized_output, dict)
    # Check that both backends are present
    assert len(result.standardized_output["backends"]) == 2
    # Check that both results are present
    assert len(result.standardized_output["results"]) == 2


def test_performance_collector_rejects_multiple_corstone_backends(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    test_tflite_model: Path,
) -> None:
    """Reject configurations the single Corstone metrics slot cannot represent."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    estimator = MagicMock()
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator", estimator
    )
    collector = EthosUPerformance(
        test_tflite_model,
        target,
        backends=["corstone-300", "corstone-310"],
    )
    collector.set_context(sample_context)

    with pytest.raises(
        ConfigurationError,
        match="at most one Corstone backend.*corstone-300, corstone-310",
    ):
        collector.collect_data()

    estimator.assert_not_called()


@pytest.mark.parametrize(
    ("backends", "result_type"),
    [
        (["vela"], VelaPerformanceResult),
        (["corstone-310"], CorstonePerformanceResult),
        (["vela", "corstone-310"], CombinedPerformanceResult),
    ],
)
def test_saved_model_performance_uses_converted_artifact_and_original_metadata(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    tmp_path: Path,
    backends: list[str],
    result_type: type,
) -> None:
    """Serialize the converted file while identifying the original SavedModel."""
    saved_model = tmp_path / "saved_model"
    (saved_model / "variables").mkdir(parents=True)
    (saved_model / "saved_model.pb").write_bytes(b"model")
    (saved_model / "variables" / "variables.data").write_bytes(b"weights")
    converted_model = tmp_path / "converted.tflite"
    converted_model.write_bytes(b"converted model")
    model_configuration = MagicMock(spec=ModelConfiguration)
    model_configuration.model_path = str(converted_model)

    target = EthosUConfiguration.load_profile("ethos-u55-256")
    performance = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    vela_metrics = VelaPerf(
        npu_cycles=3,
        sram_access_cycles=1,
        dram_access_cycles=2,
        on_chip_flash_access_cycles=3,
        off_chip_flash_access_cycles=4,
        total_cycles=4,
        batch_inference_time=1.0,
        inferences_per_second=1000.0,
        batch_size=1,
        sram_memory_area_size=1,
        dram_memory_area_size=2,
        on_chip_flash_memory_area_size=3,
        off_chip_flash_memory_area_size=4,
        layerwise_performance_info=LayerwisePerfInfo(layerwise_info=[]),
    )
    vela_serializer = MagicMock(wraps=vela_metrics.to_standardized_output)
    setattr(vela_metrics, "to_standardized_output", vela_serializer)

    corstone_metrics = CorstonePerf(
        CorstoneModelPerformanceMetrics(1, 2, 3, 4, 5, 6), []
    )
    corstone_serializer = MagicMock(wraps=corstone_metrics.to_standardized_output)
    setattr(corstone_metrics, "to_standardized_output", corstone_serializer)
    performance.corstone_metrics = corstone_metrics

    estimator = MagicMock()
    estimator.estimate.return_value = performance
    estimator.vela_perf_metrics = vela_metrics
    estimator.vela_compiler_options = None
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_legacy_model",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_tflite_model",
        MagicMock(return_value=model_configuration),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=estimator),
    )
    if "corstone-310" in backends:
        mock_corstone_backend_configuration(monkeypatch)
    collector = EthosUPerformance(saved_model, target, backends=backends)
    collector.set_context(sample_context)

    result = collector.collect_data()

    assert isinstance(result, result_type)
    assert isinstance(
        result,
        (VelaPerformanceResult, CorstonePerformanceResult, CombinedPerformanceResult),
    )
    validate_standardized_output(result.standardized_output)
    assert result.standardized_output["model"]["name"] == "saved_model"
    assert result.standardized_output["model"]["format"] == "saved_model"
    assert result.standardized_output["model"]["size_bytes"] == 12
    assert len(result.standardized_output["model"]["hash"]) == 64
    if "vela" in backends:
        assert vela_serializer.call_args.kwargs["model_path"] == converted_model
    else:
        vela_serializer.assert_not_called()
    if "corstone-310" in backends:
        assert corstone_serializer.call_args.kwargs["model_path"] == converted_model
    else:
        corstone_serializer.assert_not_called()


@pytest.mark.parametrize(
    ("backends", "error"),
    [
        (["vela"], "Vela performance metrics were not produced"),
        (["corstone-310"], "corstone-310 performance metrics were not produced"),
    ],
)
def test_performance_collector_requires_backend_metrics(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    test_tflite_model: Path,
    backends: list[str],
    error: str,
) -> None:
    """Requested backends must provide their native metrics."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )

    class MockEstimator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def estimate(self, _model: object) -> PerformanceMetrics:
            return metrics

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MockEstimator,
    )
    collector = EthosUPerformance(test_tflite_model, target, backends=backends)
    collector.set_context(sample_context)

    with pytest.raises(ConfigurationError, match=error):
        collector.collect_data()


def test_vela_serializer_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    test_tflite_model: Path,
) -> None:
    """Vela serialization errors must not fall back to target-level metrics."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    estimator = MagicMock()
    estimator.estimate.return_value = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    estimator.vela_perf_metrics.to_standardized_output.side_effect = RuntimeError(
        "Vela serialization failed"
    )
    estimator.vela_compiler_options = None
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=estimator),
    )
    collector = EthosUPerformance(test_tflite_model, target, backends=["vela"])
    collector.set_context(sample_context)

    with pytest.raises(RuntimeError, match="Vela serialization failed"):
        collector.collect_data()


def test_corstone_serializer_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
    sample_context: Context,
    test_tflite_model: Path,
) -> None:
    """Corstone serialization errors must not fabricate canonical output."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
        corstone_metrics=MagicMock(spec=CorstonePerf),
    )
    metrics.to_standardized_output = MagicMock(
        side_effect=RuntimeError("Corstone serialization failed")
    )
    estimator = MagicMock()
    estimator.estimate.return_value = metrics
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=estimator),
    )
    mock_corstone_backend_configuration(monkeypatch)
    collector = EthosUPerformance(test_tflite_model, target, backends=["corstone-310"])
    collector.set_context(sample_context)

    with pytest.raises(RuntimeError, match="Corstone serialization failed"):
        collector.collect_data()


def test_optimization_performance_collector(
    monkeypatch: pytest.MonkeyPatch,
    test_keras_model: Path,
    test_tflite_model: Path,
) -> None:
    """Test optimization performance data collector."""
    if not LEGACY_OPTIMIZATION_AVAILABLE:
        pytest.skip("Optimization performance requires legacy plugin support.")
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    mock_performance_estimation(monkeypatch, target)

    context = setup_optimization(
        [
            {"optimization_type": "pruning", "optimization_target": 0.5},
        ],
    )
    collector = EthosUOptimizationPerformance(test_keras_model, target)
    collector.set_context(context)
    result = collector.collect_data()

    assert isinstance(result, OptimizationPerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    validate_standardized_output(result.standardized_output)
    metrics = result.standardized_output["results"][0]["metrics"]
    assert {metric["qualifiers"]["phase"] for metric in metrics} == {
        "before",
        "after",
    }

    context = ExecutionContext(
        config_parameters={"common_optimizations": {"optimizations": [[]]}}
    )

    collector_no_optimizations = EthosUOptimizationPerformance(test_keras_model, target)
    collector_no_optimizations.set_context(context)
    with pytest.raises(FunctionalityNotSupportedError):
        collector_no_optimizations.collect_data()

    context = setup_optimization(
        [
            {"optimization_type": "pruning", "optimization_target": 0.5},
        ],
    )

    collector_tflite = EthosUOptimizationPerformance(test_tflite_model, target)
    collector_tflite.set_context(context)
    with pytest.raises(FunctionalityNotSupportedError):
        collector_tflite.collect_data()

    with pytest.raises(
        Exception, match="Optimization parameters expected to be a list"
    ):
        context = ExecutionContext(
            config_parameters={
                "common_optimizations": {
                    "optimizations": [{"optimization_type": "pruning"}]
                }
            }
        )

        collector_bad_config = EthosUOptimizationPerformance(test_keras_model, target)
        collector_bad_config.set_context(context)
        collector_bad_config.collect_data()


def test_optimization_factory_returns_complete_result(tmp_path: Path) -> None:
    """Optimization comparisons should own canonical output and advice."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"keras model")
    original = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 100, 4, 5, 6),
        MemoryUsage(100, 200, 300, 400),
        None,
    )
    optimized = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 80, 4, 5, 6),
        MemoryUsage(80, 150, 250, 350),
        None,
    )
    settings = [OptimizationSettings("pruning", 0.5, None)]
    action_resolver = MagicMock()
    action_resolver.operator_compatibility_details.return_value = []
    context = ExecutionContext(
        advice_category={AdviceCategory.OPTIMIZATION},
        action_resolver=action_resolver,
        output_dir=tmp_path,
    )
    collector = object.__new__(EthosUOptimizationPerformance)
    collector.model = model_path
    collector.target = target
    collector.backends = ["vela", "corstone-310"]
    collector.backend_versions = {"vela": "5.0.0", "corstone-310": "unknown"}
    collector.backend_configurations = {
        "vela": {
            "system_config": "Ethos_U55_High_End_Embedded",
            "memory_mode": "Shared_Sram",
        },
        "corstone-310": {
            "fvp": "corstone-310",
            "target": "ethos-u55",
            "mac": 256,
            "profile": "AVH",
        },
    }
    if hasattr(collector, "set_context"):
        collector.set_context(context)
    else:
        collector.context = context

    result = collector.create_optimization_performance_metrics(
        original,
        [(settings, optimized)],
    )

    assert isinstance(result, OptimizationPerformanceResult)
    assert set(vars(result)) == {"standardized_output"}
    validate_standardized_output(result.standardized_output)
    canonical_results = result.standardized_output["results"]
    results_by_producer = {
        canonical_result["producer"]: canonical_result
        for canonical_result in canonical_results
    }
    assert {
        producer: canonical_result["mode"]
        for producer, canonical_result in results_by_producer.items()
    } == {
        "vela": "predicted",
        "corstone-310": "simulated",
    }
    assert {
        backend["id"]: backend["configuration"]
        for backend in result.standardized_output["backends"]
    } == collector.backend_configurations
    for canonical_result in canonical_results:
        assert canonical_result["kind"] == "performance"
        assert {
            metric["qualifiers"]["phase"] for metric in canonical_result["metrics"]
        } == {
            "before",
            "after",
        }
        assert canonical_result["advice"]
        assert {item["category"] for item in canonical_result["advice"]} == {
            "optimization"
        }

    vela_advice = " ".join(
        item["message"] for item in results_by_producer["vela"]["advice"]
    )
    assert "SRAM used" in vela_advice
    assert "NPU total cycles" not in vela_advice

    corstone_advice = " ".join(
        item["message"] for item in results_by_producer["corstone-310"]["advice"]
    )
    assert "NPU total cycles" in corstone_advice
    assert "SRAM used" not in corstone_advice
    assert "DRAM used" not in corstone_advice


def test_optimization_estimator_records_effective_backend_configurations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optimization output metadata must identify the configurations actually used."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    assert target.compiler_options is not None
    backend_repository = MagicMock()
    backend_repository.get_backend_settings.return_value = (
        Path("backend"),
        {"profile": "AVH"},
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_backend_repository",
        lambda: backend_repository,
    )
    estimator = MagicMock()
    estimator_type = MagicMock(return_value=estimator)
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        estimator_type,
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.get_vela_version",
        lambda: "5.0.0",
    )
    collector = object.__new__(EthosUOptimizationPerformance)
    collector.target = target
    collector.backends = ["vela", "corstone-310"]
    collector.context = ExecutionContext()

    assert collector.create_estimator() is estimator

    assert collector.backend_versions == {
        "vela": "5.0.0",
        "corstone-310": "unknown",
    }
    assert collector.backend_configurations["vela"] == {
        "system_config": target.compiler_options.system_config,
        "memory_mode": target.compiler_options.memory_mode,
        "accelerator_config": str(target.compiler_options.accelerator_config),
        "max_block_dependency": target.compiler_options.max_block_dependency,
        "tensor_allocator": target.compiler_options.tensor_allocator,
        "optimization_strategy": target.compiler_options.optimization_strategy,
    }
    assert collector.backend_configurations["corstone-310"] == {
        "fvp": "corstone-310",
        "target": "ethos-u55",
        "mac": 256,
        "profile": "AVH",
    }
    backend_repository.get_backend_settings.assert_called_once_with("corstone-310")


def test_optimization_estimator_rejects_multiple_corstone_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject configurations the single Corstone metrics slot cannot represent."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    estimator = MagicMock()
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        estimator,
    )
    collector = object.__new__(EthosUOptimizationPerformance)
    collector.target = target
    collector.backends = ["corstone-300", "corstone-310"]
    collector.context = ExecutionContext()

    with pytest.raises(
        ConfigurationError,
        match="at most one Corstone backend.*corstone-300, corstone-310",
    ):
        collector.create_estimator()

    estimator.assert_not_called()


def mock_performance_estimation(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with native Vela output."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    vela_metrics = VelaPerf(
        npu_cycles=3,
        sram_access_cycles=1,
        dram_access_cycles=2,
        on_chip_flash_access_cycles=3,
        off_chip_flash_access_cycles=4,
        total_cycles=4,
        batch_inference_time=1.0,
        inferences_per_second=1000.0,
        batch_size=1,
        sram_memory_area_size=1,
        dram_memory_area_size=2,
        on_chip_flash_memory_area_size=3,
        off_chip_flash_memory_area_size=4,
        layerwise_performance_info=LayerwisePerfInfo(layerwise_info=[]),
    )
    estimator = MagicMock()
    estimator.estimate.return_value = metrics
    estimator.vela_perf_metrics = vela_metrics
    estimator.vela_compiler_options = None
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=estimator),
    )


def mock_performance_estimation_with_vela(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with Vela metrics."""
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )

    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = metrics
    mock_estimator.vela_perf_metrics = MagicMock(spec=VelaPerf)
    mock_estimator.vela_perf_metrics.to_standardized_output.return_value = {
        "schema_version": "1.0.0",
        "backends": [{"id": "vela"}],
        "results": [{"kind": "performance"}],
    }
    mock_estimator.vela_compiler_options = None

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=mock_estimator),
    )


def mock_performance_estimation_with_corstone(
    monkeypatch: pytest.MonkeyPatch,
    target: EthosUConfiguration,
    profile: str = "default",
) -> MagicMock:
    """Mock performance estimation with Corstone metrics."""
    mock_corstone_backend_configuration(monkeypatch, profile)
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    metrics.corstone_metrics = MagicMock(spec=CorstonePerf)
    serializer = MagicMock(
        return_value={
            "schema_version": "1.0.0",
            "backends": [{"id": "corstone-310"}],
            "results": [{"kind": "performance"}],
        }
    )
    setattr(metrics, "to_standardized_output", serializer)

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator.estimate",
        MagicMock(return_value=metrics),
    )
    return serializer


def mock_performance_estimation_with_both(
    monkeypatch: pytest.MonkeyPatch, target: EthosUConfiguration
) -> None:
    """Mock performance estimation with both Vela and Corstone metrics."""
    mock_corstone_backend_configuration(monkeypatch)
    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    metrics.corstone_metrics = MagicMock(spec=CorstonePerf)
    setattr(
        metrics,
        "to_standardized_output",
        MagicMock(
            return_value={
                "schema_version": "1.0.0",
                "backends": [{"id": "corstone-310"}],
                "results": [{"kind": "performance", "producer": "corstone-310"}],
            }
        ),
    )

    mock_estimator = MagicMock()
    mock_estimator.estimate.return_value = metrics
    mock_estimator.vela_perf_metrics = MagicMock(spec=VelaPerf)
    mock_estimator.vela_perf_metrics.to_standardized_output.return_value = {
        "schema_version": "1.0.0",
        "backends": [{"id": "vela"}],
        "results": [{"kind": "performance", "producer": "vela"}],
    }
    mock_estimator.vela_compiler_options = None

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MagicMock(return_value=mock_estimator),
    )


def test_operator_compatibility_pytorch_model(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, tmp_path: Path
) -> None:
    """Test operator compatibility with PyTorch model."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    pytorch_model = tmp_path / "model.pt2"
    pytorch_model.write_text("mock pytorch model")

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_pytorch_file",
        MagicMock(return_value=True),
    )

    operators = MagicMock(spec=Operators)
    operators.to_standardized_output.return_value = {}
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.supported_operators",
        MagicMock(return_value=operators),
    )

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.attach_result_advice",
        MagicMock(),
    )

    collector = EthosUOperatorCompatibility(pytorch_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, VelaCompatibilityResult)


def test_operator_compatibility_tosa_model(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, tmp_path: Path
) -> None:
    """Test operator compatibility with TOSA model."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    tosa_model = tmp_path / "model.tosa"
    tosa_model.write_text("mock tosa model")

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_pytorch_file",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tosa_file",
        MagicMock(return_value=True),
    )

    operators = MagicMock(spec=Operators)
    operators.to_standardized_output.return_value = {}
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.supported_operators",
        MagicMock(return_value=operators),
    )

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.attach_result_advice",
        MagicMock(),
    )

    collector = EthosUOperatorCompatibility(tosa_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, VelaCompatibilityResult)


def test_operator_compatibility_pte_model_is_not_supported(
    sample_context: Context, tmp_path: Path
) -> None:
    """Test operator compatibility rejects ExecuTorch models."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    pte_model = tmp_path / "model.pte"
    pte_model.write_text("mock executorch model")

    collector = EthosUOperatorCompatibility(pte_model, target)
    collector.set_context(sample_context)

    with pytest.raises(ConfigurationError, match="ExecuTorch .pte"):
        collector.collect_data()


def test_performance_collector_pytorch_model(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, tmp_path: Path
) -> None:
    """Test performance collector with PyTorch model."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    pytorch_model = tmp_path / "model.pt2"
    pytorch_model.write_text("mock pytorch model")

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_pytorch_file",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tosa_file",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tflite_model",
        MagicMock(return_value=False),
    )

    mock_performance_estimation(monkeypatch, target)

    collector = EthosUPerformance(pytorch_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, VelaPerformanceResult)
    validate_standardized_output(result.standardized_output)


def test_performance_collector_pte_rejects_unsupported_default_target(
    sample_context: Context, tmp_path: Path
) -> None:
    """Test ExecuTorch performance rejects targets without a default runner."""
    target = EthosUConfiguration.load_profile("ethos-u65-256")
    pte_model = tmp_path / "model.pte"
    pte_model.write_text("mock executorch model")

    collector = EthosUPerformance(pte_model, target)
    collector.set_context(sample_context)

    with pytest.raises(ConfigurationError, match="not supported for target"):
        collector.collect_data()


def test_performance_collector_pytorch_with_corstone_backend(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, tmp_path: Path
) -> None:
    """Test PyTorch performance can be requested on Corstone directly."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    pytorch_model = tmp_path / "model.pt2"
    pytorch_model.write_text("mock pytorch model")

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_pytorch_file",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tosa_file",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tflite_model",
        MagicMock(return_value=False),
    )

    metrics = PerformanceMetrics(
        target,
        NPUCycles(1, 2, 3, 4, 5, 6),
        MemoryUsage(1, 2, 3, 4),
        LayerwisePerfInfo(layerwise_info=[]),
    )
    metrics.corstone_metrics = CorstonePerf(
        CorstoneModelPerformanceMetrics(1, 2, 3, 4, 5, 6), []
    )
    captured_backends: list[str] | None = None

    class MockEstimator:
        def __init__(
            self,
            context: Context,
            target_config: EthosUConfiguration,
            backends: list[str],
        ) -> None:
            nonlocal captured_backends
            captured_backends = backends

        def estimate(self, model: Path) -> PerformanceMetrics:
            return metrics

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.EthosUPerformanceEstimator",
        MockEstimator,
    )
    mock_corstone_backend_configuration(monkeypatch)

    collector = EthosUPerformance(pytorch_model, target, backends=["corstone-300"])
    collector.set_context(sample_context)

    result = collector.collect_data()

    assert isinstance(result, CorstonePerformanceResult)
    validate_standardized_output(result.standardized_output)
    assert captured_backends == ["corstone-300"]


def test_performance_collector_tosa_model(
    monkeypatch: pytest.MonkeyPatch, sample_context: Context, tmp_path: Path
) -> None:
    """Test performance collector with TOSA model."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    tosa_model = tmp_path / "model.tosa"
    tosa_model.write_text("mock tosa model")

    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_pytorch_file",
        MagicMock(return_value=False),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tosa_file",
        MagicMock(return_value=True),
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.data_collection.is_tflite_model",
        MagicMock(return_value=False),
    )

    mock_performance_estimation(monkeypatch, target)

    collector = EthosUPerformance(tosa_model, target)
    collector.set_context(sample_context)

    result = collector.collect_data()
    assert isinstance(result, VelaPerformanceResult)
    validate_standardized_output(result.standardized_output)


def test_performance_collector_tosa_model_rejects_corstone_backend(
    sample_context: Context, tmp_path: Path
) -> None:
    """Test TOSA performance rejects Corstone backends."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    tosa_model = tmp_path / "model.tosa"
    tosa_model.write_text("mock tosa model")

    collector = EthosUPerformance(tosa_model, target, backends=["corstone-300"])
    collector.set_context(sample_context)

    with pytest.raises(
        ConfigurationError,
        match="TOSA performance estimation is only supported with the Vela backend",
    ):
        collector.collect_data()


def test_performance_collector_invalid_model_format(
    sample_context: Context, tmp_path: Path
) -> None:
    """Test performance collector with invalid model format."""
    target = EthosUConfiguration.load_profile("ethos-u55-256")

    invalid_model = tmp_path / "model.txt"
    invalid_model.write_text("not a model")

    collector = EthosUPerformance(invalid_model, target)
    collector.set_context(sample_context)

    with pytest.raises(ConfigurationError, match="Input must be a TFLite"):
        collector.collect_data()
