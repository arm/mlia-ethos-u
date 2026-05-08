# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Performance estimation tests."""

from typing import Any, cast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.errors import BackendUnavailableError
from mlia.core.errors import ConfigurationError
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import (
    CorstonePerformanceEstimator,
    EthosUPerformanceEstimator,
    MemoryUsage,
    NPUCycles,
    PerformanceMetrics,
    VelaPerformanceEstimator,
    merge_performance_outputs,
)
from mlia.target.ethos_u.utils.tflite_shims import ModelConfiguration


def test_performance_metrics_to_standardized_output_without_corstone_metrics(
    tmp_path: Path,
) -> None:
    """Return None when no Corstone metrics are attached."""
    target_cfg = MagicMock(target="ethos-u55", mac=256)
    metrics = PerformanceMetrics(target_cfg, None, None, None)

    model_path = tmp_path / "model.tflite"
    model_path.touch()

    assert metrics.to_standardized_output(model_path) is None


def test_performance_metrics_to_standardized_output_uses_default_backend_name(
    tmp_path: Path,
) -> None:
    """Use default backend name and build target config."""
    target_cfg = MagicMock(target="ethos-u55", mac=256)
    backend_metrics = MagicMock()

    metrics = PerformanceMetrics(
        target_cfg, None, None, None, corstone_metrics=backend_metrics
    )

    model_path = tmp_path / "model.tflite"
    model_path.touch()

    result = metrics.to_standardized_output(model_path)

    backend_metrics.to_standardized_output.assert_called_once()
    _, kwargs = backend_metrics.to_standardized_output.call_args
    assert kwargs["backend_name"] == "corstone-300"
    assert kwargs["target_config"] == {"target": "ethos-u55", "mac": 256}
    assert result is backend_metrics.to_standardized_output.return_value


def test_merge_performance_outputs_requires_output() -> None:
    """At least one standardized output must be provided."""
    with pytest.raises(ValueError, match="At least one output must be provided"):
        merge_performance_outputs(None, None)


def test_merge_performance_outputs_merges_backends_and_results() -> None:
    """Merge backends and results from both outputs."""
    vela_output = {
        "backends": ["vela-backend"],
        "results": ["vela-result"],
        "other": "keep",
    }
    corstone_output = {
        "backends": ["corstone-backend"],
        "results": ["corstone-result"],
    }

    merged = merge_performance_outputs(vela_output, corstone_output)

    assert merged["backends"] == ["vela-backend", "corstone-backend"]
    assert merged["results"] == ["vela-result", "corstone-result"]
    assert merged["other"] == "keep"


def test_merge_performance_outputs_raises_when_base_invalid() -> None:
    """Raise error when base output cannot be determined."""

    class FlakyBool:
        """Object that is truthy once then falsy."""

        def __init__(self):
            self._calls = 0

        def __bool__(self):
            self._calls += 1
            return self._calls == 1

    class AlwaysFalse:
        """Object that is always falsy."""

        def __bool__(self):
            return False

    vela_output = FlakyBool()
    corstone_output = AlwaysFalse()

    with pytest.raises(ValueError, match="No valid output provided"):
        merge_performance_outputs(
            cast(dict[str, Any] | None, vela_output),
            cast(dict[str, Any] | None, corstone_output),
        )


def test_vela_performance_estimator_raises_without_compiler_options(
    sample_context,
) -> None:
    """Vela estimator should fail when compiler options are missing."""
    target_cfg = MagicMock(compiler_options=None)
    estimator = VelaPerformanceEstimator(sample_context, target_cfg)

    with pytest.raises(BackendUnavailableError, match="Backend vela is not available"):
        estimator.estimate(Path("model.tflite"))


def test_vela_performance_estimator_estimate_populates_metrics_and_options(
    sample_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vela estimator should call backend and expose metrics and options."""
    target_cfg = MagicMock()
    target_cfg.compiler_options = MagicMock(name="compiler_options")

    vela_perf_metrics = MagicMock(
        sram_memory_area_size=1,
        dram_memory_area_size=2,
        on_chip_flash_memory_area_size=3,
        off_chip_flash_memory_area_size=4,
        layerwise_performance_info="layer-info",
    )

    estimate_mock = MagicMock(return_value=vela_perf_metrics)
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.vela_perf.estimate_performance",
        estimate_mock,
    )

    estimator = VelaPerformanceEstimator(sample_context, target_cfg)
    model_path = Path("model.tflite")

    memory_usage, layer_info = estimator.estimate(model_path)

    estimate_mock.assert_called_once_with(model_path, target_cfg.compiler_options)
    assert isinstance(memory_usage, MemoryUsage)
    assert memory_usage.sram_memory_area_size == 1
    assert memory_usage.dram_memory_area_size == 2
    assert memory_usage.on_chip_flash_memory_area_size == 3
    assert memory_usage.off_chip_flash_memory_area_size == 4
    assert layer_info == "layer-info"
    assert estimator.vela_perf_metrics is vela_perf_metrics
    assert estimator.vela_compiler_options is target_cfg.compiler_options


def test_corstone_performance_estimator_raises_without_compiler_options(
    sample_context,
) -> None:
    """Corstone estimator should fail when compiler options are missing."""
    target_cfg = MagicMock(
        compiler_options=None,
        target="ethos-u55",
        mac=256,
    )

    estimator = CorstonePerformanceEstimator(
        sample_context, target_cfg, backend="corstone-300"
    )

    with pytest.raises(BackendUnavailableError, match="Backend vela is not available"):
        estimator.estimate(Path("model.tflite"))


def test_corstone_performance_estimator_estimate_uses_compiler_and_backend(
    sample_context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Corstone estimator should invoke Vela compile and backend metrics."""
    target_cfg = MagicMock(
        compiler_options=MagicMock(name="compiler_options"),
        target="ethos-u55",
        mac=256,
    )
    backend_name = "corstone-300"

    optimized_model = tmp_path / "optimized_model.tflite"
    compile_mock = MagicMock(return_value=optimized_model)
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.vela_comp.compile_model",
        compile_mock,
    )

    corstone_metrics = MagicMock()
    estimate_perf_mock = MagicMock(return_value=corstone_metrics)
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.estimate_performance",
        estimate_perf_mock,
    )

    estimator = CorstonePerformanceEstimator(
        sample_context, target_cfg, backend=backend_name
    )

    model_path = tmp_path / "model.tflite"
    result = estimator.estimate(model_path)

    compile_mock.assert_called_once_with(model_path, target_cfg.compiler_options)
    estimate_perf_mock.assert_called_once_with(
        target_cfg.target,
        target_cfg.mac,
        optimized_model,
        backend_name,
        sample_context.output_dir,
    )

    assert isinstance(result, NPUCycles)
    assert estimator.backend_metrics is corstone_metrics


def test_corstone_performance_estimator_prepares_pte_without_conversion(
    sample_context, tmp_path: Path
) -> None:
    """ExecuTorch model files are already runner-ready artifacts."""
    target_cfg = EthosUConfiguration.load_profile("ethos-u55-256")
    estimator = CorstonePerformanceEstimator(
        sample_context, target_cfg, backend="corstone-300"
    )

    model_path = tmp_path / "model.pte"
    model_path.write_text("mock executorch model")

    assert (
        estimator._prepare_executorch_model(model_path) == model_path  # pylint: disable=protected-access
    )


def test_ethosu_performance_estimator_rejects_unsupported_backend(
    sample_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject unsupported backends at initialisation time."""
    target_cfg = MagicMock(target="ethos-u55")

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.supported_backends",
        lambda _target: ["corstone-300"],
    )

    with pytest.raises(ValueError, match="Unsupported backend 'invalid'"):
        EthosUPerformanceEstimator(sample_context, target_cfg, backends=["invalid"])


def test_ethosu_performance_estimator_uses_default_vela_backend(
    sample_context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default to Vela backend when none are provided."""
    target_cfg = MagicMock(target="ethos-u55")

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.supported_backends",
        lambda _target: [],
    )

    estimator = EthosUPerformanceEstimator(sample_context, target_cfg)
    assert estimator.backends == {"vela"}


def test_ethosu_performance_estimator_rejects_pte_with_vela(
    sample_context, tmp_path: Path
) -> None:
    """ExecuTorch performance is restricted to Corstone backends."""
    target_cfg = EthosUConfiguration.load_profile("ethos-u55-256")
    model_path = tmp_path / "model.pte"
    model_path.write_text("mock executorch model")

    estimator = EthosUPerformanceEstimator(
        sample_context, target_cfg, backends=["vela"]
    )

    with pytest.raises(ConfigurationError, match="Corstone backends"):
        estimator.estimate(model_path)


def test_ethosu_performance_estimator_rejects_pte_without_backend(
    sample_context, tmp_path: Path
) -> None:
    """Default backend selection is still Vela and rejects ExecuTorch files."""
    target_cfg = EthosUConfiguration.load_profile("ethos-u55-256")
    model_path = tmp_path / "model.pte"
    model_path.write_text("mock executorch model")

    estimator = EthosUPerformanceEstimator(sample_context, target_cfg)

    assert estimator.backends == {"vela"}
    with pytest.raises(ConfigurationError, match="Corstone backends"):
        estimator.estimate(model_path)


def test_ethosu_performance_estimator_estimate_combines_backends(
    sample_context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Combine Vela and Corstone metrics and attach raw backend data."""
    target_cfg = MagicMock(target="ethos-u55", mac=256)

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.supported_backends",
        lambda _target: ["corstone-300"],
    )

    tflite_model = MagicMock(name="tflite_model")
    get_tflite_model_mock = MagicMock(return_value=tflite_model)
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.get_tflite_model",
        get_tflite_model_mock,
    )

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.is_corstone_backend",
        lambda backend: backend.startswith("corstone-"),
    )

    class TestVelaEstimator:
        """Stub Vela estimator used by EthosUPerformanceEstimator."""

        def __init__(self, context, target_config):
            self.context = context
            self.target_config = target_config
            self.vela_perf_metrics = "vela-metrics"
            self.vela_compiler_options = "vela-options"

        def estimate(self, model):
            assert model is tflite_model
            return (
                MemoryUsage(
                    sram_memory_area_size=1,
                    dram_memory_area_size=2,
                    on_chip_flash_memory_area_size=3,
                    off_chip_flash_memory_area_size=4,
                ),
                "layer-info",
            )

    class TestCorstoneEstimator:
        """Stub Corstone estimator used by EthosUPerformanceEstimator."""

        def __init__(self, context, target_config, backend) -> None:  # noqa: D401
            self.context = context
            self.target_config = target_config
            self.backend = backend
            self.backend_metrics = "corstone-metrics"

        def estimate(self, model):
            assert model is tflite_model
            return "npu-cycles"

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.VelaPerformanceEstimator",
        TestVelaEstimator,
    )
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.CorstonePerformanceEstimator",
        TestCorstoneEstimator,
    )

    model_path = tmp_path / "model.tflite"
    model_path.touch()
    model_config = ModelConfiguration(model_path)

    estimator = EthosUPerformanceEstimator(
        sample_context, target_cfg, backends=["vela", "corstone-300"]
    )
    perf = estimator.estimate(model_config)

    get_tflite_model_mock.assert_called_once()
    called_model_path, called_ctx = get_tflite_model_mock.call_args[0]
    assert called_model_path == model_path
    assert called_ctx is sample_context

    assert isinstance(perf, PerformanceMetrics)
    assert perf.target_config is target_cfg
    assert isinstance(perf.memory_usage, MemoryUsage)
    assert perf.memory_usage.sram_memory_area_size == 1
    assert perf.layerwise_perf_info == "layer-info"
    assert perf.npu_cycles == "npu-cycles"
    assert perf.corstone_metrics == "corstone-metrics"
    assert estimator.vela_perf_metrics == "vela-metrics"
    assert estimator.vela_compiler_options == "vela-options"


def test_ethosu_performance_estimator_logs_unsupported_backend_at_estimate_time(
    sample_context, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Log a warning when encountering an unsupported backend during estimate."""
    target_cfg = MagicMock(target="ethos-u55", mac=256)

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.supported_backends",
        lambda _target: [],
    )

    estimator = EthosUPerformanceEstimator(
        sample_context, target_cfg, backends=["vela"]
    )
    # Force an unsupported backend into the estimator after initialisation
    estimator.backends = {"unknown-backend"}

    model_path = tmp_path / "model.tflite"
    model_path.touch()

    tflite_model = MagicMock(name="tflite_model")
    get_tflite_model_mock = MagicMock(return_value=tflite_model)
    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.get_tflite_model",
        get_tflite_model_mock,
    )

    monkeypatch.setattr(
        "mlia.target.ethos_u.performance.is_corstone_backend",
        lambda _backend: False,
    )

    test_logger = MagicMock()
    monkeypatch.setattr("mlia.target.ethos_u.performance.logger", test_logger)

    perf = estimator.estimate(model_path)

    assert isinstance(perf, PerformanceMetrics)
    assert perf.npu_cycles is None
    assert perf.memory_usage is None
    assert perf.layerwise_perf_info is None
    test_logger.warning.assert_called_once()
