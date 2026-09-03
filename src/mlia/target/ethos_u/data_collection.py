# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collection module for Ethos-U."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

from mlia.backend.corstone import is_corstone_backend
from mlia.backend.repo import get_backend_repository
from mlia.backend.vela.compat import VelaCompatibilityResult, supported_operators
from mlia.backend.vela.performance import get_vela_version
from mlia.core.data_collection import ContextAwareDataCollector
from mlia.core.errors import ConfigurationError
from mlia.core.performance import P, PerformanceEstimator
from mlia.target.ethos_u.utils.legacy_shims import OptimizingPerformaceDataCollector
from mlia.target.ethos_u.utils.tflite_shims import (
    LegacyChecker,
    ModelConfiguration,
    TFLiteCompatibilityResult,
    get_tflite_model,
    is_legacy_model,
)
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import (
    CombinedPerformanceResult,
    CorstonePerformanceResult,
    EthosUPerformanceEstimator,
    OptimizationPerformanceMetrics,
    OptimizationPerformanceResult,
    PerformanceMetrics,
    VelaPerformanceResult,
    merge_performance_outputs,
    optimization_performance_metrics_for_backend,
)
from mlia.target.ethos_u.result_advice import attach_result_advice
from mlia.target.ethos_u.standardized_output import (
    build_optimization_performance_output,
    build_tflite_compatibility_output,
    model_metadata,
    validate_performance_backends,
)
from mlia.target.ethos_u.utils.model_format import (
    is_pte_file,
    is_pytorch_file,
    is_tflite_model,
    is_tosa_file,
)
from mlia.utils.logging import log_action


def _default_pte_backend_for_target(target: str) -> str:
    """Return the default Corstone backend for ExecuTorch performance."""
    default_backend_by_target = {
        "ethos-u55": "corstone-300",
        "ethos-u85": "corstone-320",
    }

    try:
        return default_backend_by_target[target]
    except KeyError as err:
        raise ConfigurationError(
            f"ExecuTorch .pte performance is not supported for target '{target}'."
        ) from err


def _vela_backend_configuration(compiler_options: Any) -> dict[str, Any]:
    """Extract the effective Vela compiler configuration."""
    return {
        "system_config": compiler_options.system_config,
        "memory_mode": compiler_options.memory_mode,
        "accelerator_config": str(compiler_options.accelerator_config)
        if compiler_options.accelerator_config
        else None,
        "max_block_dependency": compiler_options.max_block_dependency,
        "tensor_allocator": compiler_options.tensor_allocator,
        "optimization_strategy": compiler_options.optimization_strategy,
    }


def _corstone_backend_configuration(
    backend_name: str, target: EthosUConfiguration
) -> dict[str, Any]:
    """Return the effective Corstone runtime configuration."""
    _backend_path, settings = get_backend_repository().get_backend_settings(
        backend_name
    )
    if not settings or "profile" not in settings:
        raise ConfigurationError(f"Unable to configure backend {backend_name}.")
    return {
        "fvp": backend_name,
        "target": target.target,
        "mac": target.mac,
        "profile": settings["profile"],
    }


def _performance_backend_configurations(
    target: EthosUConfiguration, backends: list[str]
) -> dict[str, dict[str, Any]]:
    """Return effective configurations for performance backends."""
    configurations: dict[str, dict[str, Any]] = {}
    for backend_name in backends:
        if backend_name == "vela":
            if target.compiler_options is None:
                raise ConfigurationError("Vela compiler options are unavailable.")
            configurations[backend_name] = _vela_backend_configuration(
                target.compiler_options
            )
        elif is_corstone_backend(backend_name):
            configurations[backend_name] = _corstone_backend_configuration(
                backend_name, target
            )
    return configurations


class EthosUOperatorCompatibility(ContextAwareDataCollector):
    """Collect operator compatibility information."""

    def __init__(self, model: Path, target_config: EthosUConfiguration) -> None:
        """Init operator compatibility data collector."""
        self.model = model
        self.target_config = target_config

    def collect_data(
        self,
    ) -> VelaCompatibilityResult | TFLiteCompatibilityResult:
        """Collect operator compatibility information."""
        if is_pte_file(self.model):
            raise ConfigurationError(
                "Operator compatibility is not supported for ExecuTorch .pte files."
            )
        if is_pytorch_file(self.model) or is_tosa_file(self.model):
            model_path = self.model
        else:
            if is_legacy_model(self.model):
                with log_action("Checking TensorFlow Lite compatibility ..."):
                    legacy_checker = LegacyChecker()
                    tflite_compat = legacy_checker.check_compatibility(self.model)

                    if not tflite_compat.compatible:
                        cli_args = (
                            [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []
                        )
                        standardized_output = build_tflite_compatibility_output(
                            self.model,
                            self.target_config,
                            tflite_compat,
                            cli_args,
                        )
                        attach_result_advice(
                            standardized_output, tflite_compat, self.context
                        )
                        return TFLiteCompatibilityResult(standardized_output)

            tflite_model = get_tflite_model(self.model, self.context)
            model_path = Path(tflite_model.model_path)

        with log_action("Checking operator compatibility ..."):
            operators = supported_operators(
                model_path, self.target_config.compiler_options
            )

        # Generate standardized output
        target_config = {
            "profile_name": self.target_config.profile_name,
            "target": self.target_config.target,
            "mac": self.target_config.mac,
        }
        # Get compiler options for backend configuration
        backend_config = (
            _vela_backend_configuration(self.target_config.compiler_options)
            if self.target_config.compiler_options
            else {}
        )

        # Clean CLI arguments to use basename for executable
        cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

        standardized_output = operators.to_standardized_output(
            model_path=model_path,
            target_config=target_config,
            backend_config=backend_config,
            cli_arguments=cli_args,
        )
        standardized_output["model"] = model_metadata(self.model).to_dict()

        attach_result_advice(standardized_output, operators, self.context)
        return VelaCompatibilityResult(standardized_output)

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_operator_compatibility"


class EthosUPerformance(ContextAwareDataCollector):
    """Collect performance metrics."""

    def __init__(
        self,
        model: Path,
        target_config: EthosUConfiguration,
        backends: list[str] | None = None,
    ) -> None:
        """Init performance data collector."""
        self.model = model
        self.target_config = target_config
        self.backends = backends

    def collect_data(
        self,
    ) -> VelaPerformanceResult | CorstonePerformanceResult | CombinedPerformanceResult:
        """Collect model performance metrics and return canonical output."""
        if not any(
            [
                is_tflite_model(self.model),
                is_legacy_model(self.model),
                is_tosa_file(self.model),
                is_pytorch_file(self.model),
                is_pte_file(self.model),
            ]
        ):
            raise ConfigurationError(
                "Input must be a TFLite, TOSA, ExecuTorch .pte or PyTorch .pt2 file."
            )

        is_pte_model = is_pte_file(self.model)
        requested_backends = (
            [_default_pte_backend_for_target(self.target_config.target)]
            if is_pte_model and self.backends is None
            else self.backends
        )
        backends = validate_performance_backends(requested_backends)
        any_corstone_backend = any(is_corstone_backend(backend) for backend in backends)
        if is_tosa_file(self.model) and any_corstone_backend:
            raise ConfigurationError(
                "TOSA performance estimation is only supported with the Vela backend. "
                "Use '-b vela' or provide a TFLite/.pte/.pt2 model for Corstone."
            )

        model_to_estimate: Path | Any
        if is_pte_model:
            if any(not is_corstone_backend(backend) for backend in backends):
                raise ConfigurationError(
                    "ExecuTorch .pte performance is only supported with "
                    "Corstone backends."
                )
            model_to_estimate = self.model
        elif is_pytorch_file(self.model) or is_tosa_file(self.model):
            model_to_estimate = self.model
        else:
            model_to_estimate = get_tflite_model(self.model, self.context)

        serialization_model = (
            Path(model_to_estimate.model_path)
            if isinstance(model_to_estimate, ModelConfiguration)
            else Path(model_to_estimate)
        )
        output_model = model_metadata(self.model).to_dict()

        estimator = EthosUPerformanceEstimator(
            self.context,
            self.target_config,
            backends,
        )
        performance = estimator.estimate(model_to_estimate)
        cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []

        vela_result = None
        if "vela" in backends:
            vela_metrics = getattr(estimator, "vela_perf_metrics", None)
            if vela_metrics is None:
                raise ConfigurationError(
                    "Vela performance metrics were not produced by the estimator."
                )
            compiler_options = getattr(estimator, "vela_compiler_options", None)
            vela_output = vela_metrics.to_standardized_output(
                model_path=serialization_model,
                target_config={
                    "profile_name": self.target_config.profile_name,
                    "target": self.target_config.target,
                    "mac": self.target_config.mac,
                },
                backend_config=(
                    _vela_backend_configuration(compiler_options)
                    if compiler_options
                    else {}
                ),
                cli_arguments=cli_args,
            )
            if not isinstance(vela_output, dict):
                raise ConfigurationError(
                    "Vela performance serialization did not produce canonical output."
                )
            vela_output["model"] = output_model.copy()
            vela_result = VelaPerformanceResult(vela_output)
            attach_result_advice(vela_output, vela_result, self.context)

        corstone_backend = next(
            (backend for backend in backends if is_corstone_backend(backend)), None
        )
        corstone_result = None
        if corstone_backend is not None:
            if performance.corstone_metrics is None:
                raise ConfigurationError(
                    f"{corstone_backend} performance metrics were not produced by "
                    "the estimator."
                )
            corstone_output = performance.to_standardized_output(
                model_path=serialization_model,
                backend_name=corstone_backend,
                cli_arguments=cli_args,
                backend_config=_corstone_backend_configuration(
                    corstone_backend, self.target_config
                ),
            )
            if not isinstance(corstone_output, dict):
                raise ConfigurationError(
                    f"{corstone_backend} performance serialization did not produce "
                    "canonical output."
                )
            corstone_output["model"] = output_model.copy()
            corstone_result = CorstonePerformanceResult(corstone_output)
            attach_result_advice(corstone_output, corstone_result, self.context)

        if vela_result is not None and corstone_result is not None:
            return CombinedPerformanceResult(
                merge_performance_outputs(
                    vela_result.standardized_output,
                    corstone_result.standardized_output,
                )
            )
        if vela_result is not None:
            return vela_result
        if corstone_result is not None:
            return corstone_result

        raise ConfigurationError("No performance backends were configured.")

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_performance"


class EthosUOptimizationPerformance(OptimizingPerformaceDataCollector):
    """Collect performance metrics for performance optimizations."""

    def create_estimator(self) -> PerformanceEstimator:
        """Create a PerformanceEstimator, to be overridden in subclasses."""
        backends = validate_performance_backends(self.backends)
        target = cast(EthosUConfiguration, self.target)
        estimator = EthosUPerformanceEstimator(self.context, target, backends)
        self.backend_configurations = _performance_backend_configurations(
            target, backends
        )
        self.backend_versions = {
            backend_name: get_vela_version() if backend_name == "vela" else "unknown"
            for backend_name in backends
        }
        return estimator

    def create_optimization_performance_metrics(
        self, original_metrics: P, optimizations_perf_metrics: list[P]
    ) -> OptimizationPerformanceResult:
        """Create a completed optimization comparison result."""
        comparison = OptimizationPerformanceMetrics(
            original_perf_metrics=cast(PerformanceMetrics, original_metrics),
            optimizations_perf_metrics=cast(Any, optimizations_perf_metrics),
        )
        cli_args = [Path(sys.argv[0]).name] + sys.argv[1:] if sys.argv else []
        standardized_output = build_optimization_performance_output(
            self.model,
            cast(EthosUConfiguration, self.target),
            comparison,
            self.backends,
            self.backend_configurations,
            self.backend_versions,
            cli_args,
        )
        for canonical_result in standardized_output["results"]:
            single_result_output = {
                **standardized_output,
                "results": [canonical_result],
            }
            backend_comparison = optimization_performance_metrics_for_backend(
                comparison, canonical_result["producer"]
            )
            attach_result_advice(single_result_output, backend_comparison, self.context)
        return OptimizationPerformanceResult(standardized_output)

    @classmethod
    def name(cls) -> str:
        """Return name of the collector."""
        return "ethos_u_model_optimizations"
