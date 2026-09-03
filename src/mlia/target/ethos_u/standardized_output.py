# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Canonical output builders for Ethos-U target-owned fallback results."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.corstone import is_corstone_backend
from mlia.core.errors import ConfigurationError
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.performance import (
    MemorySizeType,
    OptimizationPerformanceMetrics,
    PerformanceMetrics,
)
from mlia.target.ethos_u.performance_warnings import NPU_ONLY_PERFORMANCE_WARNING
from mlia.target.ethos_u.utils.tflite_shims import TFLiteCompatibilityInfo
from mlia.utils.filesystem import file_chunks, sha256

_DIRECTORY_DIGEST_DOMAIN = b"mlia-model-directory-manifest-v1\0"


def build_tflite_compatibility_output(
    model_path: Path,
    target_config: EthosUConfiguration,
    compatibility: TFLiteCompatibilityInfo,
    cli_arguments: list[str],
) -> dict[str, Any]:
    """Build output for a model that failed the TensorFlow Lite precheck."""
    failed = compatibility.check_failed_with_unknown_error
    result = schema.Result(
        kind=schema.ResultKind.COMPATIBILITY,
        status=(
            schema.ResultStatus.FAILED if failed else schema.ResultStatus.INCOMPATIBLE
        ),
        producer="tflite-converter",
        metrics=[
            schema.Metric(
                name=schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
                value=None,
                unit=schema.UNIT_PERCENT,
                availability=schema.MetricAvailability.UNAVAILABLE,
                reason="Operator placement was not analyzed because conversion failed.",
            )
        ],
        checks=[
            schema.Check(
                id="tflite_conversion",
                status=schema.CheckStatus.FAIL,
                details=_tflite_compatibility_details(compatibility),
            )
        ],
    )
    return _standardized_output(
        model_path,
        target_config,
        backends=[
            schema.Backend(
                id="tflite-converter",
                name="TensorFlow Lite Converter",
                version="unknown",
                configuration={},
            )
        ],
        results=[result],
        cli_arguments=cli_arguments,
    )


def validate_performance_backends(
    backends: list[str] | None,
) -> list[str]:
    """Return unique backends after validating performance data ownership."""
    performance_backends = list(
        dict.fromkeys(["vela"] if backends is None else backends)
    )
    if not performance_backends:
        raise ConfigurationError("No performance backends were configured.")
    corstone_backends = [
        backend for backend in performance_backends if is_corstone_backend(backend)
    ]
    if len(corstone_backends) > 1:
        requested = ", ".join(corstone_backends)
        raise ConfigurationError(
            "Performance collection supports at most one Corstone backend; "
            f"requested: {requested}."
        )
    return performance_backends


def build_optimization_performance_output(
    model_path: Path,
    target_config: EthosUConfiguration,
    comparison: OptimizationPerformanceMetrics,
    backends: list[str] | None,
    backend_configurations: dict[str, dict[str, Any]],
    backend_versions: dict[str, str],
    cli_arguments: list[str],
) -> dict[str, Any]:
    """Build backend-specific optimization comparison results."""
    performance_backends = validate_performance_backends(backends)
    backend_metadata: list[schema.Backend] = []
    results: list[schema.Result] = []

    for backend_name in performance_backends:
        try:
            backend_configuration = backend_configurations[backend_name]
        except KeyError as err:
            raise ConfigurationError(
                f"Missing optimization configuration for backend {backend_name!r}."
            ) from err
        try:
            backend_version = backend_versions[backend_name]
        except KeyError as err:
            raise ConfigurationError(
                f"Missing optimization version for backend {backend_name!r}."
            ) from err
        backend, mode = _optimization_backend(
            backend_name, backend_version, backend_configuration
        )
        backend_metadata.append(backend)
        metrics: list[schema.Metric] = []

        for index, (settings, optimized_metrics) in enumerate(
            comparison.optimizations_perf_metrics
        ):
            optimization_qualifiers = {
                "optimization_index": index,
                "optimizations": [_optimization_settings(item) for item in settings],
            }
            metrics.extend(
                _performance_metrics(
                    comparison.original_perf_metrics,
                    backend_name=backend_name,
                    qualifiers={"phase": "before", **optimization_qualifiers},
                )
            )
            metrics.extend(
                _performance_metrics(
                    optimized_metrics,
                    backend_name=backend_name,
                    qualifiers={"phase": "after", **optimization_qualifiers},
                )
            )

        results.append(
            schema.Result(
                kind=schema.ResultKind.PERFORMANCE,
                status=schema.ResultStatus.OK,
                producer=backend_name,
                mode=mode,
                warnings=[NPU_ONLY_PERFORMANCE_WARNING],
                metrics=metrics,
            )
        )

    return _standardized_output(
        model_path,
        target_config,
        backends=backend_metadata,
        results=results,
        cli_arguments=cli_arguments,
    )


def _optimization_backend(
    backend_name: str,
    version: str,
    configuration: dict[str, Any],
) -> tuple[schema.Backend, schema.ModeType]:
    """Return canonical backend metadata and measurement mode."""
    if backend_name == "vela":
        return (
            schema.Backend(
                id="vela",
                name="Vela Compiler",
                version=version,
                configuration=dict(configuration),
            ),
            schema.ModeType.PREDICTED,
        )
    if is_corstone_backend(backend_name):
        return (
            schema.Backend(
                id=backend_name,
                name=backend_name.replace("-", " ").title(),
                version=version,
                configuration=dict(configuration),
            ),
            schema.ModeType.SIMULATED,
        )
    raise ValueError(f"Unsupported Ethos-U performance backend: {backend_name}")


def _standardized_output(
    model_path: Path,
    target_config: EthosUConfiguration,
    *,
    backends: list[schema.Backend],
    results: list[schema.Result],
    cli_arguments: list[str],
) -> dict[str, Any]:
    """Build the shared canonical output envelope."""
    return schema.StandardizedOutput(
        schema_version=schema.SCHEMA_VERSION,
        run_id=schema.StandardizedOutput.create_run_id(),
        timestamp=schema.StandardizedOutput.create_timestamp(),
        tool=schema.Tool(name="mlia", version=mlia.__version__),
        target=_target_metadata(target_config),
        model=model_metadata(model_path),
        context=schema.Context(cli_arguments=cli_arguments),
        backends=backends,
        results=results,
    ).to_dict()


def _target_metadata(target_config: EthosUConfiguration) -> schema.Target:
    """Return canonical Ethos-U target metadata."""
    configuration = {
        "target": target_config.target,
        "mac": target_config.mac,
    }
    return schema.Target(
        profile_name=target_config.profile_name,
        target_type="npu",
        components=[
            schema.Component(
                type=schema.ComponentType.NPU,
                family=target_config.target,
                variant=str(target_config.mac),
            )
        ],
        configuration=configuration,
    )


def model_metadata(model_path: Path) -> schema.Model:
    """Return canonical metadata for a model file or SavedModel directory."""
    model_hash, size_bytes = _model_digest(model_path)
    return schema.Model(
        name=model_path.name,
        format=(
            "saved_model"
            if model_path.is_dir()
            else model_path.suffix.lstrip(".").lower() or "unknown"
        ),
        hash=model_hash,
        size_bytes=size_bytes,
    )


def _model_digest(model_path: Path) -> tuple[str, int]:
    """Return a deterministic digest and size for a model file or directory."""
    if model_path.is_file():
        return sha256(model_path), model_path.stat().st_size
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    files = sorted(
        (
            path.relative_to(model_path).as_posix(),
            path,
        )
        for path in model_path.rglob("*")
        if path.is_file()
    )
    digest = hashlib.sha256()
    digest.update(_DIRECTORY_DIGEST_DOMAIN)
    digest.update(len(files).to_bytes(8, "big"))
    size_bytes = 0

    for relative_path, path in files:
        content_digest = hashlib.sha256()
        content_size = 0
        for chunk in file_chunks(path):
            content_digest.update(chunk)
            content_size += len(chunk)

        relative_path_bytes = relative_path.encode("utf-8")
        digest.update(len(relative_path_bytes).to_bytes(8, "big"))
        digest.update(relative_path_bytes)
        digest.update(content_size.to_bytes(8, "big"))
        digest.update(content_digest.digest())
        size_bytes += content_size

    return digest.hexdigest(), size_bytes


def _tflite_compatibility_details(
    compatibility: TFLiteCompatibilityInfo,
) -> dict[str, Any]:
    """Return structured details for a TensorFlow Lite conversion failure."""
    details: dict[str, Any] = {
        "status": getattr(compatibility.status, "name", str(compatibility.status)),
    }
    conversion_errors = getattr(compatibility, "conversion_errors", None) or []
    if conversion_errors:
        details["conversion_errors"] = [
            {
                "code": getattr(error.code, "name", str(error.code)),
                "operator": error.operator,
                "location": list(error.location),
                "message": error.message,
            }
            for error in conversion_errors
        ]
    conversion_exception = getattr(compatibility, "conversion_exception", None)
    if conversion_exception is not None:
        details["exception"] = str(conversion_exception)
    return details


def _performance_metrics(
    performance: PerformanceMetrics,
    *,
    backend_name: str,
    qualifiers: dict[str, Any],
) -> list[schema.Metric]:
    """Return one backend's complete qualified optimization metric set."""
    metrics: list[schema.Metric] = []
    if backend_name == "vela" and performance.memory_usage is not None:
        memory = performance.memory_usage
        multiplier = 1024 if memory.memory_size_type == MemorySizeType.KILOBYTES else 1
        metrics.extend(
            schema.Metric(
                name=name,
                value=value * multiplier,
                unit="bytes",
            )
            for name, value in (
                ("sram_memory_area_size", memory.sram_memory_area_size),
                ("dram_memory_area_size", memory.dram_memory_area_size),
                (
                    "on_chip_flash_memory_area_size",
                    memory.on_chip_flash_memory_area_size,
                ),
                (
                    "off_chip_flash_memory_area_size",
                    memory.off_chip_flash_memory_area_size,
                ),
            )
        )
    elif is_corstone_backend(backend_name) and performance.npu_cycles is not None:
        metrics.append(
            schema.Metric(
                name="npu_total_cycles",
                value=performance.npu_cycles.npu_total_cycles,
                unit="cycles",
            )
        )

    complete_metrics = schema.ensure_standard_performance_metrics(metrics)
    return [
        replace(metric, qualifiers={**metric.qualifiers, **qualifiers})
        for metric in complete_metrics
    ]


def _optimization_settings(settings: Any) -> dict[str, Any]:
    """Return JSON-compatible optimization settings."""
    result: dict[str, Any] = {
        "optimization_type": settings.optimization_type,
        "optimization_target": settings.optimization_target,
    }
    layers = getattr(settings, "layers_to_optimize", None)
    if layers is not None:
        result["layers_to_optimize"] = list(layers)
    dataset = getattr(settings, "dataset", None)
    if dataset is not None:
        result["dataset"] = str(dataset)
    return result
