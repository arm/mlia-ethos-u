# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ethos-U target-owned standardized output."""

from pathlib import Path
from typing import Any

import mlia.core.output_schema as schema
import pytest

from mlia.core.errors import ConfigurationError
from mlia.core.output_validation import validate_standardized_output
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.performance import (
    MemoryUsage,
    NPUCycles,
    OptimizationPerformanceMetrics,
    PerformanceMetrics,
)
from mlia.target.ethos_u.performance_warnings import NPU_ONLY_PERFORMANCE_WARNING
from mlia.target.ethos_u.standardized_output import (
    build_optimization_performance_output,
    model_metadata,
)


def _performance_metrics(
    target: EthosUConfiguration,
    *,
    total_cycles: int,
    memory_base: int,
) -> PerformanceMetrics:
    """Build metrics containing both Vela and Corstone measurements."""
    return PerformanceMetrics(
        target_config=target,
        npu_cycles=NPUCycles(1, 2, total_cycles, 4, 5, 6),
        memory_usage=MemoryUsage(
            memory_base,
            memory_base + 1,
            memory_base + 2,
            memory_base + 3,
        ),
        layerwise_perf_info=None,
    )


def _backend_configurations() -> dict[str, dict[str, Any]]:
    """Return distinct configurations for both optimization backends."""
    return {
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


def _backend_versions() -> dict[str, str]:
    """Return backend versions used by optimization output tests."""
    return {"vela": "5.0.0", "corstone-310": "unknown"}


def _optimization_comparison(
    target: EthosUConfiguration,
) -> OptimizationPerformanceMetrics:
    """Build a comparison with two independently qualified optimizations."""
    return OptimizationPerformanceMetrics(
        original_perf_metrics=_performance_metrics(
            target, total_cycles=100, memory_base=100
        ),
        optimizations_perf_metrics=[
            (
                [OptimizationSettings("pruning", 0.5, None)],
                _performance_metrics(target, total_cycles=80, memory_base=80),
            ),
            (
                [OptimizationSettings("clustering", 32, None)],
                _performance_metrics(target, total_cycles=70, memory_base=70),
            ),
        ],
    )


def _build_optimization_output(tmp_path: Path, backends: list[str]) -> dict[str, Any]:
    """Build validated optimization output for the requested backends."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"keras model")
    target = EthosUConfiguration.load_profile("ethos-u55-256")
    output = build_optimization_performance_output(
        model_path,
        target,
        _optimization_comparison(target),
        backends,
        _backend_configurations(),
        _backend_versions(),
        ["mlia", "optimize"],
    )
    validate_standardized_output(output)
    return output


def test_optimization_results_split_mixed_backend_modes(tmp_path: Path) -> None:
    """Vela predictions and Corstone simulations must be separate results."""
    output = _build_optimization_output(tmp_path, ["vela", "corstone-310"])

    assert [backend["id"] for backend in output["backends"]] == [
        "vela",
        "corstone-310",
    ]
    assert {
        backend["id"]: backend["configuration"] for backend in output["backends"]
    } == _backend_configurations()
    backends_by_id = {backend["id"]: backend for backend in output["backends"]}
    assert backends_by_id["vela"]["version"] == "5.0.0"
    results_by_producer = {result["producer"]: result for result in output["results"]}
    assert results_by_producer["vela"]["mode"] == "predicted"
    assert results_by_producer["corstone-310"]["mode"] == "simulated"

    vela_metric_names = {
        metric["name"] for metric in results_by_producer["vela"]["metrics"]
    }
    corstone_metric_names = {
        metric["name"] for metric in results_by_producer["corstone-310"]["metrics"]
    }
    assert {
        "sram_memory_area_size",
        "dram_memory_area_size",
        "on_chip_flash_memory_area_size",
        "off_chip_flash_memory_area_size",
    }.issubset(vela_metric_names)
    assert "npu_total_cycles" not in vela_metric_names
    assert "npu_total_cycles" in corstone_metric_names
    assert "sram_memory_area_size" not in corstone_metric_names


def test_optimization_single_backend_modes(tmp_path: Path) -> None:
    """Single-backend comparisons must retain their backend's measurement mode."""
    vela_output = _build_optimization_output(tmp_path / "vela", ["vela"])
    corstone_output = _build_optimization_output(
        tmp_path / "corstone", ["corstone-310"]
    )

    assert vela_output["results"][0]["mode"] == "predicted"
    assert corstone_output["results"][0]["mode"] == "simulated"


def test_optimization_output_rejects_explicit_empty_backend_list(
    tmp_path: Path,
) -> None:
    """An explicit empty backend selection must not silently enable Vela."""
    with pytest.raises(ConfigurationError, match="No performance backends"):
        _build_optimization_output(tmp_path, [])


def test_optimization_output_rejects_multiple_corstone_backends(
    tmp_path: Path,
) -> None:
    """Do not serialize ambiguous data from multiple Corstone backends."""
    with pytest.raises(
        ConfigurationError,
        match="at most one Corstone backend.*corstone-300, corstone-310",
    ):
        _build_optimization_output(tmp_path, ["corstone-300", "corstone-310"])


def test_optimization_metric_sets_are_complete_and_qualified(tmp_path: Path) -> None:
    """Every before/after comparison includes all standard performance metrics."""
    output = _build_optimization_output(tmp_path, ["vela", "corstone-310"])
    standard_metric_names = {
        definition.name for definition in schema.STANDARD_PERFORMANCE_METRICS
    }
    expected_optimizations = {
        0: [
            {
                "optimization_type": "pruning",
                "optimization_target": 0.5,
            }
        ],
        1: [
            {
                "optimization_type": "clustering",
                "optimization_target": 32,
            }
        ],
    }

    for result in output["results"]:
        metric_sets: dict[tuple[int, str], list[dict[str, object]]] = {}
        for metric in result["metrics"]:
            qualifiers = metric["qualifiers"]
            key = (qualifiers["optimization_index"], qualifiers["phase"])
            metric_sets.setdefault(key, []).append(metric)
            assert qualifiers["optimizations"] == expected_optimizations[key[0]]

        assert set(metric_sets) == {
            (0, "before"),
            (0, "after"),
            (1, "before"),
            (1, "after"),
        }
        for metrics in metric_sets.values():
            metrics_by_name = {metric["name"]: metric for metric in metrics}
            assert standard_metric_names.issubset(metrics_by_name)
            for metric_name in standard_metric_names:
                metric = metrics_by_name[metric_name]
                assert metric["availability"] == "unavailable"
                assert metric["reason"]


def test_optimization_results_include_npu_only_warning(tmp_path: Path) -> None:
    """Optimization performance must state that figures cover the NPU only."""
    output = _build_optimization_output(tmp_path, ["vela", "corstone-310"])

    assert all(
        result["warnings"] == [NPU_ONLY_PERFORMANCE_WARNING]
        for result in output["results"]
    )


def test_canonical_output_preserves_target_profile_identity(tmp_path: Path) -> None:
    """Canonical target metadata must preserve the selected profile's exact name."""
    model_path = tmp_path / "model.h5"
    model_path.write_bytes(b"keras model")
    target = EthosUConfiguration(
        profile_name="lab-ethos-u55-256",
        target="ethos-u55",
        mac=256,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
    )
    target.verify()

    output = build_optimization_performance_output(
        model_path,
        target,
        _optimization_comparison(target),
        ["vela"],
        _backend_configurations(),
        _backend_versions(),
        [],
    )

    assert target.profile_name == "lab-ethos-u55-256"
    assert output["target"]["profile_name"] == "lab-ethos-u55-256"


def test_directory_digest_uses_unambiguous_manifest_framing(tmp_path: Path) -> None:
    """Path/content boundary shifts must not collide in directory model hashes."""
    first_model = tmp_path / "first"
    second_model = tmp_path / "second"
    first_model.mkdir()
    second_model.mkdir()

    # Without framing, both manifests concatenate to b"a\0bcd\0e".
    (first_model / "a").write_bytes(b"bc")
    (first_model / "d").write_bytes(b"e")
    (second_model / "a").write_bytes(b"b")
    (second_model / "cd").write_bytes(b"e")

    assert model_metadata(first_model).hash != model_metadata(second_model).hash


def test_profile_name_is_reconstructed_only_when_not_supplied() -> None:
    """Direct configurations reconstruct the canonical MAC-qualified profile name."""
    target = EthosUConfiguration(
        target="ethos-u55",
        mac=256,
        system_config="Ethos_U55_High_End_Embedded",
        memory_mode="Shared_Sram",
    )

    assert target.profile_name == "ethos-u55-256"
