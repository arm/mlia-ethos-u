# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U event handler API integration behavior."""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.vela.compat import Operators
from mlia.backend.vela.compat import VelaCompatibilityResult
from mlia.core.advice_generation import Advice
from mlia.core.context import ExecutionContext
from mlia.core.events import (
    AdviceStageFinishedEvent,
    CollectedDataEvent,
    ExecutionStartedEvent,
)
from mlia.core.output_schema import AdviceCategory, AdviceSeverity
from mlia.core.reporting import JSONReporter
from mlia.target.ethos_u.config import EthosUConfiguration
from mlia.target.ethos_u.events import EthosUAdvisorStartedEvent
from mlia.target.ethos_u.handlers import EthosUEventHandler
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.performance import OptimizationPerformanceMetrics
from mlia.target.ethos_u.performance import PerformanceMetrics
from mlia.target.ethos_u.performance import VelaPerformanceResult


def _workflow_events_handler_supports_collect_only() -> bool:
    parameters = inspect.signature(EthosUEventHandler.__mro__[1].__init__).parameters
    return "collect_only" in parameters


class _FakeLegacyInfo:
    """Minimal legacy info object for formatter resolution."""


class _FakeStandardizedResult:
    """Minimal result object with standardized output."""

    legacy_info = _FakeLegacyInfo()
    standardized_output = {"schema_version": "1.0.0", "results": []}


def _make_compatibility_result() -> VelaCompatibilityResult:
    """Create a real Vela compatibility wrapper with minimal schema output."""
    return VelaCompatibilityResult(
        legacy_info=Operators([]),
        standardized_output={"schema_version": "1.0.0", "results": [{}]},
    )


def _make_performance_result() -> VelaPerformanceResult:
    """Create a real Vela performance wrapper with minimal schema output."""
    return VelaPerformanceResult(
        legacy_info=PerformanceMetrics(
            target_config=EthosUConfiguration(
                target="ethos-u55",
                mac=128,
                system_config="Ethos_U55_High_End_Embedded",
                memory_mode="Shared_Sram",
            ),
            npu_cycles=None,
            memory_usage=None,
            layerwise_perf_info=None,
        ),
        standardized_output={"schema_version": "1.0.0", "results": [{}]},
    )


def _make_optimization_performance_metrics() -> OptimizationPerformanceMetrics:
    """Create optimization performance metrics without standardized output."""
    base_metrics = PerformanceMetrics(
        target_config=EthosUConfiguration(
            target="ethos-u55",
            mac=128,
            system_config="Ethos_U55_High_End_Embedded",
            memory_mode="Shared_Sram",
        ),
        npu_cycles=None,
        memory_usage=None,
        layerwise_perf_info=None,
    )
    optimization = OptimizationSettings(
        optimization_type="pruning",
        optimization_target=0.5,
        layers_to_optimize=None,
    )
    return OptimizationPerformanceMetrics(
        original_perf_metrics=base_metrics,
        optimizations_perf_metrics=[([optimization], base_metrics)],
    )


def test_ethos_u_event_handler_collect_only_uses_json_reporter(tmp_path: Path) -> None:
    """Collect-only Ethos-U handler should build in-memory JSON output."""
    handler = EthosUEventHandler(collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())

    assert isinstance(handler.reporter, JSONReporter)
    assert handler.collect_only is True


def test_ethos_u_event_handler_collect_only_skips_target_config_submission(
    tmp_path: Path,
) -> None:
    """Collect-only mode should not submit CLI-only target details."""
    handler = EthosUEventHandler(collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_ethos_u_advisor_started(
        EthosUAdvisorStartedEvent(
            model=tmp_path / "model.tflite",
            target_config=EthosUConfiguration(
                target="ethos-u55",
                mac=128,
                system_config="Ethos_U55_High_End_Embedded",
                memory_mode="Shared_Sram",
            ),
        )
    )

    assert handler.reporter.missing_standardized_output is False


def test_ethos_u_event_handler_collect_only_early_returns_after_parent_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Collect-only mode should return before any sidecar JSON write attempts."""
    handler = EthosUEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.reporter.submit(_FakeStandardizedResult())
    write_json = MagicMock()
    monkeypatch.setattr(handler, "_write_json_with_advice", write_json)

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    write_json.assert_not_called()


def test_ethos_u_event_handler_collect_only_skips_output_files(tmp_path: Path) -> None:
    """Collect-only Ethos-U handler should not emit output files."""
    handler = EthosUEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.reporter.submit(_FakeStandardizedResult())
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=AdviceCategory.COMPATIBILITY,
        )
    ]
    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    assert not list(tmp_path.glob("*.json"))


def test_ethos_u_event_handler_collect_only_compatibility_result_no_file(
    tmp_path: Path,
) -> None:
    """Collect-only mode should keep real compatibility results in memory."""
    handler = EthosUEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(CollectedDataEvent(_make_compatibility_result()))
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=AdviceCategory.COMPATIBILITY,
        )
    ]

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    assert handler.vela_compatibility_result is not None
    assert not list(tmp_path.glob("*.json"))


def test_ethos_u_event_handler_collect_only_performance_result_no_file(
    tmp_path: Path,
) -> None:
    """Collect-only mode should keep real performance results in memory."""
    handler = EthosUEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())
    handler.on_collected_data(CollectedDataEvent(_make_performance_result()))
    handler.advice = [
        Advice(
            id="0",
            message="msg",
            severity=AdviceSeverity.INFO,
            category=AdviceCategory.PERFORMANCE,
        )
    ]

    handler.on_advice_stage_finished(AdviceStageFinishedEvent())

    assert handler.output is not None
    assert handler.vela_performance_result is not None
    assert not list(tmp_path.glob("*.json"))


def test_ethos_u_event_handler_collect_only_skips_optimization_metrics_submission(
    tmp_path: Path,
) -> None:
    """Collect-only mode should not submit legacy optimization metrics tables."""
    handler = EthosUEventHandler(tmp_path, collect_only=True)
    handler.set_context(ExecutionContext(output_format="json", output_dir=tmp_path))
    handler.on_execution_started(ExecutionStartedEvent())

    handler.on_collected_data(
        CollectedDataEvent(_make_optimization_performance_metrics())
    )

    assert handler.reporter.missing_standardized_output is False
