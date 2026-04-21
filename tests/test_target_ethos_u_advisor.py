# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Ethos-U MLIA module."""

from __future__ import annotations

import builtins
import types
from pathlib import Path
from typing import Any

import pytest
from mlia.core.common import AdviceCategory
from mlia.core.context import ExecutionContext
from mlia.core.data_collection import DataCollector

from mlia.target.ethos_u.advice_generation import (
    EthosUAdviceProducer,
    EthosUStaticAdviceProducer,
)
from mlia.target.ethos_u.advisor import (
    _OPTIMIZATION_COLLECTOR_NAME,
    EthosUInferenceAdvisor,
    _get_config_parameters,
    _get_optimization_collector_name,
    configure_and_get_ethosu_advisor,
)
from mlia.target.ethos_u.data_analysis import EthosUDataAnalyzer
from mlia.target.ethos_u.data_collection import (
    EthosUOperatorCompatibility,
    EthosUOptimizationPerformance,
    EthosUPerformance,
)
from mlia.target.ethos_u.events import EthosUAdvisorStartedEvent
from mlia.target.ethos_u.handlers import EthosUEventHandler
from mlia.target.ethos_u.pattern_analysis import ActivationFunctionPatternAnalyzer


def fake_add_common_optimization_params(params: dict, extra: dict) -> None:
    targets = extra.get("optimization_targets")
    if targets is None:
        return

    params.setdefault("common_optimizations", {})["optimizations"] = [targets]


def fake_optimization_performance_init(
    self,
    model: Path,
    target: Any,
    backends: list[str] | None = None,
) -> None:
    """Stub optimization collector init when legacy support is unavailable."""
    self.model = model
    self.target = target
    self.backends = backends


def test_advisor_metadata() -> None:
    """Test advisor metadata."""
    assert EthosUInferenceAdvisor.name() == "ethos_u_inference_advisor"


def test_get_collectors_uses_overridden_performance_collector(
    tmp_path: Path,
    test_tflite_model: Path,
) -> None:
    """Performance collection should honor a subclass collector override."""

    class CustomPerformanceCollector(EthosUPerformance):
        """Test collector override."""

    class CustomAdvisor(EthosUInferenceAdvisor):
        """Advisor subclass using a custom performance collector."""

        performance_collector_cls = CustomPerformanceCollector

    ctx = ExecutionContext(
        output_dir=tmp_path,
        advice_category={AdviceCategory.PERFORMANCE},
        config_parameters={
            "ethos_u_inference_advisor": {
                "model": str(test_tflite_model),
                "target_profile": "ethos-u55-256",
                "backends": ["vela"],
            }
        },
    )

    collectors = CustomAdvisor().get_collectors(ctx)

    assert [type(collector) for collector in collectors] == [CustomPerformanceCollector]


def test_get_collectors_uses_overridden_performance_collector_for_non_tflite(
    tmp_path: Path,
    test_keras_model: Path,
) -> None:
    """Non-TFLite performance collection should honor a subclass override."""

    class CustomPerformanceCollector(EthosUPerformance):
        """Test collector override."""

    class CustomAdvisor(EthosUInferenceAdvisor):
        """Advisor subclass using a custom performance collector."""

        performance_collector_cls = CustomPerformanceCollector

    ctx = ExecutionContext(
        output_dir=tmp_path,
        advice_category={AdviceCategory.PERFORMANCE},
        config_parameters={
            "ethos_u_inference_advisor": {
                "model": str(test_keras_model),
                "target_profile": "ethos-u55-256",
                "backends": ["vela"],
            }
        },
    )

    collectors = CustomAdvisor().get_collectors(ctx)

    assert [type(collector) for collector in collectors] == [CustomPerformanceCollector]


@pytest.mark.parametrize(
    "categories, model_fixture, optimization_targets, expected_collectors",
    [
        pytest.param(
            {AdviceCategory.COMPATIBILITY},
            "test_tflite_model",
            None,
            [EthosUOperatorCompatibility],
            id="tflite-compatibility-only",
        ),
        pytest.param(
            {AdviceCategory.PERFORMANCE},
            "test_tflite_model",
            None,
            [EthosUPerformance],
            id="tflite-performance-only",
        ),
        pytest.param(
            {AdviceCategory.COMPATIBILITY, AdviceCategory.PERFORMANCE},
            "test_tflite_model",
            None,
            [EthosUOperatorCompatibility, EthosUPerformance],
            id="tflite-compatibility-and-performance",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION},
            "test_tflite_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOptimizationPerformance],
            id="tflite-optimization-only",
        ),
        pytest.param(
            {AdviceCategory.COMPATIBILITY, AdviceCategory.OPTIMIZATION},
            "test_tflite_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOperatorCompatibility, EthosUOptimizationPerformance],
            id="tflite-compatibility-and-optimization",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION, AdviceCategory.PERFORMANCE},
            "test_tflite_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOptimizationPerformance, EthosUPerformance],
            id="tflite-optimization-and-performance",
        ),
        pytest.param(
            {
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            },
            "test_tflite_model",
            [{"optimization_type": "rewrite"}],
            [
                EthosUOperatorCompatibility,
                EthosUOptimizationPerformance,
                EthosUPerformance,
            ],
            id="tflite-compatibility-optimization-and-performance",
        ),
    ],
)
def test_get_collectors_returns_expected_collectors_for_tflite(
    monkeypatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    categories: set[AdviceCategory],
    model_fixture: str,
    optimization_targets: list[dict[str, Any]] | None,
    expected_collectors: list[type[DataCollector]],
):
    """Test that advisor configures collectors for supported categories when tflite."""

    from mlia.target.ethos_u import advisor as adv_mod

    monkeypatch.setattr(adv_mod, "LEGACY_OPTIMIZATION_AVAILABLE", True)
    monkeypatch.setattr(
        adv_mod, "add_common_optimization_params", fake_add_common_optimization_params
    )
    monkeypatch.setattr(
        EthosUOptimizationPerformance,
        "__init__",
        fake_optimization_performance_init,
    )

    model: Path = request.getfixturevalue(model_fixture)

    ctx = ExecutionContext(output_dir=tmp_path, advice_category=categories)

    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
        optimization_targets=optimization_targets,
    )

    collectors = advisor.get_collectors(ctx)
    assert [type(collector) for collector in collectors] == expected_collectors


@pytest.mark.parametrize(
    "categories, model_fixture, optimization_targets, expected_collectors",
    [
        pytest.param(
            {AdviceCategory.COMPATIBILITY},
            "test_keras_model",
            None,
            [EthosUOperatorCompatibility],
            id="non-tflite-compatibility-only",
        ),
        pytest.param(
            {AdviceCategory.PERFORMANCE},
            "test_keras_model",
            None,
            [EthosUPerformance],
            id="non-tflite-performance-only",
        ),
        pytest.param(
            {AdviceCategory.COMPATIBILITY, AdviceCategory.PERFORMANCE},
            "test_keras_model",
            None,
            [EthosUOperatorCompatibility, EthosUPerformance],
            id="non-tflite-compatibility-and-performance",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION},
            "test_keras_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOptimizationPerformance],
            id="non-tflite-optimization-only",
        ),
        pytest.param(
            {AdviceCategory.COMPATIBILITY, AdviceCategory.OPTIMIZATION},
            "test_keras_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOperatorCompatibility, EthosUOptimizationPerformance],
            id="non-tflite-compatibility-and-optimization",
        ),
        pytest.param(
            {AdviceCategory.OPTIMIZATION, AdviceCategory.PERFORMANCE},
            "test_keras_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOptimizationPerformance],
            id="non-tflite-optimization-and-performance",
        ),
        pytest.param(
            {
                AdviceCategory.COMPATIBILITY,
                AdviceCategory.OPTIMIZATION,
                AdviceCategory.PERFORMANCE,
            },
            "test_keras_model",
            [{"optimization_type": "rewrite"}],
            [EthosUOperatorCompatibility, EthosUOptimizationPerformance],
            id="non-tflite-compatibility-optimization-and-performance",
        ),
    ],
)
def test_get_collectors_returns_expected_collectors_for_non_tflite(
    monkeypatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    categories: set[AdviceCategory],
    model_fixture: str,
    optimization_targets: list[dict[str, Any]] | None,
    expected_collectors: list[type[DataCollector]],
):
    """Test that advisor configures collectors for supported categories when not tflite."""

    from mlia.target.ethos_u import advisor as adv_mod

    monkeypatch.setattr(adv_mod, "LEGACY_OPTIMIZATION_AVAILABLE", True)
    monkeypatch.setattr(
        adv_mod, "add_common_optimization_params", fake_add_common_optimization_params
    )
    monkeypatch.setattr(
        EthosUOptimizationPerformance,
        "__init__",
        fake_optimization_performance_init,
    )

    model: Path = request.getfixturevalue(model_fixture)

    ctx = ExecutionContext(output_dir=tmp_path, advice_category=categories)

    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
        optimization_targets=optimization_targets,
    )

    collectors = advisor.get_collectors(ctx)
    assert [type(collector) for collector in collectors] == expected_collectors


@pytest.mark.parametrize(
    "model_fixture,categories,optimization_targets",
    [
        pytest.param(
            "test_tflite_model",
            {AdviceCategory.OPTIMIZATION},
            [{"optimization_type": "rewrite"}],
            id="tflite-optimization-enabled",
        ),
        pytest.param(
            "test_tflite_model",
            {AdviceCategory.COMPATIBILITY, AdviceCategory.OPTIMIZATION},
            [{"optimization_type": "rewrite"}],
            id="tflite-compatibility-and-optimization-enabled",
        ),
        pytest.param(
            "test_keras_model",
            {AdviceCategory.OPTIMIZATION},
            [{"optimization_type": "rewrite"}],
            id="non-tflite-optimization-enabled",
        ),
        pytest.param(
            "test_keras_model",
            {AdviceCategory.COMPATIBILITY, AdviceCategory.OPTIMIZATION},
            [{"optimization_type": "rewrite"}],
            id="non-tflite-compatibility-and-optimization-enabled",
        ),
    ],
)
def test_get_collectors_raises_runtime_error_when_optimization_requires_legacy_plugin(
    monkeypatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    model_fixture: str,
    categories: set[AdviceCategory],
    optimization_targets: list[dict[str, Any]],
):
    """Test that advisor raises an error when optimization requires the legacy plugin."""
    model: Path = request.getfixturevalue(model_fixture)

    from mlia.target.ethos_u import advisor as adv_mod

    monkeypatch.setattr(adv_mod, "LEGACY_OPTIMIZATION_AVAILABLE", False)
    monkeypatch.setattr(
        adv_mod,
        "add_common_optimization_params",
        fake_add_common_optimization_params,
    )

    ctx = ExecutionContext(output_dir=tmp_path, advice_category=categories)

    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
        optimization_targets=optimization_targets,
    )

    with pytest.raises(
        RuntimeError,
        match="Optimization requires the legacy plugin \\(mlia-legacy\\)\\.",
    ):
        advisor.get_collectors(ctx)


@pytest.mark.parametrize(
    "model_fixture,categories,optimization_targets",
    [
        pytest.param(
            "test_tflite_model",
            {AdviceCategory.OPTIMIZATION},
            [{"optimization_type": "Not-rewrite"}],
            id="tflite-optimization-enabled",
        ),
    ],
)
def test_get_collectors_raises_runtime_error_when_tflite_optimization_type_is_not_rewrite(
    monkeypatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    model_fixture: str,
    categories: set[AdviceCategory],
    optimization_targets: list[dict[str, Any]],
):
    """Test that advisor raises an error when optimization requires the legacy plugin."""
    model: Path = request.getfixturevalue(model_fixture)

    from mlia.target.ethos_u import advisor as adv_mod

    monkeypatch.setattr(adv_mod, "LEGACY_OPTIMIZATION_AVAILABLE", True)
    monkeypatch.setattr(
        adv_mod,
        "add_common_optimization_params",
        fake_add_common_optimization_params,
    )

    ctx = ExecutionContext(output_dir=tmp_path, advice_category=categories)

    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
        optimization_targets=optimization_targets,
    )

    with pytest.raises(
        RuntimeError,
        match="Only 'rewrite' is supported for TensorFlow Lite files.",
    ):
        advisor.get_collectors(ctx)


def test_get_analyzers_returns_ethosu_data_analyzer(
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    ctx = ExecutionContext(output_dir=tmp_path)
    model: Path = request.getfixturevalue("test_tflite_model")
    advisor = configure_and_get_ethosu_advisor(ctx, "ethos-u55-256", str(model))

    analyzers = advisor.get_analyzers(ctx)

    assert len(analyzers) == 1
    assert isinstance(analyzers[0], EthosUDataAnalyzer)


def test_get_pattern_analyzers_returns_activation_function_pattern_analyzer(
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    ctx = ExecutionContext(output_dir=tmp_path)
    model: Path = request.getfixturevalue("test_tflite_model")
    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
    )

    pattern_analyzers = advisor.get_pattern_analyzers(ctx)

    assert len(pattern_analyzers) == 2
    assert isinstance(
        pattern_analyzers[0],
        ActivationFunctionPatternAnalyzer,
    )


def test_get_producers_returns_expected_ethosu_advice_producers(
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    ctx = ExecutionContext(output_dir=tmp_path)
    model: Path = request.getfixturevalue("test_tflite_model")
    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
    )

    producers = advisor.get_producers(ctx)

    assert [type(producer) for producer in producers] == [
        EthosUAdviceProducer,
        EthosUStaticAdviceProducer,
    ]


def test_get_events_returns_ethosu_advisor_started_event_with_model_and_target_config(
    tmp_path: Path,
    request: pytest.FixtureRequest,
):
    ctx = ExecutionContext(output_dir=tmp_path)
    model: Path = request.getfixturevalue("test_tflite_model")
    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(model),
    )

    model = advisor.get_model(ctx)
    target_config = advisor._get_target_config(ctx)

    events = advisor.get_events(ctx)

    assert len(events) == 1
    assert isinstance(events[0], EthosUAdvisorStartedEvent)
    assert events[0].model == model
    assert events[0].target_config == target_config


def test_configure_and_get_ethosu_advisor_does_not_update_context_unnecessarily(
    tmp_path: Path,
):
    """Test that advisor configuration does not update context when parameters are not None."""
    context = ExecutionContext(output_dir=tmp_path)
    context.event_handlers = [EthosUEventHandler(output_dir=context.output_dir)]
    context.config_parameters = {"preconfigured": True}

    original_handlers = context.event_handlers
    original_config = context.config_parameters

    target_profile = "tgt_profile"
    model = "model"

    advisor = configure_and_get_ethosu_advisor(
        context,
        target_profile,
        model,
    )

    assert isinstance(advisor, EthosUInferenceAdvisor)
    assert context.event_handlers is original_handlers
    assert context.config_parameters is original_config


def test_configure_and_get_ethosu_advisor_correctly_updates_context(
    tmp_path: Path,
):
    """Test that advisor configuration updates context when parameters None."""
    context = ExecutionContext(output_dir=tmp_path)

    target_profile = "tgt_profile"
    model = "model"

    advisor = configure_and_get_ethosu_advisor(
        context,
        target_profile,
        model,
    )

    assert isinstance(advisor, EthosUInferenceAdvisor)

    assert len(context.event_handlers) == 1
    assert isinstance(context.event_handlers[0], EthosUEventHandler)

    expected_params = _get_config_parameters(model, target_profile)
    assert context.config_parameters == expected_params


def test_configure_and_get_ethosu_advisor_passes_backends(tmp_path: Path) -> None:
    """Test that backends are forwarded into config parameters."""
    context = ExecutionContext(output_dir=tmp_path)
    target_profile = "tgt_profile"
    model = "model"
    backends = ["vela", "corstone-310"]

    configure_and_get_ethosu_advisor(
        context,
        target_profile,
        model,
        backends=backends,
    )

    assert context.config_parameters is not None
    advisor_params = context.config_parameters["ethos_u_inference_advisor"]
    assert advisor_params["backends"] == backends


def test_configure_and_get_ethosu_advisor_invalid_backends(tmp_path: Path) -> None:
    """Test that invalid backends format raises a ValueError."""
    context = ExecutionContext(output_dir=tmp_path)

    with pytest.raises(ValueError, match="Backends value has wrong format."):
        configure_and_get_ethosu_advisor(
            context,
            "tgt_profile",
            "model",
            backends="vela",
        )


def test_get_events_returns_advisor_started_event(
    tmp_path: Path,
    test_tflite_model: Path,
) -> None:
    """Test that get_events returns EthosUAdvisorStartedEvent with expected data."""
    ctx = ExecutionContext(output_dir=tmp_path)

    advisor = configure_and_get_ethosu_advisor(
        ctx,
        "ethos-u55-256",
        str(test_tflite_model),
    )

    events = advisor.get_events(ctx)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, EthosUAdvisorStartedEvent)
    assert Path(event.model) == test_tflite_model
    assert event.target_config.compiler_options is not None
    assert event.target_config.compiler_options.output_dir == tmp_path / "mlia-output"


def fail_import(name, globals=None, locals=None, fromlist=(), level=0):
    raise ModuleNotFoundError(name)


def succeed_import(name, globals=None, locals=None, fromlist=(), level=0):
    fake_mod = types.ModuleType("mlia.target.common.optimization")

    class OptimizingDataCollector:
        @staticmethod
        def name():
            return "legacy-ok"

    fake_mod.OptimizingDataCollector = OptimizingDataCollector
    return fake_mod


@pytest.mark.parametrize(
    "import_outcome,expected",
    [
        pytest.param(fail_import, _OPTIMIZATION_COLLECTOR_NAME, id="fallback"),
        pytest.param(succeed_import, "legacy-ok", id="success"),
    ],
)
def test_get_optimization_collector_name(monkeypatch, import_outcome, expected):
    monkeypatch.setattr(builtins, "__import__", import_outcome)
    assert _get_optimization_collector_name() == expected
