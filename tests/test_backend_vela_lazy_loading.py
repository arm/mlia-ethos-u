# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for retryable lazy Vela dependency loading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlia.backend.errors import BackendUnavailableError
from mlia.backend.vela import compat, compiler, performance


def test_compiler_vela_deps_are_retried_after_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Vela import must not be cached before auto-install can run."""
    deps = compiler.VelaCompilerDeps(
        ModelReaderOptions=object,
        read_model=object(),
        Graph=object,
        NetworkType=object,
        CustomType=SimpleNamespace(ExistingNpuOp=object()),
        main=object(),
    )
    calls = iter(
        [BackendUnavailableError("Backend vela is not available", "vela"), deps]
    )

    def load_once_then_succeed() -> compiler.VelaCompilerDeps:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(compiler, "_VELA_DEPS_CACHE", None)
    monkeypatch.setattr(compiler, "_load_vela_deps", load_once_then_succeed)

    with pytest.raises(BackendUnavailableError):
        compiler._get_vela_deps()

    assert compiler._get_vela_deps() is deps


def test_compat_vela_deps_are_retried_after_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility lazy imports should retry after auto-install."""
    deps = compat.VelaDeps(
        ethosu_vela_version="5.0.0",
        Op=SimpleNamespace(
            Placeholder=object(), SubgraphInput=object(), Const=object()
        ),
        optype_to_builtintype=object(),
        TFLiteSemantic=object,
        TFLiteSupportedOperators=object,
        generate_supported_ops=object(),
        VelaCompiler=object,
        layer_metrics=(),
        parse_layerwise_perf_csv=object(),
    )
    calls = iter(
        [BackendUnavailableError("Backend vela is not available", "vela"), deps]
    )

    def load_once_then_succeed() -> compat.VelaDeps:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(compat, "_VELA_DEPS_CACHE", None)
    monkeypatch.setattr(compat, "_load_vela_deps", load_once_then_succeed)

    with pytest.raises(BackendUnavailableError):
        compat._get_vela_deps()

    assert compat._get_vela_deps() is deps


def test_performance_vela_version_is_retried_after_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version lookup should retry after an initial missing Vela import."""
    calls = iter(
        [
            BackendUnavailableError("Backend vela is not available", "vela"),
            "5.0.0",
        ]
    )

    def load_once_then_succeed() -> str:
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(performance, "_VELA_VERSION_CACHE", None)
    monkeypatch.setattr(performance, "_load_vela_version", load_once_then_succeed)

    with pytest.raises(BackendUnavailableError):
        performance._get_vela_version()

    assert performance._get_vela_version() == "5.0.0"
