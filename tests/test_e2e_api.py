# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Python API end-to-end tests for Ethos-U."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mlia import (
    list_backend_options,
    list_backends,
    list_target_profiles,
    list_targets,
    supported_backends,
)
from mlia.testing import e2e as mlia_e2e


ETHOS_U55_128 = "ethos-u55-128"
ETHOS_U55_256 = "ethos-u55-256"
CORSTONE_300 = "corstone-300"
CORSTONE_310 = "corstone-310"
VELA = "vela"

MLIA_API_E2E_MODEL = "MLIA_API_E2E_MODEL"
REPRESENTATIVE_MODEL = (
    "e2e_config/tflite_models/ds_cnn_large_fully_quantized_int8.tflite"
)

pytestmark = pytest.mark.slow


@dataclass(frozen=True)
class ParityCase:
    """One CLI JSON versus Python API parity case.

    ``backends`` are the explicit user-facing CLI/API backend arguments for the
    case. ``required_backends`` are the environment prerequisites that must be
    declared and installed before the case runs. The default-backend case leaves
    ``backends`` empty while still requiring Vela to be available.
    """

    name: str
    advice_category: str
    target_profile: str
    backends: tuple[str, ...] = ()
    required_backends: tuple[str, ...] = ()


PARITY_CASES = (
    ParityCase(
        name="compatibility_default_backends",
        advice_category="compatibility",
        target_profile=ETHOS_U55_256,
        required_backends=(VELA,),
    ),
    ParityCase(
        name="compatibility_vela",
        advice_category="compatibility",
        target_profile=ETHOS_U55_256,
        backends=(VELA,),
        required_backends=(VELA,),
    ),
    ParityCase(
        name="performance_vela_corstone_300",
        advice_category="performance",
        target_profile=ETHOS_U55_256,
        backends=(VELA, CORSTONE_300),
        required_backends=(VELA, CORSTONE_300),
    ),
    ParityCase(
        name="performance_vela_corstone_310",
        advice_category="performance",
        target_profile=ETHOS_U55_128,
        backends=(VELA, CORSTONE_310),
        required_backends=(VELA, CORSTONE_310),
    ),
)


def _require_api_e2e_configuration() -> None:
    """Skip local runs unless the e2e artifact configuration is present."""
    if os.environ.get(MLIA_API_E2E_MODEL):
        return
    if mlia_e2e.prepared_artifact_path(REPRESENTATIVE_MODEL) is not None:
        return
    pytest.skip(
        f"Set {mlia_e2e.MLIA_E2E_ARTIFACTS} or {MLIA_API_E2E_MODEL} "
        "to run API e2e tests."
    )


def _representative_model() -> Path:
    """Resolve the representative TensorFlow Lite model used for API parity."""
    override = os.environ.get(MLIA_API_E2E_MODEL)
    if override:
        model = Path(override)
    else:
        model = mlia_e2e.prepared_artifact_path(REPRESENTATIVE_MODEL)
        if model is None:
            pytest.skip(
                f"Set {mlia_e2e.MLIA_E2E_ARTIFACTS} or {MLIA_API_E2E_MODEL} "
                "to run API e2e."
            )

    if not model.is_file():
        pytest.fail(f"Representative API e2e model does not exist: {model}")
    return model


def _cli_json_output(case: ParityCase, model: Path) -> dict[str, Any]:
    """Run the equivalent CLI command and return its standardized JSON output."""
    argv = [
        "mlia",
        "check",
        str(model),
        f"--{case.advice_category}",
        "--target-profile",
        case.target_profile,
        "--json",
    ]
    for backend in case.backends:
        argv.extend(["--backend", backend])

    result = subprocess.run(
        argv,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI e2e command failed: {' '.join(argv)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return json.loads(_extract_json(result.stdout))


def _extract_json(output: str) -> str:
    """Extract the JSON object from command output that may include log lines."""
    start = output.find("{")
    end = output.rfind("}")
    assert start >= 0 and end >= start, f"No JSON object found in output:\n{output}"
    return output[start : end + 1]


def _api_output(case: ParityCase, model: Path) -> dict[str, Any]:
    """Run the Python API in a subprocess and return its serialized JSON output."""
    payload = {
        "advice_category": case.advice_category,
        "target_profile": case.target_profile,
        "model": str(model),
        "backends": list(case.backends),
    }
    code = """
import json
import sys
from pathlib import Path

from mlia import ValidationMode, run_advisor

payload = json.loads(sys.argv[1])
kwargs = {}
if payload["backends"]:
    kwargs["backends"] = payload["backends"]
result = run_advisor(
    advice_category=payload["advice_category"],
    target_profile=payload["target_profile"],
    model=Path(payload["model"]),
    validation=ValidationMode.OFF,
    **kwargs,
)
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, (
        f"Python API e2e command failed for {case.name}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return json.loads(_extract_json(result.stdout))


def _normalize_output(output: dict[str, Any]) -> dict[str, Any]:
    """Remove expected volatile CLI/API differences before parity comparison."""
    normalized = deepcopy(output)
    normalized.pop("run_id", None)
    normalized.pop("timestamp", None)

    context = normalized.get("context")
    if isinstance(context, dict):
        context.pop("cli_arguments", None)

    for result in normalized.get("results", []):
        if isinstance(result, dict):
            _normalize_advices(result.get("advices", []))

    return normalized


def _normalize_advices(advices: Any) -> None:
    """Trim CLI-only advice suffixes that are not emitted by the Python API."""
    if not isinstance(advices, list):
        return

    cli_only_suffixes = (
        " Check the estimated performance by running the following command:",
        " Try running the following command to verify that:",
        " Note: you will need a Keras model for that.",
    )
    for advice in advices:
        if not isinstance(advice, dict):
            continue
        message = advice.get("message")
        if isinstance(message, str):
            for suffix in cli_only_suffixes:
                message = message.split(suffix, maxsplit=1)[0]
            advice["message"] = message


def test_python_api_e2e_queries() -> None:
    """Check Ethos-U plugin surfaces through the public query APIs."""
    _require_api_e2e_configuration()
    mlia_e2e.ensure_backends_available((VELA,))

    targets = list_targets()
    target_names = {item["target"] for item in targets}
    assert {"ethos-u55", "ethos-u65", "ethos-u85"}.issubset(target_names)

    ethos_u55 = next(item for item in targets if item["target"] == "ethos-u55")
    assert ETHOS_U55_128 in ethos_u55["profiles"]
    assert ETHOS_U55_256 in ethos_u55["profiles"]
    assert {VELA, CORSTONE_300, CORSTONE_310}.issubset(
        set(ethos_u55["supported_backends"])
    )

    profiles = list_target_profiles()
    assert "ethos-u55" in profiles
    assert {ETHOS_U55_128, ETHOS_U55_256}.issubset(
        {item["name"] for item in profiles["ethos-u55"]}
    )

    backends = {item["name"]: item for item in list_backends()}
    assert {VELA, CORSTONE_300, CORSTONE_310}.issubset(backends)
    assert backends[VELA]["installed"] is True

    backend_options = list_backend_options()
    assert isinstance(backend_options, list)

    # supported_backends reports target capability metadata, not installed state.
    assert VELA in supported_backends(ETHOS_U55_256)
    assert CORSTONE_300 in supported_backends(ETHOS_U55_256)
    assert CORSTONE_310 in supported_backends(ETHOS_U55_128)


@pytest.mark.parametrize("case", PARITY_CASES, ids=[case.name for case in PARITY_CASES])
def test_python_api_e2e_matches_cli_json(case: ParityCase) -> None:
    """Check normalized Python API output matches equivalent CLI JSON output."""
    model = _representative_model()
    mlia_e2e.ensure_backends_available(case.required_backends)

    cli_output = _normalize_output(_cli_json_output(case, model))
    api_output = _normalize_output(_api_output(case, model))

    assert api_output == cli_output
