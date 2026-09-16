# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Module for backend integration."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import mlia
import mlia.core.output_schema as schema
from mlia.backend.errors import BackendExecutionFailed
from mlia.backend.repo import get_backend_repository
from mlia.target.ethos_u.performance_warnings import NPU_ONLY_PERFORMANCE_WARNING
from mlia.target.ethos_u.utils.model_format import is_pte_file
from mlia.utils.filesystem import get_mlia_resource_dirs, get_mlia_resources, sha256
from mlia.utils.proc import Command, OutputLogger, process_command_output

logger = logging.getLogger(__name__)


_FVP_VERSION_BY_BACKEND = {
    "corstone-300": "300",
    "corstone-310": "310",
    "corstone-320": "320",
}

_TARGET_BY_NAME = {
    "ethos-u55": "U55",
    "ethos-u65": "U65",
    "ethos-u85": "U85",
}

_APPLICATION_VERSION = "26.03.0"
_APPLICATION_BINARY = "mlek_inference_runner.axf"

_SUPPORTED_EXECUTORCH_APPLICATIONS = {
    ("corstone-300", "ethos-u55"),
    ("corstone-320", "ethos-u85"),
}


def _validate_non_negative_number(value: Any, description: str) -> None:
    """Validate a parsed numeric counter value."""
    if value is not None and value < 0:
        raise ValueError(f"Parsed metrics contain negative {description}: {value}")


@dataclass
class CorstoneModelPerformanceMetrics:
    """Model performance metrics."""

    npu_active_cycles: int
    npu_idle_cycles: int
    npu_total_cycles: int
    npu_axi0_rd_data_beat_received: int
    npu_axi0_wr_data_beat_written: int
    npu_axi1_rd_data_beat_received: int
    npu_axi1_wr_data_beat_written: int | None = None

    @classmethod
    def from_fvp_metrics(
        cls,
        target: str,
        fvp_metrics: dict[str, Any],
    ) -> CorstoneModelPerformanceMetrics:
        """Create CorstoneModelPerformanceMetrics from FVP metrics."""
        # Mapping from FVP metric names to class members.
        # Must be in the same order as in the class definition
        target_metric_maps = {
            "default": [
                "NPU ACTIVE",
                "NPU IDLE",
                "NPU TOTAL",
                "NPU AXI0_RD_DATA_BEAT_RECEIVED",
                "NPU AXI0_WR_DATA_BEAT_WRITTEN",
                "NPU AXI1_RD_DATA_BEAT_RECEIVED",
            ],
            "corstone-320": [
                "NPU ACTIVE",
                "NPU IDLE",
                "NPU TOTAL",
                "NPU ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED",
                "NPU ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN",
                "NPU ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED",
                "NPU ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN",
            ],
        }
        metric_names = target_metric_maps.get(target, target_metric_maps["default"])
        class_fields = list(cls.__dataclass_fields__.keys())
        class_kwargs = {}
        for idx, metric_name in enumerate(metric_names):
            if metric_name in fvp_metrics and idx < len(class_fields):
                class_kwargs[class_fields[idx]] = fvp_metrics[metric_name]
            else:
                raise KeyError(f"Metric {metric_name} not found in parsed data.")
        return cls(**class_kwargs)


def _validate_non_negative_model_stats(
    model_stats: CorstoneModelPerformanceMetrics,
) -> None:
    """Validate Corstone model-level performance counters."""
    for field_info in fields(model_stats):
        _validate_non_negative_number(
            getattr(model_stats, field_info.name),
            field_info.name,
        )


def _target_utilization(model_stats: CorstoneModelPerformanceMetrics) -> float:
    """Return target utilization from Corstone cycle counters."""
    if model_stats.npu_total_cycles == 0:
        return 0.0
    return model_stats.npu_active_cycles / model_stats.npu_total_cycles * 100


def _build_model_metrics(
    model_stats: CorstoneModelPerformanceMetrics,
) -> list[schema.Metric]:
    """Build Corstone model-level performance metrics."""
    metrics = [
        schema.Metric(
            name="npu_active_cycles",
            value=model_stats.npu_active_cycles,
            unit="cycles",
        ),
        schema.Metric(
            name="npu_idle_cycles",
            value=model_stats.npu_idle_cycles,
            unit="cycles",
        ),
        schema.Metric(
            name="npu_total_cycles",
            value=model_stats.npu_total_cycles,
            unit="cycles",
        ),
        schema.Metric(
            name="npu_axi0_rd_data_beat_received",
            value=model_stats.npu_axi0_rd_data_beat_received,
            unit="beats",
        ),
        schema.Metric(
            name="npu_axi0_wr_data_beat_written",
            value=model_stats.npu_axi0_wr_data_beat_written,
            unit="beats",
        ),
        schema.Metric(
            name="npu_axi1_rd_data_beat_received",
            value=model_stats.npu_axi1_rd_data_beat_received,
            unit="beats",
        ),
    ]

    if model_stats.npu_axi1_wr_data_beat_written is not None:
        metrics.append(
            schema.Metric(
                name="npu_axi1_wr_data_beat_written",
                value=model_stats.npu_axi1_wr_data_beat_written,
                unit="beats",
            )
        )

    return metrics


def _build_standard_corstone_metrics(
    model_stats: CorstoneModelPerformanceMetrics,
) -> list[schema.Metric]:
    """Build standard metrics derived from Corstone model counters."""
    return [
        schema.Metric(
            name=schema.METRIC_NAME_TARGET_UTILIZATION,
            value=_target_utilization(model_stats),
            unit=schema.UNIT_PERCENT,
        )
    ]


@dataclass
class CorstonePerformanceMetrics:
    """Model-wide performance metrics parsed from Corstone FVP output."""

    npu_model_stats: CorstoneModelPerformanceMetrics

    @classmethod
    def from_fvp_out(
        cls, target: str, metrics: dict[str, Any]
    ) -> CorstonePerformanceMetrics:
        """Create model-wide performance metrics from FVP output."""
        model_stats = CorstoneModelPerformanceMetrics.from_fvp_metrics(target, metrics)
        _validate_non_negative_model_stats(model_stats)
        return cls(model_stats)

    def to_standardized_output(
        self,
        model_path: Path,
        backend_name: str,
        target_config: dict[str, Any],
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
        backend_config: dict[str, Any] | None = None,
    ) -> Any:  # Returns StandardizedOutput but avoid circular import
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            backend_name: Name of the Corstone backend (e.g., "corstone-300")
            target_config: Target configuration parameters
            run_id: Optional run ID (will be generated if not provided)
            timestamp: Optional ISO 8601 timestamp (will be generated if not provided)
            cli_arguments: Optional CLI arguments used for the run
            backend_config: Optional backend configuration parameters

        Returns:
            StandardizedOutput object with performance results
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend
        backend_label = backend_name.replace("-", " ").title()
        backend = schema.Backend(
            id=backend_name,
            name=backend_label,
            version="unknown",  # Corstone version comes from FVP executable
            configuration=backend_config or {},
        )

        # Extract target info from config
        target_type = target_config.get("target_type", backend_name)
        mac_config = target_config.get("mac", "unknown")
        profile_name = target_config.get("profile_name")
        if profile_name is None:
            profile_target = target_config.get("target", target_type)
            profile_name = (
                f"{profile_target}-{mac_config}"
                if mac_config != "unknown"
                else profile_target
            )

        # Create target components - only NPU
        components = []

        # Add NPU component if target info available
        npu_target = target_config.get("npu_target") or target_config.get("target")
        if npu_target:
            npu_family = npu_target.split("-")[0] if "-" in npu_target else "ethos-u"
            npu_model = npu_target.split("-")[1] if "-" in npu_target else None

            components.append(
                schema.Component(
                    type=schema.ComponentType.NPU,
                    family=npu_family,
                    model=npu_model,
                    variant=str(mac_config) if mac_config != "unknown" else None,
                )
            )

        target = schema.Target(
            profile_name=profile_name,
            target_type=target_type,
            components=components,
            configuration=target_config,
            description=f"Corstone {backend_name} FVP simulation",
        )

        # Create model info
        model_hash = sha256(model_path)
        model_size = model_path.stat().st_size if model_path.exists() else None
        suffix = model_path.suffix.lower()
        if suffix == ".tflite":
            model_format = "tflite"
        elif suffix in [".vela", ".tflite.vela"]:
            model_format = "vela"
        else:
            model_format = suffix.lstrip(".") or "unknown"

        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
            size_bytes=model_size,
        )

        # Create context
        context = schema.Context(
            cli_arguments=cli_arguments or [],
            runtime_configuration=None,
            git=None,
            notes=None,
        )

        metrics = _build_model_metrics(self.npu_model_stats)
        metrics.extend(_build_standard_corstone_metrics(self.npu_model_stats))
        metrics = schema.ensure_standard_performance_metrics(metrics)

        # Corstone owns FVP model-wide measurements. Per-layer estimates belong
        # to the Vela result and are intentionally not included here.
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer=backend.id,
            warnings=[NPU_ONLY_PERFORMANCE_WARNING],
            errors=[],
            metrics=metrics,
            mode=schema.ModeType.SIMULATED,  # Corstone is simulation
        )

        return schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
            extensions={},
        ).to_dict()


class GenericInferenceOutputParser:
    """Generic inference runner output parser."""

    pattern = re.compile(r"<metrics>(.*)</metrics>")
    # Match runner error records, not warnings or incidental mentions of errors.
    error_pattern = re.compile(
        r"^(?:ERROR - |E \[ExecuTorch:|"
        r"TFLM - (?:Failed to |Didn't find op for ))"
    )

    def __init__(self) -> None:
        """Init parser."""
        self.base64_data: list[str] = []
        self.errors: list[str] = []
        self.output_tail: deque[str] = deque(maxlen=8)

    def __call__(self, line: str) -> None:
        """Collect metrics and bounded diagnostics from the app output."""
        if res_b64 := self.pattern.search(line):
            self.base64_data.append(res_b64.group(1))
            return
        diagnostic = line.strip()[:1024]
        if diagnostic:
            self.output_tail.append(diagnostic)
        if self.error_pattern.match(diagnostic):
            # Retain the first causes while bounding the user-facing diagnostic.
            if len(self.errors) < 8 and diagnostic not in self.errors:
                self.errors.append(diagnostic)

    def get_metrics(
        self, _output_dir: Path, target: str = "default"
    ) -> CorstonePerformanceMetrics:
        """Parse model-wide FVP metrics from the collected output."""
        try:
            return CorstonePerformanceMetrics.from_fvp_out(target, self._parse_data())
        except Exception as err:
            raise ValueError(f"Unable to parse output and get metrics: {err}") from err

    def _parse_data(self) -> dict[str, int]:
        """Parse the data."""
        parsed_metrics: dict[str, int] = {}

        for base64_item in self.base64_data:
            res_json = base64.b64decode(base64_item, validate=True)

            for profiling_group in json.loads(res_json):
                for metric in profiling_group["samples"]:
                    metric_name = metric["name"]
                    metric_value = int(metric["value"][0])

                    if metric_name in parsed_metrics:
                        raise KeyError(f"Duplicate key {metric_name}")

                    parsed_metrics[metric_name] = metric_value

        return parsed_metrics


@dataclass
class FVPMetadata:
    """Metadata for FVP."""

    executable: str
    generic_inf_app: Path


def get_generic_inference_app_path(fvp: str, target: str, is_pte: bool) -> Path:
    """Return path to the generic inference runner binary."""
    try:
        fvp_version = _FVP_VERSION_BY_BACKEND[fvp]
        target_name = _TARGET_BY_NAME[target]
    except KeyError as err:
        raise ValueError(
            f"No inference runner is available for backend '{fvp}' and target "
            f"'{target}'."
        ) from err

    runtime_tag = "tflm"
    if is_pte:
        if (fvp, target) not in _SUPPORTED_EXECUTORCH_APPLICATIONS:
            raise ValueError(
                f"No inference runner is available for backend '{fvp}' and target "
                f"'{target}'."
            )
        runtime_tag = "executorch"

    app_dir = (
        f"inference_runner-sse-{fvp_version}-"
        f"{_APPLICATION_VERSION}-{runtime_tag}-ethos-{target_name}-Default-noTA"
    )
    relative_app_path = Path(
        "backends",
        "applications",
        app_dir,
        _APPLICATION_BINARY,
    )

    for resources_dir in get_mlia_resource_dirs():
        candidate = resources_dir / relative_app_path
        if candidate.is_file():
            return candidate

    return get_mlia_resources() / relative_app_path


def get_executable_name(fvp: str, profile: str, target: str) -> str:
    """Return name of the executable for selected FVP and profile."""
    executable_name_mapping = {
        ("corstone-300", "AVH", "ethos-u55"): "VHT_Corstone_SSE-300_Ethos-U55",
        ("corstone-300", "AVH", "ethos-u65"): "VHT_Corstone_SSE-300_Ethos-U65",
        ("corstone-300", "default", "ethos-u55"): "FVP_Corstone_SSE-300_Ethos-U55",
        ("corstone-300", "default", "ethos-u65"): "FVP_Corstone_SSE-300_Ethos-U65",
        ("corstone-310", "AVH", "ethos-u55"): "VHT_Corstone_SSE-310",
        ("corstone-310", "AVH", "ethos-u65"): "VHT_Corstone_SSE-310_Ethos-U65",
        ("corstone-310", "default", "ethos-u55"): "FVP_Corstone_SSE-310",
        ("corstone-310", "default", "ethos-u65"): "FVP_Corstone_SSE-310_Ethos-U65",
        ("corstone-320", "AVH", "ethos-u85"): "VHT_Corstone_SSE-320",
        ("corstone-320", "default", "ethos-u85"): "FVP_Corstone_SSE-320",
    }

    return executable_name_mapping[(fvp, profile, target)]


def get_fvp_metadata(fvp: str, profile: str, target: str, is_pte: bool) -> FVPMetadata:
    """Return metadata for selected Corstone backend."""
    executable_name = get_executable_name(fvp, profile, target)

    app = get_generic_inference_app_path(fvp, target, is_pte)

    return FVPMetadata(executable_name, app)


@dataclass
class CorstoneRunConfig:
    """Configuration for running Corstone FVP generic inference."""

    output_dir: Path
    backend_path: Path
    fvp: str
    target: str
    mac: int
    model: Path
    is_pte: bool
    profile: str = "default"


def _corstone_320_runtime_env(backend_path: Path) -> dict[str, str]:
    """Return environment for the bundled Corstone-320 Python runtime."""
    python_home = backend_path / "python"
    python_lib = python_home / "lib"
    env = os.environ.copy()
    env["PYTHONHOME"] = python_home.as_posix()

    existing_library_path = os.environ.get("LD_LIBRARY_PATH")
    library_path_entries = [python_lib.as_posix()]
    if existing_library_path:
        library_path_entries.append(existing_library_path)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path_entries)

    return env


def build_corstone_command(cfg: CorstoneRunConfig) -> Command:
    """Build command to run Corstone FVP."""
    fvp_metadata = get_fvp_metadata(cfg.fvp, cfg.profile, cfg.target, cfg.is_pte)
    env = None

    if cfg.fvp == "corstone-320":
        if cfg.profile == "default":
            env = _corstone_320_runtime_env(cfg.backend_path)
        cmd = [
            cfg.backend_path.joinpath(fvp_metadata.executable).as_posix(),
            "-a",
            fvp_metadata.generic_inf_app.as_posix(),
            "--data",
            f"{cfg.model}@0x90000000",
            "-C",
            f"mps4_board.subsystem.ethosu.num_macs={cfg.mac}",
            "-C",
            "mps4_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps4_board.uart0.out_file='-'",
            "-C",
            "mps4_board.uart0.shutdown_on_eot=1",
            "-C",
            "mps4_board.visualisation.disable-visualisation=1",
            "-C",
            "vis_hdlcd.disable_visualisation=1",
            "--stat",
        ]
    else:
        cmd = [
            cfg.backend_path.joinpath(fvp_metadata.executable).as_posix(),
            "-a",
            fvp_metadata.generic_inf_app.as_posix(),
            "--data",
            f"{cfg.model}@0x90000000",
            "-C",
            f"ethosu.num_macs={cfg.mac}",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps3_board.uart0.out_file='-'",
            "-C",
            "mps3_board.uart0.shutdown_on_eot=1",
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
            "--stat",
        ]
    return Command(cmd, env=env)


def get_metrics(cfg: CorstoneRunConfig) -> CorstonePerformanceMetrics:
    """Run generic inference and return perf metrics."""
    try:
        command = build_corstone_command(cfg)
    except Exception as err:  # noqa: BLE001 - we want to wrap any construction errors
        raise BackendExecutionFailed(
            f"Unable to construct a command line for {cfg.fvp}"
        ) from err

    output_parser = GenericInferenceOutputParser()
    output_logger = OutputLogger(logger)
    failure_context = f"Backend execution failed. {cfg.fvp}, model '{cfg.model}': "

    try:
        process_command_output(
            command,
            [output_parser, output_logger],
        )
    except subprocess.CalledProcessError as err:
        details = "; ".join(output_parser.errors or output_parser.output_tail)
        raise BackendExecutionFailed(
            f"{failure_context}exit code {err.returncode}. {details}".rstrip()
        ) from err

    # The simulator can exit successfully even when the inference runner fails.
    if output_parser.errors:
        raise BackendExecutionFailed(failure_context + "; ".join(output_parser.errors))

    return output_parser.get_metrics(cfg.output_dir, cfg.fvp)


def estimate_performance(
    target: str, mac: int, model: Path, backend: str, output_dir: Path
) -> CorstonePerformanceMetrics:
    """Get performance estimations."""
    backend_repo = get_backend_repository()
    backend_path, settings = backend_repo.get_backend_settings(backend)

    if not settings or "profile" not in settings:
        raise BackendExecutionFailed(f"Unable to configure backend {backend}.")

    cfg = CorstoneRunConfig(
        output_dir=output_dir,
        backend_path=backend_path,
        fvp=backend,
        target=target,
        mac=mac,
        model=model,
        is_pte=is_pte_file(model),
        profile=settings["profile"],
    )
    return get_metrics(cfg)
