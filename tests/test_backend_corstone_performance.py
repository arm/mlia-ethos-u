# SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module backend/manager."""

from __future__ import annotations

import base64
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

import mlia.core.output_schema as schema
from mlia.backend.corstone.performance import (
    CorstoneModelPerformanceMetrics,
    CorstonePerformanceMetrics,
    CorstoneRunConfig,
    GenericInferenceOutputParser,
    build_corstone_command,
    estimate_performance,
    get_generic_inference_app_path,
    get_metrics,
)
from mlia.backend.errors import BackendExecutionFailed
from mlia.core.output_validation import validate_standardized_output
from mlia.utils.proc import Command


def encode_b64(data: str) -> str:
    """Encode data in base64 format."""
    return base64.b64encode(data.encode()).decode()


def valid_fvp_output() -> list[str]:
    """Return valid FVP output that could be successfully parsed."""
    json_data = """[
    {
        "profiling_group": "Inference",
        "count": 1,
        "samples": [
            {"name": "NPU IDLE", "value": [2]},
            {"name": "NPU AXI0_RD_DATA_BEAT_RECEIVED", "value": [4]},
            {"name": "NPU AXI0_WR_DATA_BEAT_WRITTEN", "value": [5]},
            {"name": "NPU AXI1_RD_DATA_BEAT_RECEIVED", "value": [6]},
            {"name": "NPU ACTIVE", "value": [1]},
            {"name": "NPU TOTAL", "value": [3]}
        ]
    }
]"""

    return [
        "some output",
        f"<metrics>{encode_b64(json_data)}</metrics>",
        "some_output",
    ]


def duplicate_key_output() -> list[str]:
    """Return FVP output with duplicate keys."""
    json_data = """[
    {
        "profiling_group": "Inference",
        "count": 1,
        "samples": [
            {"name": "NPU IDLE", "value": [2]},
            {"name": "NPU IDLE", "value": [2]},
            {"name": "NPU AXI0_RD_DATA_BEAT_RECEIVED", "value": [4]},
            {"name": "NPU AXI0_WR_DATA_BEAT_WRITTEN", "value": [5]},
            {"name": "NPU AXI1_RD_DATA_BEAT_RECEIVED", "value": [6]}
        ]
    }
]"""

    return [
        f"<metrics>{encode_b64(json_data)}</metrics>",
    ]


def negative_counter_output() -> list[str]:
    """Return FVP output with a negative counter."""
    json_data = """[
    {
        "profiling_group": "Inference",
        "count": 1,
        "samples": [
            {"name": "NPU IDLE", "value": [2]},
            {"name": "NPU AXI0_RD_DATA_BEAT_RECEIVED", "value": [4]},
            {"name": "NPU AXI0_WR_DATA_BEAT_WRITTEN", "value": [5]},
            {"name": "NPU AXI1_RD_DATA_BEAT_RECEIVED", "value": [6]},
            {"name": "NPU ACTIVE", "value": [-1]},
            {"name": "NPU TOTAL", "value": [3]}
        ]
    }
]"""

    return [
        f"<metrics>{encode_b64(json_data)}</metrics>",
    ]


@pytest.fixture
def mock_mlia_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock MLIA resource lookup for command construction tests."""
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resource_dirs", lambda: []
    )
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resources", lambda: Path("apps")
    )


def test_generic_inference_output_parser_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test successful generic inference output parsing."""
    output_parser = GenericInferenceOutputParser()
    monkeypatch.setattr(
        "mlia.backend.corstone.performance._parse_per_layer_csv",
        MagicMock(return_value=[{"operator": "op", "cycles": 1000}]),
    )
    per_layer_file = tmp_path / "model_per-layer.csv"
    per_layer_file.touch()
    for line in valid_fvp_output():
        output_parser(line)

    assert output_parser.get_metrics(tmp_path) == CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(1, 2, 3, 4, 5, 6),
        [{"operator": "op", "cycles": 1000}],
    )


@pytest.mark.parametrize(
    "wrong_fvp_output",
    [
        [],
        ["NPU IDLE: 123"],
        ["<metrics>123</metrics>"],
    ],
)
def test_generic_inference_output_parser_failure(
    tmp_path: Path, wrong_fvp_output: list[str]
) -> None:
    """Test unsuccessful generic inference output parsing."""
    output_parser = GenericInferenceOutputParser()

    for line in wrong_fvp_output:
        output_parser(line)

    with pytest.raises(ValueError, match="Unable to parse output and get metrics"):
        output_parser.get_metrics(tmp_path)


def test_generic_inference_output_parser_rejects_negative_model_counter(
    tmp_path: Path,
) -> None:
    """Negative model counters should fail before collection returns metrics."""
    output_parser = GenericInferenceOutputParser()
    for line in negative_counter_output():
        output_parser(line)

    with pytest.raises(ValueError, match="negative npu_active_cycles"):
        output_parser.get_metrics(tmp_path)


def test_generic_inference_output_parser_rejects_negative_per_layer_counter(
    tmp_path: Path,
) -> None:
    """Negative per-layer counters should fail before collection returns metrics."""
    per_layer_file = tmp_path / "model_per-layer.csv"
    per_layer_file.write_text(
        "NNG Operator,Name,Staging Usage,Op Cycles\nop,op_name,128,-1\n",
        encoding="utf-8",
    )
    output_parser = GenericInferenceOutputParser()
    for line in valid_fvp_output():
        output_parser(line)

    with pytest.raises(ValueError, match="negative operation cycles"):
        output_parser.get_metrics(tmp_path)


def test_generic_inference_output_parser_rejects_negative_per_layer_memory(
    tmp_path: Path,
) -> None:
    """Negative per-layer memory usage should fail before collection returns metrics."""
    per_layer_file = tmp_path / "model_per-layer.csv"
    per_layer_file.write_text(
        "NNG Operator,Name,Staging Usage,Op Cycles\nop,op_name,-128,1\n",
        encoding="utf-8",
    )
    output_parser = GenericInferenceOutputParser()
    for line in valid_fvp_output():
        output_parser(line)

    with pytest.raises(ValueError, match="negative memory usage"):
        output_parser.get_metrics(tmp_path)


def test_generic_inference_output_parser_rejects_non_numeric_per_layer_metric(
    tmp_path: Path,
) -> None:
    """Non-numeric per-layer metrics should fail before collection returns metrics."""
    per_layer_file = tmp_path / "model_per-layer.csv"
    per_layer_file.write_text(
        "NNG Operator,Name,NPU\nop,op_name,unknown\n",
        encoding="utf-8",
    )
    output_parser = GenericInferenceOutputParser()
    for line in valid_fvp_output():
        output_parser(line)

    with pytest.raises(ValueError, match="non-numeric metric.*NPU"):
        output_parser.get_metrics(tmp_path)


@dataclass(frozen=True)
class BuildCmdCase:
    """Build Command Case function."""

    backend_path: Path
    fvp: str
    target: str
    mac: int
    model: Path
    is_pte: bool
    profile: str
    expected_command: Command


@pytest.mark.parametrize(
    "case",
    [
        BuildCmdCase(
            backend_path=Path("backend_path"),
            fvp="corstone-300",
            target="ethos-u55",
            mac=256,
            model=Path("model.tflite"),
            is_pte=False,
            profile="default",
            expected_command=Command(
                [
                    "backend_path/FVP_Corstone_SSE-300_Ethos-U55",
                    "-a",
                    "apps/backends/applications/"
                    "inference_runner-sse-300-26.03.0-tflm-ethos-U55-Default-noTA/"
                    "mlek_inference_runner.axf",
                    "--data",
                    "model.tflite@0x90000000",
                    "-C",
                    "ethosu.num_macs=256",
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
            ),
        ),
        BuildCmdCase(
            backend_path=Path("backend_path"),
            fvp="corstone-300",
            target="ethos-u55",
            mac=256,
            model=Path("model.pte"),
            is_pte=True,
            profile="default",
            expected_command=Command(
                [
                    "backend_path/FVP_Corstone_SSE-300_Ethos-U55",
                    "-a",
                    "apps/backends/applications/"
                    "inference_runner-sse-300-26.03.0-executorch-ethos-U55-Default-noTA/"
                    "mlek_inference_runner.axf",
                    "--data",
                    "model.pte@0x90000000",
                    "-C",
                    "ethosu.num_macs=256",
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
            ),
        ),
    ],
)
@pytest.mark.usefixtures("mock_mlia_resources")
def test_build_corstone_command(
    tmp_path: Path,
    case: BuildCmdCase,
) -> None:
    """Test command construction with static process environment."""
    command = build_corstone_command(
        CorstoneRunConfig(
            tmp_path,
            case.backend_path,
            case.fvp,
            case.target,
            case.mac,
            case.model,
            case.is_pte,
            case.profile,
        )
    )
    assert command == case.expected_command


@pytest.mark.usefixtures("mock_mlia_resources")
def test_build_corstone_320_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test command construction for backend `corstone-320`."""
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    backend_path = Path("backend_path")
    expected_env = os.environ.copy()
    expected_env["PYTHONHOME"] = (backend_path / "python").as_posix()
    expected_env["LD_LIBRARY_PATH"] = (backend_path / "python" / "lib").as_posix()

    command = build_corstone_command(
        CorstoneRunConfig(
            Path("output_path"),
            backend_path,
            "corstone-320",
            "ethos-u85",
            1024,
            Path("model.tflite"),
            False,
            "default",
        )
    )

    assert command == Command(
        [
            "backend_path/FVP_Corstone_SSE-320",
            "-a",
            "apps/backends/applications/"
            "inference_runner-sse-320-26.03.0-tflm-ethos-U85-Default-noTA/"
            "mlek_inference_runner.axf",
            "--data",
            "model.tflite@0x90000000",
            "-C",
            "mps4_board.subsystem.ethosu.num_macs=1024",
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
        ],
        env=expected_env,
    )


@pytest.mark.usefixtures("mock_mlia_resources")
def test_corstone_320_command_uses_bundled_python_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test backend `corstone-320` command uses the installed Python runtime."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/lib")

    command = build_corstone_command(
        CorstoneRunConfig(
            tmp_path,
            tmp_path,
            "corstone-320",
            "ethos-u85",
            1024,
            Path("model.tflite"),
            False,
            "default",
        )
    )

    assert command.env is not None
    assert command.env["PYTHONHOME"] == (tmp_path / "python").as_posix()
    assert command.env["LD_LIBRARY_PATH"] == (
        f"{(tmp_path / 'python' / 'lib').as_posix()}:/host/lib"
    )


@pytest.mark.usefixtures("mock_mlia_resources")
def test_build_corstone_320_avh_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test command construction for backend `corstone-320` with profile `AVH`."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/lib")

    command = build_corstone_command(
        CorstoneRunConfig(
            Path("output_path"),
            Path("backend_path"),
            "corstone-320",
            "ethos-u85",
            1024,
            Path("model.tflite"),
            False,
            "AVH",
        )
    )

    assert command == Command(
        [
            "backend_path/VHT_Corstone_SSE-320",
            "-a",
            "apps/backends/applications/"
            "inference_runner-sse-320-26.03.0-tflm-ethos-U85-Default-noTA/"
            "mlek_inference_runner.axf",
            "--data",
            "model.tflite@0x90000000",
            "-C",
            "mps4_board.subsystem.ethosu.num_macs=1024",
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
    )


def test_get_generic_inference_app_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test generic inference app lookup across configured MLIA resource dirs."""
    first_resources_dir = tmp_path / "resources-first"
    second_resources_dir = tmp_path / "resources-second"
    app_path = (
        second_resources_dir
        / "backends"
        / "applications"
        / "inference_runner-sse-300-26.03.0-tflm-ethos-U55-Default-noTA"
        / "mlek_inference_runner.axf"
    )
    app_path.parent.mkdir(parents=True)
    app_path.write_text("fake axf", encoding="utf-8")

    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resource_dirs",
        lambda: [first_resources_dir, second_resources_dir],
    )
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_mlia_resources",
        lambda: tmp_path / "fallback",
    )

    assert (
        get_generic_inference_app_path("corstone-300", "ethos-u55", False) == app_path
    )


def test_get_metrics_wrong_fvp(tmp_path: Path) -> None:
    """Test that command construction should fail for wrong FVP."""
    with pytest.raises(
        BackendExecutionFailed, match=r"Unable to construct a command line for some_fvp"
    ):
        get_metrics(
            CorstoneRunConfig(
                tmp_path,
                Path("backend_path"),
                "some_fvp",
                "ethos-u55",
                256,
                Path("model.tflite"),
                False,
                "default",
            )
        )


def test_get_metrics_pte_parse_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test .pte parser failures include ExecuTorch compatibility guidance."""
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.build_corstone_command",
        MagicMock(return_value=Command(["fvp"])),
    )
    monkeypatch.setattr(
        "mlia.backend.corstone.performance.process_command_output",
        MagicMock(),
    )

    with pytest.raises(
        BackendExecutionFailed,
        match="Ensure .pte file is compatible with ExecuTorch Corstone FVP",
    ):
        get_metrics(
            CorstoneRunConfig(
                tmp_path,
                Path("backend_path"),
                "corstone-300",
                "ethos-u55",
                256,
                Path("model.pte"),
                True,
                "default",
            )
        )


@pytest.mark.parametrize(
    "target, per_layer_csv, model_metrics, expected_model_stats, expected_per_layer",
    [
        (
            "default",
            """TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,100,50,200,10,200,50,10,5,0,1000,20,40,loc0
CONV_2D,Conv2DBias,120,60,250,15,250,60,15,8,0,1200,25,45,loc1""",  # noqa: E501
            {
                "NPU IDLE": 100,
                "NPU AXI0_RD_DATA_BEAT_RECEIVED": 200,
                "NPU AXI0_WR_DATA_BEAT_WRITTEN": 150,
                "NPU AXI1_RD_DATA_BEAT_RECEIVED": 180,
                "NPU ACTIVE": 1000,
                "NPU TOTAL": 1100,
            },
            CorstoneModelPerformanceMetrics(
                npu_active_cycles=1000,
                npu_idle_cycles=100,
                npu_total_cycles=1100,
                npu_axi0_rd_data_beat_received=200,
                npu_axi0_wr_data_beat_written=150,
                npu_axi1_rd_data_beat_received=180,
                npu_axi1_wr_data_beat_written=None,
            ),
            [
                {
                    "TFLite_operator": "CONV_2D",
                    "NNG Operator": "Conv2DBias",
                    "SRAM Usage": "100",
                    "Peak%": "50",
                    "Op Cycles": "200",
                    "Network%": "20",
                    "NPU": "200",
                    "SRAM AC": "50",
                    "DRAM AC": "10",
                    "OnFlash AC": "5",
                    "OffFlash AC": "0",
                    "MAC Count": "1000",
                    "Util%": "40",
                    "Name": "loc0",
                },
                {
                    "TFLite_operator": "CONV_2D",
                    "NNG Operator": "Conv2DBias",
                    "SRAM Usage": "120",
                    "Peak%": "60",
                    "Op Cycles": "250",
                    "Network%": "25",
                    "NPU": "250",
                    "SRAM AC": "60",
                    "DRAM AC": "15",
                    "OnFlash AC": "8",
                    "OffFlash AC": "0",
                    "MAC Count": "1200",
                    "Util%": "45",
                    "Name": "loc1",
                },
            ],
        ),
        (
            "corstone-320",
            """Original Operator,NNG Operator,Target,Staging Usage,Peak% (Staging),Op Cycles,Network% (cycles),NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network% (MAC),Util% (MAC),Name
Conv2D,Conv2D,NPU,150,40,300,30,300,70,20,0,0,1500,18,35,loc0
Conv2D,Relu,NPU,180,50,400,35,400,80,25,0,0,2000,22,42,loc1""",  # noqa: E501
            {
                "NPU ACTIVE": 2000,
                "NPU ETHOSU_PMU_SRAM_RD_DATA_BEAT_RECEIVED": 250,
                "NPU ETHOSU_PMU_SRAM_WR_DATA_BEAT_WRITTEN": 120,
                "NPU ETHOSU_PMU_EXT_RD_DATA_BEAT_RECEIVED": 300,
                "NPU ETHOSU_PMU_EXT_WR_DATA_BEAT_WRITTEN": 80,
                "NPU IDLE": 50,
                "NPU TOTAL": 2050,
            },
            CorstoneModelPerformanceMetrics(
                npu_active_cycles=2000,
                npu_idle_cycles=50,
                npu_total_cycles=2050,
                npu_axi0_rd_data_beat_received=250,
                npu_axi0_wr_data_beat_written=120,
                npu_axi1_rd_data_beat_received=300,
                npu_axi1_wr_data_beat_written=80,
            ),
            [
                {
                    "Original Operator": "Conv2D",
                    "NNG Operator": "Conv2D",
                    "Target": "NPU",
                    "Staging Usage": "150",
                    "Peak% (Staging)": "40",
                    "Op Cycles": "300",
                    "Network% (cycles)": "30",
                    "NPU": "300",
                    "SRAM AC": "70",
                    "DRAM AC": "20",
                    "OnFlash AC": "0",
                    "OffFlash AC": "0",
                    "MAC Count": "1500",
                    "Network% (MAC)": "18",
                    "Util% (MAC)": "35",
                    "Name": "loc0",
                },
                {
                    "Original Operator": "Conv2D",
                    "NNG Operator": "Relu",
                    "Target": "NPU",
                    "Staging Usage": "180",
                    "Peak% (Staging)": "50",
                    "Op Cycles": "400",
                    "Network% (cycles)": "35",
                    "NPU": "400",
                    "SRAM AC": "80",
                    "DRAM AC": "25",
                    "OnFlash AC": "0",
                    "OffFlash AC": "0",
                    "MAC Count": "2000",
                    "Network% (MAC)": "22",
                    "Util% (MAC)": "42",
                    "Name": "loc1",
                },
            ],
        ),
    ],
)
def test_build_metrics_from_fvp_output(
    target: str,
    per_layer_csv: str,
    model_metrics: dict[str, int],
    expected_model_stats: CorstoneModelPerformanceMetrics,
    expected_per_layer: list[dict[str, str]],
    tmp_path: Path,
) -> None:
    """Test from_fvp_out method."""
    per_layer_file = tmp_path / "per-layer.csv"
    with open(per_layer_file, "w", encoding="utf-8") as file:
        file.write(per_layer_csv)

    perf_metrics = CorstonePerformanceMetrics.from_fvp_out(
        target, model_metrics, per_layer_file
    )
    model_stats = perf_metrics.npu_model_stats
    per_layer_stats = perf_metrics.npu_per_layer_stats

    # Verify model_stats match expected values
    assert model_stats == expected_model_stats

    # Verify layer_stats are correctly parsed from CSV
    assert len(per_layer_stats) == 2
    assert per_layer_stats == expected_per_layer


def test_corstone_model_performance_metrics_missing_metric() -> None:
    """Test if KeyError is raised if a metric is missing."""
    fvp_metrics = {"NPU ACTIVE": 1000}

    with pytest.raises(KeyError, match=r"^'Metric .+ not found in parsed data.'$"):
        CorstoneModelPerformanceMetrics.from_fvp_metrics("default", fvp_metrics)


def test_estimate_performance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test function estimate_performance."""
    mock_repository = MagicMock()
    mock_repository.get_backend_settings.return_value = (
        Path("backend_path"),
        {"profile": "default"},
    )

    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_backend_repository",
        lambda: mock_repository,
    )

    monkeypatch.setattr(
        "mlia.backend.corstone.performance._parse_per_layer_csv",
        MagicMock(return_value=[{"operator": "op", "cycles": 1000}]),
    )
    per_layer_file = tmp_path / "model_per-layer.csv"
    per_layer_file.touch()

    def command_output_mock(_command: Command) -> Generator[str, None, None]:
        """Mock FVP output."""
        yield from valid_fvp_output()

    monkeypatch.setattr("mlia.utils.proc.command_output", command_output_mock)

    result = estimate_performance(
        "ethos-u55", 256, Path("model.tflite"), "corstone-300", tmp_path
    )
    assert result == CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(1, 2, 3, 4, 5, 6),
        [{"operator": "op", "cycles": 1000}],
    )

    mock_repository.get_backend_settings.assert_called_once()

    # Check if BackendExecutionFailed is raised if the corstone command fails
    mock_check_call = MagicMock(
        side_effect=subprocess.CalledProcessError(returncode=1, cmd="fvp")
    )

    monkeypatch.setattr("mlia.utils.proc.command_output", mock_check_call)

    with pytest.raises(BackendExecutionFailed, match="Backend execution failed."):
        _ = estimate_performance(
            "ethos-u55", 256, Path("model.tflite"), "corstone-300", tmp_path
        )

    # Check if BackendExecutionFailed is raised if get_backend_settings
    # returns invalid results
    mock_backend_repo = MagicMock()
    mock_backend_repo.get_backend_settings.return_value = (None, None)

    monkeypatch.setattr(
        "mlia.backend.corstone.performance.get_backend_repository",
        MagicMock(return_value=mock_backend_repo),
    )

    with pytest.raises(BackendExecutionFailed, match="Unable to configure backend"):
        _ = estimate_performance(
            "ethos-u55", 256, Path("model.tflite"), "corstone-300", tmp_path
        )


@pytest.mark.parametrize(
    "model_file, expected_model_format",
    [
        ("model.tflite", "tflite"),
        ("model.tflite.vela", "vela"),
        ("model", "unknown"),
        ("model.pt", "pt"),
    ],
)
def test_performance_metrics_to_standardized_output(
    model_file: Path, expected_model_format: str, tmp_path: Path
) -> None:
    """Test conversion of PerformanceMetrics to standardized output."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=500,
            npu_total_cycles=1500,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=75,
        ),
        [
            {
                "NNG Operator": "op",
                "Name": "op_name",
                "NPU": 1000,
                "Staging Usage": "150",
                "Op Cycles": "300",
            },
            {
                "NNG Operator": "op2",
                "Name": "op2_name",
                "NPU": 500,
                "Staging Usage": "180",
                "Op Cycles": "100",
            },
        ],
    )

    # Create a model file for hash computation
    model_file = tmp_path / model_file
    model_file.touch()
    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )
    validate_standardized_output(output)

    assert output["model"]["format"] == expected_model_format

    # Structure checks
    for key in ("schema_version", "backends", "target", "model", "context", "results"):
        assert key in output
    assert output["schema_version"] == schema.SCHEMA_VERSION
    # Backend checks

    # Target/component checks
    components = output["target"]["components"]

    assert any(
        c["type"] == "npu" and "ethos" in c.get("family", "").lower()
        for c in components
    )
    # Check target description mentions corstone
    assert "corstone" in output["target"]["description"].lower()
    # Results/metrics checks
    result = output["results"][0]
    assert result["warnings"] == [
        "The performance figures above refer to NPU only",
    ]
    metrics = result["metrics"]
    metrics_dict = {m["name"]: m for m in metrics}
    assert metrics_dict["npu_active_cycles"]["value"] == 1000
    assert metrics_dict["npu_idle_cycles"]["value"] == 500
    assert metrics_dict["npu_total_cycles"]["value"] == 1500
    assert metrics_dict["npu_axi0_rd_data_beat_received"]["value"] == 200
    assert metrics_dict["npu_axi0_wr_data_beat_written"]["value"] == 100
    assert metrics_dict["npu_axi1_rd_data_beat_received"]["value"] == 150
    assert metrics_dict["npu_axi1_wr_data_beat_written"]["value"] == 75
    assert metrics_dict[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": pytest.approx(1000 / 1500 * 100),
        "unit": schema.UNIT_PERCENT,
    }
    assert metrics_dict[schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY] == {
        "name": schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        "value": 180.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics_dict[schema.METRIC_NAME_AVERAGE_MEMORY] == {
        "name": schema.METRIC_NAME_AVERAGE_MEMORY,
        "value": pytest.approx(((150 * 300) + (180 * 100)) / (300 + 100)),
        "unit": schema.UNIT_BYTES,
    }
    assert metrics_dict[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY] == {
        "name": schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        "unit": schema.UNIT_BYTES,
        "availability": "unavailable",
        "reason": "Model weight memory data is not available.",
    }
    for metric_name, unit in (
        (schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE, schema.UNIT_PERCENT),
        (schema.METRIC_NAME_INFERENCES_PER_SECOND, schema.UNIT_INFERENCES_PER_SECOND),
        (schema.METRIC_NAME_CPU_UTILIZATION, schema.UNIT_PERCENT),
    ):
        metric = metrics_dict[metric_name]
        assert metric["unit"] == unit
        assert metric["availability"] == "unavailable"
        assert "value" not in metric
        assert metric["reason"]

    assert result["entities"][0] == {
        "id": "source_operator/op_name",
        "kind": "source_operator",
        "name": "op",
        "placement": "NPU",
    }
    breakdown = result["breakdowns"][0]
    assert breakdown["entity_id"] == "source_operator/op_name"
    assert breakdown["metrics"] == [
        {"name": "npu", "value": 1000.0, "unit": "cycles"},
        {"name": "staging_usage", "value": 150.0, "unit": "bytes"},
        {"name": "op_cycles", "value": 300.0, "unit": "cycles"},
    ]


@pytest.mark.parametrize(
    "backend_name, target_config, model_stats, expected_metrics",
    [
        (
            "corstone-300",
            {"mac": 256, "target": "ethos-u55"},
            CorstoneModelPerformanceMetrics(
                npu_active_cycles=1000,
                npu_idle_cycles=500,
                npu_total_cycles=1500,
                npu_axi0_rd_data_beat_received=200,
                npu_axi0_wr_data_beat_written=100,
                npu_axi1_rd_data_beat_received=150,
                npu_axi1_wr_data_beat_written=None,
            ),
            {
                "npu_active_cycles": {
                    "name": "npu_active_cycles",
                    "value": 1000,
                    "unit": "cycles",
                },
                "npu_idle_cycles": {
                    "name": "npu_idle_cycles",
                    "value": 500,
                    "unit": "cycles",
                },
                "npu_total_cycles": {
                    "name": "npu_total_cycles",
                    "value": 1500,
                    "unit": "cycles",
                },
                "npu_axi0_rd_data_beat_received": {
                    "name": "npu_axi0_rd_data_beat_received",
                    "value": 200,
                    "unit": "beats",
                },
                "npu_axi0_wr_data_beat_written": {
                    "name": "npu_axi0_wr_data_beat_written",
                    "value": 100,
                    "unit": "beats",
                },
                "npu_axi1_rd_data_beat_received": {
                    "name": "npu_axi1_rd_data_beat_received",
                    "value": 150,
                    "unit": "beats",
                },
            },
        ),
        (
            "corstone-320",
            {"mac": 1024, "target": "ethos-u85"},
            CorstoneModelPerformanceMetrics(
                npu_active_cycles=2000,
                npu_idle_cycles=50,
                npu_total_cycles=2050,
                npu_axi0_rd_data_beat_received=250,
                npu_axi0_wr_data_beat_written=120,
                npu_axi1_rd_data_beat_received=300,
                npu_axi1_wr_data_beat_written=80,
            ),
            {
                "npu_active_cycles": {
                    "name": "npu_active_cycles",
                    "value": 2000,
                    "unit": "cycles",
                },
                "npu_idle_cycles": {
                    "name": "npu_idle_cycles",
                    "value": 50,
                    "unit": "cycles",
                },
                "npu_total_cycles": {
                    "name": "npu_total_cycles",
                    "value": 2050,
                    "unit": "cycles",
                },
                "npu_axi0_rd_data_beat_received": {
                    "name": "npu_axi0_rd_data_beat_received",
                    "value": 250,
                    "unit": "beats",
                },
                "npu_axi0_wr_data_beat_written": {
                    "name": "npu_axi0_wr_data_beat_written",
                    "value": 120,
                    "unit": "beats",
                },
                "npu_axi1_rd_data_beat_received": {
                    "name": "npu_axi1_rd_data_beat_received",
                    "value": 300,
                    "unit": "beats",
                },
                "npu_axi1_wr_data_beat_written": {
                    "name": "npu_axi1_wr_data_beat_written",
                    "value": 80,
                    "unit": "beats",
                },
            },
        ),
    ],
    ids=["corstone-300", "corstone-320"],
)
def test_performance_metrics_emits_corstone_model_counters(
    backend_name: str,
    target_config: dict[str, int | str],
    model_stats: CorstoneModelPerformanceMetrics,
    expected_metrics: dict[str, dict[str, int | str]],
    tmp_path: Path,
) -> None:
    """Corstone model-level FVP counters should remain integer JSON values."""
    perf_metrics = CorstonePerformanceMetrics(model_stats, [])
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name=backend_name,
        target_config=target_config,
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert {name: metrics[name] for name in expected_metrics} == expected_metrics
    assert isinstance(metrics["npu_active_cycles"]["value"], int)
    assert isinstance(metrics["npu_idle_cycles"]["value"], int)
    assert isinstance(metrics["npu_total_cycles"]["value"], int)
    assert metrics[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY] == {
        "name": schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        "unit": schema.UNIT_BYTES,
        "availability": "unavailable",
        "reason": "Model weight memory data is not available.",
    }


def test_performance_metrics_preserves_supported_corstone_layer_statistics(
    tmp_path: Path,
) -> None:
    """Supported numeric Corstone per-layer CSV fields should become metrics."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=500,
            npu_total_cycles=1500,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=None,
        ),
        [
            {
                "Original Operator": "Conv2D",
                "NNG Operator": "Conv2DBias",
                "Target": "NPU",
                "Staging Usage": "150",
                "Peak% (Staging)": "40",
                "Op Cycles": "300",
                "Network% (cycles)": "30",
                "NPU": "300",
                "SRAM AC": "70",
                "DRAM AC": "20",
                "OnFlash AC": "0",
                "OffFlash AC": "0",
                "MAC Count": "1500",
                "Network% (MAC)": "18",
                "Util% (MAC)": "35",
                "SRAM Usage": "160",
                "Peak%": "50",
                "Network%": "32",
                "Util%": "36",
                "Name": "loc0",
            }
        ],
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-320",
        target_config={"mac": 1024, "target": "ethos-u85"},
    )

    result = output["results"][0]
    breakdown = result["breakdowns"][0]
    assert breakdown["entity_id"] == "source_operator/loc0"
    assert result["entities"] == [
        {
            "id": "source_operator/loc0",
            "kind": schema.ENTITY_KIND_SOURCE_OPERATOR,
            "name": "Conv2DBias",
            "placement": schema.PlacementType.NPU.value,
        }
    ]
    assert {metric["name"]: metric for metric in breakdown["metrics"]} == {
        "staging_usage": {"name": "staging_usage", "value": 150.0, "unit": "bytes"},
        "peak_staging": {"name": "peak_staging", "value": 40.0, "unit": "%"},
        "op_cycles": {"name": "op_cycles", "value": 300.0, "unit": "cycles"},
        "network_cycles": {"name": "network_cycles", "value": 30.0, "unit": "%"},
        "npu": {"name": "npu", "value": 300.0, "unit": "cycles"},
        "sram_ac": {"name": "sram_ac", "value": 70.0, "unit": "accesses"},
        "dram_ac": {"name": "dram_ac", "value": 20.0, "unit": "accesses"},
        "onflash_ac": {"name": "onflash_ac", "value": 0.0, "unit": "accesses"},
        "offflash_ac": {"name": "offflash_ac", "value": 0.0, "unit": "accesses"},
        "mac_count": {"name": "mac_count", "value": 1500.0, "unit": "operations"},
        "network_mac": {"name": "network_mac", "value": 18.0, "unit": "%"},
        "util_mac": {"name": "util_mac", "value": 35.0, "unit": "%"},
        "sram_usage": {"name": "sram_usage", "value": 160.0, "unit": "bytes"},
        "peak": {"name": "peak", "value": 50.0, "unit": "%"},
        "network": {"name": "network", "value": 32.0, "unit": "%"},
        "util": {"name": "util", "value": 36.0, "unit": "%"},
    }


def test_performance_metrics_to_standardized_output_with_null_axi1_wr(
    tmp_path: Path,
) -> None:
    """Test conversion when axi1_wr_data_beat_written is None."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=500,
            npu_total_cycles=1500,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=None,
        ),
        [],
    )

    # Create a model file for hash computation
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output: dict = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )

    # Verify that the optional AXI1 write metric is absent while standard
    # performance fields are still represented.
    results = output["results"]
    assert len(results) == 1
    metrics = results[0]["metrics"]
    metrics_by_name = {metric["name"]: metric for metric in metrics}
    metric_names = set(metrics_by_name)
    assert "npu_axi1_wr_data_beat_written" not in metric_names
    assert schema.METRIC_NAME_TARGET_UTILIZATION in metric_names
    assert schema.METRIC_NAME_INFERENCES_PER_SECOND in metric_names
    for metric_name in (
        schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        schema.METRIC_NAME_AVERAGE_MEMORY,
    ):
        metric = metrics_by_name[metric_name]
        assert metric["availability"] == "unavailable"
        assert "value" not in metric
        assert metric["reason"]


def test_performance_metrics_memory_uses_second_numeric_alias(
    tmp_path: Path,
) -> None:
    """Corstone memory metrics should fall back to the next numeric alias."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=500,
            npu_total_cycles=1500,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=None,
        ),
        [
            {
                "NNG Operator": "op",
                "Name": "op_name",
                "Staging Usage": "",
                "SRAM Usage": "256",
                "Op Cycles": "100",
            }
        ],
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics[schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY] == {
        "name": schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        "value": 256.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics[schema.METRIC_NAME_AVERAGE_MEMORY] == {
        "name": schema.METRIC_NAME_AVERAGE_MEMORY,
        "value": 256.0,
        "unit": schema.UNIT_BYTES,
    }


def test_performance_metrics_to_standardized_output_validates(
    tmp_path: Path,
) -> None:
    """Test Corstone performance output against the MLIA output schema."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=500,
            npu_total_cycles=1500,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=75,
        ),
        [
            {
                "NNG Operator": "op",
                "Name": "op_name",
                "NPU": "1000",
                "Staging Usage": "150",
                "Op Cycles": "300",
            }
        ],
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )

    validate_standardized_output(output)


def test_performance_metrics_to_standardized_output_reports_zero_utilization(
    tmp_path: Path,
) -> None:
    """Corstone target utilization should handle zero total cycles."""
    perf_metrics = CorstonePerformanceMetrics(
        CorstoneModelPerformanceMetrics(
            npu_active_cycles=1000,
            npu_idle_cycles=0,
            npu_total_cycles=0,
            npu_axi0_rd_data_beat_received=200,
            npu_axi0_wr_data_beat_written=100,
            npu_axi1_rd_data_beat_received=150,
            npu_axi1_wr_data_beat_written=None,
        ),
        [],
    )
    model_file = tmp_path / "model.tflite"
    model_file.write_bytes(b"test model content")

    output = perf_metrics.to_standardized_output(
        model_path=model_file,
        backend_name="corstone-300",
        target_config={"mac": 256, "target": "ethos-u55"},
    )

    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }
