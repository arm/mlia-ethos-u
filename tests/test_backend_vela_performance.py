# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for module vela/performance."""

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict, get_type_hints
from unittest.mock import MagicMock

import pytest

try:
    import ethosu.vela  # noqa: F401
except ImportError:
    pytest.skip(
        "All tests require ethosu.vela package to be installed", allow_module_level=True
    )
else:
    # Only reference ethosu.vela if it was successfully imported
    _ = ethosu.vela

import mlia.backend.vela.compiler as vela_compiler_module  # noqa: E402
import mlia.core.output_schema as schema
from mlia.backend.vela.compiler import (
    VelaCompilerOptions,
    VelaSummary,
    compile_model,  # noqa: E402
)
from mlia.backend.vela.performance import (
    LayerPerfInfo,  # noqa: E402
    LayerwisePerfInfo,  # noqa: E402
    PerformanceMetrics,  # noqa: E402
    _debug_db_performance_locations,  # noqa: E402
    estimate_performance,  # noqa: E402
    layer_metrics,  # noqa: E402
    parse_layerwise_perf_csv,  # noqa: E402
)
from mlia.core.output_validation import validate_standardized_output  # noqa: E402
from mlia.target.ethos_u.config import EthosUConfiguration  # noqa: E402
from mlia.utils.filesystem import recreate_directory  # noqa: E402


def patch_vela_main(monkeypatch: pytest.MonkeyPatch, main_mock: Any) -> None:
    """Patch the lazily-loaded Vela main entry point."""
    deps = vela_compiler_module._get_vela_deps()
    monkeypatch.setattr(
        vela_compiler_module,
        "_VELA_DEPS_CACHE",
        replace(deps, main=main_mock),
    )


class ExpectedMetric(TypedDict):
    """Expected standardized metric representation."""

    name: str
    value: int | float
    unit: str


class ExpectedBreakdown(TypedDict):
    """Expected standardized breakdown representation."""

    entity_id: str
    metrics: dict[str, ExpectedMetric]


class _FakeVelaCompiler:
    """Small Vela compiler fake that returns prepared summary data."""

    def __init__(self, summary_data: VelaSummary, compiled_model_path: Path) -> None:
        self.summary_data = summary_data
        self.compiled_model_path = compiled_model_path
        self.model_path: Path | None = None
        self.force_regeneration: bool | None = None

    def compile_model(
        self,
        model_path: Path,
        force_regeneration: bool = False,
    ) -> tuple[VelaSummary, Path]:
        """Record the compile call and return the prepared summary data."""
        self.model_path = model_path
        self.force_regeneration = force_regeneration
        return self.summary_data, self.compiled_model_path


def test_estimate_performance(test_tflite_model: Path) -> None:
    """Test getting performance estimations."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert target_config.compiler_options is not None, (
        "Vela should be available in tests"
    )
    perf_metrics = estimate_performance(
        test_tflite_model, target_config.compiler_options
    )

    assert isinstance(perf_metrics, PerformanceMetrics)


def test_estimate_performance_csv_parser_called(
    monkeypatch: pytest.MonkeyPatch, test_tflite_model: Path
) -> None:
    """Test that estimate_performance from backend.vela.performance is called."""
    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert target_config.compiler_options is not None, (
        "Vela should be available in tests"
    )
    csv_file_name = target_config.compiler_options.output_dir / (
        test_tflite_model.stem + "_per-layer.csv"
    )
    debug_db_path = target_config.compiler_options.output_dir / (
        test_tflite_model.stem + "_debug.xml"
    )
    mock = MagicMock()
    monkeypatch.setattr("mlia.backend.vela.performance.parse_layerwise_perf_csv", mock)
    estimate_performance(test_tflite_model, target_config.compiler_options)
    mock.assert_called_with(
        vela_csv_file=csv_file_name,
        metrics=layer_metrics,
        debug_db_path=debug_db_path,
    )


LAYERWISE_TMP_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

LAYERWISE_MULTI_HEADER_TMP_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

LAYERWISE_ALT_ALIAS_TMP_DATA_STR = """
Original Operator,NNG Operator,Staging Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network% (MAC),Util% (MAC),Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

LAYERWISE_BAD_NUM_VALUE_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,bad_float,sequential/conv1/Relu;sequential/conv1/Conv2D
""".strip()  # noqa: E501

LAYERWISE_NEGATIVE_OP_CYCLES_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,11936,54.65201465201465,-1.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
""".strip()  # noqa: E501

LAYERWISE_MIXED_ALIAS_TMP_DATA_STR = """
TFLite_operator,NNG Operator,Staging Usage,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network% (1),Util% (MAC),Name
CONV_2D,Conv2DBias,11936,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,10944,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

LAYERWISE_REQUIRED_ONLY_TMP_DATA_STR = """
TFLite_operator,NNG Operator,SRAM Usage,Op Cycles,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Util%,Name
CONV_2D,Conv2DBias,11936,7312.0,7312.0,2000.0,0.0,0.0,0.0,73008,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,10944,2992.0,1330.0,2992.0,0.0,0.0,0.0,6912,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

LAYERWISE_TARGET_TMP_DATA_STR = """
TFLite_operator,NNG Operator,Target,SRAM Usage,Op Cycles,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Util%,Name
CONV_2D,Conv2DBias,NPU,11936,7312.0,7312.0,2000.0,0.0,0.0,0.0,73008,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,Passthrough,CPU,0,0.0,0.0,0.0,0.0,0.0,0.0,0,100.0,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501

EXPECTED_ROWS = [
    {
        "name": "sequential/conv1/Relu;sequential/conv1/Conv2D",
        "tflite_operator": "CONV_2D",
        "sram_usage": 11936,
        "op_cycles": 7312,
        "npu_cycles": 7312,
        "sram_access_cycles": 2000,
        "dram_access_cycles": 0,
        "on_chip_flash_access_cycles": 0,
        "off_chip_flash_access_cycles": 0,
        "mac_count": 73008,
        "util_mac_percentage": 3.9002666849015313,
        "placement": schema.PlacementType.UNKNOWN.value,
        "source_locations": [],
    },
    {
        "name": "sequential/max_pooling2d/MaxPool",
        "tflite_operator": "MAX_POOL_2D",
        "sram_usage": 10944,
        "op_cycles": 2992,
        "npu_cycles": 1330,
        "sram_access_cycles": 2992,
        "dram_access_cycles": 0,
        "on_chip_flash_access_cycles": 0,
        "off_chip_flash_access_cycles": 0,
        "mac_count": 6912,
        "util_mac_percentage": 0.9024064171122994,
        "placement": schema.PlacementType.UNKNOWN.value,
        "source_locations": [],
    },
]


@pytest.mark.parametrize(
    "input_csv_content",
    [
        LAYERWISE_TMP_DATA_STR,
        LAYERWISE_MULTI_HEADER_TMP_DATA_STR,
        LAYERWISE_ALT_ALIAS_TMP_DATA_STR,
        LAYERWISE_MIXED_ALIAS_TMP_DATA_STR,
    ],
    ids=["single-header", "multi-header", "alt-aliases", "mixed-aliases"],
)
def test_parse_layerwise_csv_populates_fields_correctly(
    test_csv_file: Path, input_csv_content: str
) -> None:
    """Ensure that parse_layerwise_perf_csv
    populates LayerPerfInfo objects correctly."""

    # Create the test file and parse it
    with open(test_csv_file, "w", encoding="utf8", newline="") as csv_file:
        csv_file.write(input_csv_content)
    layerwise = parse_layerwise_perf_csv(test_csv_file, layer_metrics)
    assert isinstance(layerwise, LayerwisePerfInfo)

    items = layerwise.layerwise_info
    assert items, "No parsed layers found"
    assert len(items) == len(EXPECTED_ROWS), (
        f"Row count mismatch: got {len(items)} vs expected {len(EXPECTED_ROWS)}"
    )

    # Guard against out-of-date EXPECTED_ROWS
    hints = get_type_hints(LayerPerfInfo)
    dc_keys = set(hints.keys())
    for row in EXPECTED_ROWS:
        exp_keys = set(row.keys())
        assert exp_keys == dc_keys, (
            f"EXPECTED_ROWS keys != dataclass fields:\n{exp_keys ^ dc_keys}"
        )

    # Check we got the expected values, with the appropriate types
    for got, exp in zip(items, EXPECTED_ROWS):
        for field_name, expected_type in hints.items():
            got_val = getattr(got, field_name)
            exp_val = exp[field_name]
            if field_name == "source_locations":
                assert isinstance(got_val, list)
            else:
                assert isinstance(got_val, expected_type), (
                    f"{field_name} has wrong type: {type(got_val)} != {expected_type}"
                )
            if expected_type is float:
                assert got_val == pytest.approx(exp_val)
            else:
                assert got_val == exp_val
    additional_metrics = [
        {metric.name: metric.to_dict() for metric in layer_metrics}
        for layer_metrics in layerwise.additional_layer_metrics
    ]
    assert additional_metrics == [
        {
            "peak_sram_usage_percentage": {
                "name": "peak_sram_usage_percentage",
                "value": 54.65201465201465,
                "unit": schema.UNIT_PERCENT,
            },
            "op_cycles_network_percentage": {
                "name": "op_cycles_network_percentage",
                "value": 17.648194632168373,
                "unit": schema.UNIT_PERCENT,
            },
            "mac_count_network_percentage": {
                "name": "mac_count_network_percentage",
                "value": 8.653353814644136,
                "unit": schema.UNIT_PERCENT,
            },
        },
        {
            "peak_sram_usage_percentage": {
                "name": "peak_sram_usage_percentage",
                "value": 50.10989010989011,
                "unit": schema.UNIT_PERCENT,
            },
            "op_cycles_network_percentage": {
                "name": "op_cycles_network_percentage",
                "value": 7.22147132651091,
                "unit": schema.UNIT_PERCENT,
            },
            "mac_count_network_percentage": {
                "name": "mac_count_network_percentage",
                "value": 0.819252432155658,
                "unit": schema.UNIT_PERCENT,
            },
        },
    ]


def test_parse_layerwise_csv_omits_absent_optional_metrics(
    test_csv_file: Path,
) -> None:
    """Vela layer CSVs may omit optional percentage columns."""
    with open(test_csv_file, "w", encoding="utf8", newline="") as csv_file:
        csv_file.write(LAYERWISE_REQUIRED_ONLY_TMP_DATA_STR)

    layerwise = parse_layerwise_perf_csv(test_csv_file, layer_metrics)

    assert layerwise.layerwise_info == [LayerPerfInfo(**row) for row in EXPECTED_ROWS]
    assert layerwise.additional_layer_metrics == [[], []]


def test_parse_layerwise_csv_preserves_target_placement(test_csv_file: Path) -> None:
    """Vela's Target column is the source of truth for layer placement."""
    test_csv_file.write_text(LAYERWISE_TARGET_TMP_DATA_STR, encoding="utf8")

    layerwise = parse_layerwise_perf_csv(test_csv_file, layer_metrics)

    assert [layer.placement for layer in layerwise.layerwise_info] == [
        schema.PlacementType.NPU.value,
        schema.PlacementType.CPU.value,
    ]


LAYERWISE_TMP_DATA_MISSING_HEADER_STR = """
TFLite_operator,NNG Operator,Peak%,Op Cycles,Network%,NPU,SRAM AC,DRAM AC,OnFlash AC,OffFlash AC,MAC Count,Network%,Util%,Name
CONV_2D,Conv2DBias,54.65201465201465,7312.0,17.648194632168373,7312.0,2000.0,0.0,0.0,0.0,73008,8.653353814644136,3.9002666849015313,sequential/conv1/Relu;sequential/conv1/Conv2D
MAX_POOL_2D,MaxPool,50.10989010989011,2992.0,7.22147132651091,1330.0,2992.0,0.0,0.0,0.0,6912,0.819252432155658,0.9024064171122994,sequential/max_pooling2d/MaxPool
""".strip()  # noqa: E501


def test_estimate_performance_parse_layerwise_csv_file_with_missing_headers(
    test_csv_file: Path,
) -> None:
    """Test that ensures a KeyError
    is raised when a csv file is parsed with missing headers.
    """
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(LAYERWISE_TMP_DATA_MISSING_HEADER_STR)
    with pytest.raises(KeyError, match="Generated CSV missing expected headers"):
        parse_layerwise_perf_csv(test_csv_file, layer_metrics)


def test_estimate_performance_parse_layerwise_csv_file_missing_file() -> None:
    """Test that ensures a FileNotFoundError
    is raised when a non-existent csv file is parsed.
    """
    with pytest.raises(
        FileNotFoundError, match="CSV File not found at missing_file.csv"
    ):
        parse_layerwise_perf_csv(Path("missing_file.csv"), layer_metrics)


def test_estimate_performance_parse_layerwise_csv_file_invalid_number(
    test_csv_file: Path,
) -> None:
    """Test if ValueError is raised if a bad numeric value is present in a CSV file."""
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(LAYERWISE_BAD_NUM_VALUE_DATA_STR)
    with pytest.raises(ValueError):
        parse_layerwise_perf_csv(test_csv_file, layer_metrics)


def test_estimate_performance_parse_layerwise_csv_file_negative_op_cycles(
    test_csv_file: Path,
) -> None:
    """Test if ValueError is raised if op_cycles is negative."""
    with open(test_csv_file, "w", encoding="utf8") as csv_file:
        csv_file.write(LAYERWISE_NEGATIVE_OP_CYCLES_DATA_STR)

    with pytest.raises(ValueError, match="negative op_cycles"):
        parse_layerwise_perf_csv(test_csv_file, layer_metrics)


def test_estimate_performance_parse_layerwise_empty_csv_file(
    empty_test_csv_file: Path,
) -> None:
    """Test that ensures that if an empty csv file
    is parsed, we return an empty layerwise object.
    """
    empty_test_csv_file.touch()
    layerwise_object = parse_layerwise_perf_csv(empty_test_csv_file, layer_metrics)
    assert isinstance(layerwise_object, LayerwisePerfInfo)
    assert len(layerwise_object.layerwise_info) == 0


def test_read_invalid_model(test_tflite_invalid_model: Path) -> None:
    """Test that reading invalid model should fail with exception."""
    with pytest.raises(
        Exception,
        match=(
            f"Unable to read model {test_tflite_invalid_model}"
            "|Model could not be optimized with Vela compiler"
        ),
    ):
        target_config = EthosUConfiguration.load_profile("ethos-u55-256")
        assert target_config.compiler_options is not None, (
            "Vela should be available in tests"
        )
        estimate_performance(test_tflite_invalid_model, target_config.compiler_options)


def test_no_csv_file_found(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests if FileNotFound is raised if per-layer CSV file is not found."""
    # Mock a VelaCompiler object that does nothing
    mock_vela_compiler = MagicMock()
    mock_vela_compiler.compile_model.return_value = (VelaSummary, test_tflite_model)
    mock_vela_compiler_class = MagicMock(return_value=mock_vela_compiler)
    monkeypatch.setattr(
        "mlia.backend.vela.compiler.VelaCompiler",
        mock_vela_compiler_class,
    )

    target_config = EthosUConfiguration.load_profile("ethos-u55-256")
    assert target_config.compiler_options is not None, (
        "Vela should be available in tests"
    )
    with pytest.raises(FileNotFoundError, match="Vela per-layer CSV file not found"):
        # Pass an empty directory as output_dir, so that per-layer CSV file
        # cannot be found
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_config.compiler_options.output_dir = Path(tmp_dir)
            estimate_performance(test_tflite_model, target_config.compiler_options)


def test_compile_invalid_model(
    test_tflite_model: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that if model could not be compiled then correct exception raised."""

    mock_compiler = MagicMock()
    mock_compiler.side_effect = Exception("Bad model!")

    patch_vela_main(monkeypatch, mock_compiler)

    model_path = tmp_path / "optimized_model.tflite"
    with pytest.raises(
        Exception, match="Model could not be optimized with Vela compiler"
    ):
        target_config = EthosUConfiguration.load_profile("ethos-u55-256")
        assert target_config.compiler_options is not None, (
            "Vela should be available in tests"
        )
        recreate_directory(Path(target_config.compiler_options.output_dir))
        compile_model(test_tflite_model, target_config.compiler_options)

    assert not model_path.exists()


def _get_perf_metrics() -> PerformanceMetrics:
    layer_info = [
        LayerPerfInfo(
            name="sequential/conv1/Relu;sequential/conv1/Conv2D",
            tflite_operator="CONV_2D",
            sram_usage=11936,
            op_cycles=7312,
            npu_cycles=7312,
            sram_access_cycles=2000,
            dram_access_cycles=0,
            on_chip_flash_access_cycles=0,
            off_chip_flash_access_cycles=0,
            mac_count=73008,
            util_mac_percentage=3.9002666849015313,
            placement=schema.PlacementType.NPU.value,
            source_locations=["operator/0"],
        ),
        LayerPerfInfo(
            name="sequential/max_pooling2d/MaxPool",
            tflite_operator="MAX_POOL_2D",
            sram_usage=10944,
            op_cycles=2992,
            npu_cycles=1330,
            sram_access_cycles=2992,
            dram_access_cycles=0,
            on_chip_flash_access_cycles=0,
            off_chip_flash_access_cycles=0,
            mac_count=6912,
            util_mac_percentage=0.9024064171122994,
            placement=schema.PlacementType.NPU.value,
            source_locations=["operator/1"],
        ),
    ]

    layerwise_perf_info = LayerwisePerfInfo(
        layerwise_info=layer_info,
        additional_layer_metrics=[
            [
                schema.Metric(
                    name="peak_sram_usage_percentage",
                    value=54.65201465201465,
                    unit=schema.UNIT_PERCENT,
                ),
                schema.Metric(
                    name="op_cycles_network_percentage",
                    value=17.648194632168373,
                    unit=schema.UNIT_PERCENT,
                ),
                schema.Metric(
                    name="mac_count_network_percentage",
                    value=8.653353814644136,
                    unit=schema.UNIT_PERCENT,
                ),
            ],
            [
                schema.Metric(
                    name="peak_sram_usage_percentage",
                    value=50.10989010989011,
                    unit=schema.UNIT_PERCENT,
                ),
                schema.Metric(
                    name="op_cycles_network_percentage",
                    value=7.22147132651091,
                    unit=schema.UNIT_PERCENT,
                ),
                schema.Metric(
                    name="mac_count_network_percentage",
                    value=0.819252432155658,
                    unit=schema.UNIT_PERCENT,
                ),
            ],
        ],
    )

    return PerformanceMetrics(
        npu_cycles=10304,
        sram_access_cycles=4992,
        dram_access_cycles=0,
        on_chip_flash_access_cycles=0,
        off_chip_flash_access_cycles=0,
        total_cycles=41416,
        batch_inference_time=0.207,
        inferences_per_second=4830.9,
        batch_size=1,
        sram_memory_area_size=11936.0,
        dram_memory_area_size=0.0,
        on_chip_flash_memory_area_size=0.0,
        off_chip_flash_memory_area_size=0.0,
        layerwise_performance_info=layerwise_perf_info,
        additional_summary_metrics=[
            schema.Metric(
                name="total_original_weights",
                value=16.0,
                unit=schema.UNIT_BYTES,
            ),
            schema.Metric(
                name="total_npu_encoded_weights",
                value=8.0,
                unit=schema.UNIT_BYTES,
            ),
            schema.Metric(
                name=schema.METRIC_NAME_INFERENCE_TIME,
                value=0.207,
                unit=schema.UNIT_MILLISECONDS,
            ),
            schema.Metric(name="dram_total_bytes", value=12.0, unit=schema.UNIT_BYTES),
        ],
    )


def test_to_standardized_output(
    test_tflite_model: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test to_standardized_output."""
    perf_metrics = _get_perf_metrics()

    monkeypatch.setattr(
        "mlia.core.output_schema.StandardizedOutput.create_run_id",
        MagicMock(return_value="fake_id"),
    )
    monkeypatch.setattr(
        "mlia.core.output_schema.StandardizedOutput.create_timestamp",
        MagicMock(return_value="fake_timestamp"),
    )
    monkeypatch.setattr("mlia.core.output_schema.SCHEMA_VERSION", "1.1.0")

    standardized_output = perf_metrics.to_standardized_output(test_tflite_model)

    assert standardized_output["schema_version"] == "1.1.0"
    assert standardized_output["run_id"] == "fake_id"
    assert standardized_output["timestamp"] == "fake_timestamp"
    assert standardized_output["tool"] == {
        "name": "mlia",
        "version": standardized_output["tool"]["version"],
    }
    assert standardized_output["target"] == {
        "profile_name": "ethos-u",
        "target_type": "npu",
        "components": [{"type": "npu", "family": "ethos-u"}],
        "configuration": {},
    }
    assert standardized_output["model"]["name"] == test_tflite_model.name
    assert standardized_output["model"]["format"] == "tflite"
    assert not standardized_output["context"]
    assert standardized_output["backends"] == [
        {
            "id": "vela",
            "name": "Vela Compiler",
            "version": standardized_output["backends"][0]["version"],
            "configuration": {},
        }
    ]
    assert len(standardized_output["results"]) == 1
    results = standardized_output["results"][0]
    assert results["warnings"] == [
        "The performance figures above refer to NPU only",
    ]
    result_metrics = {metric["name"]: metric for metric in results["metrics"]}
    assert result_metrics["npu_cycles"] == {
        "name": "npu_cycles",
        "value": 10304,
        "unit": "cycles",
    }
    assert result_metrics[schema.METRIC_NAME_INFERENCES_PER_SECOND] == {
        "name": schema.METRIC_NAME_INFERENCES_PER_SECOND,
        "value": 4830.9,
        "unit": schema.UNIT_INFERENCES_PER_SECOND,
    }
    assert result_metrics[schema.METRIC_NAME_INFERENCE_TIME] == {
        "name": schema.METRIC_NAME_INFERENCE_TIME,
        "value": 0.207,
        "unit": schema.UNIT_MILLISECONDS,
    }
    assert result_metrics["batch_inference_time"] == {
        "name": "batch_inference_time",
        "value": 0.207,
        "unit": schema.UNIT_MILLISECONDS,
    }
    assert result_metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": pytest.approx((10304 / 41416) * 100),
        "unit": schema.UNIT_PERCENT,
    }
    assert result_metrics[schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY] == {
        "name": schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        "value": 11936,
        "unit": schema.UNIT_BYTES,
    }
    assert result_metrics[schema.METRIC_NAME_AVERAGE_MEMORY] == {
        "name": schema.METRIC_NAME_AVERAGE_MEMORY,
        "value": pytest.approx(((11936 * 7312) + (10944 * 2992)) / (7312 + 2992)),
        "unit": schema.UNIT_BYTES,
    }
    assert result_metrics[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY] == {
        "name": schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        "value": 8,
        "unit": schema.UNIT_BYTES,
    }
    assert result_metrics["total_original_weights"] == {
        "name": "total_original_weights",
        "value": 16.0,
        "unit": schema.UNIT_BYTES,
    }
    assert result_metrics["total_npu_encoded_weights"] == {
        "name": "total_npu_encoded_weights",
        "value": 8.0,
        "unit": schema.UNIT_BYTES,
    }
    assert result_metrics["dram_total_bytes"] == {
        "name": "dram_total_bytes",
        "value": 12.0,
        "unit": schema.UNIT_BYTES,
    }
    for unavailable_metric in [
        schema.METRIC_NAME_ACCELERATOR_OPERATOR_PERCENTAGE,
        schema.METRIC_NAME_CPU_UTILIZATION,
    ]:
        metric = result_metrics[unavailable_metric]
        assert metric["availability"] == "unavailable"
        assert "value" not in metric
        assert metric["reason"]

    assert len(results["breakdowns"]) == 2
    breakdowns = results["breakdowns"]

    expected_entities = [
        {
            "id": "source_operator/operator/0",
            "kind": "source_operator",
            "name": "CONV_2D",
            "placement": "NPU",
            "attributes": {
                "layer_name": "sequential/conv1/Relu;sequential/conv1/Conv2D"
            },
        },
        {
            "id": "source_operator/operator/1",
            "kind": "source_operator",
            "name": "MAX_POOL_2D",
            "placement": "NPU",
            "attributes": {"layer_name": "sequential/max_pooling2d/MaxPool"},
        },
    ]
    assert results["entities"] == expected_entities

    expected_breakdowns: list[ExpectedBreakdown] = [
        {
            "entity_id": "source_operator/operator/0",
            "metrics": {
                "op_cycles": {"name": "op_cycles", "value": 7312, "unit": "cycles"},
                "npu_cycles": {"name": "npu_cycles", "value": 7312, "unit": "cycles"},
                "sram_access_cycles": {
                    "name": "sram_access_cycles",
                    "value": 2000,
                    "unit": "cycles",
                },
                "dram_access_cycles": {
                    "name": "dram_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "on_chip_flash_access_cycles": {
                    "name": "on_chip_flash_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "off_chip_flash_access_cycles": {
                    "name": "off_chip_flash_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "sram_usage": {"name": "sram_usage", "value": 11936, "unit": "bytes"},
                "peak_sram_usage_percentage": {
                    "name": "peak_sram_usage_percentage",
                    "value": 54.65201465201465,
                    "unit": schema.UNIT_PERCENT,
                },
                "mac_count": {"name": "mac_count", "value": 73008, "unit": "count"},
                "op_cycles_network_percentage": {
                    "name": "op_cycles_network_percentage",
                    "value": 17.648194632168373,
                    "unit": schema.UNIT_PERCENT,
                },
                "mac_count_network_percentage": {
                    "name": "mac_count_network_percentage",
                    "value": 8.653353814644136,
                    "unit": schema.UNIT_PERCENT,
                },
                "util_mac_percentage": {
                    "name": "util_mac_percentage",
                    "value": 3.9002666849015313,
                    "unit": "percent",
                },
            },
        },
        {
            "entity_id": "source_operator/operator/1",
            "metrics": {
                "op_cycles": {"name": "op_cycles", "value": 2992, "unit": "cycles"},
                "npu_cycles": {"name": "npu_cycles", "value": 1330, "unit": "cycles"},
                "sram_access_cycles": {
                    "name": "sram_access_cycles",
                    "value": 2992,
                    "unit": "cycles",
                },
                "dram_access_cycles": {
                    "name": "dram_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "on_chip_flash_access_cycles": {
                    "name": "on_chip_flash_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "off_chip_flash_access_cycles": {
                    "name": "off_chip_flash_access_cycles",
                    "value": 0,
                    "unit": "cycles",
                },
                "sram_usage": {"name": "sram_usage", "value": 10944, "unit": "bytes"},
                "peak_sram_usage_percentage": {
                    "name": "peak_sram_usage_percentage",
                    "value": 50.10989010989011,
                    "unit": schema.UNIT_PERCENT,
                },
                "mac_count": {"name": "mac_count", "value": 6912, "unit": "count"},
                "op_cycles_network_percentage": {
                    "name": "op_cycles_network_percentage",
                    "value": 7.22147132651091,
                    "unit": schema.UNIT_PERCENT,
                },
                "mac_count_network_percentage": {
                    "name": "mac_count_network_percentage",
                    "value": 0.819252432155658,
                    "unit": schema.UNIT_PERCENT,
                },
                "util_mac_percentage": {
                    "name": "util_mac_percentage",
                    "value": 0.9024064171122994,
                    "unit": "percent",
                },
            },
        },
    ]

    for i, expected in enumerate(expected_breakdowns):
        assert breakdowns[i]["entity_id"] == expected["entity_id"]
        assert len(breakdowns[i]["metrics"]) == len(expected["metrics"])

        metrics = {m["name"]: m for m in breakdowns[i]["metrics"]}
        for metric_name, expected_metric in expected["metrics"].items():
            assert metrics[metric_name] == expected_metric


def test_to_standardized_output_preserves_layer_placement(
    test_tflite_model: Path,
) -> None:
    """Performance entities use Vela placement rather than assuming NPU."""
    perf_metrics = _get_perf_metrics()
    perf_metrics.layerwise_performance_info.layerwise_info[
        1
    ].placement = schema.PlacementType.CPU.value

    output = perf_metrics.to_standardized_output(test_tflite_model)

    entities = {entity["id"]: entity for entity in output["results"][0]["entities"]}
    assert entities["source_operator/operator/0"]["placement"] == "NPU"
    assert entities["source_operator/operator/1"]["placement"] == "CPU"


def test_performance_metrics_preserves_vela_summary_statistics(
    monkeypatch: pytest.MonkeyPatch,
    test_tflite_model: Path,
    tmp_path: Path,
) -> None:
    """Vela summary statistics should become result-level metrics."""
    summary_data = VelaSummary(
        cycles_total=2.0,
        cycles_npu=1.0,
        cycles_sram_access=0.0,
        cycles_dram_access=0.0,
        cycles_on_chip_flash_access=0.0,
        cycles_off_chip_flash_access=0.0,
        core_clock=10_000.0,
        dram_memory_used=512.0,
        sram_memory_used=1024.0,
        on_chip_flash_memory_used=0.0,
        off_chip_flash_memory_used=0.0,
        batch_size=1,
        memory_mode="Shared_Sram",
        system_config="Ethos_U55_High_End_Embedded",
        accelerator_configuration="Ethos_U55_256",
        arena_cache_size=4096.0,
        dram_bandwidth=4.0,
        inference_time=0.0002,
        passes_before_fusing=7.0,
        passes_after_fusing=2.0,
        total_original_weights=64.0,
        total_npu_encoded_weights=32.0,
        dram_feature_map_read_bytes=16.0,
        dram_weight_write_bytes=8.0,
        nn_macs=128.0,
        nn_tops=0.25,
    )
    compiler_options = VelaCompilerOptions(
        accelerator_config="ethos-u55-256",
        output_dir=tmp_path,
    )
    per_layer_csv = tmp_path / f"{test_tflite_model.stem}_per-layer.csv"
    per_layer_csv.touch()
    fake_vela_compiler = _FakeVelaCompiler(
        summary_data,
        tmp_path / "compiled.tflite",
    )
    mock_parse_layerwise_perf_csv = MagicMock(
        return_value=LayerwisePerfInfo(layerwise_info=[])
    )
    # The summary metrics are attached inside estimate_performance, so this test
    # replaces only the backend execution and per-layer parsing steps.
    monkeypatch.setattr(
        "mlia.backend.vela.compiler.VelaCompiler",
        lambda _compiler_options: fake_vela_compiler,
    )
    monkeypatch.setattr(
        "mlia.backend.vela.performance.parse_layerwise_perf_csv",
        mock_parse_layerwise_perf_csv,
    )

    perf_metrics = estimate_performance(test_tflite_model, compiler_options)

    output = perf_metrics.to_standardized_output(test_tflite_model)

    assert fake_vela_compiler.model_path == test_tflite_model
    assert fake_vela_compiler.force_regeneration is False
    mock_parse_layerwise_perf_csv.assert_called_once_with(
        vela_csv_file=per_layer_csv,
        metrics=layer_metrics,
    )
    metrics = {metric["name"]: metric for metric in output["results"][0]["metrics"]}
    assert metrics["total_original_weights"] == {
        "name": "total_original_weights",
        "value": 64.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics["total_npu_encoded_weights"] == {
        "name": "total_npu_encoded_weights",
        "value": 32.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY] == {
        "name": schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        "value": 32,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics[schema.METRIC_NAME_INFERENCE_TIME] == {
        "name": schema.METRIC_NAME_INFERENCE_TIME,
        "value": pytest.approx(0.2),
        "unit": schema.UNIT_MILLISECONDS,
    }
    assert metrics["dram_feature_map_read_bytes"] == {
        "name": "dram_feature_map_read_bytes",
        "value": 16.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics["dram_weight_write_bytes"] == {
        "name": "dram_weight_write_bytes",
        "value": 8.0,
        "unit": schema.UNIT_BYTES,
    }
    assert metrics["nn_macs"] == {
        "name": "nn_macs",
        "value": 128.0,
        "unit": "operations",
    }
    assert metrics["nn_tops"] == {
        "name": "nn_tops",
        "value": 0.25,
        "unit": "TOPS",
    }
    for configuration_metric in (
        "core_clock",
        "arena_cache_size",
        "dram_bandwidth",
        "passes_before_fusing",
        "passes_after_fusing",
    ):
        assert configuration_metric not in metrics


def test_debug_db_performance_locations_use_source_ext_key(tmp_path: Path) -> None:
    """Vela debug DB ext_key values provide canonical TFLite operator locations."""
    debug_db = tmp_path / "model_debug.xml"
    debug_db.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<debug source="model.tflite" optimised="model_vela.tflite">
  <table name='source'><![CDATA["id","operator","kernel_w","kernel_h","ofm_w","ofm_h","ofm_d","ext_key"
0,"Conv2D",1,1,1,1,1,7
1,"Softmax",1,1,1,1,1,68
]]></table>
  <table name='perf'><![CDATA["source_id","name"
0,"untrusted name"
1,"another untrusted name"
]]></table>
</debug>
""",
        encoding="utf-8",
    )

    assert _debug_db_performance_locations(debug_db) == [
        ["operator/7"],
        ["operator/68"],
    ]


def test_to_standardized_output_validates_against_schema(
    test_tflite_model: Path,
) -> None:
    """Test Vela performance output against the MLIA output schema."""
    perf_metrics = _get_perf_metrics()

    standardized_output = perf_metrics.to_standardized_output(test_tflite_model)

    validate_standardized_output(standardized_output)


def test_to_standardized_output_reports_zero_target_utilization(
    test_tflite_model: Path,
) -> None:
    """Test target utilization when total cycles are unavailable."""
    perf_metrics = _get_perf_metrics()
    perf_metrics.total_cycles = 0

    standardized_output = perf_metrics.to_standardized_output(test_tflite_model)

    result_metrics: dict[str, dict[str, Any]] = {
        metric["name"]: metric
        for metric in standardized_output["results"][0]["metrics"]
    }
    assert result_metrics[schema.METRIC_NAME_TARGET_UTILIZATION] == {
        "name": schema.METRIC_NAME_TARGET_UTILIZATION,
        "value": 0.0,
        "unit": schema.UNIT_PERCENT,
    }


def test_to_standardized_output_marks_memory_metrics_unavailable_without_layers(
    test_tflite_model: Path,
) -> None:
    """Test memory metric availability when per-layer source data is absent."""
    perf_metrics = _get_perf_metrics()
    perf_metrics.layerwise_performance_info = LayerwisePerfInfo(layerwise_info=[])

    standardized_output = perf_metrics.to_standardized_output(test_tflite_model)

    result_metrics: dict[str, dict[str, Any]] = {
        metric["name"]: metric
        for metric in standardized_output["results"][0]["metrics"]
    }
    for metric_name in (
        schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
        schema.METRIC_NAME_AVERAGE_MEMORY,
    ):
        metric = result_metrics[metric_name]
        assert metric["availability"] == "unavailable"
        assert "value" not in metric
        assert metric["reason"]


def test_to_standardized_output_marks_model_weight_memory_unavailable_without_source(
    test_tflite_model: Path,
) -> None:
    """Test model weight metric availability when encoded weights are absent."""
    perf_metrics = _get_perf_metrics()
    perf_metrics.additional_summary_metrics = [
        metric
        for metric in perf_metrics.additional_summary_metrics
        if metric.name != "total_npu_encoded_weights"
    ]

    standardized_output = perf_metrics.to_standardized_output(test_tflite_model)

    result_metrics: dict[str, dict[str, Any]] = {
        metric["name"]: metric
        for metric in standardized_output["results"][0]["metrics"]
    }
    assert result_metrics[schema.METRIC_NAME_MODEL_WEIGHT_MEMORY] == {
        "name": schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
        "unit": schema.UNIT_BYTES,
        "availability": "unavailable",
        "reason": "Model weight memory data is not available.",
    }


def test_to_standardized_output_rejects_negative_total_layer_op_cycles(
    test_tflite_model: Path,
) -> None:
    """Test the internal invariant for average memory weighting."""
    perf_metrics = _get_perf_metrics()
    perf_metrics.layerwise_performance_info.layerwise_info[0].op_cycles = -20_000

    with pytest.raises(ValueError, match="negative value"):
        perf_metrics.to_standardized_output(test_tflite_model)


def test_to_standarized_output_kwargs(test_tflite_model: Path) -> None:
    """Test optional kwargs to test_tflite_model."""
    perf_metrics = _get_perf_metrics()
    standardized_output = perf_metrics.to_standardized_output(
        test_tflite_model,
        target_config={"target": "my_target"},
        backend_config={"backend_key": "backend_val"},
        run_id="12",
        timestamp="34",
        cli_arguments=["--arg1"],
    )

    assert standardized_output["run_id"] == "12"
    assert standardized_output["timestamp"] == "34"
    assert standardized_output["target"]["profile_name"] == "my_target"
    assert standardized_output["backends"][0]["configuration"] == {
        "backend_key": "backend_val"
    }
    assert standardized_output["context"] == {"cli_arguments": ["--arg1"]}
