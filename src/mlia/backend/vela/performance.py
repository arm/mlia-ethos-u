# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela performance module."""

from __future__ import annotations

import csv
import io
import logging
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mlia.backend.errors import BackendUnavailableError

if TYPE_CHECKING:
    from mlia.backend.vela.compiler import VelaCompilerOptions, VelaSummary

import mlia
import mlia.core.output_schema as schema
from mlia.target.ethos_u.performance_warnings import NPU_ONLY_PERFORMANCE_WARNING
from mlia.utils.filesystem import sha256

logger = logging.getLogger(__name__)

_VELA_VERSION_CACHE: str | None = None

_VELA_SUMMARY_METRIC_UNITS = {
    "inference_time": "ms",
    "sram_memory_used": "bytes",
    "dram_memory_used": "bytes",
    "on_chip_flash_memory_used": "bytes",
    "off_chip_flash_memory_used": "bytes",
    "total_original_weights": "bytes",
    "total_npu_encoded_weights": "bytes",
    "sram_feature_map_read_bytes": "bytes",
    "sram_feature_map_write_bytes": "bytes",
    "sram_weight_read_bytes": "bytes",
    "sram_weight_write_bytes": "bytes",
    "sram_total_bytes": "bytes",
    "dram_feature_map_read_bytes": "bytes",
    "dram_feature_map_write_bytes": "bytes",
    "dram_weight_read_bytes": "bytes",
    "dram_weight_write_bytes": "bytes",
    "dram_total_bytes": "bytes",
    "on_chip_flash_feature_map_read_bytes": "bytes",
    "on_chip_flash_feature_map_write_bytes": "bytes",
    "on_chip_flash_weight_read_bytes": "bytes",
    "on_chip_flash_weight_write_bytes": "bytes",
    "on_chip_flash_total_bytes": "bytes",
    "off_chip_flash_feature_map_read_bytes": "bytes",
    "off_chip_flash_feature_map_write_bytes": "bytes",
    "off_chip_flash_weight_read_bytes": "bytes",
    "off_chip_flash_weight_write_bytes": "bytes",
    "off_chip_flash_total_bytes": "bytes",
    "nn_macs": "operations",
    "nn_tops": "TOPS",
}

_VELA_RESULT_METRIC_ALIASES = {
    "cycles_npu": "npu_cycles",
    "cycles_sram_access": "sram_access_cycles",
    "cycles_dram_access": "dram_access_cycles",
    "cycles_on_chip_flash_access": "on_chip_flash_access_cycles",
    "cycles_off_chip_flash_access": "off_chip_flash_access_cycles",
    "cycles_total": "total_cycles",
    "inferences_per_second": "inferences_per_second",
}

_VELA_OPTIONAL_LAYER_METRIC_UNITS = {
    "peak_sram_usage_percentage": schema.UNIT_PERCENT,
    "op_cycles_network_percentage": schema.UNIT_PERCENT,
    "mac_count_network_percentage": schema.UNIT_PERCENT,
}

_VELA_MODEL_WEIGHT_MEMORY_SOURCE_METRIC_NAME = "total_npu_encoded_weights"


def _summary_metric_value(field_name: str, value: float | int) -> float | int:
    """Return the MLIA output value for a Vela summary metric."""
    if field_name == schema.METRIC_NAME_INFERENCE_TIME:
        return value * 1000
    return value


def _byte_count_value(value: int | float) -> int | float:
    """Return integral byte counts as integers."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _model_weight_memory_metric(
    metrics: list[schema.Metric],
) -> schema.Metric | None:
    """Return the standard model weight memory metric from Vela encoded weights."""
    for metric in metrics:
        if metric.name != _VELA_MODEL_WEIGHT_MEMORY_SOURCE_METRIC_NAME:
            continue
        if metric.value is None:
            return None
        return schema.Metric(
            name=schema.METRIC_NAME_MODEL_WEIGHT_MEMORY,
            value=_byte_count_value(metric.value),
            unit=schema.UNIT_BYTES,
        )
    return None


def _split_locations(value: str) -> list[str]:
    """Split semicolon-separated source-model locations."""
    return [part for part in value.split(";") if part]


_VELA_PERFORMANCE_LAYER_KIND = "vela_performance_layer"
_VELA_PERFORMANCE_GROUP_KIND = "vela_performance_group"


def _source_operator_entity_id(source_reference: str) -> str:
    """Return the canonical source-operator entity ID."""
    return f"source_operator/{source_reference}"


def _performance_layer_entity_id(
    layer_index: int, source_index: int | None = None
) -> str:
    """Return a result-local entity ID for one Vela performance layer."""
    suffix = (
        str(layer_index)
        if source_index is None
        else f"{layer_index}/source/{source_index}"
    )
    return f"{_VELA_PERFORMANCE_LAYER_KIND}/{suffix}"


def _performance_group_entity_id(index: int) -> str:
    """Return a result-local entity ID for one aggregate performance row."""
    return f"{_VELA_PERFORMANCE_GROUP_KIND}/{index}"


def _has_unambiguous_tflite_operator_provenance(model_path: Path) -> bool:
    """Return whether Vela references can identify source TFLite operators."""
    if model_path.suffix.lower() != ".tflite":
        return False

    try:
        from ethosu.vela.tflite.Model import Model as TFLiteModel

        model_buffer = model_path.read_bytes()
        if not TFLiteModel.ModelBufferHasIdentifier(model_buffer, 0):
            return False
        model = TFLiteModel.GetRootAs(model_buffer)
    except Exception as exc:  # Provenance failures must conservatively withhold links.
        logger.debug("Unable to inspect TFLite source provenance: %s", exc)
        return False

    # Vela records only Operation.op_index in debug DB ext_key. TFLite operator
    # indexes are local to a subgraph, but the database omits subgraph_index, so
    # only a proven single-subgraph model has an unambiguous source identity.
    return model.SubgraphsLength() == 1


def _read_debug_db_table(debug_db_path: Path, table_name: str) -> list[dict[str, str]]:
    """Read a Vela debug XML table as CSV rows."""
    if not debug_db_path.is_file():
        raise FileNotFoundError(f"Vela debug database not found: {debug_db_path}")

    root = ET.parse(debug_db_path).getroot()
    table = root.find(f"./table[@name='{table_name}']")
    if table is None or table.text is None:
        raise ValueError(f"Vela debug database missing table: {table_name}")
    return list(csv.DictReader(io.StringIO(table.text.strip())))


def _debug_db_performance_source_references(
    debug_db_path: Path,
) -> list[list[str]]:
    """Return Vela source references associated with each performance row."""
    source_rows = _read_debug_db_table(debug_db_path, "source")
    perf_rows = _read_debug_db_table(debug_db_path, "perf")

    ext_key_by_source_id = {
        row["id"]: row.get("ext_key", "")
        for row in source_rows
        if row.get("id") is not None
    }

    references: list[list[str]] = []
    for row in perf_rows:
        source_id = row.get("source_id")
        ext_key = ext_key_by_source_id.get(str(source_id))
        if ext_key is None or ext_key == "" or ext_key == "-1":
            references.append([])
            continue
        references.append([f"operator/{int(ext_key)}"])
    return references


def _average_memory_usage(layerwise_info: LayerwisePerfInfo) -> float | None:
    """Return per-layer memory usage weighted by operation cycles."""
    total_cycles = sum(layer.op_cycles for layer in layerwise_info.layerwise_info)
    if total_cycles < 0:
        raise ValueError(
            f"Layer operation cycles must not sum to a negative value: {total_cycles}"
        )
    if total_cycles == 0:
        return None

    weighted_memory_usage = sum(
        layer.sram_usage * layer.op_cycles for layer in layerwise_info.layerwise_info
    )
    return weighted_memory_usage / total_cycles


def _peak_memory_usage(layerwise_info: LayerwisePerfInfo) -> int | None:
    """Return the highest per-layer memory usage."""
    memory_values = [layer.sram_usage for layer in layerwise_info.layerwise_info]
    if not memory_values:
        return None
    return max(memory_values)


def _load_vela_version() -> str:
    """Load Vela version on demand."""
    try:
        import importlib

        importlib.invalidate_caches()
        from ethosu.vela import __version__ as ethosu_vela_version
    except ImportError as exc:
        raise BackendUnavailableError("Backend vela is not available", "vela") from exc
    return ethosu_vela_version


def _get_vela_version() -> str:
    """Return cached Vela version or load it."""
    global _VELA_VERSION_CACHE

    if _VELA_VERSION_CACHE is None:
        _VELA_VERSION_CACHE = _load_vela_version()
    return _VELA_VERSION_CACHE


@dataclass
class PerformanceMetrics:
    """Contains all the performance metrics Vela generates in a run."""

    npu_cycles: int
    sram_access_cycles: int
    dram_access_cycles: int
    on_chip_flash_access_cycles: int
    off_chip_flash_access_cycles: int
    total_cycles: int
    batch_inference_time: float
    inferences_per_second: float
    batch_size: int
    sram_memory_area_size: float
    dram_memory_area_size: float
    on_chip_flash_memory_area_size: float
    off_chip_flash_memory_area_size: float
    layerwise_performance_info: LayerwisePerfInfo
    additional_summary_metrics: list[schema.Metric] = field(default_factory=list)

    def to_standardized_output(
        self,
        model_path: Path,
        target_config: dict[str, Any] | None = None,
        backend_config: dict[str, Any] | None = None,
        run_id: str | None = None,
        timestamp: str | None = None,
        cli_arguments: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert to standardized output format.

        Args:
            model_path: Path to the model file
            target_config: Target configuration dict (target, mac, etc.)
            backend_config: Backend configuration
            run_id: Optional run ID (generated if not provided)
            timestamp: Optional timestamp (generated if not provided)
            cli_arguments: Optional CLI arguments used for the run

        Returns:
            Standardized output dictionary
        """
        # Generate run_id and timestamp if not provided
        if run_id is None:
            run_id = schema.StandardizedOutput.create_run_id()
        if timestamp is None:
            timestamp = schema.StandardizedOutput.create_timestamp()

        # Create tool info
        tool = schema.Tool(name="mlia", version=mlia.__version__)

        # Create backend with version
        try:
            backend_version = _get_vela_version()
        except Exception as exc:
            logger.warning("Failed to get vela version: %s", exc)
            backend_version = "unknown"

        backend = schema.Backend(
            id="vela",
            name="Vela Compiler",
            version=backend_version,
            configuration=backend_config or {},
        )

        # Create target with NPU component
        target_type = (target_config or {}).get("target", "ethos-u")
        mac = (target_config or {}).get("mac", "unknown")

        npu_component = schema.Component(
            type=schema.ComponentType.NPU,
            family=target_type,
            model=None,
            variant=mac if mac != "unknown" else None,
        )

        target = schema.Target(
            profile_name=target_type,
            target_type="npu",
            components=[npu_component],
            configuration=target_config or {},
        )

        # Create model
        model_hash = sha256(model_path)
        model_format = model_path.suffix.lstrip(".") if model_path.suffix else "unknown"
        model = schema.Model(
            name=model_path.name,
            format=model_format,
            hash=model_hash,
        )

        # Create context
        context = schema.Context(cli_arguments=cli_arguments or [])

        # Backend-specific raw Vela metrics.
        model_metrics = [
            schema.Metric(name="npu_cycles", value=self.npu_cycles, unit="cycles"),
            schema.Metric(
                name="sram_access_cycles", value=self.sram_access_cycles, unit="cycles"
            ),
            schema.Metric(
                name="dram_access_cycles", value=self.dram_access_cycles, unit="cycles"
            ),
            schema.Metric(
                name="on_chip_flash_access_cycles",
                value=self.on_chip_flash_access_cycles,
                unit="cycles",
            ),
            schema.Metric(
                name="off_chip_flash_access_cycles",
                value=self.off_chip_flash_access_cycles,
                unit="cycles",
            ),
            schema.Metric(name="total_cycles", value=self.total_cycles, unit="cycles"),
            schema.Metric(
                name="batch_inference_time",
                value=self.batch_inference_time,
                unit=schema.UNIT_MILLISECONDS,
            ),
            schema.Metric(name="batch_size", value=self.batch_size, unit="count"),
            schema.Metric(
                name="model_size", value=model_path.stat().st_size, unit="bytes"
            ),
            schema.Metric(
                name="sram_memory_area_size",
                value=self.sram_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="dram_memory_area_size",
                value=self.dram_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="on_chip_flash_memory_area_size",
                value=self.on_chip_flash_memory_area_size,
                unit="bytes",
            ),
            schema.Metric(
                name="off_chip_flash_memory_area_size",
                value=self.off_chip_flash_memory_area_size,
                unit="bytes",
            ),
        ]

        # Standard performance metrics defined by the output schema.
        model_metrics.extend(
            [
                schema.Metric(
                    name=schema.METRIC_NAME_INFERENCES_PER_SECOND,
                    value=self.inferences_per_second,
                    unit=schema.UNIT_INFERENCES_PER_SECOND,
                ),
                schema.Metric(
                    name=schema.METRIC_NAME_TARGET_UTILIZATION,
                    value=(
                        (self.npu_cycles / self.total_cycles) * 100
                        if self.total_cycles
                        else 0.0
                    ),
                    unit=schema.UNIT_PERCENT,
                ),
            ]
        )
        existing_metric_names = {metric.name for metric in model_metrics}
        for metric in self.additional_summary_metrics:
            if metric.name in existing_metric_names:
                continue
            model_metrics.append(metric)
            existing_metric_names.add(metric.name)
        # Add the standardized model weight metric separately because it is
        # derived from the Vela encoded-weight metric preserved above.
        model_weight_memory_metric = _model_weight_memory_metric(
            self.additional_summary_metrics
        )
        if (
            model_weight_memory_metric is not None
            and model_weight_memory_metric.name not in existing_metric_names
        ):
            model_metrics.append(model_weight_memory_metric)
        peak_memory_usage = _peak_memory_usage(self.layerwise_performance_info)
        if peak_memory_usage is not None:
            model_metrics.append(
                schema.Metric(
                    name=schema.METRIC_NAME_PEAK_ACTIVATION_MEMORY,
                    value=peak_memory_usage,
                    unit=schema.UNIT_BYTES,
                )
            )
        average_memory_usage = _average_memory_usage(self.layerwise_performance_info)
        if average_memory_usage is not None:
            model_metrics.append(
                schema.Metric(
                    name=schema.METRIC_NAME_AVERAGE_MEMORY,
                    value=average_memory_usage,
                    unit=schema.UNIT_BYTES,
                )
            )
        model_metrics = schema.ensure_standard_performance_metrics(model_metrics)

        entities: list[schema.Entity] = []
        breakdowns = []
        source_operator_provenance_is_unambiguous = (
            _has_unambiguous_tflite_operator_provenance(model_path)
        )
        uses_performance_layer_entities = False
        performance_group_child_kinds: set[str] = set()
        for layer_index, layer_info in enumerate(
            self.layerwise_performance_info.layerwise_info
        ):
            breakdown_metrics = [
                schema.Metric(
                    name="op_cycles",
                    value=layer_info.op_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="npu_cycles",
                    value=layer_info.npu_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="sram_access_cycles",
                    value=layer_info.sram_access_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="dram_access_cycles",
                    value=layer_info.dram_access_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="on_chip_flash_access_cycles",
                    value=layer_info.on_chip_flash_access_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="off_chip_flash_access_cycles",
                    value=layer_info.off_chip_flash_access_cycles,
                    unit="cycles",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="sram_usage",
                    value=layer_info.sram_usage,
                    unit="bytes",
                    aggregation=schema.AggregationType.MAX,
                ),
                schema.Metric(
                    name="mac_count",
                    value=layer_info.mac_count,
                    unit="count",
                    aggregation=schema.AggregationType.SUM,
                ),
                schema.Metric(
                    name="util_mac_percentage",
                    value=layer_info.util_mac_percentage,
                    unit="percent",
                ),
            ]
            if layer_index < len(
                self.layerwise_performance_info.additional_layer_metrics
            ):
                breakdown_metrics.extend(
                    self.layerwise_performance_info.additional_layer_metrics[
                        layer_index
                    ]
                )
            source_locations = list(dict.fromkeys(layer_info.source_locations))
            display_name = layer_info.tflite_operator or layer_info.name
            if source_operator_provenance_is_unambiguous and source_locations:
                source_operator_ids = [
                    _source_operator_entity_id(location)
                    for location in source_locations
                ]
                for source_operator_id in source_operator_ids:
                    existing_entity = next(
                        (
                            entity
                            for entity in entities
                            if entity.id == source_operator_id
                        ),
                        None,
                    )
                    if existing_entity is not None:
                        if existing_entity.placement != layer_info.placement:
                            raise ValueError(
                                "Vela performance rows assign conflicting placements "
                                f"to {source_operator_id!r}."
                            )
                        continue
                    entities.append(
                        schema.Entity(
                            id=source_operator_id,
                            kind=schema.ENTITY_KIND_SOURCE_OPERATOR,
                            name=display_name,
                            placement=layer_info.placement,
                            attributes={"layer_name": layer_info.name},
                        )
                    )
                if len(source_operator_ids) == 1:
                    entity_id = source_operator_ids[0]
                else:
                    entity_id = _performance_group_entity_id(layer_index)
                    performance_group_child_kinds.add(
                        schema.ENTITY_KIND_SOURCE_OPERATOR
                    )
                    entities.append(
                        schema.Entity(
                            id=entity_id,
                            kind=_VELA_PERFORMANCE_GROUP_KIND,
                            name=display_name,
                            child_ids=source_operator_ids,
                            placement=layer_info.placement,
                            attributes={"layer_name": layer_info.name},
                        )
                    )
                    for source_operator_id in source_operator_ids:
                        source_entity = next(
                            entity
                            for entity in entities
                            if entity.id == source_operator_id
                        )
                        if entity_id not in source_entity.parent_ids:
                            source_entity.parent_ids.append(entity_id)
            else:
                # Do not manufacture source_operator identities from ext_key here.
                # For non-TFLite inputs it is not a TFLite identity, and for
                # multi-subgraph TFLite it lacks the subgraph index needed to
                # distinguish repeated operator indexes such as operator/0.
                uses_performance_layer_entities = True
                if len(source_locations) <= 1:
                    entity_id = _performance_layer_entity_id(layer_index)
                    attributes = {"layer_name": layer_info.name}
                    if source_locations:
                        attributes["vela_source_reference"] = source_locations[0]
                    entities.append(
                        schema.Entity(
                            id=entity_id,
                            kind=_VELA_PERFORMANCE_LAYER_KIND,
                            name=display_name,
                            placement=layer_info.placement,
                            attributes=attributes,
                        )
                    )
                else:
                    entity_id = _performance_group_entity_id(layer_index)
                    child_ids = [
                        _performance_layer_entity_id(layer_index, source_index)
                        for source_index in range(len(source_locations))
                    ]
                    performance_group_child_kinds.add(_VELA_PERFORMANCE_LAYER_KIND)
                    entities.append(
                        schema.Entity(
                            id=entity_id,
                            kind=_VELA_PERFORMANCE_GROUP_KIND,
                            name=display_name,
                            child_ids=child_ids,
                            placement=layer_info.placement,
                            attributes={"layer_name": layer_info.name},
                        )
                    )
                    entities.extend(
                        schema.Entity(
                            id=child_id,
                            kind=_VELA_PERFORMANCE_LAYER_KIND,
                            name=f"{display_name} ({source_location})",
                            parent_ids=[entity_id],
                            placement=layer_info.placement,
                            attributes={
                                "layer_name": layer_info.name,
                                "vela_source_reference": source_location,
                            },
                        )
                        for child_id, source_location in zip(
                            child_ids, source_locations
                        )
                    )
            breakdowns.append(
                schema.Breakdown(
                    entity_id=entity_id,
                    metrics=breakdown_metrics,
                )
            )

        entity_kinds = []
        if uses_performance_layer_entities:
            entity_kinds.append(schema.EntityKind(id=_VELA_PERFORMANCE_LAYER_KIND))
        if performance_group_child_kinds:
            entity_kinds.append(
                schema.EntityKind(
                    id=_VELA_PERFORMANCE_GROUP_KIND,
                    child_kinds=sorted(performance_group_child_kinds),
                )
            )

        # Create result
        result = schema.Result(
            kind=schema.ResultKind.PERFORMANCE,
            status=schema.ResultStatus.OK,
            producer="vela",
            warnings=[NPU_ONLY_PERFORMANCE_WARNING],
            errors=[],
            metrics=model_metrics,
            breakdowns=breakdowns,
            entities=entities,
            entity_kinds=entity_kinds,
        )

        # Build StandardizedOutput
        output = schema.StandardizedOutput(
            schema_version=schema.SCHEMA_VERSION,
            run_id=run_id,
            timestamp=timestamp,
            tool=tool,
            target=target,
            model=model,
            context=context,
            backends=[backend],
            results=[result],
        )

        return output.to_dict()


@dataclass
class LayerPerfInfo:
    """Contains metrics from a row from the per-layer csv file from Vela."""

    name: str
    tflite_operator: str
    sram_usage: int
    op_cycles: int
    npu_cycles: int
    sram_access_cycles: int
    dram_access_cycles: int
    on_chip_flash_access_cycles: int
    off_chip_flash_access_cycles: int
    mac_count: int
    util_mac_percentage: float
    placement: str
    source_locations: list[str]


@dataclass
class LayerwisePerfInfo:
    """Contains all the per-layer metrics from the per-layer csv file from Vela."""

    layerwise_info: list[LayerPerfInfo]
    additional_layer_metrics: list[list[schema.Metric]] = field(default_factory=list)


complete_layer_metrics = [
    ("tflite_operator", ["TFLite_operator", "Original Operator"], "TFLite Operator"),
    ("nng_operator", "NNG Operator", "NNG Operator"),
    ("sram_usage", ["SRAM Usage", "Staging Usage"], "SRAM Usage"),
    ("peak_sram_usage_percentage", "Peak%", "Peak SRAM Usage (%)"),
    ("op_cycles", "Op Cycles", "OP Cycles"),
    ("op_cycles_network_percentage", "Network%", "OP Cycles in Network (%)"),
    ("npu_cycles", "NPU", "NPU Cycles"),
    ("sram_access_cycles", "SRAM AC", "SRAM AC"),
    ("dram_access_cycles", "DRAM AC", "DRAM AC"),
    ("on_chip_flash_access_cycles", "OnFlash AC", "OnFlash AC"),
    ("off_chip_flash_access_cycles", "OffFlash AC", "OffFlash AC"),
    ("mac_count", "MAC Count", "MAC Count"),
    (
        "mac_count_network_percentage",
        ["Network% (1)", "Network% (MAC)"],
        "MAC Count in Network (%)",
    ),
    ("util_mac_percentage", ["Util%", "Util% (MAC)"], "MAC Util (%)"),
    ("name", "Name", "Layer Name"),
]

OUTPUT_METRICS = [field.name for field in fields(LayerPerfInfo)]
OPTIONAL_OUTPUT_METRICS = list(_VELA_OPTIONAL_LAYER_METRIC_UNITS)

layer_metrics = [
    layer_metric
    for layer_metric in complete_layer_metrics
    if layer_metric[0] in OUTPUT_METRICS
]

layer_metrics.sort(key=lambda e: OUTPUT_METRICS.index(e[0]))

optional_layer_metrics = [
    layer_metric
    for layer_metric in complete_layer_metrics
    if layer_metric[0] in OPTIONAL_OUTPUT_METRICS
]

optional_layer_metrics.sort(key=lambda e: OPTIONAL_OUTPUT_METRICS.index(e[0]))


def extract_metrics_from_row(row_as_dict: dict, metrics: list, key_types: dict) -> dict:
    """Extract metrics from a CSV row."""
    ids_to_metrics = {}
    for key, title_options, _ in metrics:
        title_found = False
        for title in (
            title_options if isinstance(title_options, list) else [title_options]
        ):
            try:
                ids_to_metrics[key] = key_types[key](row_as_dict[title])
                title_found = True
                break
            except KeyError:
                continue
            except ValueError as err:
                if "invalid literal for int() with base 10" in str(err):
                    ids_to_metrics[key] = key_types[key](float(row_as_dict[title]))
                    title_found = True
                    break
                raise
        if not title_found:
            raise KeyError(f"Title not found for metric key: {key}")
    return ids_to_metrics


def _layer_placement(raw_target: str | None) -> str:
    """Return standardized placement from Vela's optional Target column."""
    if raw_target is None or raw_target == "":
        # Older Vela CSV formats did not include Target. Preserve compatibility
        # without claiming that those rows ran on the NPU.
        return schema.PlacementType.UNKNOWN.value
    try:
        return schema.PlacementType(raw_target.upper()).value
    except ValueError as err:
        raise ValueError(
            f"Vela per-layer CSV contains unsupported Target value: {raw_target!r}"
        ) from err


def _extract_optional_layer_metrics(row_as_dict: dict) -> list[schema.Metric]:
    """Extract optional Vela per-layer metrics when the CSV includes them."""
    metrics = []
    for key, title_options, _ in optional_layer_metrics:
        titles = title_options if isinstance(title_options, list) else [title_options]
        for title in titles:
            if title not in row_as_dict:
                continue
            if row_as_dict[title] == "":
                continue
            value = float(row_as_dict[title])
            # Optional Vela percentage columns can contain NaN when no NPU work
            # was measured. Omit those values rather than constructing a metric
            # that cannot be represented by strict JSON.
            if math.isfinite(value):
                metrics.append(
                    schema.Metric(
                        name=key,
                        value=value,
                        unit=_VELA_OPTIONAL_LAYER_METRIC_UNITS[key],
                    )
                )
            break
    return metrics


def parse_layerwise_perf_csv(
    vela_csv_file: Path,
    metrics: list,
    debug_db_path: Path | None = None,
) -> LayerwisePerfInfo:
    """Parse the per-layer csv file from backend vela."""
    if not vela_csv_file.is_file():
        raise FileNotFoundError(f"CSV File not found at {vela_csv_file}\n")
    debug_locations = (
        _debug_db_performance_source_references(debug_db_path)
        if debug_db_path is not None
        else []
    )
    debug_location_index = 0
    layerwise_info = []  # type: list[LayerPerfInfo]
    additional_layer_metrics = []  # type: list[list[schema.Metric]]
    with open(vela_csv_file, encoding="UTF-8") as csv_file:
        layerwise_reader = csv.reader(csv_file, delimiter=",")
        try:
            headers = list(next(layerwise_reader))
        except StopIteration:
            return LayerwisePerfInfo(layerwise_info=layerwise_info)
        headers_to_check_cpu_ops = headers.copy()
        multiple_header_count = Counter(headers)
        # Deal with multiple of the same values in CSV header.
        for idx, header in enumerate(reversed(headers)):
            if multiple_header_count[header] > 1:
                headers[len(headers) - idx - 1] = (
                    headers[len(headers) - idx - 1]
                    + " ("
                    + str(multiple_header_count[header] - 1)
                    + ")"
                )
                multiple_header_count[header] -= 1
        for row in layerwise_reader:
            row_as_dict = dict(zip(headers, row))
            if row == headers_to_check_cpu_ops:
                continue
            try:
                key_types = {
                    field.name: eval(field.type)  # type: ignore[arg-type]
                    for field in fields(LayerPerfInfo)
                }
                ids_to_metrics = extract_metrics_from_row(
                    row_as_dict, metrics, key_types
                )
                source_locations = []
                if debug_db_path is not None:
                    try:
                        source_locations = debug_locations[debug_location_index]
                    except IndexError as err:
                        raise ValueError(
                            "Vela debug database has fewer performance rows than "
                            "the per-layer CSV"
                        ) from err
                    debug_location_index += 1
                layer_info = LayerPerfInfo(
                    **ids_to_metrics,
                    placement=_layer_placement(row_as_dict.get("Target")),
                    source_locations=source_locations,
                )
                layer_metrics_from_row = _extract_optional_layer_metrics(row_as_dict)
                if layer_info.op_cycles < 0:
                    raise ValueError(
                        "Per-layer CSV contains negative op_cycles "
                        f"for layer {layer_info.name!r}: {layer_info.op_cycles}"
                    )
                layerwise_info.append(layer_info)
                additional_layer_metrics.append(layer_metrics_from_row)
            except KeyError as err:
                raise KeyError("Generated CSV missing expected headers") from err
    if debug_db_path is not None and debug_location_index != len(debug_locations):
        raise ValueError(
            "Vela debug database has more performance rows than the per-layer CSV"
        )
    return LayerwisePerfInfo(
        layerwise_info=layerwise_info,
        additional_layer_metrics=additional_layer_metrics,
    )


def _find_per_layer_csv(output_dir: Path, model_name: str) -> Path | None:
    """Return the model's generated per-layer CSV, if present."""
    return next(
        (
            path
            for path in sorted(output_dir.glob("*per-layer.csv"))
            if model_name in path.name
        ),
        None,
    )


def estimate_performance(
    model_path: Path, compiler_options: VelaCompilerOptions
) -> PerformanceMetrics:
    """Return performance estimations for the model/target.

    Logic for this function comes from Vela module stats_writer.py
    """
    logger.debug(
        "Estimate performance for the model %s on %s",
        model_path,
        compiler_options.accelerator_config,
    )
    from mlia.backend.vela.compiler import VelaCompiler

    vela_compiler = VelaCompiler(compiler_options)
    output_dir = Path(compiler_options.output_dir)
    model_name = model_path.stem
    summary_path = (
        output_dir / f"{model_name}_summary_{compiler_options.system_config}.csv"
    )
    debug_db_path = output_dir / f"{model_name}_debug.xml"
    cached_csv_path = _find_per_layer_csv(output_dir, model_name)
    cache_is_complete = (
        summary_path.is_file()
        and debug_db_path.is_file()
        and cached_csv_path is not None
    )
    summary_data, _ = vela_compiler.compile_model(model_path, cache_is_complete)

    csv_path = _find_per_layer_csv(output_dir, model_name)
    if csv_path is None:
        raise FileNotFoundError("Vela per-layer CSV file not found")
    layerwise_performance_info = parse_layerwise_perf_csv(
        vela_csv_file=csv_path,
        metrics=layer_metrics,
        debug_db_path=debug_db_path,
    )

    return _performance_metrics(layerwise_performance_info, summary_data)


def _performance_metrics(
    layerwise_performance_info: LayerwisePerfInfo, summary_data: VelaSummary
) -> PerformanceMetrics:
    """Return performance metrics for optimized model."""
    midpoint_fps = 0.0
    midpoint_inference_time = summary_data.cycles_total / summary_data.core_clock
    if midpoint_inference_time > 0:
        midpoint_fps = 1 / midpoint_inference_time

    return PerformanceMetrics(
        npu_cycles=int(summary_data.cycles_npu),
        sram_access_cycles=int(summary_data.cycles_sram_access),
        dram_access_cycles=int(summary_data.cycles_dram_access),
        on_chip_flash_access_cycles=int(summary_data.cycles_on_chip_flash_access),
        off_chip_flash_access_cycles=int(summary_data.cycles_off_chip_flash_access),
        total_cycles=int(summary_data.cycles_total),
        batch_inference_time=midpoint_inference_time * 1000,
        inferences_per_second=midpoint_fps,
        batch_size=summary_data.batch_size,
        sram_memory_area_size=float(summary_data.sram_memory_used),
        dram_memory_area_size=float(summary_data.dram_memory_used),
        on_chip_flash_memory_area_size=float(summary_data.on_chip_flash_memory_used),
        off_chip_flash_memory_area_size=float(summary_data.off_chip_flash_memory_used),
        layerwise_performance_info=layerwise_performance_info,
        additional_summary_metrics=_summary_metrics(summary_data),
    )


def _summary_metrics(summary_data: VelaSummary) -> list[schema.Metric]:
    """Return additional result-level metrics parsed from the Vela summary CSV."""
    metrics = []
    for field_name, unit in _VELA_SUMMARY_METRIC_UNITS.items():
        source_name = _VELA_RESULT_METRIC_ALIASES.get(field_name, field_name)
        value = getattr(summary_data, field_name, None)
        if value is None:
            continue
        metric_value = _summary_metric_value(field_name, value)
        # Vela uses NaN when a statistic cannot be calculated, for example when
        # a model has no NPU cycles. That is missing data, not a JSON number, so
        # omit backend-specific metrics and let the standard-metric helper add
        # an explicit unavailable entry where the metric has a standard name.
        if not math.isfinite(metric_value):
            continue
        metrics.append(
            schema.Metric(
                name=source_name,
                value=metric_value,
                unit=unit,
            )
        )
    return metrics
