# SPDX-FileCopyrightText: Copyright 2022-2024, 2026, Arm Limited
# and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Ethos-U data analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod

from mlia.backend.vela.compat import Operators, VelaCompatibilityResult
from mlia.core.common import DataItem
from mlia.core.data_analysis import (
    Fact,
    FactExtractor,
    LayerCompatibilityIssue,
    LayerPerformanceIssue,
    register_fact_type,
)
from mlia.target.ethos_u.common_reporters import analyze_tflite_compatibility_common
from mlia.target.ethos_u.optimization_shims import OptimizationSettings
from mlia.target.ethos_u.utils.tflite_shims import TFLiteCompatibilityInfo
from mlia.target.ethos_u.performance import (
    CombinedPerformanceResult,
    CorstonePerformanceResult,
    VelaPerformanceResult,
    OptimizationPerformanceMetrics,
)


@dataclass
class HasCPUOnlyOperators(Fact):
    """Model has CPU only operators."""

    cpu_only_ops: list[str]


@dataclass
class HasUnsupportedOnNPUOperators(Fact):
    """Model has unsupported on NPU operators."""

    npu_unsupported_ratio: float


@dataclass
class AllOperatorsSupportedOnNPU(Fact):
    """All model's operators supported on NPU."""


@register_fact_type(
    "ethos_u_layer_compatibility",
    "layer",
    "Ethos-U specific layer compatibility information",
)
@dataclass
class EthosULayerCompatibilityIssue(LayerCompatibilityIssue):
    """Ethos-U specific layer compatibility fact.

    Extends base LayerCompatibilityIssue with NPU placement information.
    """

    npu_placement: str = "unknown"  # 'npu', 'cpu', or 'unknown'

    def to_dict(self) -> dict:
        """Convert to dictionary with proper serialization."""
        result = super().to_dict()
        result["npu_placement"] = self.npu_placement
        return result


@register_fact_type(
    "ethos_u_layer_suboptimal_activation",
    "layer",
    "Layer uses suboptimal activation function on Ethos-U",
)
@dataclass
class EthosULayerSuboptimalActivation(LayerCompatibilityIssue):
    """Fact indicating a layer uses suboptimal activation function."""

    activation_type: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary with proper serialization."""
        result = super().to_dict()
        result["activation_type"] = self.activation_type
        return result


@register_fact_type(
    "ethos_u_layer_high_network_share",
    "layer",
    "Layer has a high number of op cycles compared to others",
)
@dataclass
class EthosULayerHighNetworkShare(LayerPerformanceIssue):
    """Fact indicating a layer has high network share."""


@register_fact_type(
    "ethos_u_layer_high_op_cycles",
    "layer",
    "Layer has a high number of op cycles",
)
@dataclass
class EthosULayerHighOpCycles(LayerPerformanceIssue):
    """Fact indicating a layer has high op cycles."""


@register_fact_type(
    "ethos_u_layer_high_memory_pressure",
    "layer",
    "Layer op cycles dominated by memory access cycles",
)
@dataclass
class EthosULayerHighMemoryPressure(LayerPerformanceIssue):
    """Fact indicating memory access dominates layer execution time."""

    mem_to_npu_ratio: float


@register_fact_type(
    "ethos_u_layer_low_mac_util",
    "layer",
    "Layer has lower mac util than expected",
)
@dataclass
class EthosULayerLowMacUtil(LayerPerformanceIssue):
    """Fact indicating mac util is lower than expected for that op."""

    severity: str


@dataclass
class PerfMetricDiff:
    """Performance metric difference."""

    original_value: int | float
    optimized_value: int | float

    @property
    def diff(self) -> float:
        """Difference between metrics."""
        if self.original_value == 0:
            return 0

        return 100 - ((self.optimized_value / self.original_value) * 100)

    @property
    def improved(self) -> bool:
        """Return true if metric improved."""
        return self.diff > 0

    @property
    def degraded(self) -> bool:
        """Return true if metric degraded."""
        return self.diff < 0

    @property
    def same(self) -> bool:
        """Return true if metric stays the same."""
        return self.diff == 0


@dataclass
class OptimizationDiff:
    """Optimization performance impact."""

    opt_type: list[OptimizationSettings]
    opt_diffs: dict[str, PerfMetricDiff]


@dataclass
class OptimizationResults(Fact):
    """Optimization results."""

    diffs: list[OptimizationDiff]


@dataclass
class LutPatternRule:
    """Activation funciton operator patterns."""

    op_seq: list[str]
    op_label: str


PATTERN_RULES = [
    LutPatternRule(
        ["greater", "fully_connected", "exp", "sub", "mul", "select"], "SELU"
    ),
    LutPatternRule(["greater", "mul", "exp", "sub", "mul", "select"], "SELU"),
    LutPatternRule(["exp", "add", "log", "tanh", "mul"], "MISH"),
    LutPatternRule(["exp", "add", "log"], "SOFTPLUS"),
    LutPatternRule(["logistic", "mul"], "LOGISTIC BASED OPERATION"),
    LutPatternRule(["exp"], "EXPONENTIAL"),
    LutPatternRule(["gelu"], "GELU"),
    LutPatternRule(["elu"], "ELU"),
    LutPatternRule(["tanh"], "TANH"),
    LutPatternRule(["logistic"], "SIGMOID"),
    LutPatternRule(["softmax"], "SOFTMAX"),
]


class EthosUDataAnalyzer(FactExtractor):
    """Ethos-U data analyzer."""

    @singledispatchmethod
    def analyze_data(self, data_item: DataItem) -> None:  # type: ignore
        """Analyze the data."""
        print(
            f"DEBUG: Unhandled data_item type: {type(data_item)} - "
            f"{data_item.__class__.__module__}.{data_item.__class__.__name__}"
        )

    @analyze_data.register
    def analyze_vela_compatibility(self, vela_result: VelaCompatibilityResult) -> None:
        """Analyze Vela compatibility result and extract operator information."""
        # Extract the Operators object from VelaCompatibilityResult
        self.analyze_operator_compatibility(vela_result.legacy_info)
        # Analyze activation usage patterns
        self._analyze_activation_function(vela_result.legacy_info)

    def _sequence_matches(self, ops: list, idx: int, function_ops: list[str]) -> bool:
        """Check operators sequence matches activation function operator pattern."""
        if idx + len(function_ops) > len(ops):
            return False
        for i, function_op in enumerate(function_ops):
            # Check compatibility operators sequentially against activation function
            if not ops[idx + i].op_type.lower() == function_op:
                return False
        return True

    def _analyze_activation_function(self, operators: Operators) -> None:
        """Analyze operators from compatibility check for activation functions."""
        ops = operators.ops
        idx = 0
        while idx < len(ops):
            # Check if activation function pattern occers
            for rule in PATTERN_RULES:
                if self._sequence_matches(ops, idx, rule.op_seq):
                    operator = ops[idx]
                    lut_fact = EthosULayerSuboptimalActivation(
                        operator_name=operator.name,
                        location=f"operator/{idx}",
                        operator_type=operator.op_type,
                        is_supported=operator.run_on_npu.supported,
                        reasons=operator.run_on_npu.reasons,
                        activation_type=rule.op_label,
                    )
                    self.add_fact(lut_fact)
                    # Skip to next operation not in detected pattern
                    idx += len(rule.op_seq)
                    break
            else:
                idx += 1

    @analyze_data.register
    def analyze_operator_compatibility(self, operators: Operators) -> None:
        """Analyse operator compatibility information."""
        for idx, operator in enumerate(operators.ops):
            # Determine NPU placement
            if operator.cpu_only:
                npu_placement = "cpu"
            elif operator.run_on_npu.supported:
                npu_placement = "npu"
            else:
                npu_placement = "unknown"

            # Create layer compatibility fact
            layer_fact = EthosULayerCompatibilityIssue(
                operator_name=operator.name,
                location=f"operator/{idx}",
                operator_type=operator.op_type,
                is_supported=operator.run_on_npu.supported,
                reasons=operator.run_on_npu.reasons,
                npu_placement=npu_placement,
            )
            self.add_fact(layer_fact)

        # Keep network-level facts for backward compatibility
        cpu_only = [operator.op_type for operator in operators.ops if operator.cpu_only]
        if cpu_only:
            self.add_fact(HasCPUOnlyOperators(cpu_only))

        if operators.npu_unsupported_ratio != 0:
            self.add_fact(HasUnsupportedOnNPUOperators(operators.npu_unsupported_ratio))

        if operators.npu_unsupported_ratio == 0:
            self.add_fact(AllOperatorsSupportedOnNPU())

    @analyze_data.register
    def analyze_optimization_results(
        self, optimization_results: OptimizationPerformanceMetrics
    ) -> None:
        """Analyse optimization performance metrics."""
        optimizations = optimization_results.optimizations_perf_metrics
        if not optimizations:
            return

        orig = optimization_results.original_perf_metrics
        orig_memory = orig.memory_usage
        orig_cycles = orig.npu_cycles

        diffs: list[OptimizationDiff] = []
        for opt_type, opt_perf_metrics in optimizations:
            opt = opt_perf_metrics
            opt_memory = opt.memory_usage
            opt_cycles = opt.npu_cycles

            opt_diffs: dict[str, PerfMetricDiff] = {}

            if orig_memory and opt_memory:
                opt_diffs.update(
                    {
                        "sram": PerfMetricDiff(
                            orig_memory.sram_memory_area_size,
                            opt_memory.sram_memory_area_size,
                        ),
                        "dram": PerfMetricDiff(
                            orig_memory.dram_memory_area_size,
                            opt_memory.dram_memory_area_size,
                        ),
                        "on_chip_flash": PerfMetricDiff(
                            orig_memory.on_chip_flash_memory_area_size,
                            opt_memory.on_chip_flash_memory_area_size,
                        ),
                        "off_chip_flash": PerfMetricDiff(
                            orig_memory.off_chip_flash_memory_area_size,
                            opt_memory.off_chip_flash_memory_area_size,
                        ),
                    }
                )
            if orig_cycles and opt_cycles:
                opt_diffs["npu_total_cycles"] = PerfMetricDiff(
                    orig_cycles.npu_total_cycles,
                    opt_cycles.npu_total_cycles,
                )

            diff = OptimizationDiff(opt_type=opt_type, opt_diffs=opt_diffs)
            diffs.append(diff)

        self.add_fact(OptimizationResults(diffs))

    @analyze_data.register
    def analyze_tflite_compatibility(self, data_item: TFLiteCompatibilityInfo) -> None:
        """Analyze TensorFlow Lite compatibility information."""
        analyze_tflite_compatibility_common(self, data_item)

    @analyze_data.register
    def analyze_combined_performance_result(
        self, data_item: CombinedPerformanceResult
    ) -> None:
        """Analyze combined performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            print(
                f"DEBUG: No standardized output for: {type(data_item)} - "
                f"{data_item.__class__.__module__}.{data_item.__class__.__name__}"
            )
            return
        corstone_result = 1
        breakdowns = standard_out["results"][corstone_result]["breakdowns"]
        self.analyze_network_share(breakdowns)
        self.analyze_memory_pressure(breakdowns)
        self.analyze_mac_util(breakdowns)

    @analyze_data.register
    def analyze_corstone_performance_result(
        self, data_item: CorstonePerformanceResult
    ) -> None:
        """Analyze Corstone-only performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            print(
                f"DEBUG: No standardized output for: {type(data_item)} - "
                f"{data_item.__class__.__module__}.{data_item.__class__.__name__}"
            )
            return
        corstone_result = 0
        breakdowns = standard_out["results"][corstone_result]["breakdowns"]
        self.analyze_network_share(breakdowns)
        self.analyze_memory_pressure(breakdowns)
        self.analyze_mac_util(breakdowns)

    @analyze_data.register
    def analyze_vela_performance_result(self, data_item: VelaPerformanceResult) -> None:
        """Analyze combined performance results."""
        standard_out = getattr(data_item, "standardized_output", None)
        if not standard_out or not isinstance(standard_out, dict):
            print(
                f"DEBUG: No standardized output for: {type(data_item)} - "
                f"{data_item.__class__.__module__}.{data_item.__class__.__name__}"
            )
            return
        vela_result = 0
        breakdowns = standard_out["results"][vela_result]["breakdowns"]
        self.analyze_op_cycles(breakdowns)
        self.analyze_memory_pressure(breakdowns)
        self.analyze_mac_util(breakdowns)

    def get_metric_tup(
        self, layer: dict, metric_names: list[str]
    ) -> tuple[str, float, str] | None:
        """Return the first matching metric as (name, float(value), unit)."""
        metrics = layer["metrics"]
        for metric in metrics:
            name = metric["name"]
            if name in metric_names:
                try:
                    return (metric["name"], float(metric["value"]), metric["unit"])
                except (TypeError, ValueError):
                    return None
        return None

    METRIC_ALIASES = {
        "op_cycles": ["op_cycles"],
        "npu_cycles": ["npu_cycles", "npu"],
        "sram_cycles": ["sram_access_cycles", "sram_ac"],
        "dram_cycles": ["dram_access_cycles", "dram_ac"],
        "on_flash_cycles": ["on_chip_flash_access_cycles", "onflash_ac"],
        "off_flash_cycles": ["off_chip_flash_access_cycles", "offflash_ac"],
        "mac_count": ["mac_count"],
        "mac_util": ["util_mac_percentage", "util_mac"],
        "sram_usage": ["sram_usage", "staging_usage"],
    }
    RELATIVE_FLOOR = 0.10
    MAX_ITEMS = 10

    def analyze_network_share(self, breakdowns: list) -> None:
        """Detect layers with largest network share."""
        relative_floor = self.RELATIVE_FLOOR
        max_items = self.MAX_ITEMS

        # Collect (network_share_pct, layer) pairs for easy sorting
        layers_scored = []
        for layer in breakdowns:
            try:
                network_metric = 3
                network_pct = float(layer["metrics"][network_metric]["value"])
                layers_scored.append((network_pct, layer))
            except (KeyError, IndexError, TypeError, ValueError):
                continue

        if not layers_scored:
            return

        value = 0
        layers_scored.sort(key=lambda x: x[value], reverse=True)

        # Determine threshold relative to highest share
        highest_op = 0
        max_pct = layers_scored[highest_op][value]
        threshold = max_pct * relative_floor

        selected = [(pct, layer) for pct, layer in layers_scored if pct >= threshold]
        selected = selected[:max_items]

        for pct, layer in selected:
            layer_fact = EthosULayerHighNetworkShare(
                operator_name=layer["name"],
                location=layer["location"],
                metric="network_share",
                metric_value=pct,
                metric_unit="%",
            )
            self.add_fact(layer_fact)

    def analyze_op_cycles(self, breakdowns: list) -> None:
        """Detect the top 10 layers with the highest op cycles."""
        max_items = self.MAX_ITEMS

        layers_scored = []
        for layer in breakdowns:
            metric = self.get_metric_tup(layer, self.METRIC_ALIASES["op_cycles"])
            if metric is None:
                continue

            metric_name, op_cycles, metric_unit = metric
            layers_scored.append((op_cycles, metric_name, metric_unit, layer))

        if not layers_scored:
            return

        # sort by op_cycles, high to low
        value = 0
        layers_scored.sort(key=lambda item: item[value], reverse=True)

        # only emit fact for the the top MAX_ITEMS
        for op_cycles, metric_name, metric_unit, layer in layers_scored[:max_items]:
            layer_fact = EthosULayerHighOpCycles(
                operator_name=layer["name"],
                location=layer["location"],
                metric=metric_name,
                metric_value=op_cycles,
                metric_unit=metric_unit,
            )
            self.add_fact(layer_fact)

    def analyze_memory_pressure(self, breakdowns: list) -> None:
        """Detect if op cycles is memory dominated."""
        for layer in breakdowns:
            try:
                npu_metric = self.get_metric_tup(
                    layer, self.METRIC_ALIASES["npu_cycles"]
                )
                if npu_metric is None:
                    continue

                npu_cycles = npu_metric[1]
                # get sram_ac, dram_ac, onfalsh_ac, offflash_ac
                candidates = [
                    self.get_metric_tup(layer, self.METRIC_ALIASES["sram_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["dram_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["on_flash_cycles"]),
                    self.get_metric_tup(layer, self.METRIC_ALIASES["off_flash_cycles"]),
                ]
                valid_candidates: list[tuple[str, float, str]] = [
                    candidate for candidate in candidates if candidate is not None
                ]
                if not valid_candidates:
                    continue

                max_name, max_val, max_unit = max(valid_candidates, key=lambda t: t[1])

                if npu_cycles and (max_val / npu_cycles) > 1:
                    layer_fact = EthosULayerHighMemoryPressure(
                        operator_name=layer["name"],
                        location=layer["location"],
                        metric=max_name,
                        metric_value=max_val,
                        metric_unit=max_unit,
                        mem_to_npu_ratio=max_val / npu_cycles,
                    )
                    self.add_fact(layer_fact)

            except (KeyError, IndexError, TypeError, ValueError):
                continue

    def analyze_mac_util(self, breakdowns: list) -> None:
        """Detect if mac util is lower then expected."""
        for layer in breakdowns:
            mac_util_metric = self.get_metric_tup(
                layer, self.METRIC_ALIASES["mac_util"]
            )
            if mac_util_metric is None:
                continue

            severity = None

            if layer["name"] == "Conv2D" and mac_util_metric[1] < 30:
                severity = "very low"
            elif layer["name"] == "Conv2D" and mac_util_metric[1] < 55:
                severity = "low"
            elif layer["name"] == "DepthwiseConv2D" and mac_util_metric[1] < 3:
                severity = "low"

            metric, value, unit = mac_util_metric

            if severity is not None:
                layer_fact = EthosULayerLowMacUtil(
                    operator_name=layer["name"],
                    location=layer["location"],
                    metric=metric,
                    metric_value=value,
                    metric_unit=unit,
                    severity=severity,
                )
                self.add_fact(layer_fact)
