# SPDX-FileCopyrightText: Copyright 2022-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Vela compiler wrapper module."""

from __future__ import annotations

import csv
import logging
import re
import sys
from dataclasses import dataclass, fields
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from mlia.backend.errors import BackendUnavailableError
from mlia.transformers.error import TransformerNotFoundError
from mlia.transformers.registry import (
    TransformRequest,
    transform_model,
)
from mlia.utils.filesystem import get_vela_config
from mlia.utils.logging import redirect_output, redirect_raw_output

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ethosu.vela.nn_graph import Graph, NetworkType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VelaCompilerDeps:
    """Resolved Vela compiler dependencies."""

    ModelReaderOptions: Any
    read_model: Any
    Graph: Any
    NetworkType: Any
    CustomType: Any
    main: Any


_VELA_DEPS_CACHE: VelaCompilerDeps | None = None


def _load_vela_deps() -> VelaCompilerDeps:
    """Load Vela modules on demand."""
    try:
        import importlib

        importlib.invalidate_caches()
        from ethosu.vela.model_reader import ModelReaderOptions, read_model
        from ethosu.vela.nn_graph import Graph, NetworkType
        from ethosu.vela.operation import CustomType
        from ethosu.vela.vela import main
    except ImportError as exc:
        raise BackendUnavailableError("Backend vela is not available", "vela") from exc

    return VelaCompilerDeps(
        ModelReaderOptions=ModelReaderOptions,
        read_model=read_model,
        Graph=Graph,
        NetworkType=NetworkType,
        CustomType=CustomType,
        main=main,
    )


def _get_vela_deps() -> VelaCompilerDeps:
    """Return cached Vela deps or load them."""
    global _VELA_DEPS_CACHE

    if _VELA_DEPS_CACHE is None:
        _VELA_DEPS_CACHE = _load_vela_deps()
    return _VELA_DEPS_CACHE


# File extensions that Vela natively supports
_VELA_SUPPORTED_FILE_EXTENSIONS = [".tflite", ".tosa", ".tosamlir"]

# TOSA format file extensions (TOSA outputs to raw .npz format after compilation)
_TOSA_FILE_FORMAT_EXTENSIONS = [".tosa", ".tosamlir"]


@dataclass
class VelaInitMemoryData:
    """Memory Data from vela.ini."""

    clock_scale: float | None
    burst_length: int | None
    read_latency: int | None
    write_latency: int | None


@dataclass
class VelaInitData:
    """Data gathered from the vela.ini file we provide to vela."""

    system_config: str
    core_clock: float
    axi0_port: str
    axi1_port: str
    sram_memory_data: VelaInitMemoryData
    dram_memory_data: VelaInitMemoryData
    off_chip_flash_memory_data: VelaInitMemoryData
    on_chip_flash_memory_data: VelaInitMemoryData
    memory_mode: str
    const_mem_area: str
    arena_mem_area: str
    cache_mem_area: str
    arena_cache_size: int | None


@dataclass
class VelaSummary:
    """Data gathered from the summary CSV file that Vela produces."""

    cycles_total: float
    cycles_npu: float
    cycles_sram_access: float
    cycles_dram_access: float
    cycles_on_chip_flash_access: float
    cycles_off_chip_flash_access: float
    core_clock: float
    dram_memory_used: float
    sram_memory_used: float
    on_chip_flash_memory_used: float
    off_chip_flash_memory_used: float
    batch_size: int
    memory_mode: str
    system_config: str
    accelerator_configuration: str
    arena_cache_size: float
    sram_bandwidth: float | None = None
    dram_bandwidth: float | None = None
    on_chip_flash_bandwidth: float | None = None
    off_chip_flash_bandwidth: float | None = None
    inferences_per_second: float | None = None
    inference_time: float | None = None
    passes_before_fusing: float | None = None
    passes_after_fusing: float | None = None
    total_original_weights: float | None = None
    total_npu_encoded_weights: float | None = None
    sram_feature_map_read_bytes: float | None = None
    sram_feature_map_write_bytes: float | None = None
    sram_weight_read_bytes: float | None = None
    sram_weight_write_bytes: float | None = None
    sram_total_bytes: float | None = None
    dram_feature_map_read_bytes: float | None = None
    dram_feature_map_write_bytes: float | None = None
    dram_weight_read_bytes: float | None = None
    dram_weight_write_bytes: float | None = None
    dram_total_bytes: float | None = None
    on_chip_flash_feature_map_read_bytes: float | None = None
    on_chip_flash_feature_map_write_bytes: float | None = None
    on_chip_flash_weight_read_bytes: float | None = None
    on_chip_flash_weight_write_bytes: float | None = None
    on_chip_flash_total_bytes: float | None = None
    off_chip_flash_feature_map_read_bytes: float | None = None
    off_chip_flash_feature_map_write_bytes: float | None = None
    off_chip_flash_weight_read_bytes: float | None = None
    off_chip_flash_weight_write_bytes: float | None = None
    off_chip_flash_total_bytes: float | None = None
    nn_macs: float | None = None
    nn_tops: float | None = None

    def __repr__(self) -> str:
        """Return String Representation of VelaSummary object."""
        header_values = dict(summary_metrics)
        string_to_check = ""
        for field in fields(self):
            string_to_check += (
                f"{header_values[field.name]}: {getattr(self, field.name)}, "
            )
        return string_to_check


complete_summary_metrics = [
    ("experiment", "experiment"),
    ("network", "network"),
    ("accelerator_configuration", "accelerator_configuration"),
    ("system_config", "system_config"),
    ("memory_mode", "memory_mode"),
    ("core_clock", "core_clock"),
    ("arena_cache_size", "arena_cache_size"),
    ("sram_bandwidth", "sram_bandwidth"),
    ("dram_bandwidth", "dram_bandwidth"),
    ("on_chip_flash_bandwidth", "on_chip_flash_bandwidth"),
    ("off_chip_flash_bandwidth", "off_chip_flash_bandwidth"),
    ("weights_storage_area", "weights_storage_area"),
    ("feature_map_storage_area", "feature_map_storage_area"),
    ("inferences_per_second", "inferences_per_second"),
    ("batch_size", "batch_size"),
    ("inference_time", "inference_time"),
    ("passes_before_fusing", "passes_before_fusing"),
    ("passes_after_fusing", "passes_after_fusing"),
    ("sram_memory_used", "sram_memory_used"),
    ("dram_memory_used", "dram_memory_used"),
    (
        "on_chip_flash_memory_used",
        "on_chip_flash_memory_used",
    ),
    ("off_chip_flash_memory_used", "off_chip_flash_memory_used"),
    ("total_original_weights", "total_original_weights"),
    ("total_npu_encoded_weights", "total_npu_encoded_weights"),
    ("sram_feature_map_read_bytes", "sram_feature_map_read_bytes"),
    ("sram_feature_map_write_bytes", "sram_feature_map_write_bytes"),
    ("sram_weight_read_bytes", "sram_weight_read_bytes"),
    ("sram_weight_write_bytes", "sram_weight_write_bytes"),
    ("sram_total_bytes", "sram_total_bytes"),
    ("dram_feature_map_read_bytes", "dram_feature_map_read_bytes"),
    ("dram_feature_map_write_bytes", "dram_feature_map_write_bytes"),
    ("dram_weight_read_bytes", "dram_weight_read_bytes"),
    ("dram_weight_write_bytes", "dram_weight_write_bytes"),
    ("dram_total_bytes", "dram_total_bytes"),
    (
        "on_chip_flash_feature_map_read_bytes",
        "on_chip_flash_feature_map_read_bytes",
    ),
    ("on_chip_flash_feature_map_write_bytes", "on_chip_flash_feature_map_write_bytes"),
    ("on_chip_flash_weight_read_bytes", "on_chip_flash_weight_read_bytes"),
    ("on_chip_flash_weight_write_bytes", "on_chip_flash_weight_write_bytes"),
    ("on_chip_flash_total_bytes", "on_chip_flash_total_bytes"),
    ("off_chip_flash_feature_map_read_bytes", "off_chip_flash_feature_map_read_bytes"),
    (
        "off_chip_flash_feature_map_write_bytes",
        "off_chip_flash_feature_map_write_bytes",
    ),
    ("off_chip_flash_weight_read_bytes", "off_chip_flash_weight_read_bytes"),
    ("off_chip_flash_weight_write_bytes", "off_chip_flash_weight_write_bytes"),
    ("off_chip_flash_total_bytes", "off_chip_flash_total_bytes"),
    ("nn_macs", "nn_macs"),
    ("nn_tops", "nn_tops"),
    ("cycles_npu", "cycles_npu"),
    ("cycles_sram_access", "cycles_sram_access"),
    ("cycles_dram_access", "cycles_dram_access"),
    ("cycles_on_chip_flash_access", "cycles_on_chip_flash_access"),
    ("cycles_off_chip_flash_access", "cycles_off_chip_flash_access"),
    ("cycles_total", "cycles_total"),
]

OUTPUT_METRICS = [field.name for field in fields(VelaSummary)]

summary_metrics = [
    summary_metric
    for summary_metric in complete_summary_metrics
    if summary_metric[0] in OUTPUT_METRICS
]
summary_metrics.sort(key=lambda e: OUTPUT_METRICS.index(e[0]))

# Some Vela summary fields are optional in the parsed model because backend
# output can omit them or leave them blank. Required fields still raise an
# error so unexpected CSV changes are noticed.
_OPTIONAL_SUMMARY_METRICS = {
    field.name for field in fields(VelaSummary) if "|" in str(field.type)
}
_SUMMARY_FIELD_TYPES = {field.name: str(field.type) for field in fields(VelaSummary)}
_SUMMARY_VALUE_CONVERTERS = {"float": float, "int": int, "str": str}


def _convert_summary_value(field_name: str, raw_value: str) -> int | float | str | None:
    """Convert a Vela summary CSV value for the matching dataclass field."""
    if raw_value == "" and field_name in _OPTIONAL_SUMMARY_METRICS:
        return None

    type_name = _SUMMARY_FIELD_TYPES[field_name].split("|", maxsplit=1)[0].strip()
    return _SUMMARY_VALUE_CONVERTERS[type_name](raw_value)


@dataclass
class Model:
    """Model metadata."""

    # Use string annotations to avoid import errors when ethosu.vela is not available
    nng: Graph
    network_type: NetworkType

    @property
    def optimized(self) -> bool:
        """Return true if model is already optimized."""
        deps = _get_vela_deps()
        return any(
            op.attrs.get("custom_type") == deps.CustomType.ExistingNpuOp
            for sg in self.nng.subgraphs
            for op in sg.get_all_ops()
        )


AcceleratorConfigType = Literal[
    "ethos-u55-32",
    "ethos-u55-64",
    "ethos-u55-128",
    "ethos-u55-256",
    "ethos-u65-256",
    "ethos-u65-512",
    "ethos-u85-128",
    "ethos-u85-256",
    "ethos-u85-512",
    "ethos-u85-1024",
    "ethos-u85-2048",
]

TensorAllocatorType = Literal["LinearAlloc", "Greedy", "HillClimb"]

OptimizationStrategyType = Literal["Performance", "Size"]


@dataclass
class VelaCompilerOptions:
    """Vela compiler options."""

    config_file: str | None = None
    system_config: str = "internal-default"
    memory_mode: str = "internal-default"
    accelerator_config: AcceleratorConfigType | None = None
    max_block_dependency: int = 3
    arena_cache_size: int | None = None
    tensor_allocator: TensorAllocatorType = "HillClimb"
    cpu_tensor_alignment: int = 16
    optimization_strategy: OptimizationStrategyType = "Performance"
    output_dir: Path = Path("output")
    recursion_limit: int = 1000
    verbose_performance: bool = True


class VelaCompiler:
    """Vela compiler wrapper."""

    def __init__(self, compiler_options: VelaCompilerOptions):
        """Init Vela wrapper instance."""
        self.config_file = compiler_options.config_file
        self.system_config = compiler_options.system_config
        self.memory_mode = compiler_options.memory_mode
        self.accelerator_config = compiler_options.accelerator_config
        self.max_block_dependency = compiler_options.max_block_dependency
        self.arena_cache_size = compiler_options.arena_cache_size
        self.tensor_allocator = compiler_options.tensor_allocator
        self.cpu_tensor_alignment = compiler_options.cpu_tensor_alignment
        self.optimization_strategy = compiler_options.optimization_strategy
        self.output_dir = Path(compiler_options.output_dir)
        self.recursion_limit = compiler_options.recursion_limit
        self.verbose_performance = compiler_options.verbose_performance

        sys.setrecursionlimit(self.recursion_limit)

    def _empty_summary(self) -> VelaSummary:
        """Build a fallback summary when Vela summary CSV is unavailable."""
        return VelaSummary(
            cycles_total=0.0,
            cycles_npu=0.0,
            cycles_sram_access=0.0,
            cycles_dram_access=0.0,
            cycles_on_chip_flash_access=0.0,
            cycles_off_chip_flash_access=0.0,
            core_clock=0.0,
            dram_memory_used=0.0,
            sram_memory_used=0.0,
            on_chip_flash_memory_used=0.0,
            off_chip_flash_memory_used=0.0,
            batch_size=1,
            memory_mode=str(self.memory_mode),
            system_config=str(self.system_config),
            accelerator_configuration=str(self.accelerator_config),
            arena_cache_size=float(self.arena_cache_size or 0.0),
        )

    def _preprocess_model(self, model_path: Path) -> Path:
        """Preprocess model file to supported format if needed.

        Vela natively supports TFLite (.tflite) and TOSA (.tosa, .tosamlir) files.
        PyTorch (.pt2) files are converted to TOSA format first.
        """
        if model_path.suffix.lower() in _VELA_SUPPORTED_FILE_EXTENSIONS:
            return model_path

        request = TransformRequest(
            model=model_path,
            target_format="tosa",
            output_dir=self.output_dir,
            transform_options={},
        )

        try:
            return transform_model(request)
        except TransformerNotFoundError as err:
            if model_path.suffix.lower() == ".pt2":
                err.args = (
                    f"{err}\n"
                    "PyTorch to TOSA conversion could not be resolved. "
                    "Try installing the 'mlia-converters-pytorch' plugin.",
                )
                raise
            raise

    def compile_model(
        self, model_path: Path, already_compiled: bool = False
    ) -> tuple[VelaSummary, Path]:
        """Compile the model.

        Supports TFLite (.tflite), TOSA (.tosa), and PyTorch (.pt2) files.
        PyTorch files are automatically converted to TOSA before compilation.
        """
        deps = _get_vela_deps()
        processed_model_path = self._preprocess_model(model_path)

        if not processed_model_path.is_file():
            raise RuntimeError(
                f"Unable to read model {processed_model_path} (original: {model_path})"
            )

        try:
            with redirect_raw_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                tmp = sys.stdout
                output_message = StringIO()
                sys.stdout = output_message
                try:
                    is_tosa_input = (
                        processed_model_path.suffix.lower()
                        in _TOSA_FILE_FORMAT_EXTENSIONS
                    )
                    output_format = "raw" if is_tosa_input else "tflite"
                    output_extension = "_vela.npz" if is_tosa_input else "_vela.tflite"
                    main_args = [
                        "--output-dir",
                        str(self.output_dir.as_posix()),
                        "--tensor-allocator",
                        str(self.tensor_allocator),
                        "--cpu-tensor-alignment",
                        str(self.cpu_tensor_alignment),
                        "--accelerator-config",
                        str(self.accelerator_config),
                        "--system-config",
                        str(self.system_config),
                        "--memory-mode",
                        str(self.memory_mode),
                        "--max-block-dependency",
                        str(self.max_block_dependency),
                        "--optimise",
                        str(self.optimization_strategy),
                        "--output-format",
                        output_format,
                        processed_model_path.as_posix(),
                        "--debug-force-regor",
                    ]
                    if self.config_file:
                        main_args.extend(["--config", str(self.config_file)])
                    if self.verbose_performance:
                        main_args.append("--verbose-performance")
                    if not already_compiled:
                        deps.main(main_args)
                    optimized_model_path = Path(
                        self.output_dir.as_posix()
                        + "/"
                        + processed_model_path.stem
                        + output_extension
                    )
                finally:
                    sys.stdout = tmp
                if (
                    "Warning: SRAM target for arena memory area exceeded."
                    in output_message.getvalue()
                ):
                    raise MemoryError("Model is too large and uses too much RAM")
            summary_csv_path = Path(
                self.output_dir.as_posix()
                + "/"
                + processed_model_path.stem
                + "_summary_"
                + self.system_config
                + ".csv"
            )
            if not summary_csv_path.is_file():
                summary_candidates = sorted(
                    self.output_dir.glob(f"{processed_model_path.stem}_summary_*.csv")
                )
                if summary_candidates:
                    summary_csv_path = summary_candidates[0]

            if summary_csv_path.is_file():
                summary_data = parse_summary_csv_file(summary_csv_path)
            else:
                logger.debug(
                    "Vela summary CSV not found for model '%s', using empty summary.",
                    processed_model_path,
                )
                summary_data = self._empty_summary()
            return summary_data, optimized_model_path
        except MemoryError as err:
            raise err
        except (SystemExit, Exception) as err:
            output_text = output_message.getvalue()
            # Check for various forms of invalid model errors
            if isinstance(err, FileNotFoundError) or (
                isinstance(err, SystemExit)
                and (
                    "Error: Invalid tflite file." in output_text
                    or "Error: Invalid TFLite file."
                    in output_text  # Case-sensitive fix
                    or "struct.error" in output_text
                    or "parsing" in output_text
                )
            ):
                raise RuntimeError(
                    f"Unable to read model {processed_model_path} "
                    f"(original: {model_path})"
                ) from err
            raise RuntimeError(
                "Model could not be optimized with Vela compiler."
            ) from err

    @staticmethod
    def _read_model(model: str | Path) -> tuple[Graph, NetworkType]:
        """Read TensorFlow Lite model."""
        model_path = str(model) if isinstance(model, Path) else model
        deps = _get_vela_deps()
        try:
            with redirect_output(
                logger, stdout_level=logging.DEBUG, stderr_level=logging.DEBUG
            ):
                return deps.read_model(model_path, deps.ModelReaderOptions())
        except (SystemExit, Exception) as err:
            raise RuntimeError(f"Unable to read model {model_path}.") from err


def resolve_compiler_config(
    vela_compiler_options: VelaCompilerOptions,
) -> VelaInitData:
    """Resolve passed compiler options.

    Vela has number of configuration parameters that being
    resolved during passing compiler options. E.g. Vela
    reads configuration parameters from vela.ini and fills
    it's internal structures with resolved values (memory mode,
    system mode, etc.).

    In order to get this information we need to create
    instance of the Vela compiler first.
    """
    config_file = vela_compiler_options.config_file or get_vela_config()
    return parse_vela_initialisation_file(
        Path(config_file),
        vela_compiler_options.system_config,
        vela_compiler_options.memory_mode,
    )


def compile_model(model_path: Path, compiler_options: VelaCompilerOptions) -> Path:
    """Compile model."""
    _get_vela_deps()

    vela_compiler = VelaCompiler(compiler_options)
    # output dir could be a path or str, cast to Path object
    output_dir = Path(compiler_options.output_dir)
    if Path(
        output_dir.as_posix()
        + "/"
        + model_path.stem
        + "_summary_"
        + compiler_options.system_config
        + ".csv"
    ).is_file():
        _, optimized_model_path = vela_compiler.compile_model(model_path, True)
    else:
        _, optimized_model_path = vela_compiler.compile_model(model_path)
    return optimized_model_path


def parse_summary_csv_file(vela_summary_csv_file: Path) -> VelaSummary:
    """Parse the summary csv file from Vela."""
    if not vela_summary_csv_file.is_file():
        raise FileNotFoundError(f"CSV File not found at {vela_summary_csv_file}")

    with open(vela_summary_csv_file, encoding="UTF-8") as csv_file:
        summary_reader = csv.DictReader(csv_file, delimiter=",")
        try:
            row = next(summary_reader)
        except StopIteration as err:
            raise RuntimeError("Generated Vela Summary CSV is empty") from err
        try:
            summary_values: dict[str, Any] = {}
            for key, title in summary_metrics:
                if title in row:
                    summary_values[key] = _convert_summary_value(key, row[title])
                elif key in _OPTIONAL_SUMMARY_METRICS:
                    summary_values[key] = None
                else:
                    raise KeyError(title)
            summary_data = VelaSummary(**summary_values)
        except KeyError as err:
            raise KeyError(
                f"Generated Vela Summary CSV missing expected header: {err.args[0]}."
            ) from err
    return summary_data


def parse_vela_initialisation_file(
    vela_init_file: Path, system_config: str, memory_mode: str
) -> VelaInitData:
    """Parse the vela.ini to retrieve data for the target information table."""
    if not vela_init_file.is_file():
        raise FileNotFoundError(
            f"Vela Initialisation File not found at {vela_init_file}"
        )

    lines = []
    with open(vela_init_file, encoding="UTF-8") as init_file:
        lines = init_file.readlines()

    if len(lines) == 0:
        raise OSError("vela.ini File Is Empty")

    lines = [line.strip("\n][ ") for line in lines]

    idxs_memory_mode = [
        idx for idx, item in enumerate(lines) if re.search("^Memory_Mode.*", item)
    ]

    if len(idxs_memory_mode) == 0:
        raise IndexError("No memory modes are present in vela.ini file.")

    idxs_system_config = [
        idx for idx, item in enumerate(lines) if re.search("^System_Config.*", item)
    ] + [idxs_memory_mode[0]]

    if len(idxs_system_config) <= 1:
        raise IndexError("No system configs are present in vela.ini file.")

    try:
        idx_config = lines.index("System_Config." + system_config)
    except ValueError as err:
        raise ValueError(
            f"System Config: {system_config} not present in vela.ini file."
        ) from err

    lines_to_probe = lines[
        idx_config : idxs_system_config[  # noqa: E203
            idxs_system_config.index(idx_config) + 1
        ]
    ]

    def collect_memory_mode_lines(memory_mode: str) -> list[str]:
        try:
            idx_memory_mode = lines.index("Memory_Mode." + memory_mode)
        except ValueError as err:
            raise ValueError(
                f"Memory Mode: {memory_mode} not present in vela.ini file."
            ) from err
        if idxs_memory_mode.index(idx_memory_mode) == len(idxs_memory_mode) - 1:
            lines_to_probe = lines[idx_memory_mode:]
        else:
            lines_to_probe = lines[
                idx_memory_mode : idxs_memory_mode[  # noqa: E203
                    idxs_memory_mode.index(idx_memory_mode) + 1
                ]
            ]
        return lines_to_probe

    lines_to_probe_memory_mode = collect_memory_mode_lines(memory_mode)
    extra_memory_mode_lines = []
    for line in lines_to_probe_memory_mode:
        if "inherit=Memory_Mode." in line:
            extra_memory_mode = line[line.rindex(".") + 1 :]  # noqa: E203
            extra_memory_mode_lines = collect_memory_mode_lines(extra_memory_mode)

    lines_to_probe += extra_memory_mode_lines + lines_to_probe_memory_mode

    init_dict = {}
    for line in lines_to_probe:
        if "=" in line:
            init_dict[line[: line.index("=")]] = line[
                line.index("=") + 1 :  # noqa: E203
            ]
    try:
        init_data = VelaInitData(
            system_config=system_config,
            core_clock=float(init_dict["core_clock"]),
            axi0_port=str(init_dict["axi0_port"]),
            axi1_port=str(init_dict["axi1_port"]),
            memory_mode=memory_mode,
            sram_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["Sram_clock_scale"])
                if "Sram_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["Sram_burst_length"])
                if "Sram_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["Sram_read_latency"])
                if "Sram_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["Sram_write_latency"])
                if "Sram_write_latency" in init_dict
                else None,
            ),
            dram_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["Dram_clock_scale"])
                if "Dram_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["Dram_burst_length"])
                if "Dram_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["Dram_read_latency"])
                if "Dram_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["Dram_write_latency"])
                if "Dram_write_latency" in init_dict
                else None,
            ),
            off_chip_flash_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["OffChipFlash_clock_scale"])
                if "OffChipFlash_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["OffChipFlash_burst_length"])
                if "OffChipFlash_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["OffChipFlash_read_latency"])
                if "OffChipFlash_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["OffChipFlash_write_latency"])
                if "OffChipFlash_write_latency" in init_dict
                else None,
            ),
            on_chip_flash_memory_data=VelaInitMemoryData(
                clock_scale=float(init_dict["OnChipFlash_clock_scale"])
                if "OnChipFlash_clock_scale" in init_dict
                else None,
                burst_length=int(init_dict["OnChipFlash_burst_length"])
                if "OnChipFlash_burst_length" in init_dict
                else None,
                read_latency=int(init_dict["OnChipFlash_read_latency"])
                if "OnChipFlash_read_latency" in init_dict
                else None,
                write_latency=int(init_dict["OnChipFlash_write_latency"])
                if "OnChipFlash_write_latency" in init_dict
                else None,
            ),
            const_mem_area=str(init_dict["const_mem_area"]),
            arena_mem_area=str(init_dict["arena_mem_area"]),
            cache_mem_area=str(init_dict["cache_mem_area"]),
            arena_cache_size=int(init_dict["arena_cache_size"])
            if "arena_cache_size" in init_dict
            else None,
        )

    except KeyError as err:
        raise KeyError(f"Vela.ini file missing expected header: {err.args[0]}") from err

    return init_data
