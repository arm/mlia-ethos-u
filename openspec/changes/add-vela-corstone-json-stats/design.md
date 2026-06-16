## Context

MLIA-1667 asks for Vela and Corstone performance JSON to include all metrics
returned by the selected backend runs. Existing Ethos-U work already adds shared
standardized performance fields and keeps missing standard fields explicit, but
that field set is intentionally smaller than the full Vela and Corstone
statistics payloads.

The same ticket also covers the Ethos-U mapping for the
`memoryProfile.modelWeightMemory` design-input field. The raw backend statistic
should remain available with its source-derived name where useful, but MLIA
should also expose a stable MLIA-standard result metric for model weight memory
as a numeric metric when a supported source is available and as unavailable
when the selected backend cannot provide one.

The current MLIA standardized output schema already has the two locations needed
for this work:

- `results[*].metrics` for model-level or result-level performance statistics.
- `results[*].breakdowns[*].metrics` for per-operator or per-layer statistics.

Metric names are not enumerated by the schema, so backend-specific metrics can
remain plugin-owned while still validating as standardized output. The shared
`model_weight_memory` metric is handled in core because it represents a standard
MLIA performance field rather than a Vela-only or Corstone-only source
statistic.

## Goals / Non-Goals

**Goals:**

- Preserve Vela summary CSV statistics in performance result metrics.
- Preserve Vela per-layer CSV statistics in performance breakdown metrics.
- Preserve Corstone model-level FVP statistics in performance result metrics.
- Preserve Corstone per-layer CSV statistics in performance breakdown metrics.
- Include supported memory and traffic statistics such as model weight memory,
  memory area usage, and memory read/write traffic.
- Add the MLIA-standard result metric that maps to
  `memoryProfile.modelWeightMemory`, using a numeric value when a supported
  Ethos-U source value is available and an unavailable entry otherwise.
- Serialize Corstone integer cycle counters as JSON integer values.
- Validate representative Vela and Corstone outputs against the MLIA
  standardized output schema.

**Non-Goals:**

- Do not add backend-specific Vela or Corstone metric definitions to core.
- Do not add other shared core standard metrics beyond `model_weight_memory`.
- Do not redesign result, breakdown, extension, or warning structures.
- Do not treat consumer-specific schemas as the MLIA implementation contract.
- Do not address unrelated schema-alignment issues unless they directly block
  this statistics-completeness change.

## Decisions

### Backend statistics use metrics and breakdown metrics

Vela and Corstone model-level statistics should be emitted as
`results[*].metrics`. Per-layer or per-operator statistics should be emitted as
`results[*].breakdowns[*].metrics`.

This keeps the output queryable through the existing standardized JSON shape and
avoids a backend-specific extension block for data that is already metric-like.
Extensions should remain a fallback only if a future statistic cannot be
represented as a numeric metric or unavailable metric.

### Metric names are source-derived unless they are standard metrics

The plugin should use stable, machine-readable metric names in standardized
JSON. Existing snake_case names should be preserved where they already exist.
New names derived from CSV headers should use the same lowercase snake_case
normalization used by current Corstone per-layer metrics.

Backend-specific statistics should keep source-derived names. Standard MLIA
metrics should use MLIA's standardized metric naming style, not the
consumer-facing camelCase design-input field name.

Units should reflect the source statistic. Percentages should use `%`, memory
sizes and traffic byte counts should use `bytes`, cycle counts should use
`cycles`, and data-beat counters should keep `beats`.

### Vela summary completeness starts from parsed summary fields

Vela currently parses only a subset of the summary CSV into `VelaSummary`, even
though the known CSV contains additional fields such as storage areas, memory
traffic, model weight memory, MAC counts, and throughput-related values.

The implementation should expand the parsed summary fields required for
MLIA-1667 and emit them as result-level metrics when Vela provides numeric
values. Existing shared standard metrics should remain present through the core
helper, but backend-specific metrics should not be forced into the standard
field list.

String-valued storage area and configuration fields should remain outside the
metric list. Numeric configuration values such as clocks, cache sizes, and
bandwidth settings should also stay out of this metric-completeness change
unless a later requirement defines a backend configuration section for them.

### Vela per-layer completeness starts from known layer CSV headers

Vela per-layer output already preserves several layer metrics. MLIA-1667 should
check the known layer CSV headers against the emitted breakdown metrics and add
any supported numeric statistics that are currently parsed but not surfaced.

The supported Vela layer metrics are SRAM usage, peak SRAM usage percentage,
operation cycles, operation-cycle network percentage, NPU cycles, memory access
cycles, MAC count, MAC-count network percentage, and MAC utilization percentage.
The Vela `NNG Operator` string is not a numeric metric and should stay out of
`breakdowns[*].metrics`.

String identifiers such as operator name and layer name should stay as breakdown
identity fields, not metrics.

### Corstone completeness starts from parsed FVP and CSV stats

Corstone model-level FVP statistics should remain result-level metrics. Existing
optional handling for absent write counters should be preserved: absent optional
backend counters should not produce fabricated numeric values.

Corstone cycle counters that represent integral cycle counts should be emitted
as JSON integer values. This includes `npu_active_cycles`, `npu_idle_cycles`,
and `npu_total_cycles`; serializing these values as JSON numbers with a `.0`
suffix is a type mismatch with their source semantics.

Corstone per-layer CSV statistics should remain breakdown metrics. Metrics with
numeric source values should be emitted with units from the existing per-layer
unit map. Non-numeric source values should not be emitted as numeric metrics.

### Model weight memory gets a standard result metric

The Vela and Corstone raw model-weight statistics are part of this
stats-completeness work. In addition to preserving appropriate source-derived
backend statistics, the implementation should emit one MLIA-standard
result-level metric that maps to `memoryProfile.modelWeightMemory` with unit
`bytes`. The metric should be numeric when a supported source value is available
and availability-aware when the selected backend cannot provide a supported
source value.

If Vela and Corstone expose different candidate source values, the
implementation should choose and document the source for each backend in code
or tests. The standard metric should be explicit when the backend output does
not provide a supported source value; this work should not fabricate a value.

For Vela, the standard `model_weight_memory` result metric should use
`total_npu_encoded_weights`. This represents the encoded weights in the compiled
target-side model. `total_original_weights` should still be preserved as a
source-derived Vela summary metric, but it should not be used for the standard
model-weight-memory mapping.

For Corstone, the current FVP model-level statistics expose NPU cycle counters
and data-beat counters, and the per-layer CSV exposes per-operator memory,
cycle, access, MAC, and percentage statistics. No supported Corstone source
value has been identified for compiled model weight memory. Corstone output
should therefore emit `model_weight_memory` as unavailable until a backend
source with the right semantics is available rather than using model file size,
per-layer SRAM usage, or data-beat counters as a proxy.

### Intentional omissions

The following backend fields remain outside this metric-completeness change:

- Vela string identity and configuration fields such as `experiment`,
  `network`, `accelerator_configuration`, `system_config`, `memory_mode`,
  `weights_storage_area`, and `feature_map_storage_area`.
- Vela numeric setup or compiler-configuration fields such as clock,
  cache-size, memory-bandwidth, and pass-count values.
- Vela layer string fields such as `NNG Operator`; operator names remain
  breakdown identity fields rather than numeric metrics.
- Corstone per-layer string fields such as original operator, target, NNG
  operator, and layer name; operator and layer names remain breakdown identity
  fields where applicable.
- Corstone numeric `model_weight_memory`, because the current supported
  Corstone source data does not include a model-weight-memory counter. The
  standard metric should still be emitted as unavailable.

### Unavailable markers stay scoped

Unavailable markers stay scoped to MLIA's shared standard metric set and to
specific optional backend metrics only if this change deliberately defines them
that way. `model_weight_memory` is part of the shared standard set, but
MLIA-1667 should not add unavailable markers for every possible backend-specific
statistic that a Vela or Corstone version might omit.

If the implementation defines a specific optional backend metric as part of the
contract and the backend run reports that it is unsupported, then an unavailable
metric can be used. Otherwise, omitted backend-specific source values should be
treated as absent rather than synthesized.

## Risks / Trade-offs

- "All metrics" depends on backend version and output format. The implementation
  should define completeness against the parsed Vela summary CSV, Vela per-layer
  CSV, Corstone FVP stats, and Corstone per-layer CSV fields that this plugin
  supports.
- Adding every backend-specific metric as a core standard field would create an
  ownership problem. Keeping these metrics plugin-owned preserves the existing
  core/plugin boundary.
- Corstone FVP runs can be slow. Unit-level parser and serializer tests should
  provide the main evidence, with schema validation over representative payloads.
- Large generated JSON files would be noisy. If a literal sample is needed, keep
  it small and neutral.
