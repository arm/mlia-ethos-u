## Why

MLIA-1667 requires Ethos-U performance JSON to preserve the backend statistics
that Vela and Corstone already report. The current standardized output includes
the shared performance field set and selected backend-specific metrics, but it
does not yet define a complete plugin-owned contract for Vela summary stats,
Vela per-layer stats, Corstone model stats, and Corstone per-layer stats.
This ticket also covers the MLIA-standard metric equivalent for the
`memoryProfile.modelWeightMemory` design-input field. Vela uses the confirmed
encoded-weight source value, while Corstone emits the standard metric as
unavailable because no supported Corstone source value has been identified.

This change keeps backend-specific Vela and Corstone statistics plugin-owned
while using the shared core standard metric set for `model_weight_memory`. It
treats the MLIA standardized output schema as the implementation contract and
uses existing result-level `metrics` and per-operator `breakdowns[*].metrics`
where they fit the source data.

## What Changes

- Add an Ethos-U plugin-owned contract for Vela and Corstone performance
  statistics in standardized JSON.
- Preserve Vela summary CSV statistics as result-level metrics when Vela reports
  the source values.
- Preserve Vela per-layer CSV statistics as operator breakdown metrics when
  Vela reports the source values.
- Preserve Corstone model-level FVP statistics as result-level metrics when
  Corstone reports the source values.
- Preserve Corstone per-layer CSV statistics as operator breakdown metrics when
  Corstone reports the source values.
- Include supported memory and traffic statistics, including model weight memory
  and memory read/write traffic.
- Add an MLIA-standard model-weight-memory result metric that maps to the
  `memoryProfile.modelWeightMemory` design-input field, using an
  availability-aware metric entry when a supported Ethos-U source value is not
  available.
- Emit Corstone integer cycle counters, such as `npu_active_cycles`,
  `npu_idle_cycles`, and `npu_total_cycles`, as JSON integer values.
- Use focused unit tests and schema validation for the JSON shape, with
  heavyweight Corstone runs kept out of the normal validation path.

## Out of Scope

- Do not redesign the full MLIA standardized output schema.
- Do not add backend-specific Vela or Corstone statistics to `mlia` core; only
  the shared `model_weight_memory` standard metric belongs in core.
- Do not use this change to address unrelated schema-alignment bugs such as
  advice serialization, target component variant typing, or Corstone CSV string
  values.
- Do not add numeric CPU utilization values unless supported backend source data
  is identified.
- Do not require unavailable markers for every possible backend statistic that a
  backend version might omit.

## Capabilities

### New Capabilities

- `ethos-u-performance-stats-json`: Defines Vela and Corstone statistics
  completeness for Ethos-U performance standardized JSON.

### Modified Capabilities

- None.

## Impact

- Vela performance standardized JSON.
- Corstone performance standardized JSON.
- Ethos-U plugin tests and schema-validation coverage.
- Companion core `mlia` change adding `model_weight_memory` to the shared
  standard performance metric set.
