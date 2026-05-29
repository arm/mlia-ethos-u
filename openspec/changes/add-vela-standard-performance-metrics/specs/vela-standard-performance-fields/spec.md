## ADDED Requirements

### Requirement: Vela performance output preserves model-level metrics

Vela SHALL preserve model-level result metrics when building standardized performance output.

#### Scenario: Breakdown metrics do not overwrite result metrics

- **WHEN** Vela standardized performance output includes layer breakdowns
- **THEN** the performance result's `metrics` contain the model-level metrics rather than the final layer's breakdown metrics.

#### Scenario: Existing Vela throughput metric remains present

- **WHEN** Vela has a model-level inference throughput value
- **THEN** the performance result includes `inferences_per_second` with unit `inferences/s`.

### Requirement: Vela integrates with core standardized performance fields

Vela SHALL use the core MLIA standardized performance metric contract when building standardized performance results.

#### Scenario: Core helper fills missing standardized metrics

- **WHEN** Vela passes a performance result metric list through the core standardized performance metric helper
- **THEN** supplied numeric values are preserved and missing standardized metrics are represented as availability-aware metric entries.

#### Scenario: Helper is called explicitly

- **WHEN** Vela builds standardized performance output
- **THEN** Vela explicitly calls the core helper rather than relying on core reporting to modify the result later.

### Requirement: Vela maps only trustworthy source values

Vela SHALL emit numeric standardized metrics only when Vela source data matches the standardized metric semantics.

#### Scenario: Target utilization source is available

- **WHEN** Vela can use suitable NPU and total cycle counters for a performance result
- **THEN** Vela emits `target_utilization` with value `(npu_cycles / total_cycles) * 100 if total_cycles else 0.0` and unit `%`.

#### Scenario: CPU utilization source is absent

- **WHEN** Vela has no trustworthy source for CPU utilization
- **THEN** Vela emits `cpu_utilization` as an availability-aware metric entry with unit `%` and a reason.

#### Scenario: Accelerator operator percentage source is available

- **WHEN** Vela operator compatibility data is available to the payload being built
- **THEN** Vela emits `accelerator_operator_percentage` as `Operators.npu_supported_ratio * 100` with unit `%`.

#### Scenario: Accelerator operator percentage source is absent

- **WHEN** Vela operator compatibility data is not available to the payload being built
- **THEN** Vela emits `accelerator_operator_percentage` as an availability-aware metric entry with unit `%` and a reason.

#### Scenario: Peak activation memory source is available

- **WHEN** Vela has per-layer SRAM or staging usage for a single inference run
- **THEN** Vela emits `peak_activation_memory` as a result-level metric with unit `bytes`.

#### Scenario: Average memory source is available

- **WHEN** Vela has per-layer SRAM or staging usage and per-layer operation cycles
- **THEN** Vela emits `average_memory` as the per-layer memory usage weighted by operation cycles, with unit `bytes`.

#### Scenario: Average memory source is absent

- **WHEN** Vela has no trustworthy per-layer memory and cycle source data
- **THEN** Vela emits `average_memory` as an availability-aware metric entry with unit `bytes` and a reason.

#### Scenario: Optional memory-profile stats remain out of scope

- **WHEN** model weight memory or other optional Vela/Corstone memory statistics are needed
- **THEN** Vela handles them in a separate statistics-completeness change rather than this representative standard-fields integration.

### Requirement: Vela validation covers representative payload behavior

Vela SHALL include tests that prove the representative standardized performance payload behavior.

#### Scenario: Result-level standardized metrics are tested

- **WHEN** Vela standardized performance output is tested
- **THEN** tests assert the result-level standardized metrics and availability-aware entries, not only layer breakdowns.

#### Scenario: Breakdown metrics are still tested

- **WHEN** Vela standardized performance output includes layer breakdowns
- **THEN** tests assert that breakdown metrics are preserved after standardized result metrics are added.
