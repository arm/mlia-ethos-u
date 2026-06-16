## ADDED Requirements

### Requirement: Vela performance result preserves summary statistics

Vela performance standardized JSON SHALL preserve supported numeric statistics
from the Vela summary CSV as result-level metrics.

The emitted metric names SHALL be stable lowercase snake_case names and the
units SHALL match the source statistic semantics.

#### Scenario: Vela summary statistic is available

- **WHEN** Vela reports a supported numeric summary statistic for a performance run
- **THEN** the performance result includes that statistic in `results[*].metrics`.

#### Scenario: Vela model weight memory is available

- **WHEN** Vela reports `total_npu_encoded_weights` for a performance run
- **THEN** the performance result includes the `model_weight_memory` metric with unit `bytes`.
- **AND** the performance result preserves `total_npu_encoded_weights` as a source-derived result-level metric.
- **AND** the performance result does not use `total_original_weights` for the `model_weight_memory` value.

#### Scenario: Vela memory traffic is available

- **WHEN** Vela reports memory read or write traffic for a performance run
- **THEN** the performance result includes result-level metrics for the supported traffic values with unit `bytes`.

#### Scenario: Vela summary statistic is absent

- **WHEN** a backend version does not report a backend-specific Vela summary statistic
- **THEN** the performance result does not fabricate a numeric value for that statistic.

### Requirement: Ethos-U performance result exposes standard model weight memory

Ethos-U performance standardized JSON SHALL expose an MLIA-standard result-level
metric that maps to the `memoryProfile.modelWeightMemory` design-input field
as either a numeric metric or an availability-aware metric entry.

The standard model-weight-memory metric SHALL use unit `bytes`.

#### Scenario: Supported model weight memory source is available

- **WHEN** Vela or Corstone reports a supported model weight memory source value
- **THEN** the performance result includes the MLIA-standard model-weight-memory metric in `results[*].metrics`.

#### Scenario: Supported model weight memory source is absent

- **WHEN** a backend output does not include a supported source value for model weight memory
- **THEN** the performance result includes `model_weight_memory` as an availability-aware metric entry.
- **AND** the performance result does not fabricate a numeric model-weight-memory value.

### Requirement: Vela performance breakdowns preserve layer statistics

Vela performance standardized JSON SHALL preserve supported numeric statistics
from the Vela per-layer CSV as operator breakdown metrics.

String fields used to identify the layer or operator SHALL remain breakdown
identity fields rather than numeric metrics.

#### Scenario: Vela layer statistic is available

- **WHEN** Vela reports a supported numeric per-layer statistic
- **THEN** the corresponding operator breakdown includes that statistic in `metrics`.

#### Scenario: Vela layer statistic is absent

- **WHEN** a Vela per-layer CSV does not contain a supported optional statistic
- **THEN** the corresponding breakdown does not fabricate a numeric value for that statistic.

### Requirement: Corstone performance result preserves model statistics

Corstone performance standardized JSON SHALL preserve supported model-level FVP
statistics as result-level metrics.

#### Scenario: Corstone model statistic is available

- **WHEN** Corstone reports a supported model-level FVP statistic
- **THEN** the performance result includes that statistic in `results[*].metrics`.

#### Scenario: Corstone cycle counter is integral

- **WHEN** Corstone reports an integral cycle counter such as `npu_active_cycles`, `npu_idle_cycles`, or `npu_total_cycles`
- **THEN** the performance result emits that metric as a JSON integer value.

#### Scenario: Optional Corstone model statistic is absent

- **WHEN** a Corstone backend does not report an optional model-level statistic
- **THEN** the performance result does not fabricate a numeric value for that statistic.

### Requirement: Corstone performance breakdowns preserve layer statistics

Corstone performance standardized JSON SHALL preserve supported numeric
statistics from Corstone per-layer CSV output as operator breakdown metrics.

#### Scenario: Corstone layer statistic is available

- **WHEN** Corstone reports a supported numeric per-layer statistic
- **THEN** the corresponding operator breakdown includes that statistic in `metrics`.

#### Scenario: Corstone layer statistic is non-numeric

- **WHEN** a Corstone per-layer source value cannot be represented as a numeric metric
- **THEN** the corresponding breakdown does not emit that value as a numeric metric.

### Requirement: Ethos-U performance statistics output validates

Ethos-U Vela and Corstone performance statistics output SHALL validate against
the MLIA standardized output schema.

#### Scenario: Vela statistics output is serialized

- **WHEN** Vela emits performance standardized JSON with backend-specific statistics
- **THEN** the output validates against the MLIA standardized output schema.

#### Scenario: Corstone statistics output is serialized

- **WHEN** Corstone emits performance standardized JSON with backend-specific statistics
- **THEN** the output validates against the MLIA standardized output schema.
