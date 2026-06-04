## ADDED Requirements

### Requirement: Ethos-U performance output includes interpretation warnings

Ethos-U performance standardized output SHALL include the known performance interpretation note in result warnings.

#### Scenario: Vela performance warning is emitted

- **WHEN** Vela emits standardized performance output
- **THEN** the performance result includes the Ethos-U performance interpretation note in `warnings`.

#### Scenario: Corstone performance warning is emitted

- **WHEN** Corstone emits standardized performance output
- **THEN** the performance result includes the Ethos-U performance interpretation note in `warnings`.

### Requirement: Vela compatibility emits accelerator operator percentage

Vela compatibility standardized output SHALL emit the accelerator operator percentage when operator-placement data is available.

The metric SHALL be sourced from compatibility/operator-placement data rather than inferred from a performance-only payload.

#### Scenario: Operator placement data is available

- **WHEN** Vela compatibility output has at least one operator
- **THEN** the compatibility result emits `accelerator_operator_percentage` as `Operators.npu_supported_ratio * 100` with unit `%`.

#### Scenario: Operator placement data is unavailable

- **WHEN** Vela compatibility output has no operators
- **THEN** the compatibility result emits `accelerator_operator_percentage` as an availability-aware metric entry with unit `%` and a reason.

### Requirement: Corstone integrates with core standardized performance fields

Corstone performance standardized output SHALL use the core standard performance metric contract.

#### Scenario: Target utilization source is available

- **WHEN** Corstone has active and total NPU cycle counters
- **THEN** Corstone emits `target_utilization` with value `(npu_active_cycles / npu_total_cycles) * 100 if npu_total_cycles else 0.0` and unit `%`.

#### Scenario: Corstone peak activation memory source is available

- **WHEN** Corstone has per-layer staging or SRAM usage data
- **THEN** Corstone emits `peak_activation_memory` as the highest per-layer memory usage value with unit `bytes`.

#### Scenario: Corstone average memory source is available

- **WHEN** Corstone has per-layer staging or SRAM usage data and per-layer operation cycles
- **THEN** Corstone emits `average_memory` as the per-layer memory usage weighted by operation cycles, with unit `bytes`.

#### Scenario: Standard field source is absent

- **WHEN** Corstone has no supported source for a standard performance metric
- **THEN** Corstone emits that metric as an availability-aware metric entry with the standard unit and a reason.

#### Scenario: Corstone standardized output validates

- **WHEN** Corstone emits standardized performance output with standard performance metrics
- **THEN** the output validates against the MLIA standardized output schema.
