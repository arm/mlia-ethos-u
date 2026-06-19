## MODIFIED Requirements

### Requirement: Vela performance fields

Vela SHALL emit supported standardized performance fields through MLIA
standardized output using the shared MLIA metric names and units.

#### Scenario: Vela summary latency emitted in milliseconds

- **WHEN** Vela summary CSV data contains `inference_time`
- **THEN** MLIA emits `inference_time` under `results[*].metrics` with unit `ms`
- **AND** the emitted value is converted from seconds to milliseconds.

#### Scenario: Vela derived batch inference time unit matches value

- **WHEN** MLIA emits Vela `batch_inference_time`
- **THEN** the metric unit is `ms`.

#### Scenario: Vela summary latency missing

- **WHEN** Vela summary CSV data does not contain `inference_time`
- **THEN** MLIA does not fabricate a numeric latency value.
