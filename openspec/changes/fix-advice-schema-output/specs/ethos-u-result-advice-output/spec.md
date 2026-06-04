## ADDED Requirements

### Requirement: Ethos-U sidecar output uses result-level advice

Ethos-U target-specific JSON sidecar output SHALL write collected advice under
`results[*].advice`.

#### Scenario: Sidecar output contains advice

- **WHEN** the Ethos-U event handler writes standardized output with collected
  advice
- **THEN** each result object includes `advice`
- **AND** no result object includes `advices`.

### Requirement: Ethos-U sidecar advice uses schema enum values

Ethos-U target-specific JSON sidecar output SHALL serialize advice `category`
and `severity` as lower-case schema values.

#### Scenario: Sidecar advice is serialized

- **WHEN** the Ethos-U event handler writes an advice entry with performance
  category and info severity
- **THEN** the written JSON contains `"category": "performance"`
- **AND** the written JSON contains `"severity": "info"`.

### Requirement: Ethos-U e2e normalization follows the standardized field

Ethos-U CLI/Python API e2e normalization SHALL inspect `results[*].advice`
rather than `results[*].advices`.

#### Scenario: Advice messages contain CLI-only suffixes

- **WHEN** normalized e2e output contains result-level advice
- **THEN** CLI-only advice suffixes are trimmed from entries under `advice`.
