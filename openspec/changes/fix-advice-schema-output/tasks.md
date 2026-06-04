## 1. Sidecar Output

- [x] 1.1 Update `EthosUEventHandler._write_json_with_advice()` to write
  `result["advice"]`.
- [x] 1.2 Use core schema advice serialization for sidecar advice entries.
- [x] 1.3 Do not emit the legacy `result["advices"]` field.

## 2. E2E Normalization

- [x] 2.1 Update e2e output normalization to read `result.get("advice", [])`.
- [x] 2.2 Leave unrelated supported-advice category terminology unchanged.

## 3. Validation

- [x] 3.1 Add or update focused handler tests for sidecar advice output.
- [x] 3.2 Run targeted handler and e2e normalization tests.
- [x] 3.3 Run OpenSpec validation for `fix-advice-schema-output`.
- [x] 3.4 Run pre-commit for the committed diff.

## 4. Dependency Floor

- [x] 4.1 Identify the published core package containing the schema `1.1.0`
  advice change.
- [x] 4.2 Update the `mlia` dependency floor to `mlia>=0.11.0.dev27`.
- [x] 4.3 Validate against the published package dependency.
