## Why

Core MLIA schema `1.1.0` now defines result-level advice as
`results[*].advice` with lower-case advice enum values. Ethos-U owns a
target-specific JSON file writer and e2e normalization path that still use the
legacy `results[*].advices` spelling, so this repository must align with the
core output contract before its JSON outputs can validate consistently.

## What Changes

- Update the Ethos-U target-specific JSON file writer to emit result-level
  advice under `results[*].advice`.
- Use core schema advice serialization so sidecar JSON advice entries use
  lower-case `category` and `severity` values.
- Update e2e output normalization to inspect `advice` rather than `advices`.
- Add focused handler tests for the target-specific JSON writer.
- Leave unrelated advice-category concepts such as `supported_advice`
  unchanged.
- Update the `mlia` dependency floor to the published package containing the
  schema change: `mlia>=0.11.0.dev27`.

## Capabilities

### New Capabilities

- `ethos-u-result-advice-output`: Ethos-U alignment with the standardized
  result-level advice field in MLIA schema `1.1.0`.

### Modified Capabilities

- None.

## Impact

- `src/mlia/target/ethos_u/handlers.py`
- `tests/test_target_ethos_u_handlers.py`
- `tests/test_e2e_api.py`
- `pyproject.toml`
