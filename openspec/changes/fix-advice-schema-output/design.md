## Context

The core `mlia` change `fix-advice-schema-output` adds schema `1.1.0` support
for `results[*].advice` and changes advice JSON enum values to lower-case
strings. Ethos-U has two relevant downstream paths:

- `EthosUEventHandler._write_json_with_advice()` writes sidecar JSON files for
  standardized compatibility and performance output.
- `tests/test_e2e_api.py` normalizes advice entries before comparing CLI and
  Python API JSON output.

The sidecar writer currently bypasses `JSONReporter` and writes advice under
`result["advices"]`. That path must be updated in this repository because core
reporting cannot rewrite files that the target handler writes directly.

## Goals / Non-Goals

**Goals:**

- Emit sidecar advice under `results[*].advice`.
- Keep the sidecar advice JSON aligned with lower-case schema values for
  `category` and `severity`.
- Update e2e normalization to use the new result-level field.
- Cover the target-specific writer with focused tests.
- Keep the implementation independent of unrelated supported-advice registry
  terminology.

**Non-Goals:**

- Do not invent an unpublished future dependency version.
- Do not accept or normalize the legacy `advices` field in new Ethos-U output.
- Do not redesign advice generation or decide whether advice should be routed
  to a narrower subset of result objects.
- Do not change unrelated backend registry or target registry tests that use
  "advice" as a capability term.

## Decisions

### Update only result-level advice output

The change is limited to result-level advice entries in standardized JSON. Local
variables and tests that refer to supported advice categories do not represent
the JSON field and should not be renamed for this task.

### Use core schema serialization

The core advice-schema change has been published as `mlia>=0.11.0.dev27`, so
the Ethos-U dependency floor should require that version and sidecar advice
entries should use core `SchemaAdvice.to_dict()` serialization directly. This
keeps the target-specific writer aligned with the core schema contract without
duplicating enum conversion logic locally.

### Preserve existing advice routing

The current writer attaches collected advice to each result in a standardized
output object. This change keeps that behavior and changes the field name while
relying on core for advice entry serialization.

## Risks / Trade-offs

- **Risk:** The dependency floor could drift from the core schema behavior this
  branch relies on. **Mitigation:** Use the published development package from
  the core release workflow and validate against the declared package floor.
- **Risk:** Broad renaming could touch supported-advice category tests.
  **Mitigation:** Search for `advices` and update only result-level JSON output
  paths or tests.
- **Risk:** Full e2e output may still contain unrelated schema warnings.
  **Mitigation:** Validate the focused writer behavior here and keep broader
  schema issues tracked separately.
