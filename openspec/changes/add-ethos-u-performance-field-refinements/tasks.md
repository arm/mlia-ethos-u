## 1. Performance Interpretation Warnings

- [x] 1.1 Add a shared Ethos-U performance warning string.
- [x] 1.2 Emit the warning from Vela performance standardized JSON.
- [x] 1.3 Emit the warning from Corstone performance standardized JSON.
- [x] 1.4 Preserve the existing report note text.
- [x] 1.5 Add tests for the warning in standardized output.

## 2. Vela Compatibility Percentage

- [x] 2.1 Add `accelerator_operator_percentage` to Vela compatibility output when operators are available.
- [x] 2.2 Use `Operators.npu_supported_ratio * 100` with unit `%`.
- [x] 2.3 Emit an unavailable metric entry when no operators are available.
- [x] 2.4 Add tests for partial, full, and no-operator behavior.
- [x] 2.5 Record that no maintained sample JSON artifact was identified for the
  compatibility percentage change; tests and schema validation cover the
  representative JSON shape.

## 3. Corstone Standard Fields

- [x] 3.1 Add Corstone `target_utilization` from active and total cycle counters.
- [x] 3.2 Call the core standard performance helper for Corstone metrics.
- [x] 3.3 Keep missing Corstone standard fields unavailable.
- [x] 3.4 Add zero-total-cycle behavior coverage.
- [x] 3.5 Add Corstone schema-validation coverage.

## 4. Scope Boundaries

- [x] 4.1 Keep Vela/Corstone all-stats completeness out of this change.
- [x] 4.2 Avoid heavyweight Corstone FVP execution for this data-shaping slice.
