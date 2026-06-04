## Why

The core MLIA standard performance-field contract is in place, and the first Vela performance integration proves the basic helper path. The remaining Ethos-U slice needs plugin-owned refinements so the standardized JSON output better represents the agreed field set without pulling Vela/Corstone statistics-completeness work into the current change.

## What Changes

- Add the existing Ethos-U performance interpretation note to Vela and Corstone standardized JSON results as `warnings`.
- Add Vela compatibility result coverage for `accelerator_operator_percentage` using the existing operator-placement ratio.
- Keep Vela performance output's `accelerator_operator_percentage` unavailable unless compatibility data is available to that payload.
- Add Corstone standard performance-field helper coverage.
- Add Corstone `target_utilization` from `npu_active_cycles / npu_total_cycles * 100 if npu_total_cycles else 0.0`.
- Add Corstone `peak_activation_memory` and `average_memory` from per-layer memory usage where supported source data is available.
- Keep Corstone throughput and CPU utilization unavailable unless supported source data is identified.
- Add tests and schema-validation coverage for the new standardized output behavior.

## Out of Scope

- Do not implement Vela/Corstone all-stats completeness.
- Do not implement `memoryProfile.modelWeightMemory`, DRAM traffic standardization, or optional memory-profile fields.
- Do not add numeric CPU utilization or memory values without backend-supported source data.
- Do not add heavyweight Corstone end-to-end coverage for this unit-level standardized-output change.

## Capabilities

### New Capabilities

- `ethos-u-performance-field-refinements`: Completes the next Ethos-U standardized performance-field slice after the Vela representative integration.

### Modified Capabilities

- None.

## Impact

- Vela compatibility standardized output.
- Vela and Corstone performance standardized output warnings.
- Corstone performance standardized metrics.
- Ethos-U plugin tests and schema-validation coverage.
