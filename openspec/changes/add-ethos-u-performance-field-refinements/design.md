## Context

The current core and Vela representative changes define and prove the shared standard performance-field helper. The next Ethos-U slice covers three plugin-owned gaps: performance interpretation warnings, Vela compatibility operator percentage, and Corstone standard-field helper coverage.

The source requirements require analysis limitations, throughput, target utilization, peak activation memory, and average memory to be represented. They also track CPU utilization and accelerator operator percentage. Current implementation should emit numeric values only where the plugin owns suitable source data.

## Goals / Non-Goals

**Goals:**

- Represent the existing Ethos-U performance limitation note in standardized JSON warnings.
- Populate Vela compatibility `accelerator_operator_percentage` from existing operator-placement data.
- Keep unavailable performance entries explicit where the performance payload does not own source data.
- Add Corstone standard-field helper coverage, Corstone target utilization, and Corstone result-level memory metrics where per-layer source data is available.
- Validate Corstone through unit-level standardized-output tests rather than a slow FVP run.

**Non-Goals:**

- Do not implement Vela/Corstone all-stats completeness.
- Do not standardize optional memory-profile or memory-traffic metrics.
- Do not fabricate numeric memory or CPU utilization values from unsupported source data.
- Do not broaden this change into Corstone end-to-end coverage.

## Decisions

### Performance notes use result warnings

The existing Ethos-U text report notes that performance figures refer to NPU-only data. Standardized JSON should carry the same interpretation note in `results[*].warnings` for Vela and Corstone performance results.

Availability reasons remain on individual unavailable metric entries. They should not be used as a replacement for result-level interpretation warnings.

### Vela compatibility owns operator percentage

Vela compatibility output already owns operator-placement data through `Operators.npu_supported_ratio`. The compatibility result should emit `accelerator_operator_percentage` as `npu_supported_ratio * 100` with unit `%` when operators are available.

This is intentionally tied to compatibility output. The source requirement context refers to pass, fail, and partial status, and those statuses are produced by compatibility checks over operator placement. Vela performance output has performance metrics, but it does not own the compatibility checks or the pass/fail/partial operator-support decision.

If no operators are available, the metric should be represented as unavailable rather than as a misleading `0%` value.

The Vela performance payload should continue to mark `accelerator_operator_percentage` unavailable because that path does not own compatibility data.

### Corstone owns cycle-derived target utilization

Corstone performance output has `npu_active_cycles` and `npu_total_cycles`. It should emit `target_utilization` as `(npu_active_cycles / npu_total_cycles) * 100 if npu_total_cycles else 0.0`.

Corstone should call the core standard performance helper after adding any supported metrics. That keeps throughput and CPU utilization represented as unavailable until a supported source exists, while allowing Corstone to populate peak activation memory and average memory from per-layer memory usage data.

Corstone peak activation memory should use the highest per-layer staging or SRAM usage value with unit `bytes`. Corstone average memory should use per-layer staging or SRAM usage weighted by per-layer operation cycles, with unit `bytes`. If the per-layer memory and cycle data is absent, the core helper should leave the relevant metric represented as unavailable.

## Risks / Trade-offs

- Corstone FVP runs can be expensive. Unit-level standardized-output tests are the right validation for this slice because the change is data-shaping logic over parsed metrics, not backend execution.
- Vela compatibility and performance outputs now differ for `accelerator_operator_percentage`. That is deliberate because the compatibility path owns the source data.
- The warning text is shared with existing report output to avoid drift between stdout and JSON interpretation notes.
- The field-story DoD mentions sample JSON. No maintained committed sample JSON artifact was identified for this repository, so the representative JSON shape is covered by focused tests and schema validation. If reviewers require literal sample JSON, add a small neutral snippet rather than a large generated backend output file.

## Follow-Up Work

- Implement Vela/Corstone all-stats completeness under the later `MLIA-1667` OpenSpec.
- Revisit numeric CPU utilization only if backend-supported source data is identified.
