## Context

The core `mlia` change `add-standard-performance-metrics` defines a standardized JSON contract for standardized performance performance fields. This repository owns the Ethos-U Vela backend path, so it must decide which Vela source values can populate those standardized metrics and where Vela must emit explicit unavailable entries instead.

The current Vela performance output already emits several model-level metrics, including `inferences_per_second`, cycle counts, batch size, model size, and memory-area sizes. It also emits layer breakdowns from Vela's per-layer CSV. The current `PerformanceMetrics.to_standardized_output()` implementation reuses the local `metrics` variable for model-level metrics and then for layer-level metrics, which means the final performance result can receive the last layer's metrics instead of the model-level metrics.

Vela compatibility output already has operator support information through `Operators.npu_supported_ratio`, but that data is produced by the compatibility path rather than the performance result construction path.

## Goals / Non-Goals

**Goals:**

- Prove the core standardized performance metric contract against the Vela performance path.
- Preserve Vela model-level result metrics when adding standardized fields.
- Call the core helper explicitly once the helper is available from `mlia`.
- Populate standardized metrics from Vela source values only when the source semantics match the core contract.
- Emit availability-aware metric entries for standardized fields that Vela cannot provide.
- Add Vela-specific tests for source mapping, unavailable entries, and result-level metric preservation.

**Non-Goals:**

- Do not implement Vela/Corstone all-stats completeness in this change.
- Do not implement model weight memory or optional memory-stats completeness in this change; keep that with the later Vela/Corstone all-stats work.
- Do not add Corstone standardized field extraction in this change.
- Do not make core MLIA infer Vela source semantics.
- Do not standardize a new latency metric in this plugin change.
- Do not fabricate CPU utilization or memory values from unrelated Vela fields.

## Decisions

### Vela calls the core helper explicitly

The core `mlia` helper should be called from the Vela performance result construction path after Vela has built its model-level metric list. This keeps the common contract in core while making the plugin-owned source extraction explicit.

The helper should receive the metrics Vela can provide and fill missing standardized metrics as availability-aware entries. The plugin should not duplicate the core unavailable-fill rules.

Alternative considered: rely on core reporting to normalize every performance result automatically. The core spec rejected that for the initial implementation, and Vela should follow the explicit helper-call model.

### Preserve model-level metrics separately from breakdown metrics

Vela should use separate local variables for model-level result metrics and layer-level breakdown metrics. The performance result must receive the model-level metrics list, not the last layer's metrics.

This is required before the Vela payload can be used as the representative proof for the core contract. If the overwrite remains, tests for standardized result-level metrics could pass or fail based on the final layer rather than the model result.

### Source mapping is metric-specific

Vela source values should map to standardized MLIA metric names as follows:

| Standard metric | Vela source | Initial behavior |
| --- | --- | --- |
| `inferences_per_second` | `PerformanceMetrics.inferences_per_second` | Emit numeric value with unit `inferences/s`. |
| `target_utilization` | `npu_cycles` and `total_cycles` | Emit `(npu_cycles / total_cycles) * 100` with unit `%` when those counters are confirmed as suitable; otherwise emit unavailable. |
| `cpu_utilization` | None currently identified | Emit unavailable with unit `%`. |
| `accelerator_operator_percentage` | `Operators.npu_supported_ratio * 100` from Vela compatibility data | Emit numeric value only when compatibility data is available to the payload being built; otherwise emit unavailable with unit `%`. |
| `peak_activation_memory` | Highest Vela per-layer SRAM or staging usage | Emit numeric value with unit `bytes`. |
| `average_memory` | Vela per-layer SRAM or staging usage weighted by per-layer operation cycles | Emit numeric value with unit `bytes` when per-layer cycle data is available; otherwise emit unavailable. |

The implementation should not map unrelated Vela memory fields solely because they are memory values. The source requirements call for result-level peak activation and average memory metrics. Vela owns enough source data for this representative payload to emit peak activation memory from the highest per-layer SRAM or staging usage value, and average memory from per-layer SRAM or staging usage weighted by per-layer operation cycles.

Model weight memory and other optional memory-profile or memory-traffic statistics are not part of this representative Vela standard-fields change. They should stay with the later Vela/Corstone statistics-completeness work.

### Existing latency remains a partial mapping

Vela currently emits `batch_inference_time` with unit `seconds`, while `_performance_metrics()` calculates `midpoint_inference_time * 1000`. That looks like a unit or naming inconsistency, but the core change intentionally preserves latency as an existing partial mapping rather than standardizing it.

This plugin change should not broaden into latency standardization. If the Vela latency issue affects the standardized throughput mapping or tests, capture the fix narrowly; otherwise track it as follow-up.

### Interpretation notes use result warnings when needed

If Vela emits limitations needed to interpret standardized performance metrics, they should use result warnings. Examples include statements that a metric is estimated from Vela compiler output rather than measured hardware output, or that a memory field is unavailable because no source matches the standardized metric semantics.

Structured availability reasons should remain on availability-aware metric entries.

## Risks / Trade-offs

- Core helper availability may depend on a released `mlia` version. Mitigation: land this after the core change is available and update dependency floors through the normal MLIA cross-repo process.
- Vela memory fields may be tempting but semantically wrong for optional memory-profile fields. Mitigation: map only the fields whose semantics match the result-level standard metrics, and keep optional memory-profile completeness with the later all-stats work.
- Operator percentage comes from compatibility data, not the current performance object. Mitigation: emit it only when the relevant data is available to the payload; otherwise rely on the core helper to mark it unavailable.
- The Vela overwrite fix is small but changes existing standardized output behavior. Mitigation: add regression tests that assert both model-level result metrics and layer breakdown metrics are preserved.
- This representative integration does not complete Corstone or every plugin. Mitigation: keep this OpenSpec scoped to Vela and leave all-stats completeness to separate plugin-owned work.

## Migration Plan

Implement this after the core `mlia` change provides schema `1.1.0`, standardized metric constants or names, availability-aware metric support, and the shared helper.

Plugin implementation should then update the Vela performance output path, tests, and dependency metadata as needed. If the helper is only available from an unreleased core commit, use the normal MLIA cross-repo dependency process rather than adding a local compatibility copy.

## Follow-up Decisions

- Vela emits `peak_activation_memory` from the highest per-layer SRAM or staging usage value.
- Vela emits `average_memory` from per-layer SRAM or staging usage weighted by per-layer operation cycles when that per-layer source data is available.
- The Vela performance payload will not be coupled to the compatibility collection flow in this change, so `accelerator_operator_percentage` is emitted as unavailable unless the performance payload later gains trustworthy source data.
- The current Vela `batch_inference_time` unit inconsistency is tracked as a separate latency cleanup so this change stays focused on the standardized performance standardized fields.
- Model weight memory remains part of later Vela/Corstone statistics-completeness work, not this representative standard-fields integration.
