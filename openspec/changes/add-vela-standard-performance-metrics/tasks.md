## 1. Core Contract Alignment

- [x] 1.1 Confirm the core `mlia` change provides schema `1.1.0`, standardized metric names and units, availability-aware metric support, and the shared helper.
- [ ] 1.2 Update the `mlia` dependency floor only after a core package containing the helper is published.
- [x] 1.3 Import the core helper and metric identifiers from `mlia` instead of duplicating the standardized field rules locally.

## 2. Vela Result Construction

- [x] 2.1 Split Vela model-level result metrics and layer-level breakdown metrics into separate local variables.
- [x] 2.2 Ensure the performance result receives model-level metrics rather than the final layer's breakdown metrics.
- [x] 2.3 Call the core standardized performance metric helper explicitly from the Vela performance result construction path.
- [x] 2.4 Preserve existing Vela numeric result metrics, including `inferences_per_second`, when adding standardized fields.

## 3. Vela Source Mapping

- [x] 3.1 Emit `inferences_per_second` from the existing Vela throughput value with unit `inferences/s`.
- [x] 3.2 Confirm whether Vela `npu_cycles` and `total_cycles` have the required semantics for `target_utilization`.
- [x] 3.3 Emit `target_utilization` as `(npu_cycles / total_cycles) * 100 if total_cycles else 0.0` when the counters are suitable.
- [x] 3.4 Emit `cpu_utilization` as unavailable because no trustworthy Vela source has been identified.
- [x] 3.5 Decide not to couple the Vela performance payload to compatibility data in this change.
- [x] 3.6 Emit `accelerator_operator_percentage` as unavailable when compatibility data is not available to the performance payload.
- [x] 3.7 Emit `peak_activation_memory` from the highest Vela per-layer SRAM or staging usage value.
- [x] 3.8 Emit `average_memory` from Vela per-layer SRAM or staging usage weighted by per-layer operation cycles when source data is available.
- [x] 3.9 Keep `average_memory` unavailable if no matching per-layer memory and cycle source data is available.
- [x] 3.10 Do not fabricate numeric values from unrelated Vela fields.

## 4. Validation

- [x] 4.1 Add a regression test proving layer breakdown metrics do not overwrite result-level metrics.
- [x] 4.2 Add tests proving supplied numeric standardized metrics are preserved.
- [x] 4.3 Add tests proving missing standardized metrics are emitted as availability-aware entries with units and reasons.
- [x] 4.4 Add tests proving layer breakdown metrics are still present after result-level standardized metrics are added.
- [x] 4.5 Add tests proving unavailable metrics do not contain fake numeric `value` fields.
- [x] 4.6 Add a plugin-level test proving the Vela standardized output validates against the MLIA output schema.

## 5. Follow-up Boundaries

- [x] 5.1 Keep Vela/Corstone all-stats completeness, including model weight memory and optional memory statistics, separate from this change.
- [x] 5.2 Track the existing Vela `batch_inference_time` unit inconsistency as a separate latency cleanup.
