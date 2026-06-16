## 1. Source Inventory

- [x] 1.1 Compare Vela summary CSV headers with parsed `VelaSummary` fields and emitted result metrics.
- [x] 1.2 Compare Vela per-layer CSV headers with parsed `LayerPerfInfo` fields and emitted breakdown metrics.
- [x] 1.3 Compare Corstone FVP model stats with emitted result metrics for Corstone-300 and Corstone-320.
- [x] 1.4 Compare Corstone per-layer CSV headers with emitted breakdown metrics and the unit map.
- [x] 1.5 Identify the Vela source value and confirm that no supported Corstone source is available for the MLIA-standard model-weight-memory metric.
- [x] 1.6 Record any unsupported, non-numeric, or configuration-like backend fields that remain intentionally absent.

## 2. Vela JSON Completeness

- [x] 2.1 Parse supported missing Vela summary statistics needed for MLIA-1667.
- [x] 2.2 Revise the draft Vela summary metric emission so setup/configuration-like fields are not emitted as performance metrics.
- [x] 2.3 Emit supported Vela summary statistics as result-level metrics with stable names and units.
- [x] 2.4 Emit supported missing Vela per-layer statistics as breakdown metrics.
- [x] 2.5 Emit the MLIA-standard model-weight-memory metric when the chosen Vela source value is available.
- [x] 2.6 Preserve existing shared standard performance metrics and warnings.
- [x] 2.7 Add focused tests for Vela result-level metric completeness.

## 3. Corstone JSON Completeness

- [x] 3.1 Confirm Corstone-300 and Corstone-320 model-level FVP stats are emitted where available.
- [x] 3.2 Add any missing supported Corstone model-level metrics without fabricating absent optional counters.
- [x] 3.3 Normalize Corstone integer counter metrics such as `npu_active_cycles`, `npu_idle_cycles`, and `npu_total_cycles` so they are emitted as integer values rather than JSON numbers with `.0`.
- [x] 3.4 Emit the MLIA-standard model-weight-memory metric as unavailable when no supported Corstone source value is available.
- [x] 3.5 Confirm Corstone per-layer metric emission covers supported numeric CSV fields.
- [x] 3.6 Add focused tests for Corstone result-level and breakdown metric completeness.
- [x] 3.7 Keep Corstone validation unit-level unless a targeted run is needed.

## 4. Validation And Review

- [x] 4.1 Validate representative Vela standardized output against the MLIA output schema.
- [x] 4.2 Validate representative Corstone standardized output against the MLIA output schema.
- [x] 4.3 Run targeted tests for touched Vela code.
- [x] 4.4 Run the repository copyright-header check for new OpenSpec files.
- [x] 4.5 Check public artifacts against the restricted terminology list and MLIA review themes before review.
