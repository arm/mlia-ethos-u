## Why

Core MLIA is defining standardized performance fields for standardized performance output, but Vela owns the backend source values and the representative plugin payload. The Vela performance path needs a plugin-local change so it can prove the core contract without moving Vela-specific extraction or bugs into core MLIA.

## What Changes

- Integrate the Vela performance result construction with the core standardized performance metric helper once that helper is available from `mlia`.
- Preserve Vela-provided result-level performance metrics when adding standardized fields.
- Fix the Vela result-level metric overwrite issue if it would otherwise make the representative payload use the last layer's metrics instead of model-level metrics.
- Map Vela source values to standardized MLIA metric names only when the source semantics match the core contract.
- Emit availability-aware entries for standardized metrics that Vela cannot provide from trustworthy source data.
- Add Vela-specific tests for result-level metrics, breakdown preservation, source-value mapping, and unavailable entries.
- Keep Vela/Corstone all-stats completeness, including model weight memory and optional memory statistics, separate from this representative integration.

## Capabilities

### New Capabilities

- `vela-standard-performance-fields`: Vela integration for MLIA standardized performance fields and availability-aware metric entries.

### Modified Capabilities

- None.

## Impact

- `src/mlia/backend/vela/performance.py`
- Vela performance tests, especially `tests/test_backend_vela_performance.py`
- Dependency on the core `mlia` change that provides schema `1.1.0`, standardized metric names/units, availability-aware metric entries, and the shared helper
- Possible follow-up release or dependency-floor work once the core helper is available from a published `mlia` package
