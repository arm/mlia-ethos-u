## Why

Vela performance output exposes inference-time values from two paths:

- `inference_time` from the Vela summary CSV
- `batch_inference_time` derived from cycle count and core clock

The summary value has been emitted as seconds, while other MLIA performance
output uses milliseconds for inference latency. The derived
`batch_inference_time` value is already calculated in milliseconds but is
currently labelled as seconds.

This should be normalized before latency becomes a standard MLIA performance
metric with a fixed unit.

## What Changes

- Emit Vela summary `inference_time` as milliseconds.
- Keep the standardized metric name `inference_time`.
- Emit `batch_inference_time` with unit `ms`, matching the existing derived
  value.
- Add tests for the converted summary metric and corrected derived metric unit.

## Out of Scope

- Do not change how Vela parses raw summary CSV values.
- Do not remove existing backend-specific metrics.
- Do not invent latency values when Vela does not provide source data.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `vela-standard-performance-fields`: Normalizes Vela inference-time output to
  the standard MLIA latency unit.

## Impact

- `src/mlia/backend/vela/performance.py`
- Vela performance output tests
- The change depends on the core MLIA standard latency metric constants and unit
  contract.
