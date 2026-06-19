## Context

Vela summary CSV `inference_time` is parsed into `VelaSummary` without unit
conversion. The MLIA output layer decides how that value is exposed as a metric.

The derived `batch_inference_time` value is calculated as
`midpoint_inference_time * 1000`, so the stored value is already milliseconds
even though the emitted unit currently says seconds.

## Decision

Expose Vela inference-time output in milliseconds:

- Set the emitted unit for summary `inference_time` to `ms`.
- Convert the summary value from seconds to milliseconds when building the MLIA
  output metric.
- Keep `batch_inference_time` unchanged as a value, but emit unit `ms`.

The raw `VelaSummary` object remains a representation of the Vela CSV rather
than becoming an MLIA-normalized data object.

## Consequences

Vela output uses the same latency unit as the MLIA standard performance metric
contract.

The emitted `batch_inference_time` unit now matches the value that has already
been calculated.

Existing consumers that interpreted Vela `inference_time` as seconds will need
to update to the corrected output unit.
