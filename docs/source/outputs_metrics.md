<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Outputs and Metrics

## Overview

Ethos-U runs in MLIA usually give you two things at once:

- A quick human-readable summary in the console.
- Structured results that you can inspect later, save as JSON, or compare
  between runs.

The most important part is not the format itself, but what the output helps you
decide. Good Ethos-U metrics help you answer questions like:

- Is the model broadly viable for this target profile?
- Is the cost mainly compute or memory?
- Which layers or operators deserve attention first?
- Do the deeper results change what the quick estimate suggested?

## Common output forms

Typical Ethos-U outputs include:

- Console summaries printed directly by MLIA.
- JSON output when `--json` is used.
- Backend-generated artifacts such as CSV data in Corstone-oriented flows.
- Logs and intermediate files in the output directory.

Treat the console output as the summary and the JSON or artifacts as the detail
you return to when something needs explanation.

## Example JSON shape

Metrics are represented as objects in a list. Per-operator and per-layer metrics
are stored in breakdowns linked to result-local entities:

```json
{
  "backends": [{"id": "vela", "name": "Vela Compiler"}],
  "results": [
    {
      "kind": "performance",
      "producer": "vela",
      "metrics": [
        {"name": "total_cycles", "value": 123456, "unit": "cycles"},
        {"name": "inference_time", "value": 207.0, "unit": "ms"},
        {"name": "model_weight_memory", "value": 4096, "unit": "bytes"}
      ],
      "entities": [
        {
          "id": "source_operator/operator/0",
          "kind": "source_operator",
          "name": "operator/0",
          "placement": "NPU"
        }
      ],
      "breakdowns": [
        {
          "entity_id": "source_operator/operator/0",
          "metrics": [
            {
              "name": "op_cycles",
              "value": 1000,
              "unit": "cycles",
              "aggregation": "sum"
            },
            {
              "name": "sram_usage",
              "value": 4096,
              "unit": "bytes",
              "aggregation": "max"
            }
          ]
        }
      ]
    }
  ]
}
```

Vela and Corstone mark additive layer counters, such as cycles, accesses, and
MAC counts, with `aggregation: "sum"`. Layer memory usage uses
`aggregation: "max"`. Percentage metrics are descriptive and do not declare an
aggregation policy.

Vela omits non-finite backend values such as `NaN`, which cannot be represented
in strict JSON. When a standardized metric is unavailable, the output contains
an explicit unavailable metric entry where the shared schema supports one;
backend-specific optional metrics may simply be absent.

## What to read first in a Vela run

When the run uses `vela`, start with:

1. `total_cycles`
2. Memory-related cycle counters such as `sram_access_cycles` and
   `dram_access_cycles`
3. The small number of operators contributing most of the cost.

Common Vela metrics include:

- `total_cycles`
- `npu_cycles`
- `inference_time`
- `inferences_per_second`
- `target_utilization`
- `sram_access_cycles`
- `dram_access_cycles`
- `model_weight_memory`
- Memory area size metrics.
- Additional Vela summary statistics such as encoded weights or total bytes
  when Vela reports them.
- Per-operator cycle and utilisation data.

### What Vela numbers help you decide

Use them to decide whether the next step should be to:

- Fix unsupported or awkward mapping.
- Reduce memory pressure.
- Focus on a few expensive operators.
- Stop and change the model before doing deeper validation.

## What to read first in a Corstone run

When the run uses a Corstone backend, start with:

1. The total cycle picture.
2. Active-versus-idle style metrics.
3. Traffic-related counters.
4. Layer-level detail.

Common Corstone signals include:

- `npu_active_cycles`
- `npu_idle_cycles`
- `npu_total_cycles`
- AXI data-beat counters.
- Standardized memory and weight metrics when they can be derived from the
  backend result.
- Per-layer CSV metrics covering cycles, memory access, and utilisation.

### What Corstone numbers help you decide

Use them to decide whether the deeper path confirms the quick estimate, or
whether system-level behaviour is changing the story in a useful way.

## A practical reading pattern

A good review pattern is:

1. Read the top-level total first.
2. Decide whether the result looks acceptable, suspicious, or clearly poor.
3. Separate compute-heavy behaviour from memory-heavy behaviour.
4. Only then move to operator or layer detail.

This keeps you from getting lost in details before you know what kind of
problem you are dealing with.

## When numbers look bad

If total cycles are high but most of the cost comes from a few operators, focus
there first.

If memory-related metrics dominate, treat that as a sign that model layout,
tensor sizes, or data movement deserve more attention than the raw compute path.

If Vela and Corstone tell noticeably different stories, do not assume one is
wrong. Differences often mean the deeper path is revealing behaviour the quick
estimate could not show on its own.

If expected metrics are missing, or the result shape looks odd, treat that as a
troubleshooting problem before drawing performance conclusions.

## Cross-links

- See [backends.md](backends.md) for choosing between Vela and Corstone.
- See [cli.md](cli.md) for commands that produce these results.
- See [troubleshooting.md](troubleshooting.md) if output is missing,
  incomplete, or surprising.
