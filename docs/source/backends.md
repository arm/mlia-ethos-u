<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Ethos-U Backends

Use this page when you are deciding which Ethos-U backend to run for a model and what kind of information each backend is intended to give you. It is for choosing the right backend for the question you need to answer, not for learning the basic MLIA command flow.

| Backend | Use it when you want | Typical next step |
| --- | --- | --- |
| `vela` | The fastest answer about compatibility, estimated cycles, and memory pressure | Decide whether the model is worth iterating further |
| `corstone` | A deeper platform-oriented view of where time is going | Validate or investigate results that need more detail |

## Choosing between backends

If you are still deciding whether the model is a good fit for the selected
Ethos-U target, start with `vela`.

If the model already looks promising and you need deeper evidence about runtime
behaviour, move to `corstone`.

If you are unsure, the practical default is simple: use `vela` first, then only
bring in `corstone` when you need more detail than the fast estimate gives you.

## Vela backend

`vela` is the quickest way to answer questions like:

- Does the model map cleanly to the target profile?
- Is the estimated cost broadly acceptable?
- Is the result obviously limited by memory rather than compute?
- Which operators should you inspect first?

### Vela backend: good first commands

Use compatibility first when support is the main concern:

```bash
mlia check model.tflite --target-profile ethos-u55-256 --compatibility --backend vela
```

Use performance when you want a quick estimate and machine-readable output:

```bash
mlia check model.tflite --target-profile ethos-u65-512 --performance --backend vela --json
```

### Vela backend: what to look at in the result

A Vela run is most helpful when you read it in this order:

1. Total cycles.
2. Memory-related cycles.
3. The small number of operators doing most of the work.

That sequence usually tells you whether the next action should be:

- Fix compatibility issues.
- Reduce memory pressure.
- Focus on a few expensive operators.

### When Vela is enough

If the Vela result is already clearly poor, you usually do not need a heavier
backend to tell you that the model needs work. The more useful follow-up is to
work out whether the problem is unsupported mapping, memory cost, or a few
expensive layers.

## Corstone backend

`corstone` is the backend to use when you need a fuller answer to questions
like:

- Why does the run still look expensive after the quick estimate?
- Where is the system spending time beyond the top-level estimate?
- Does a model that looks reasonable in Vela still look reasonable with richer
  backend evidence?
- Do I need to run the supported ExecuTorch AOT path instead of the standard
  TFLite flow?

### Corstone backend: good first commands

Run Corstone directly when you already know you want the deeper path:

```bash
mlia check model.tflite --target-profile ethos-u65-512 --performance --backend corstone
```

Run a supported ExecuTorch AOT path through Corstone:

```bash
mlia check model.pt2 --target-profile ethos-u55-256 --performance --backend corstone-300
```

Run both backends together when you want to compare the quick estimate with the
more detailed path:

```bash
mlia check model.tflite --target-profile ethos-u65-512 --performance --backend vela --backend corstone
```

### Corstone backend: what to look at in the result

A Corstone run is most helpful when you want to move beyond a single total and
inspect:

- Layer-level evidence.
- Traffic-related signals.
- Active-versus-idle style metrics.
- Whether the deeper path changes your view of the model.
- Supported Corstone execution for `.pte` and `.pt2` ExecuTorch workloads.

### When Corstone is worth the extra time

Use Corstone when the quick estimate is not enough to make a decision. That is
usually the case when the result is surprising, when you need stronger evidence
before acting, or when you are trying to explain where the cost is really
coming from.

It is also the backend path to use when you are intentionally validating a
supported ExecuTorch AOT flow instead of the standard TFLite-based workflow.

## A practical workflow

A useful Ethos-U workflow is:

1. Start with `vela`.
2. Check whether the result is mainly about compatibility, memory, or a few
   expensive operators.
3. Bring in `corstone` if the answer still needs more detail.
4. Compare the two results to decide what to change next.

## Cross-links

- See [cli.md](cli.md) for Ethos-U command patterns.
- See [outputs_metrics.md](outputs_metrics.md) for how to interpret the numbers.
- See [troubleshooting.md](troubleshooting.md) when the issue is setup or run
  behaviour rather than backend choice.
