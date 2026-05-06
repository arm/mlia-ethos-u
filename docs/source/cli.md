<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

Use the core `mlia` CLI for Ethos-U runs. This guide focuses on the command patterns that are most useful in practice.

## If you are starting from scratch

A sensible first sequence is:

1. List what is installed with `mlia-target list` and `mlia-backend list`.
2. Run one compatibility command.
3. Run one performance command with `vela`.
4. Only then add `corstone` or `--json`.

## Compatibility

Use compatibility when the first question is whether the model can map cleanly
to the selected Ethos-U target profile.

```bash
mlia check model.tflite --target-profile ethos-u55-256 --compatibility
```

If you want to make the backend path explicit during investigation, add Vela:

```bash
mlia check model.tflite \
  --target-profile ethos-u65-512 \
  --compatibility \
  --backend vela
```

## Fast performance estimate

For most day-to-day work, start with `vela`:

```bash
mlia check model.tflite \
  --target-profile ethos-u55-256 \
  --performance \
  --backend vela
```

Use this when you want the quickest answer about whether the model looks
promising and where the main cost seems to be.

## Deeper performance investigation

Use `corstone` when you want more detail than the quick estimate gives you:

```bash
mlia check model.tflite \
  --target-profile ethos-u65-512 \
  --performance \
  --backend corstone
```

If you are trying to decide whether the deeper path changes your view of the
model, run both backends in one invocation:

```bash
mlia check model.tflite \
  --target-profile ethos-u65-512 \
  --performance \
  --backend vela \
  --backend corstone
```

## ExecuTorch AOT through Corstone

The Corstone path can also run supported ExecuTorch workloads. If
`mlia-converters-pytorch` is installed, MLIA can accept a `.pt2` input and
convert it to `.pte` before running the Corstone backend:

```bash
mlia check model.pt2 \
  --target-profile ethos-u55-256 \
  --performance \
  --backend corstone-300
```

If you already have a prepared `.pte` artifact, you can run it directly:

```bash
mlia check model.pte \
  --target-profile ethos-u85-256 \
  --performance \
  --backend corstone-320
```

This path is currently limited to supported target and backend combinations, so
it should be treated as a Corstone-specific workflow rather than a replacement
for the normal Vela-first TFLite path.

## JSON output

Use `--json` when you want to save results, compare runs, or inspect output more
carefully later.

```bash
mlia check model.tflite \
  --target-profile ethos-u55-256 \
  --performance \
  --backend vela \
  --json
```

## When a run is confusing

If the result is not making sense, try this sequence:

1. Rerun with one explicit backend at a time.
2. Start with compatibility if unsupported mapping looks likely.
3. Compare the top-level cycle counts before reading deeper detail.
4. Use `corstone` only after you know what the quick estimate is telling you.

## Quick rules of thumb

- Use `--compatibility` first if you suspect unsupported operators.
- Use `--backend vela` first if you want the fastest useful answer.
- Use `--backend corstone` when you need deeper evidence before deciding what
  to change.
- Use `.pt2` or `.pte` with Corstone only when you are explicitly exercising
  the ExecuTorch AOT path.
- Use `--json` when the result is something you expect to compare or archive.
