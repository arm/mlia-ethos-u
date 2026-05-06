<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Ethos-U Target Profiles

## Overview

This package provides the MLIA target plugin for Arm Ethos-U NPUs. It is the
part of the MLIA ecosystem responsible for Ethos-U-specific target profiles,
operator analysis, reporting, and the backend wiring needed for performance and
compatibility flows.

The plugin is intended for TensorFlow Lite models that will be analysed against
Ethos-U55, Ethos-U65, or Ethos-U85 configurations. It also participates in
supported Corstone ExecuTorch flows for `.pt2` and `.pte` inputs.

## Bundled profiles

The package ships built-in TOML target profiles under
`src/mlia/resources/target_profiles/`.

Bundled profiles include:

- `ethos-u55-128`
- `ethos-u55-256`
- `ethos-u65-256`
- `ethos-u65-512`
- `ethos-u85-128`
- `ethos-u85-256`
- `ethos-u85-512`
- `ethos-u85-1024`
- `ethos-u85-2048`

These profiles let MLIA select the correct accelerator configuration, memory
mode, and backend assumptions without requiring a custom profile for common
hardware targets.

## Supported input formats

The most common input format in this package is:

- TensorFlow Lite (`.tflite`).

Additional formats can also participate in Ethos-U workflows:

- TOSA (`.tosa`, `.tosamlir`).
- ExecuTorch exported program (`.pte`).
- PyTorch exported program (`.pt2`) when `mlia-converters-pytorch` is
  installed.

Quantized TFLite models are still the natural fit for Vela and for most
day-to-day Ethos-U analysis. The `.pt2` and `.pte` paths are specifically tied
to supported Corstone ExecuTorch AOT flows.

## Typical usage

Check compatibility:

```bash
mlia check model.tflite --target-profile ethos-u65-512 --compatibility
```

Estimate performance with Vela:

```bash
mlia check model.tflite --target-profile ethos-u55-256 --performance --backend vela
```

Run a more detailed performance flow with a Corstone backend:

```bash
mlia check model.tflite --target-profile ethos-u65-512 --performance --backend corstone
```

Run a supported ExecuTorch AOT flow through Corstone:

```bash
mlia check model.pt2 --target-profile ethos-u55-256 --performance --backend corstone-300
```

You can also point the Corstone path at a prepared `.pte` artifact directly:

```bash
mlia check model.pte --target-profile ethos-u85-256 --performance --backend corstone-320
```

## Configuration concepts

Ethos-U target profiles in MLIA typically encode:

- Accelerator or MAC configuration.
- Memory mode assumptions.
- System configuration names used by backend tooling.
- Paths to bundled or custom backend configuration assets.

When the built-in profiles are not enough, custom profiles can be layered on top
of this package's bundled backend assets.

## Performance and compatibility outputs

This package contributes the target-specific analysis and reporting pipeline that
produces:

- Operator compatibility information.
- Model-level performance estimates.
- Per-operator reporting and advice.
- Target-aware guidance for compatibility and performance findings.

The exact metrics depend on which backend is used. See [backends.md](backends.md)
for the split between Vela and Corstone responsibilities.
