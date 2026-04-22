<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Ethos-U Plugin

This package contains the MLIA target plugin for Arm Ethos-U platforms and
packages the Ethos-U backend integrations used by MLIA.

The package is distributed as `mlia-ethos-u` and contributes:

- The `ethos_u` target plugin.
- The `vela` backend plugin.
- The `corstone` backend plugin.
- Bundled Ethos-U target profiles and Vela configuration assets.

## Table of Contents

- [Overview](#overview)
- [Supported targets](#supported-targets)
- [Backends in this package](#backends-in-this-package)
- [Installation](#installation)
- [Development setup](#development-setup)
- [Common commands](#common-commands)
- [Project layout](#project-layout)
- [Documentation](#documentation)

## Overview

`mlia-ethos-u` is the main MLIA plugin package for Ethos-U inference analysis.
It extends the core MLIA framework with Arm Ethos-U target knowledge, operator
analysis, bundled target profiles, and the backend integrations required for
Vela compilation and Corstone-based performance flows.

This is the package to install when you want MLIA to analyse TFLite models
for Ethos-U55, Ethos-U65, or Ethos-U85 targets.

## Supported targets

Bundled target profiles include:

- `ethos-u55-128`
- `ethos-u55-256`
- `ethos-u65-256`
- `ethos-u65-512`
- `ethos-u85-128`
- `ethos-u85-256`
- `ethos-u85-512`
- `ethos-u85-1024`
- `ethos-u85-2048`

These profiles are shipped under `src/mlia/resources/target_profiles/`.

## Backends in this package

### Vela

The Vela backend is used for compiler-oriented analysis, compatibility checks,
and Ethos-U performance-related reporting based on the Vela toolchain.

### Corstone

The Corstone backend supports simulation-oriented performance flows for
Corstone platforms used in Ethos-U analysis and validation.

## Installation

Install into an environment that already contains `mlia`:

```bash
pip install mlia-ethos-u
```

A typical MLIA workflow then references one of the bundled profiles, for
example:

```bash
mlia check model.tflite --target-profile ethos-u55-256
```

The package depends on `mlia>=0.11.0.dev6` and is intended to be used as part
of a wider MLIA installation rather than as a standalone CLI.

## Development setup

Create a local environment with the project and all development dependencies:

```bash
uv sync --group dev
```

If you only need the test dependencies:

```bash
uv sync --group test
```

This repository currently does not use a committed lock file (`uv.lock`).

## Common commands

Run the quick test suite used in CI:

```bash
uv run pytest -m "not slow" tests/
```

Run the full test suite with coverage:

```bash
uv sync --group test
uv run pytest tests/
```

Run the local quality checks:

```bash
uv run pre-commit run --all-files
```

Build the package:

```bash
uv build
```

## Project layout

- `src/mlia/target/ethos_u/`: target integration, advisor logic, reporting, and
  analysis pipeline for Ethos-U.
- `src/mlia/backend/vela/`: Vela backend integration.
- `src/mlia/backend/corstone/`: Corstone backend integration.
- `src/mlia/resources/target_profiles/`: bundled Ethos-U target profiles.
- `src/mlia/resources/vela/`: bundled Vela configuration.
- `tests/`: unit, integration, and CLI coverage for targets and backends.

## Documentation

Additional package documentation lives in [docs/README.md](docs/README.md).
