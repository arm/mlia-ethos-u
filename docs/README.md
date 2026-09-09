<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Ethos-U Documentation

This directory contains the MkDocs content for the `mlia-ethos-u` repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/target_profiles.md`: Ethos-U variants, bundled profiles, supported inputs, and usage examples
- `source/backends.md`: role of Vela and Corstone backends in the Ethos-U flow
- `source/outputs_metrics.md`: output shapes and key metrics produced by Ethos-U workflows
- `source/cli.md`: practical CLI usage examples for common Ethos-U tasks
- `source/troubleshooting.md`: backend-specific troubleshooting notes
- `source/development.md`: local development, testing, and maintenance workflow
- `source/ethos_u_api_walkthrough.ipynb`: Jupyter notebook walkthrough for the Ethos-U Python API flow

## Build

Install the documentation dependencies in your environment, then build from the
repository root:

```bash
uv sync --no-sources --no-install-project --only-group docs
uv run --no-sync mkdocs build --strict
```

For local preview:

```bash
uv run --no-sync mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

These docs cover the Ethos-U target plugin, bundled profiles, and backend
integrations provided by `mlia-ethos-u`.

## Relationship to the core repo

The main `mlia` repository documents shared CLI behaviour, output structure,
and plugin discovery. Use this documentation for Ethos-U-specific target,
backend, metric, and troubleshooting details.
