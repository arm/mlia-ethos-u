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
uv run mkdocs build --strict
```

For local preview:

```bash
uv run mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

These docs focus on what now lives in the split `mlia-ethos-u` repo: the target
plugin, bundled profiles, and backend integrations.

## Relationship to the core repo

Core CLI behaviour, shared output structure, and plugin-discovery concepts
remain documented in the main `mlia` repo. Use this docs tree for Ethos-U-
specific target, backend, metric, and troubleshooting detail.
