<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## What counts as development in this repo

This repository provides Ethos-U target support and the Ethos-U backend path.
Changes here often affect target profiles, backend registration, standardized
output, result-owned advice, and the MLIA user workflow.

## Local setup

Use `uv` to create and sync the development environment:

```bash
uv sync --group dev
```

If you only need the test stack:

```bash
uv sync --group test
```

## Common commands

Run the quick CI-aligned test suite:

```bash
uv run pytest -m "not slow" tests/
```

Run the full test suite:

```bash
uv run pytest tests/
```

Run repository checks:

```bash
uv run pre-commit run --all-files
```

Build the package:

```bash
uv build
```

## What usually changes together

When you change one of these areas, review the others too:

- Target profile definitions and target-related tests.
- Vela integration and compatibility or performance expectations.
- Corstone integration and backend registration behaviour.
- Result-owned advice or standardized-output logic and user-facing examples.

## Good review questions

Before you consider a change complete, ask:

- Does the target still appear correctly through `mlia target list`?
- Do the expected backends still appear through `mlia backend list`?
- Did the change alter compatibility output, performance output, or advice text?
- Does the docs wording still match the actual workflow?

## Documentation expectations

Keep the top-level README and this repo's `docs/` pages in sync when adding new
Ethos-U profiles, backend options, or major workflows.

## Runtime boundary actions

Application code must not directly perform runtime network access, launch
external processes, or mutate the filesystem. Repository-owned actions must go
through a gateway in the Corstone boundary package. For example:

```python
from mlia.backend.corstone.boundaries.process import run_installer_with_notice

run_installer_with_notice(logger, corstone_name, dist_dir, command)
```

To add a boundary operation, add an intention-revealing function to the boundary
package and test its behavior. Reviewers must manually verify, with LLM
assistance where useful, that boundary operations remain inside the package and
that user-visible actions announce themselves before crossing the boundary.
Generic Core process gateways must therefore remain wrapped by an Ethos-U
gateway that performs the appropriate announcement.
Derived Vela and Corstone artifacts under the already-announced MLIA output
directory are user output. Build hooks, CI workflows, and tests are outside this
runtime policy.
