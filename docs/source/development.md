<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Development

## What counts as development in this repo

This repository provides Ethos-U target support and the Ethos-U backend path.
Changes here often affect target profiles, backend registration, reporting, and
the MLIA user workflow.

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
- Advice generation or reporting logic and user-facing output examples.

## Good review questions

Before you consider a change complete, ask:

- Does the target still appear correctly through `mlia-target list`?
- Do the expected backends still appear through `mlia-backend list`?
- Did the change alter compatibility output, performance output, or advice text?
- Does the docs wording still match the actual workflow?

## Documentation expectations

Keep the top-level README and this repo's `docs/` pages in sync when adding new
Ethos-U profiles, backend options, or major workflows.
