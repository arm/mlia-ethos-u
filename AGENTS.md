<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Repository Guidelines

## Overview

This repository provides MLIA plugins for Arm Ethos-U targets plus Vela and Corstone backends. The Python package lives under `src/mlia`, and tests live under `tests`.

## Working Rules

- Use `uv` for environment management, test execution, and builds. Do not reintroduce `tox`, `setup.cfg`, or ad hoc virtualenv instructions.
- Keep changes scoped to the task. This repo may have unrelated local edits in flight.
- Preserve bundled resources under `src/mlia/resources/` unless the task explicitly requires updating them.
- Add or update tests when behavior changes.
- Keep new files compatible with the repo's REUSE checks by including the SPDX copyright and license header used in neighboring source files.

## Setup And Validation

Install development dependencies:

```bash
uv sync --group dev
```

Common validation commands:

```bash
uv run pytest -m "not slow" tests/
uv run pre-commit run --all-files
uv build
```

For coverage runs:

```bash
uv sync --group test
uv run pytest tests/
```

## Repo Map

- `src/mlia/target/ethos_u/`: target plugin, config, advisors, data analysis, reporting.
- `src/mlia/backend/vela/`: Vela backend registration, installation, compatibility, and performance logic.
- `src/mlia/backend/corstone/`: Corstone backend registration, installation, and performance helpers.
- `src/mlia/resources/target_profiles/`: target profile TOML files used by tests and runtime configuration.
- `tests/`: regression coverage for CLI, plugin registration, backend behavior, and Ethos-U flows.
- `pre_commit_hooks/check_copyright_header.py`: local hook enforcing copyright-year consistency.

## Change Hygiene

- Prefer targeted test runs for the area you changed, then run broader validation if the change is cross-cutting.
- This repository currently does not use a committed lock file (`uv.lock`).
- If you touch packaging or dependency flows, check `pyproject.toml` and the GitHub workflows together so they stay aligned.
- If you add documentation examples, make sure the commands are executable with `uv` from the repository root.
