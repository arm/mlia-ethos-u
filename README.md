<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA Ethos Plugin

This package provides the Ethos-U target plugin and the Vela and Corstone backend plugins for MLIA.

## Requirements

- Python 3.10
- `uv`

## Development Setup

Create a local environment with the project and all development dependencies:

```bash
uv sync --group dev
```

If you only need the test dependencies:

```bash
uv sync --group test
```

This repository currently does not use a committed lock file (`uv.lock`).

## Common Commands

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

## Project Layout

- `src/mlia/target/ethos_u/`: Ethos-U target integration and advisor logic.
- `src/mlia/backend/vela/`: Vela backend integration.
- `src/mlia/backend/corstone/`: Corstone backend integration.
- `src/mlia/resources/`: bundled target profiles, Vela config, and backend assets.
- `tests/`: unit tests covering plugins, CLI integration, and target/backend behavior.
