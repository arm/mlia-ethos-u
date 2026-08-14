## Why

Corstone-320 FVP packages include a bundled Python runtime that must be
installed with the model payload and exposed when the FVP runs.

## What Changes

- Configure Corstone-320 installs to include the package `python` folder.
- Set only the required Corstone-320 process environment values at runner time:
  `PYTHONHOME` and `LD_LIBRARY_PATH` with bundled `python/lib`.
- Preserve existing Corstone-300, Corstone-310, and AVH/static backend behavior.

## Capabilities

### New Capabilities

- `corstone-320-python-runtime`: Installs the bundled Python runtime and prepares
  the Corstone-320 execution environment.

## Impact

- Corstone backend installation configuration in
  `src/mlia/backend/corstone/install.py`.
- Corstone runner command construction in
  `src/mlia/backend/corstone/performance.py`.
- Focused tests for Corstone-320 installation metadata and command environment.
