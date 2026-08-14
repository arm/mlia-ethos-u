## Context

Corstone-320 packages include a `python` runtime folder alongside the FVP model
payload. The plugin must install that folder and expose it at runtime.

## Goals / Non-Goals

**Goals:**

- Install Corstone-320's `python` folder next to the installed FVP payload.
- Add the minimal Corstone-320 process environment needed by the bundled
  runtime.
- Preserve existing command arguments and non-Corstone-320 behavior.

**Non-Goals:**

- Change Corstone-300, Corstone-310, AVH/static layouts, download, or EULA
  behavior.
- Add generic backend environment scripting.

## Decisions

1. Configure Corstone-320 with the primary model subfolder plus `python`.
2. Build a Corstone-320-only environment overlay that sets `PYTHONHOME` and
   prepends `python/lib` to `LD_LIBRARY_PATH`.
3. Keep command arguments unchanged.

## Risks / Trade-offs

- The installed layout must place `python` next to the primary payload.
- `LD_LIBRARY_PATH` ordering must keep bundled Python libraries first.
