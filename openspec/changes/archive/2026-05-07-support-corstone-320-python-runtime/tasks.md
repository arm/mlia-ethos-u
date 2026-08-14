## 1. Tests

- [x] 1.1 Verify Corstone-320 installs the primary payload plus `python`.
- [x] 1.2 Add a regression test showing Corstone-300 and Corstone-310 keep their
  single-subfolder installation configuration.
- [x] 1.3 Verify Corstone-320 command env sets `PYTHONHOME` and prepends
  `python/lib` to `LD_LIBRARY_PATH`.
- [x] 1.4 Verify host `LD_LIBRARY_PATH` is preserved and other Corstone command
  envs are unchanged.

## 2. Core Implementation

- [x] 2.1 Configure Corstone-320 with the primary payload plus `python`.
- [x] 2.2 Derive Corstone-320 `PYTHONHOME` and `LD_LIBRARY_PATH` from
  `backend_path`.
- [x] 2.3 Attach the env only to Corstone-320 commands.

## 3. Validation

- [x] 3.1 Run focused Corstone installation and performance tests.
- [x] 3.2 Run `uv run pre-commit run --all-files`.
- [ ] 3.3 Run `uv run pytest -m "not slow" tests/` if the focused tests pass.
- [x] 3.4 Run real CLI validation for Corstone-320 install layout and a
  Corstone-320 performance check with a real model.
