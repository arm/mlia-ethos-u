# corstone-320-python-runtime Specification

## Purpose
Describe Corstone-320 package installation and command execution requirements
for the bundled Python runtime shipped with the FVP package.

## Requirements
### Requirement: Corstone-320 installs bundled Python runtime

The system SHALL install the Corstone-320 package's `python` folder as a
supporting payload next to the primary FVP payload.

#### Scenario: Corstone-320 package install includes Python payload

- **WHEN** Corstone-320 is installed from a valid package layout
- **THEN** the installed backend contains the primary FVP payload and `python`

#### Scenario: Other Corstone package installs are unchanged

- **WHEN** Corstone-300 or Corstone-310 package installation is configured
- **THEN** the installation uses the existing single primary payload behavior

### Requirement: Corstone-320 (default/FVP profile) command uses bundled Python environment

The system SHALL run Corstone-320 commands for the installed FVP package (profile `default`) with an environment that exposes the installed bundled Python runtime.

#### Scenario: Python environment is configured for Corstone-320 default profile

- **WHEN** a Corstone-320 command is built for profile `default`
- **THEN** the command environment sets `PYTHONHOME` to that `python` folder
  and prepends its `lib` directory to `LD_LIBRARY_PATH`

#### Scenario: Existing library path is preserved

- **WHEN** a Corstone-320 command is built for profile `default` and the process already has
  `LD_LIBRARY_PATH` set
- **THEN** the command environment keeps the existing value after the bundled
  Python library path

#### Scenario: Other profiles/backends are unchanged

- **WHEN** a command is built for Corstone-320 profile `AVH`, or for Corstone-300 or Corstone-310
- **THEN** the command environment remains unchanged
