<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Contribution Guidelines

The ML Inference Advisor (MLIA) project is open for external contributors and
welcomes contributions. MLIA is licensed under the [Apache-2.0 license](https://spdx.org/licenses/Apache-2.0.html)
and all accepted contributions must have the same license.

This document contains the rules for contributing code to MLIA. All contributed
code must follow these rules before it can be accepted to the main branch of
MLIA.

## Setting up the MLIA plugin repo

First clone the MLIA Ethos-U repository.

```bash
    # Using SSH
    git clone "ssh://git@github.com:arm/mlia-ethos-u.git"
    # Or HTTPS
    git clone "https://github.com/arm/mlia-ethos-u.git"
    cd mlia-ethos-u
    git checkout main
    # git pull is not required upon initial clone but good practice before
    # creating a patch
    git pull
    # Set your username, this must be your real name no pseudonyms or anonymous
    # contributions are accepted.
    git config user.name "FIRST_NAME SECOND_NAME"
    # use the same e-mail you set up your github account with
    git config user.email your@email.address
```

This plugin depends on core functionality provided by the
[`arm/mlia` repository](https://github.com/arm/mlia). Consult that repository
for shared MLIA APIs and behavior; a separate checkout is not required to
contribute to this plugin.

### Pre-Commit Checks

Pre-commit checks help ensure contributions comply with the MLIA coding style
and project policies. Some checks may reformat files automatically. All checks
must pass before a contribution can be merged.

Install the development dependencies, including `pre-commit`, with `uv`:

```bash
    uv sync --group dev
```

Optionally install the Git hooks to run the checks automatically before each
commit:

```bash
    uv run pre-commit install
```

Run all checks manually before opening a pull request. This is the same command
used by continuous integration:

```bash
    uv run pre-commit run --all-files
```

### Commit Messages

For the commit messages, the codebase follows [Conventional Commits](https://www.conventionalcommits.org),
with some customizations. Header description is be capitalized, and the following
commit types are allowed: build, ci, docs, feat, fix, perf, refactor, style, test.

### Sign off

Commit your code using [sign-off](#developer-certificate-of-origin-dco) this
adds a "Signed-off-by" line, required for MLIA contributions.

```bash
    git commit -s -m "fix: your commit message"
```

### Code reviews

This project follows the conventional GitHub pull request flow. See [here](https://docs.github.com/en/pull-requests)
for details of how to create a pull request.

Contributions must go through code review on GitHub. Only reviewed contributions
can go to the main branch of MLIA.

### Reporting bugs

Report bugs by creating GitHub issues. Use the
[`arm/mlia-ethos-u` issue tracker](https://github.com/arm/mlia-ethos-u/issues)
by default.

If the bug is in shared MLIA core functionality rather than this plugin, use
the [`arm/mlia` issue tracker](https://github.com/arm/mlia/issues).

## Developer Certificate of Origin (DCO)

Before the MLIA project accepts your contribution, you need to certify its
origin and give us your permission. To manage this process we use
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/).

To indicate that you agree to the the terms of the DCO, you "sign off" your contribution
by adding a line with your name and e-mail address to every git commit message:

```bash
Signed-off-by: FIRST_NAME SECOND_NAME <your@email.address>
```

You must use your real name, no pseudonyms or anonymous contributions are accepted.

## In File Copyright Notice

In each source file, include the following copyright notice:

```bash
# SPDX-FileCopyrightText: Copyright <years changes were made> <copyright holder>.
# SPDX-License-Identifier: Apache-2.0
```

Note: if an existing file does not conform, please update the license header
as part of your contribution.

## Releases

Official releases are published through [PyPI](https://pypi.org/project/mlia-ethos-u/).

## Development Repository

The development repository is hosted on
[github.com](https://github.com/arm/mlia-ethos-u.git/).

## Continuous Integration

Contributions to MLIA go through testing at the Arm CI system. All unit,
integration and regression tests must pass before a contribution gets merged
to the MLIA main branch.
