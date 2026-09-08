# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Hatchling metadata hooks."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from hatch_build import MetadataHook


def test_metadata_hook_update_uses_commit_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README metadata links should target the current commit."""
    (tmp_path / "README.md").write_text(
        "[Docs](docs.md)\n![Image](image.png)\n[Section](#section)",
        encoding="utf-8",
    )
    (tmp_path / "docs.md").touch()
    (tmp_path / "image.png").touch()
    monkeypatch.setattr(
        "hatch_build.subprocess.run",
        Mock(
            side_effect=[
                Mock(stdout=f"{tmp_path}\n"),
                Mock(stdout="0123456789abcdef\n"),
            ]
        ),
    )
    hook = MetadataHook(str(tmp_path), {})
    metadata = {"version": "0.1.0"}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": "[Docs](https://github.com/arm/mlia-ethos-u/blob/"
        "0123456789abcdef/docs.md)\n"
        "![Image](https://raw.githubusercontent.com/arm/mlia-ethos-u/"
        "0123456789abcdef/image.png)\n"
        "[Section](https://github.com/arm/mlia-ethos-u/blob/"
        "0123456789abcdef/README.md#section)",
    }


@pytest.mark.parametrize(
    ("version", "revision"),
    [
        ("0.12.2", "v0.12.2"),
        ("0.12.3.dev9+40a5004", "40a5004"),
    ],
)
def test_metadata_hook_update_falls_back_to_version_or_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: str,
    revision: str,
) -> None:
    """README links should use the version or its hash without Git metadata."""
    (tmp_path / "README.md").write_text("[Docs](docs.md)", encoding="utf-8")
    (tmp_path / "docs.md").touch()
    monkeypatch.setattr("hatch_build.subprocess.run", Mock(side_effect=OSError))
    hook = MetadataHook(str(tmp_path), {})
    metadata = {"version": version}

    hook.update(metadata)

    assert metadata["readme"] == {
        "content-type": "text/markdown",
        "text": f"[Docs](https://github.com/arm/mlia-ethos-u/blob/{revision}/docs.md)",
    }
