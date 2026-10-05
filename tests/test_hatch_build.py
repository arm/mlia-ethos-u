# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Hatchling metadata and wheel build hooks."""

from email.parser import Parser
from pathlib import Path
import shutil
import tomllib
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from hatchling.builders.wheel import WheelBuilder

from hatch_build import MetadataHook


def test_wheel_build_uses_python_312_tag(tmp_path: Path) -> None:
    """The configured hook must tag both the filename and wheel metadata."""
    # Build wheel in a tmp dir and check
    # python tag
    root = Path(__file__).resolve().parents[1]
    project_config = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    config = {
        "project": {"name": "wheel-fixture", "version": "1.0"},
        "tool": {"hatch": {"build": project_config["tool"]["hatch"]["build"]}},
    }
    shutil.copy2(root / "hatch_build.py", tmp_path / "hatch_build.py")
    package = tmp_path / "src" / "mlia-ethos-u"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()

    builder = WheelBuilder(str(tmp_path), config=config)
    [wheel_path] = builder.build(directory=str(tmp_path / "dist"))

    assert Path(wheel_path).name == "wheel_fixture-1.0-py312-none-any.whl"
    with ZipFile(wheel_path) as wheel:
        metadata = Parser().parsestr(
            wheel.read("wheel_fixture-1.0.dist-info/WHEEL").decode("utf-8")
        )
        assert metadata.get_all("Tag") == ["py312-none-any"]


def test_metadata_hook_update_uses_commit_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README metadata links should target the current commit."""
    (tmp_path / "README.md").write_text(
        "[Docs](docs.md)\n![Image](image.png)\n[Section](#section)\n"
        "[Details](docs.md#details)",
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
        "0123456789abcdef/README.md#section)\n"
        "[Details](https://github.com/arm/mlia-ethos-u/blob/"
        "0123456789abcdef/docs.md#details)",
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
