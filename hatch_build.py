# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatchling hook used to customize package metadata."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from string import Template

from hatchling.metadata.plugin.interface import MetadataHookInterface


def _replace_markdown_relative_paths(path: Path, file_name: str, revision: str) -> str:
    """Replace relative paths in md file with links to GitHub."""
    md_url = Template("https://github.com/arm/mlia-ethos-u/blob/$rev/$link")
    img_url = Template("https://raw.githubusercontent.com/arm/mlia-ethos-u/$rev/$link")
    md_link_pattern = r"(!?\[.+?\]\((.+?)\))"

    content = path.joinpath(file_name).read_text(encoding="utf-8")
    for match, link in re.findall(md_link_pattern, content):
        if link.startswith("#") or path.joinpath(link).exists():
            if link.startswith("#"):
                new_url = md_url.substitute(rev=revision, link=file_name + link)
            else:
                template = img_url if match[0] == "!" else md_url
                new_url = template.substitute(rev=revision, link=link)
            target = f"({link})"
            md_link = match.replace(target, f"({new_url})", 1)
            content = content.replace(match, md_link)
    return content


def _get_revision(root: Path, fallback: str) -> str:
    """Return the current commit hash, or fallback when Git is unavailable."""
    try:
        worktree = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

        # The source package placed inside a git repo,
        # but not a git repo itself
        if Path(worktree.stdout.strip()).resolve() != root.resolve():
            return fallback

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback

    return revision.stdout.strip() or fallback


def _get_revision_fallback(version: str) -> str:
    """Extract the commit hash from a development version when available."""
    match = re.fullmatch(r"\d+\.\d+\.\d+\.dev\d+\+([A-Za-z0-9]+)", version)
    return match.group(1) if match else f"v{version}"


class MetadataHook(MetadataHookInterface):
    """Rewrite README links in package metadata."""

    def update(self, metadata: dict) -> None:
        """Mutate Hatch metadata with rewritten README content."""
        root = Path(self.root)
        version = str(metadata.get("version", ""))
        revision = _get_revision(root, _get_revision_fallback(version))
        pypi_md = _replace_markdown_relative_paths(root, "README.md", revision)
        if os.getenv("MLIA_DEBUG"):
            (root / "PYPI.md").write_text(pypi_md)
        metadata["readme"] = {"content-type": "text/markdown", "text": pypi_md}
