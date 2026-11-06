# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem boundary gateways for Ethos-U backend cache artifacts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

from mlia.utils.filesystem import temp_directory


@contextmanager
def temporary_supported_ops_directory() -> Generator[Path, None, None]:
    """Create and clean up an isolated directory for the Vela report."""
    with temp_directory() as directory:
        yield directory


def publish_supported_ops_cache(path: Path, report: str) -> None:
    """Atomically publish a complete supported-ops report in the user cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(report)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
