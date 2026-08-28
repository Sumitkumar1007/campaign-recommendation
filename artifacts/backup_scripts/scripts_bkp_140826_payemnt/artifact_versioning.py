from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _version_stamp(now: datetime | None = None) -> str:
    current = now.astimezone(UTC) if now else datetime.now(UTC)
    return current.strftime("%Y%m%d")


def next_versioned_path(base_path: Path, *, now: datetime | None = None) -> Path:
    stamp = _version_stamp(now)
    pattern = f"{base_path.stem}_{stamp}_v*{base_path.suffix}"
    highest = 0
    for existing in base_path.parent.glob(pattern):
        name = existing.stem
        marker = f"_{stamp}_v"
        if marker not in name:
            continue
        try:
            version = int(name.rsplit(marker, 1)[1])
        except ValueError:
            continue
        highest = max(highest, version)
    next_version = highest + 1
    return base_path.with_name(f"{base_path.stem}_{stamp}_v{next_version:03d}{base_path.suffix}")


def next_versioned_directory(base_dir: Path, *, now: datetime | None = None) -> Path:
    stamp = _version_stamp(now)
    pattern = f"{base_dir.name}_{stamp}_v*"
    highest = 0
    for existing in base_dir.parent.glob(pattern):
        marker = f"_{stamp}_v"
        if marker not in existing.name:
            continue
        try:
            version = int(existing.name.rsplit(marker, 1)[1])
        except ValueError:
            continue
        highest = max(highest, version)
    next_version = highest + 1
    return base_dir.parent / f"{base_dir.name}_{stamp}_v{next_version:03d}"


def write_versioned_json(*, payload: dict[str, Any], latest_path: Path, versioned_path: Path) -> None:
    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    versioned_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")


def copy_to_latest(*, source_path: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, latest_path)
