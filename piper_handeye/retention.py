"""Bound generated action journals without touching source or calibration data."""

from __future__ import annotations

from pathlib import Path
import re
import shutil


DEFAULT_ACTION_RETENTION = 20
_CAPTURE_STEM = re.compile(r"^(\d{8}_\d{6}_\d{6})")


def prune_old_entries(root: Path | str, *, keep: int = DEFAULT_ACTION_RETENTION) -> list[str]:
    """Keep the newest top-level run directories/files and remove older ones."""
    directory = Path(root)
    if keep < 1:
        raise ValueError("retention count must be at least one")
    if not directory.exists():
        return []
    entries = sorted(
        directory.iterdir(),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    removed = []
    for entry in entries[keep:]:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed.append(str(entry))
    return removed


def prune_capture_groups(
    root: Path | str,
    *,
    keep: int = DEFAULT_ACTION_RETENTION,
) -> list[str]:
    """Keep RGB/depth triplets for the newest capture actions as one group each."""
    directory = Path(root)
    if keep < 1:
        raise ValueError("retention count must be at least one")
    if not directory.exists():
        return []
    groups: dict[str, list[Path]] = {}
    for entry in directory.iterdir():
        match = _CAPTURE_STEM.match(entry.name)
        key = match.group(1) if match is not None else entry.name
        groups.setdefault(key, []).append(entry)
    ordered = sorted(
        groups.values(),
        key=lambda items: max(item.stat().st_mtime_ns for item in items),
        reverse=True,
    )
    removed = []
    for items in ordered[keep:]:
        for entry in items:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed.append(str(entry))
    return removed
