#!/usr/bin/env python3
"""Create an evaluation CSV from ROOT/{ON,OFF,UNCERTAIN} image folders."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--target", required=True, help="button label shared by these images")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    for state in ("ON", "OFF", "UNCERTAIN"):
        folder = root / state
        if folder.is_dir():
            rows.extend((str(path.resolve()), str(args.target), state) for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.casefold() in EXTENSIONS)
    if not rows:
        raise RuntimeError(f"no images found below {root}/ON, OFF or UNCERTAIN")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image", "target", "state", "gt_x1", "gt_y1", "gt_x2", "gt_y2"))
        for image, target, state in rows:
            writer.writerow((image, target, state, "", "", "", ""))
    print(f"wrote {len(rows)} samples to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
