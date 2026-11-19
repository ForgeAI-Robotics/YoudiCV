#!/usr/bin/env python3
"""Build the closed-slide chassis collision mesh from the vendor STL.

The vendor mesh contains eight disconnected solids.  Its second-largest
solid is the deployed slide/drawer geometry, which only occupies that space
while open.  The robot currently operates with the slide closed, so retaining
that solid in the collision mesh creates obstacles that are not physically
present.  Visual geometry is deliberately left unchanged for traceability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np


STL_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("vertices", "<f4", (3, 3)),
    ("attribute", "<u2"),
])


def read_binary_stl(path: Path) -> tuple[bytes, np.ndarray]:
    with path.open("rb") as handle:
        header = handle.read(80)
        count_bytes = handle.read(4)
        if len(count_bytes) != 4:
            raise ValueError("invalid binary STL header")
        count = struct.unpack("<I", count_bytes)[0]
        triangles = np.fromfile(handle, dtype=STL_DTYPE, count=count)
    if len(triangles) != count:
        raise ValueError("truncated binary STL")
    return header, triangles


def connected_component_ids(triangles: np.ndarray) -> np.ndarray:
    vertices = triangles["vertices"].reshape(-1, 3)
    # Vendor vertices that should meet are bit-identical in practice.  Six
    # decimal places also tolerates harmless STL round-off at shared edges.
    _unique, inverse = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    face_vertices = inverse.reshape(-1, 3)
    parent = np.arange(len(triangles), dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owner: dict[int, int] = {}
    for face_index, row in enumerate(face_vertices):
        for vertex_index in row:
            prior = owner.setdefault(int(vertex_index), face_index)
            if prior != face_index:
                union(face_index, prior)
    return np.asarray([find(index) for index in range(len(triangles))])


def build_closed_mesh(source: Path, output: Path) -> dict:
    header, triangles = read_binary_stl(source)
    component = connected_component_ids(triangles)
    identifiers, counts = np.unique(component, return_counts=True)
    if len(identifiers) != 8:
        raise RuntimeError(f"expected 8 vendor solids, found {len(identifiers)}")
    descending = np.argsort(counts)[::-1]
    removed_id = identifiers[descending[1]]
    removed = triangles[component == removed_id]
    kept = triangles[component != removed_id]
    removed_vertices = removed["vertices"].reshape(-1, 3)
    expected_faces = 47_964
    if len(removed) != expected_faces:
        raise RuntimeError(
            f"open-slide solid changed: expected {expected_faces} faces, "
            f"found {len(removed)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        label = b"Piper M5A closed-slide collision mesh"
        handle.write((label + header[len(label):])[:80].ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(kept)))
        kept.tofile(handle)
    return {
        "source": str(source),
        "output": str(output),
        "source_faces": int(len(triangles)),
        "closed_faces": int(len(kept)),
        "removed_open_slide_faces": int(len(removed)),
        "removed_open_slide_bounds_m": [
            removed_vertices.min(axis=0).tolist(),
            removed_vertices.max(axis=0).tolist(),
        ],
        "source_component_count": int(len(identifiers)),
        "closed_component_count": int(len(identifiers) - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    report = build_closed_mesh(args.source.resolve(), args.output.resolve())
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.metadata:
        args.metadata.resolve().write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
