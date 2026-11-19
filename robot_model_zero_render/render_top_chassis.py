#!/usr/bin/env python3
"""Render the full top-chassis mesh alone and assembled with the Piper arm."""

from pathlib import Path
import struct
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_zero_urdf as base_render


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "piper_top_chassis_model.png"


def load_full_binary_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        raise ValueError(f"expected binary STL: {path}")
    records = np.frombuffer(
        data,
        dtype=np.dtype([
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]),
        offset=84,
        count=count,
    )
    return np.asarray(records["vertices"], dtype=float)


def chassis_world_triangles() -> np.ndarray:
    triangles = load_full_binary_stl(base_render.MESH_ROOT / "top_chassis.STL")
    transform = np.eye(4)
    angle = -np.pi / 2.0
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    transform[:3, 3] = [-0.0649972, 0.0, -0.7642475852]
    return base_render.transform_triangles(triangles, transform)


def add_mesh(axis, triangles, color, *, alpha=1.0):
    axis.add_collection3d(Poly3DCollection(
        triangles,
        facecolor=color,
        edgecolor="none",
        linewidth=0.0,
        alpha=alpha,
        rasterized=True,
    ))


def style_axis(axis, points, *, elev, azim):
    base_render.equal_axes(axis, (points.min(axis=0), points.max(axis=0)), padding=0.035)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.view_init(elev=elev, azim=azim)
    axis.grid(True, alpha=0.20)


def main() -> None:
    chassis = chassis_world_triangles()
    items = base_render.model_triangles()

    figure = plt.figure(figsize=(17, 8), dpi=150)
    alone = figure.add_subplot(1, 2, 1, projection="3d")
    assembled = figure.add_subplot(1, 2, 2, projection="3d")

    add_mesh(alone, chassis, "#94a3b8")
    alone.scatter([0], [0], [0], color="#dc2626", s=45, depthshade=False)
    alone.text(0, 0, 0.025, "Piper base origin", color="#991b1b", fontsize=9)
    chassis_points = chassis.reshape(-1, 3)
    style_axis(alone, chassis_points, elev=27, azim=-58)
    alone.set_title("M5A top_chassis collision mesh\n0.711 × 0.437 × 0.598 m envelope")

    all_points = [chassis_points]
    add_mesh(assembled, chassis, "#94a3b8")
    for name, triangles in items:
        if name == "top_chassis_link":
            continue
        add_mesh(assembled, triangles, base_render.color_for(name), alpha=1.0)
        all_points.append(triangles.reshape(-1, 3))
    merged = np.vstack(all_points)
    style_axis(assembled, merged, elev=24, azim=-55)
    assembled.set_title("URDF assembly at q = [0, 0, 0, 0, 0, 0] rad")

    figure.suptitle(
        "Piper mounting chassis model used by the URDF / collision scene",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Gray: top_chassis   Green: arm   Orange: gripper   Blue: camera   "
        "Geometry comes directly from the project STL and fixed URDF transform",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.93))
    figure.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
