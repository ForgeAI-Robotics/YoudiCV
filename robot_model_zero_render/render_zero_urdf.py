#!/usr/bin/env python3
"""Render the installed Piper URDF collision meshes at six-axis zero."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
URDF = ROOT / "piper_zero_model.urdf"
MESH_ROOT = ROOT / "piper_description" / "meshes"
OUTPUT = ROOT / "piper_zero_collision_model.png"


def vector(text, default):
    return np.asarray([float(value) for value in (text or default).split()])


def origin_transform(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", vector(element.attrib.get("rpy"), "0 0 0")
    ).as_matrix()
    matrix[:3, 3] = vector(element.attrib.get("xyz"), "0 0 0")
    return matrix


def load_stl(path):
    data = path.read_bytes()
    triangles = None
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * count == len(data):
            values = np.frombuffer(data, dtype=np.dtype([
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]), offset=84, count=count)
            triangles = np.asarray(values["vertices"], dtype=float)
    if triangles is None:
        vertices = []
        for line in data.decode("ascii", errors="ignore").splitlines():
            fields = line.strip().split()
            if len(fields) == 4 and fields[0] == "vertex":
                vertices.append([float(value) for value in fields[1:]])
        triangles = np.asarray(vertices, dtype=float).reshape(-1, 3, 3)
    if len(triangles) > 6000:
        indices = np.linspace(0, len(triangles) - 1, 6000, dtype=int)
        triangles = triangles[indices]
    return triangles


def transform_triangles(triangles, transform):
    points = triangles.reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3].reshape(-1, 3, 3)


def color_for(link):
    if link == "top_chassis_link":
        return "#9ca3af"
    if "camera" in link:
        return "#2563eb"
    if "gripper" in link:
        return "#f59e0b"
    if link == "piper_base_link":
        return "#4b5563"
    return "#16a085"


def model_triangles():
    root = ET.parse(URDF).getroot()
    children = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(
            (child, origin_transform(joint.find("origin")))
        )
        child_links.add(child)
    roots = [link.attrib["name"] for link in root.findall("link") if link.attrib["name"] not in child_links]
    transforms = {}

    def walk(link, transform):
        transforms[link] = transform
        for child, joint_transform in children.get(link, []):
            walk(child, transform @ joint_transform)

    for link in roots:
        walk(link, np.eye(4))

    result = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        collision = link.find("collision")
        if collision is None:
            continue
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        filename = Path(mesh.attrib["filename"]).name
        scale = vector(mesh.attrib.get("scale"), "1 1 1")
        triangles = load_stl(MESH_ROOT / filename) * scale[None, None, :]
        transform = transforms[name] @ origin_transform(collision.find("origin"))
        result.append((name, transform_triangles(triangles, transform)))
    return result


def equal_axes(axis, bounds, padding=0.04):
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    center = 0.5 * (low + high)
    radius = 0.5 * float(np.max(high - low)) + padding
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def add_model(axis, items, selected):
    points = []
    for name, triangles in items:
        if name not in selected:
            continue
        points.append(triangles.reshape(-1, 3))
        axis.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolor=color_for(name),
                edgecolor="none",
                alpha=0.88,
                rasterized=True,
            )
        )
    merged = np.vstack(points)
    equal_axes(axis, (merged.min(axis=0), merged.max(axis=0)))
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.grid(True, alpha=0.25)


def main():
    items = model_triangles()
    all_links = {name for name, _ in items}
    arm_links = all_links - {"top_chassis_link"}
    figure = plt.figure(figsize=(18, 9), dpi=150)
    full = figure.add_subplot(1, 2, 1, projection="3d")
    close = figure.add_subplot(1, 2, 2, projection="3d")
    add_model(full, items, all_links)
    add_model(close, items, arm_links)
    full.view_init(elev=23, azim=-55)
    close.view_init(elev=20, azim=-62)
    full.set_title("Complete modeled body (including top_chassis)")
    close.set_title("Arm close-up: links, gripper and camera")
    figure.suptitle(
        "Piper URDF collision model at six-axis zero\n"
        "VIRTUAL CHECK ONLY — PHYSICAL ROBOT WAS NOT MOVED",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "q = [0, 0, 0, 0, 0, 0] rad   |   Orange: gripper   Blue: camera   "
        "Green: arm   Gray: chassis   |   Added rubber tip is NOT represented",
        ha="center",
        fontsize=10,
        color="#991b1b",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.92))
    figure.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
