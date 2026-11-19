#!/usr/bin/env python3
"""Render the exact URDF collision assembly at supplied joint positions."""

from pathlib import Path
import argparse
import struct
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation

import render_zero_urdf as base


VIEWS = (
    ("Front", 8, -90),
    ("Rear", 8, 90),
    ("Left", 8, 180),
    ("Right", 8, 0),
    ("Top", 89.9, -90),
    ("Isometric", 28, -52),
)


def axis_rotation(axis, angle):
    transform = np.eye(4)
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm > 1e-12 and abs(angle) > 1e-12:
        transform[:3, :3] = Rotation.from_rotvec(axis / norm * angle).as_matrix()
    return transform


def load_binary_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    if 84 + 50 * count != len(data):
        return base.load_stl(path)
    records = np.frombuffer(data, dtype=dtype, offset=84, count=count)
    return np.asarray(records["vertices"], dtype=float)


def voxel_simplify(triangles, resolution):
    """Merge nearby vertices while retaining connected surface faces."""
    points = triangles.reshape(-1, 3)
    quantized = np.rint(points / resolution).astype(np.int64)
    unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return (unique[faces[keep]].astype(float) * resolution)


def model_triangles(joints):
    root = ET.parse(base.URDF).getroot()
    children = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = base.origin_transform(joint.find("origin"))
        axis_node = joint.find("axis")
        axis = base.vector(
            None if axis_node is None else axis_node.attrib.get("xyz"), "1 0 0"
        )
        angle = float(joints.get(joint.attrib["name"], 0.0))
        transform = origin
        if joint.attrib.get("type") in {"revolute", "continuous"}:
            transform = origin @ axis_rotation(axis, angle)
        children.setdefault(parent, []).append((child, transform))
        child_links.add(child)

    roots = [
        link.attrib["name"] for link in root.findall("link")
        if link.attrib["name"] not in child_links
    ]
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
        scale = base.vector(mesh.attrib.get("scale"), "1 1 1")
        triangles = load_binary_stl(base.MESH_ROOT / filename)
        resolution = 0.004 if filename == "top_chassis.STL" else 0.003
        triangles = voxel_simplify(triangles, resolution) * scale[None, None, :]
        transform = transforms[name] @ base.origin_transform(collision.find("origin"))
        result.append((name, base.transform_triangles(triangles, transform)))
    return result


def shaded_faces(triangles, color, alpha):
    rgb = np.asarray(mcolors.to_rgb(color))
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(first, second)
    length = np.linalg.norm(normals, axis=1)
    normals[length > 0] /= length[length > 0, None]
    light = np.asarray([0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    intensity = np.clip(0.48 + 0.52 * np.abs(normals @ light), 0.35, 1.0)
    return np.column_stack((rgb[None, :] * intensity[:, None], np.full(len(triangles), alpha)))


def add_assembly(axis, items, *, closeup):
    points = []
    for name, triangles in items:
        color = base.color_for(name)
        alpha = 0.64 if name == "top_chassis_link" else 1.0
        axis.add_collection3d(Poly3DCollection(
            triangles,
            facecolors=shaded_faces(triangles, color, alpha),
            edgecolor="none",
            linewidth=0,
            rasterized=True,
        ))
        points.append(triangles.reshape(-1, 3))
    merged = np.vstack(points)
    if closeup:
        # Crop around the arm pedestal and the nearby chassis protrusions.
        axis.set_xlim(-0.32, 0.32)
        axis.set_ylim(-0.32, 0.32)
        axis.set_zlim(-0.13, 0.52)
        axis.set_box_aspect((0.64, 0.64, 0.65))
    else:
        base.equal_axes(axis, (merged.min(axis=0), merged.max(axis=0)), padding=0.02)
    axis.scatter([0], [0], [0], color="#dc2626", s=18, depthshade=False)
    axis.set_axis_off()


def render(items, output, title, subtitle, *, closeup):
    figure = plt.figure(figsize=(18, 12), dpi=170, facecolor="white")
    for number, (name, elev, azim) in enumerate(VIEWS, 1):
        axis = figure.add_subplot(2, 3, number, projection="3d")
        add_assembly(axis, items, closeup=closeup)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(name, fontsize=12, fontweight="bold", pad=2)
    figure.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    figure.text(0.5, 0.018, subtitle, ha="center", fontsize=10)
    figure.tight_layout(rect=(0.005, 0.035, 0.995, 0.955), pad=0.6)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", nargs=6, type=float, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    joints = {"piper_joint%d" % (i + 1): value for i, value in enumerate(args.q)}
    items = model_triangles(joints)
    prefix = Path(args.output_prefix)
    q_text = ", ".join("%.4f" % value for value in args.q)
    subtitle = (
        "Exact active URDF collision meshes and fixed transforms | "
        "Gray: chassis, green: arm, orange: gripper, blue: camera | q=[%s] rad" % q_text
    )
    render(
        items, prefix.with_name(prefix.name + "_full.png"),
        args.label + " — full MoveIt collision assembly", subtitle, closeup=False,
    )
    render(
        items, prefix.with_name(prefix.name + "_base_closeup.png"),
        args.label + " — pedestal/case close-up", subtitle, closeup=True,
    )
    print(prefix.with_name(prefix.name + "_full.png"))
    print(prefix.with_name(prefix.name + "_base_closeup.png"))


if __name__ == "__main__":
    main()
