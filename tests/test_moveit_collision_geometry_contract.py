from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def collision_mesh_for_link(path, link_name):
    root = ET.parse(path).getroot()
    link = next(item for item in root.findall("link") if item.attrib["name"] == link_name)
    return link.find("collision/geometry/mesh").attrib["filename"]


def test_camera_underbody_collision_mesh_is_retained():
    path = ROOT / "robot_model_zero_render/piper_description/description/piper_camera.xacro"
    assert collision_mesh_for_link(path, "piper_camera_base_link").endswith(
        "/piper_camera.STL"
    )


def test_closed_top_chassis_collision_mesh_is_used():
    path = ROOT / "robot_model_zero_render/piper_description/description/piper_arm.xacro"
    assert collision_mesh_for_link(path, "top_chassis_link").endswith(
        "/top_chassis_closed.STL"
    )


def test_open_slide_is_removed_only_from_collision_mesh():
    mesh_dir = ROOT / "robot_model_zero_render/piper_description/meshes"
    source = mesh_dir / "top_chassis.STL"
    closed = mesh_dir / "top_chassis_closed.STL"
    assert source.exists() and closed.exists()
    assert closed.stat().st_size < source.stat().st_size

    root = ET.parse(
        ROOT / "robot_model_zero_render/piper_description/description/piper_arm.xacro"
    ).getroot()
    link = next(
        item for item in root.findall("link")
        if item.attrib["name"] == "top_chassis_link"
    )
    visual = link.find("visual/geometry/mesh").attrib["filename"]
    collision = link.find("collision/geometry/mesh").attrib["filename"]
    assert visual.endswith("/top_chassis.STL")
    assert collision.endswith("/top_chassis_closed.STL")
