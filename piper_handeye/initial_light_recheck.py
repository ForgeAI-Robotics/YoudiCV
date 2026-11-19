"""Small initial-lamp view change, before any approach or contact is allowed."""
import numpy as np

from .kinematics import forward_kinematics
from .robot import PIPER_JOINT_LIMITS


def choose_initial_recheck_joints(current, original, handeye, panel, matrix,
                                  calibration_size, image_size, bbox):
    """Return a +/-0.025 rad J5 candidate; caller must collision-check it.

    Require a far, fully visible target throughout the candidate, <=10 mm TCP
    displacement and <=2 mm approach to the panel. This is not a close approach
    or a way to bypass depth safety. Maximum offset from the original is 0.05.
    """
    current = np.asarray(current, dtype=float)
    original = np.asarray(original, dtype=float)
    panel = np.asarray(panel, dtype=float)
    handeye = np.asarray(handeye, dtype=float)
    if (current.shape != (6,) or original.shape != (6,) or panel.shape != (4, 4)
            or handeye.shape != (4, 4)
            or not np.isfinite(np.r_[current, original, panel.ravel(), handeye.ravel()]).all()):
        raise ValueError("invalid initial view geometry")
    width, height = map(float, image_size)
    cw, ch = map(float, calibration_size)
    k = np.asarray(matrix, dtype=float).copy()
    box = np.asarray(bbox, dtype=float)
    if (k.shape != (3, 3) or box.shape != (4,) or min(width, height, cw, ch) <= 0
            or not np.isfinite(np.r_[k.ravel(), box]).all()
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("invalid initial view camera geometry")
    k[0] *= width / cw
    k[1] *= height / ch
    target = np.r_[panel[:3, 3], 1.0]
    normal = panel[:3, 2]
    if not .99 <= np.linalg.norm(normal) <= 1.01:
        raise ValueError("invalid panel normal")
    def project(q):
        xyz = (np.linalg.inv(forward_kinematics(q) @ handeye) @ target)[:3]
        if not np.isfinite(xyz).all() or xyz[2] <= .15:
            raise ValueError("initial view target too close to camera")
        uv = k @ xyz
        return uv[:2] / uv[2], xyz[2]
    before, depth = project(current)
    tcp = forward_kinematics(current)[:3, 3]
    side = float((tcp - target[:3]) @ normal)
    if abs(side) < .15:
        return None, {"reason": "initial_recheck_requires_15cm_panel_clearance"}
    candidates = []
    for delta in (-.025, .025):
        goal = current.copy()
        goal[4] += delta
        # Continue away from the initial pose on the second view, never bounce
        # back to the same glare image and count it as independent evidence.
        if abs(goal[4] - original[4]) <= abs(current[4] - original[4]) + .001:
            continue
        if abs(goal[4] - original[4]) > .050001:
            continue
        margin = float("inf")
        valid = True
        for fraction in np.linspace(0, 1, 11):
            q = current + fraction * (goal - current)
            if np.any(q < PIPER_JOINT_LIMITS[:, 0]) or np.any(q > PIPER_JOINT_LIMITS[:, 1]):
                valid = False; break
            point = forward_kinematics(q)[:3, 3]
            distance = float((point - target[:3]) @ normal) * np.sign(side)
            if (distance < max(.15, abs(side) - .002)
                    or np.linalg.norm(point - tcp) > .01):
                valid = False; break
            try:
                pixel, z = project(q)
            except ValueError:
                valid = False; break
            center = (box[:2] + box[2:]) / 2 + pixel - before
            half = (box[2:] - box[:2]) / 2 * depth / z
            low, high = center - half, center + half
            margin = min(margin, *low, width - high[0], height - high[1])
            if margin < 10:
                valid = False; break
        if valid:
            candidates.append((margin, goal, delta))
    if not candidates:
        return None, {"reason": "no_safe_bounded_initial_view_candidate"}
    margin, goal, delta = max(candidates, key=lambda item: item[0])
    return goal, {"delta_j5_rad": delta, "total_j5_offset_rad": float(goal[4] - original[4]),
                  "predicted_min_edge_margin_px": margin, "initial_panel_clearance_m": abs(side)}
