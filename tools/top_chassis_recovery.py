"""Pure policy helpers for escape-only top-chassis boundary recovery."""

from __future__ import annotations

import math


EXPECTED_CONTACT_PAIRS = [["piper_link2", "top_chassis_link"]]
J2_INDEX = 1
J2_RECOVERY_STEP_RAD = 0.005
J2_RECOVERY_MAX_RAD = 0.050
J2_RECOVERY_SPEED_RAD_S = 0.020
RECOVERY_COMMAND_PERIOD_SEC = 0.020
J3_INDEX = 2
JOINT_LIMITS = [(-2.618, 2.618), (0., 3.14), (-2.967, 0.03490658503988659),
                (-1.745, 1.745), (-1.22, 1.22), (-2.0944, 2.0944)]
GEOMETRIC_RECOVERY_MAX_RAD = .080
ESCAPE_REQUIRED_CLEARANCE_M = .005
ESCAPE_PLANNED_CLEARANCE_M = .007


def geometry_score(metric):
    """Signed directional exit while overlapping; true gap once separated."""
    for key in ('upward_exit_m', 'clearance_m'):
        value = float(metric.get(key, float('nan')))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError('invalid full-mesh escape metric')
    if not isinstance(metric.get('overlap'), bool):
        raise RuntimeError('full-mesh overlap flag missing')
    return -metric['upward_exit_m'] if metric['overlap'] else metric['clearance_m']


def require_expected_contacts(result):
    if not result['valid'] and result.get('contact_pairs') != EXPECTED_CONTACT_PAIRS:
        raise RuntimeError('boundary recovery rejected unexpected contacts=%r'
                           % result.get('contact_pairs'))


def select_geometric_j2_j3_escape_target(start, check_state, measure):
    start = [float(v) for v in start]
    if len(start)!=6 or not all(math.isfinite(v) for v in start):
        raise ValueError('recovery start must contain six finite joints')
    if any(not low-.005<=v<=high+.005 for v,(low,high) in zip(start,JOINT_LIMITS)):
        raise RuntimeError('recovery start exceeds joint limits')
    initial = check_state(start)
    if initial['valid']:
        return dict(needed=False,target=start,initial=initial,probe_count=0)
    require_expected_contacts(initial)
    first_metric=measure(start)
    score=geometry_score(first_metric)
    if not first_metric['overlap']:
        raise RuntimeError('full-mesh and MoveIt contact disagree')
    checked=[]
    moveit_clear=False
    for index in range(1,int(round(GEOMETRIC_RECOVERY_MAX_RAD/.001))+1):
        q=list(start);q[1]+=index*.001;q[2]-=index*.001
        if q[1]>JOINT_LIMITS[1][1] or q[2]<JOINT_LIMITS[2][0]:break
        result=check_state(q)
        require_expected_contacts(result)
        if moveit_clear and not result['valid']:
            raise RuntimeError('boundary recovery re-enters collision after escape')
        moveit_clear=moveit_clear or result['valid']
        metric=measure(q)
        current_score=geometry_score(metric)
        # 10 um covers two 5-um bisection bins; this is a geometric numerical
        # tolerance, not permission to deepen a mesh contact.
        if current_score < score-.00001:
            raise RuntimeError('full-mesh escape clearance would decrease')
        score=current_score
        checked.append(dict(joints_rad=q,state=result,geometry=metric))
        if (result['valid'] and not metric['overlap']
                and metric['clearance_m']>=ESCAPE_PLANNED_CLEARANCE_M
                and q[2]<=JOINT_LIMITS[2][1]):
            return dict(needed=True,target=q,initial=initial,target_state=result,
                        probe_count=index,checked_path=checked,moving_joints=['J2','J3'],
                        initial_geometry=first_metric,target_geometry=metric,
                        metric_kind='full_mesh_upward_exit_then_distance',
                        required_clearance_m=ESCAPE_REQUIRED_CLEARANCE_M,
                        planned_clearance_m=ESCAPE_PLANNED_CLEARANCE_M)
    raise RuntimeError('no full-mesh guarded J2/J3 escape within 0.080 rad')


def validate_escape_state(result, initial_depth):
    """Reject new contacts or overlap exceeding the initial contact depth.

    Mesh contact reports can switch triangles between adjacent poses; their
    reported depth is not a continuous signed-distance measurement. Compare
    to the fixed initial ceiling, never accept a larger initial overlap.
    """
    if result['valid']:
        return 0.0
    if result.get('contact_pairs') != EXPECTED_CONTACT_PAIRS:
        raise RuntimeError('boundary recovery rejected unexpected contacts=%r'
                           % result.get('contact_pairs'))
    depth = float(result.get('max_contact_depth_m', float('nan')))
    if not math.isfinite(depth) or depth <= 0.:
        raise RuntimeError('boundary recovery needs finite contact depth')
    if depth > initial_depth + .00005:
        raise RuntimeError('boundary recovery would increase contact penetration')
    return depth


def select_j2_j3_escape_target(start, check_state):
    """Lift the upper arm while folding the forearm: J2+, J3-.

    Every 0.001-rad path sample is checked against the initial penetration
    ceiling before enabling motion. Other
    collision pairs remain forbidden; the normal collision geometry/padding
    is never disabled. The unchanged 0.050-rad envelope applies to each joint.
    """
    start = [float(value) for value in start]
    if len(start) != 6 or not all(math.isfinite(v) for v in start):
        raise ValueError('recovery start must contain six finite joints')
    for v, (lower, upper) in zip(start, JOINT_LIMITS):
        if not lower-.005 <= v <= upper+.005:
            raise RuntimeError('recovery start exceeds joint limits')
    initial = check_state(start)
    if initial['valid']:
        return dict(needed=False, target=start, initial=initial, probe_count=0)
    depth = validate_escape_state(initial, float('inf'))
    first_clear = None
    checked = []
    for index in range(1, 51):
        target = list(start)
        target[1] += index*.001
        target[2] -= index*.001
        # A tiny encoder overshoot may only move inward to the legal limit.
        if target[1] > JOINT_LIMITS[1][1] or target[2] < JOINT_LIMITS[2][0]:
            break
        result = check_state(target)
        validate_escape_state(result, depth)
        checked.append(dict(joints_rad=target, state=result))
        if result['valid'] and target[2] <= JOINT_LIMITS[2][1] and first_clear is None:
            first_clear = index
        if first_clear is not None and not result['valid']:
            raise RuntimeError('boundary recovery re-enters collision after escape')
        # Continue 0.005 rad beyond the first clear state within the same
        # strictly validated envelope, avoiding the numerical boundary.
        if first_clear is not None and index >= first_clear+5:
            return dict(needed=True, target=target, initial=initial,
                        target_state=result, probe_count=index,
                        checked_path=checked, moving_joints=['J2', 'J3'])
    raise RuntimeError('no guarded J2/J3 escape within 0.050 rad')


def interpolate_j2_j3_escape(start, target):
    start, target = list(start), list(target)
    if len(start) != 6 or len(target) != 6:
        raise ValueError('recovery states must contain six joints')
    if any(abs(target[i]-start[i]) > 1e-12 for i in (0, 3, 4, 5)):
        raise ValueError('boundary recovery may only change J2/J3')
    up, fold = target[1]-start[1], start[2]-target[2]
    if not 0. < up <= GEOMETRIC_RECOVERY_MAX_RAD+1e-12 or abs(up-fold) > 1e-12:
        raise ValueError('J2+ and J3- must travel equally, at most 0.080 rad')
    count = max(2, int(math.ceil(up/J2_RECOVERY_SPEED_RAD_S/RECOVERY_COMMAND_PERIOD_SEC)))
    return [[start[i]+(target[i]-start[i])*index/count for i in range(6)]
            for index in range(1, count+1)]


def select_j2_escape_target(start, check_state):
    """Find the first collision-free state by moving only J2 upward.

    Recovery is deliberately unavailable for every other collision pair.  A
    valid starting state is a no-op, which lets callers use this before every
    return without weakening normal MoveIt validation.
    """
    start = [float(value) for value in start]
    if len(start) != 6 or not all(math.isfinite(value) for value in start):
        raise ValueError("recovery start must contain six finite joints")
    initial = check_state(start)
    if initial["valid"]:
        return {
            "needed": False,
            "target": start,
            "initial": initial,
            "probe_count": 0,
        }
    if initial.get("contact_pairs") != EXPECTED_CONTACT_PAIRS:
        raise RuntimeError(
            "boundary recovery rejected unexpected contacts=%r"
            % initial.get("contact_pairs")
        )
    probes = int(round(J2_RECOVERY_MAX_RAD / J2_RECOVERY_STEP_RAD))
    for index in range(1, probes + 1):
        target = list(start)
        target[J2_INDEX] += index * J2_RECOVERY_STEP_RAD
        result = check_state(target)
        if result["valid"]:
            return {
                "needed": True,
                "target": target,
                "initial": initial,
                "target_state": result,
                "probe_count": index,
            }
        if result.get("contact_pairs") != EXPECTED_CONTACT_PAIRS:
            raise RuntimeError(
                "boundary recovery encountered unexpected contacts=%r"
                % result.get("contact_pairs")
            )
    raise RuntimeError("no collision-free J2 escape within 0.050 rad")


def interpolate_j2_escape(start, target):
    start, target = list(start), list(target)
    if len(start) != 6 or len(target) != 6:
        raise ValueError("recovery states must contain six joints")
    for index in range(6):
        if index != J2_INDEX and abs(float(target[index]) - float(start[index])) > 1e-12:
            raise ValueError("boundary recovery may only change J2")
    travel = float(target[J2_INDEX]) - float(start[J2_INDEX])
    if not 0.0 < travel <= J2_RECOVERY_MAX_RAD + 1e-12:
        raise ValueError("boundary recovery must move J2 upward by at most 0.050 rad")
    count = max(
        2,
        int(math.ceil(
            travel / J2_RECOVERY_SPEED_RAD_S / RECOVERY_COMMAND_PERIOD_SEC
        )),
    )
    samples = []
    for index in range(1, count + 1):
        sample = list(start)
        sample[J2_INDEX] = float(start[J2_INDEX]) + travel * index / count
        samples.append(sample)
    return samples


def validate_escape_transition_pose(start, actual):
    """Bound passive mode-transition drift before a fresh full path preflight.

    Only an inward J3 shift and non-lowering J2 are eligible. Other joints
    retain the existing 0.010-rad drift bound. This alone authorizes no motion.
    """
    if len(start)!=6 or len(actual)!=6 or not all(math.isfinite(x) for x in actual):
        raise RuntimeError('invalid escape transition feedback')
    if any(abs(actual[i]-start[i])>.010 for i in (0,3,4,5)):
        raise RuntimeError('escape transition moved a fixed joint')
    up=actual[1]-start[1];fold=start[2]-actual[2]
    if not -.001<=up<=GEOMETRIC_RECOVERY_MAX_RAD or not -.001<=fold<=GEOMETRIC_RECOVERY_MAX_RAD:
        raise RuntimeError('escape transition moved outside recovery direction/envelope')
    if any(not low-.005<=q<=high+.005 for q,(low,high) in zip(actual,JOINT_LIMITS)):
        raise RuntimeError('escape transition feedback exceeds joint limits')
