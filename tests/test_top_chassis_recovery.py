import pytest

from tools.top_chassis_recovery import (
    EXPECTED_CONTACT_PAIRS,
    J2_RECOVERY_SPEED_RAD_S,
    RECOVERY_COMMAND_PERIOD_SEC,
    interpolate_j2_escape,
    select_j2_escape_target,
    select_j2_j3_escape_target,
    interpolate_j2_j3_escape,
)


START = [0.04, 0.075, 0.0, 0.051, 0.204, 0.0]


def test_valid_start_is_noop():
    result = select_j2_escape_target(
        START, lambda _joints: {"valid": True, "contact_pairs": []}
    )
    assert result["needed"] is False
    assert result["target"] == START


def test_only_expected_chassis_link2_contact_can_recover():
    with pytest.raises(RuntimeError, match="unexpected contacts"):
        select_j2_escape_target(
            START,
            lambda _joints: {
                "valid": False,
                "contact_pairs": [["piper_link3", "top_chassis_link"]],
            },
        )


def test_first_valid_upward_j2_probe_is_selected():
    def check(joints):
        valid = joints[1] >= 0.090 - 1e-12
        return {
            "valid": valid,
            "contact_pairs": [] if valid else EXPECTED_CONTACT_PAIRS,
        }

    result = select_j2_escape_target(START, check)
    assert result["needed"] is True
    assert result["probe_count"] == 3
    assert result["target"][1] == pytest.approx(0.090)
    assert result["target"][:1] + result["target"][2:] == START[:1] + START[2:]


def test_no_valid_state_within_strict_envelope_is_rejected():
    with pytest.raises(RuntimeError, match="within 0.050 rad"):
        select_j2_escape_target(
            START,
            lambda _joints: {
                "valid": False,
                "contact_pairs": EXPECTED_CONTACT_PAIRS,
            },
        )


def test_interpolation_moves_only_j2_at_bounded_speed():
    target = list(START)
    target[1] += 0.030
    samples = interpolate_j2_escape(START, target)
    prior = START
    for sample in samples:
        assert sample[0] == START[0]
        assert sample[2:] == START[2:]
        assert 0.0 < sample[1] - prior[1] <= (
            J2_RECOVERY_SPEED_RAD_S * RECOVERY_COMMAND_PERIOD_SEC + 1e-12
        )
        prior = sample
    assert samples[-1] == pytest.approx(target)


def test_interpolation_rejects_other_joint_motion():
    target = list(START)
    target[0] += 0.001
    target[1] += 0.010
    with pytest.raises(ValueError, match="only change J2"):
        interpolate_j2_escape(START, target)


def test_coupled_escape_checks_penetration_and_keeps_margin():
    visited = []
    def check(joints):
        visited.append(joints)
        rise = joints[1]-START[1]
        valid = rise >= .015-1e-12
        return dict(valid=valid, contact_pairs=[] if valid else EXPECTED_CONTACT_PAIRS,
                    max_contact_depth_m=max(0., .003-.2*rise))
    plan = select_j2_j3_escape_target(START, check)
    assert plan['probe_count'] == 20
    assert plan['target'][1] == pytest.approx(START[1]+.020)
    assert plan['target'][2] == pytest.approx(START[2]-.020)
    assert len(visited) == 21
    samples = interpolate_j2_j3_escape(START, plan['target'])
    prior = START
    for q in samples:
        assert q[1] >= prior[1] and q[2] <= prior[2]
        assert abs(q[1]-prior[1]) <= .02*.02+1e-12
        assert abs(q[2]-prior[2]) <= .02*.02+1e-12
        assert [q[i] for i in (0,3,4,5)] == [START[i] for i in (0,3,4,5)]
        prior = q


def test_coupled_escape_rejects_worsening_depth():
    def check(q):
        return dict(valid=False, contact_pairs=EXPECTED_CONTACT_PAIRS,
                    max_contact_depth_m=.001+(q[1]-START[1]))
    with pytest.raises(RuntimeError, match='increase contact penetration'):
        select_j2_j3_escape_target(START, check)


def test_coupled_escape_rejects_other_collision_and_missing_depth():
    with pytest.raises(RuntimeError, match='unexpected contacts'):
        select_j2_j3_escape_target(START, lambda q: dict(
            valid=False, contact_pairs=[['piper_link3','top_chassis_link']],
            max_contact_depth_m=.001))
    with pytest.raises(RuntimeError, match='finite contact depth'):
        select_j2_j3_escape_target(START, lambda q: dict(
            valid=False, contact_pairs=EXPECTED_CONTACT_PAIRS))


def test_coupled_escape_respects_joint_limit_and_envelope():
    def collision(q):
        return dict(valid=False, contact_pairs=EXPECTED_CONTACT_PAIRS,
                    max_contact_depth_m=.001)
    q=list(START);q[2]=-2.965
    with pytest.raises(RuntimeError, match='within 0.050'):
        select_j2_j3_escape_target(q, collision)
    with pytest.raises(RuntimeError, match='within 0.050'):
        select_j2_j3_escape_target(START, collision)


def test_coupled_escape_does_not_move_when_already_clear():
    result=select_j2_j3_escape_target(START, lambda q: dict(valid=True))
    assert not result['needed']
    assert result['target'] == START
