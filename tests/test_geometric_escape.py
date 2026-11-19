import pytest
from tools.top_chassis_recovery import (
    select_geometric_j2_j3_escape_target, interpolate_j2_j3_escape,
    EXPECTED_CONTACT_PAIRS, geometry_score,
)
START = [0., .07, .031, 0., .23, .1]
def check(q):
    travel=q[1]-START[1]
    return dict(valid=travel>=.03, contact_pairs=[] if travel>=.03 else EXPECTED_CONTACT_PAIRS,
                max_contact_depth_m=.001 if travel<.0015 else .018)
def measure(q):
    gap=.2*(q[1]-START[1])-.006
    return dict(overlap=gap<0, upward_exit_m=max(0.,-gap), clearance_m=max(0.,gap))
def test_depth_jumps_do_not_override_complete_mesh():
    p=select_geometric_j2_j3_escape_target(START,check,measure)
    assert .007<=p['target_geometry']['clearance_m']<.00721
    assert p['probe_count']>50
    previous=START
    for q in interpolate_j2_j3_escape(START,p['target']):
        assert q[1]-previous[1]<=.020*.020+1e-12
        assert previous[2]-q[2]<=.020*.020+1e-12
        assert [q[i] for i in (0,3,4,5)]==[START[i] for i in (0,3,4,5)]
        previous=q
    assert p['required_clearance_m']==.005
    assert p['target'][2]<.03490658503988659

def test_worsening_mesh_is_rejected():
    with pytest.raises(RuntimeError,match='clearance would decrease'):
        select_geometric_j2_j3_escape_target(START,check,lambda q:dict(
            overlap=True, upward_exit_m=.006+q[1]-START[1],clearance_m=0.))
def test_other_contact_is_rejected():
    with pytest.raises(RuntimeError,match='unexpected contacts'):
        select_geometric_j2_j3_escape_target(START,lambda q:dict(valid=False,
            contact_pairs=[['piper_link3','top_chassis_link']]),measure)
def test_clear_start_does_not_load_geometry():
    p=select_geometric_j2_j3_escape_target(START,lambda q:dict(valid=True),lambda q:pytest.fail())
    assert not p['needed'] and p['target']==START

def test_insufficient_gap_is_rejected():
    with pytest.raises(RuntimeError,match='within 0.080'):
        select_geometric_j2_j3_escape_target(START,check,lambda q:dict(
            overlap=q==START,upward_exit_m=.006 if q==START else 0.,clearance_m=0. if q==START else .004))
def test_nonfinite_or_missing_metric_rejected():
    for m in ({},dict(overlap=True,upward_exit_m=float('nan'),clearance_m=0.)):
        with pytest.raises(RuntimeError):geometry_score(m)
def test_reentry_rejected():
    def state(q):
        clear=.03<=q[1]-START[1]<.04
        return dict(valid=clear,contact_pairs=[] if clear else EXPECTED_CONTACT_PAIRS)
    with pytest.raises(RuntimeError,match='re-enters'):
        select_geometric_j2_j3_escape_target(START,state,measure)
def test_contact_geometry_disagreement_rejected():
    with pytest.raises(RuntimeError,match='disagree'):
        select_geometric_j2_j3_escape_target(START,check,lambda q:dict(
            overlap=False,upward_exit_m=0.,clearance_m=.01))
