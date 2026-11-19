"""Measured feedback, safety precedence and bounded waits; no robot access."""
import pytest
from piper_handeye.joint_convergence import wait_for_joint_convergence, JointConvergenceError


class Clock:
    def __init__(self): self.t = 100.
    def __call__(self): return self.t
    def sleep(self, sec): self.t += sec


def run(clock, snapshot, safety=lambda: None, **kwargs):
    return wait_for_joint_convergence([0.]*6, snapshot, safety, clock=clock, sleep=clock.sleep, **kwargs)


def test_actual_first_seven_degree_error_then_one_point_four_degrees_converges():
    c = Clock()
    def feedback():
        error = .1225 if c.t < 100.12 else .025599856
        return round((c.t-100)*100), c.t, [error]*6
    result = run(c, feedback)
    assert result['first_error_rad'] == .1225
    assert result['final_error_rad'] == .025599856
    assert result['tolerance_rad'] == .03 and .19 <= result['elapsed_sec'] < .3


def test_normal_completion_returns_early_without_whole_timeout_delay():
    c = Clock()
    audit = run(c, lambda: (round(c.t*100), c.t, [.001]*6))
    assert .08 <= audit['elapsed_sec'] < .11


@pytest.mark.parametrize('error', [.03001, .1225])
def test_persistent_error_does_not_widen_original_tolerance(error):
    c = Clock()
    with pytest.raises(JointConvergenceError) as caught:
        run(c, lambda: (round(c.t*100), c.t, [error]*6))
    assert caught.value.audit['outcome'] == 'timeout'
    assert 1 <= caught.value.audit['elapsed_sec'] < 1.02
    assert caught.value.audit['final_error_rad'] == error


@pytest.mark.parametrize('joints,received', [([0.]*6,99.8), ([float('nan')]*6,100.), ([0.]*5,100.)])
def test_stale_invalid_feedback_fails_immediately(joints, received):
    c = Clock()
    with pytest.raises(JointConvergenceError, match='invalid_or_stale'):
        run(c, lambda: (1,received,joints))
    assert c.t == 100.


def test_duplicate_frame_never_confirms_and_eventually_expires():
    c = Clock()
    with pytest.raises(JointConvergenceError, match='stale') as caught:
        run(c, lambda: (1,100.,[0.]*6))
    assert caught.value.audit['distinct_samples'] == 1


def test_cached_pre_completion_frame_not_counted():
    c = Clock()
    with pytest.raises(JointConvergenceError) as caught:
        run(c, lambda: (1,99.99,[0.]*6))
    assert caught.value.audit['distinct_samples'] == 0


def test_out_of_tolerance_sample_clears_stable_window():
    c = Clock()
    def feedback():
        return round(c.t*100),c.t,[.05 if 100.05 <= c.t <100.07 else .01]*6
    assert run(c,feedback)['elapsed_sec'] >= .15


def test_safety_stop_wins_even_at_confirmation_boundary():
    c = Clock()
    calls=[]
    def stop():
        calls.append(c.t)
        if c.t >= 100.07: raise RuntimeError('emergency stop')
    with pytest.raises(RuntimeError, match='emergency stop') as caught:
        run(c, lambda:(round(c.t*100),c.t,[0.]*6), stop)
    assert caught.value.convergence_audit['outcome'] == 'safety_or_callback_failure'


def test_encoder_sequence_reset_rejected():
    c = Clock()
    with pytest.raises(JointConvergenceError, match='reversed'):
        run(c,lambda:(10 if c.t == 100. else 1,c.t,[0.]*6))


def test_scheduler_gap_cannot_join_two_separated_good_samples_into_confirmation():
    c=Clock();calls=[]
    def sleep(sec):
        c.t += .2 if not calls else sec
        calls.append(1)
    result=wait_for_joint_convergence([0.]*6,lambda:(round(c.t*100),c.t,[0.]*6),lambda:None,
        clock=c,sleep=sleep)
    assert result['confirmation_gap_resets']==1
    assert result['elapsed_sec']>=.28
