import threading
import inspect
import pytest
from piper_handeye.press_preparation import BackgroundPressPreparation


def test_cpu_plan_can_complete_during_approach_and_is_joined_before_use():
    began=threading.Event();release=threading.Event()
    result=object()
    def builder():
        began.set();assert release.wait(2.);return result
    with BackgroundPressPreparation() as prep:
        prep.start(builder)
        assert began.wait(1.)
        assert not prep.future.done()
        # The simulated arm is still approaching; no hardware interface is
        # exposed to this builder. Its finished plan is not an execution permit.
        release.set()
        plan,audit=prep.finish()
        assert plan is result and audit['build_sec']>=0
    assert prep.future.done()


def test_background_planning_failure_is_propagated():
    with BackgroundPressPreparation() as prep:
        prep.start(lambda: (_ for _ in ()).throw(RuntimeError('IK unavailable')))
        with pytest.raises(RuntimeError,match='IK unavailable'):prep.finish()


def test_cannot_silently_reuse_a_previous_background_plan():
    with BackgroundPressPreparation() as prep:
        with pytest.raises(RuntimeError,match='not scheduled'):prep.finish()
        prep.start(lambda:'plan')
        with pytest.raises(RuntimeError,match='already started'):prep.start(lambda:'different')


def test_production_receding_entry_accepts_preparation_hook():
    from run_button_light_closed_loop import execute_receding_prepress
    assert 'replacement_ready' in inspect.signature(execute_receding_prepress).parameters
