from unittest.mock import MagicMock, patch
from tools.navigation_task_gui import Gui
from piper_handeye.navigation_gui_state import NavigationState


def gui():
    g=Gui.__new__(Gui);g.state=NavigationState()
    g.status=MagicMock();g.floor=MagicMock();g.arrival=MagicMock()
    g.telemetry={'goal_revision':7,'goal':[1,2,0,1]};g.record=MagicMock()
    g.fresh=MagicMock(return_value=True);g.send=MagicMock()
    return g


def test_manual_button_queues_once_and_does_not_send_immediately():
    g=gui();g.state.enable('liftin',(-1.829298,2.820949,.708453,.705758))
    with patch('tools.navigation_task_gui.time.monotonic',return_value=100):
        g.manual_start();g.manual_start()
    assert g.state.floor=='manual' and g.state.deadline==103
    assert g.goal_revision_floor==7 and 'goal' not in g.telemetry
    g.send.assert_not_called()


def test_manual_button_requires_enable_and_fresh_feedback():
    g=gui();g.manual_start();assert g.state.deadline is None
    g.state.enable();g.fresh.return_value=False;g.manual_start()
    assert g.state.deadline is None;g.send.assert_not_called()
