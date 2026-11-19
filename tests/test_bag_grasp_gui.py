"""使用系统 Python/Tk 的 GUI 控制协议测试；不会连接机器人。"""
import queue
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from run_bag_detection_gui import BagGUI


class BagTaskGuiTests(unittest.TestCase):
    def gui(self):
        g=BagGUI.__new__(BagGUI)
        g.args=SimpleNamespace(source=None,backend_python=Path('/env/bin/python'))
        g.arm_process=None;g.arm_events=queue.Queue();g.arm_session=0
        g.arm_terminal=False;g.close_pending=False
        g.abort_return_pending=False;g.return_request_id=0;g.return_job=None
        g.arm_action=None;g.arm_last_report=None;g.root=MagicMock()
        g.cycle_auto_return=False;g.pending_cycle_report=None
        g.abort_return_button=MagicMock()
        g.arm_buttons=[MagicMock(),MagicMock(),MagicMock()]
        g.arm_stop_button=MagicMock();g.arm_status=MagicMock();g.target_info=MagicMock()
        g.conf=MagicMock();g.conf.get.return_value='.65'
        g._destroy=MagicMock()
        return g

    def test_only_touch_action_can_request_execute(self):
        import tempfile
        for action in ['save','plan','touch','cycle']:
            with self.subTest(action=action),tempfile.TemporaryDirectory() as tmp:
                g=self.gui()
                with patch('run_bag_detection_gui.ROOT',Path(tmp)),patch('run_bag_detection_gui.subprocess.Popen') as p,patch('run_bag_detection_gui.threading.Thread'):
                    g.start_arm_task(action)
                cmd=p.call_args.args[0]
                self.assertEqual('--execute' in cmd,action in ('touch','cycle'))
                self.assertIn('--gui-heartbeat',cmd)
                self.assertEqual(cmd[cmd.index('--action')+1],action)

    def test_double_click_cannot_launch_parallel_motion(self):
        g=self.gui();g.arm_process=MagicMock()
        with patch('run_bag_detection_gui.subprocess.Popen') as p:g.start_arm_task('touch')
        p.assert_not_called()

    def test_offline_image_view_cannot_move_arm(self):
        g=self.gui();g.args.source=Path('example.jpg')
        with patch('run_bag_detection_gui.subprocess.Popen') as p:g.start_arm_task('touch')
        p.assert_not_called()

    def test_stop_is_sent_to_existing_task_without_starting_return(self):
        g=self.gui();g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        g.stop_arm_task()
        g.arm_process.stdin.write.assert_called_once_with('{"command":"stop"}\n')
        g.arm_process.stdin.flush.assert_called_once()

    def test_close_waits_for_task_stop_before_destroying_window(self):
        g=self.gui();g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        g.close()
        self.assertTrue(g.close_pending)
        g._destroy.assert_not_called()
        g.arm_process.stdin.write.assert_called_once_with('{"command":"stop"}\n')

    def test_late_session_events_cannot_change_active_task(self):
        g=self.gui();g.arm_session=3
        g.arm_events.put((2,{'type':'error','message':'old'}))
        g.poll_arm()
        g.arm_status.set.assert_not_called()

    def test_heartbeat_and_exit_cleanup(self):
        g=self.gui();proc=MagicMock();proc.poll.return_value=None
        g.arm_process=proc;g.arm_terminal=True
        g.arm_events.put((0,{'type':'exit','code':0}))
        g.poll_arm()
        self.assertEqual(__import__('json').loads(proc.stdin.write.call_args.args[0])['command'],'heartbeat')
        proc.stdin.close.assert_called_once()
        self.assertIsNone(g.arm_process)


    def test_abort_return_waits_for_exit_before_starting_return(self):
        g=self.gui();proc=MagicMock();proc.poll.return_value=None
        g.arm_process=proc;g.arm_action='touch'
        g.start_arm_task=MagicMock()
        g.abort_and_return()
        self.assertTrue(g.abort_return_pending)
        g.start_arm_task.assert_not_called()
        proc.stdin.write.assert_called_once_with('{"command":"stop"}\n')
        g.arm_events.put((0,{'type':'error','message':'stopped','report':{'independent_stop':{'status':'stopped'}}}))
        g.arm_events.put((0,{'type':'exit','code':2}))
        g.poll_arm()
        g.start_arm_task.assert_not_called()
        g.begin_pending_return(g.return_request_id)
        g.start_arm_task.assert_called_once_with('return')

    def test_plain_stop_cancels_late_return_callback(self):
        g=self.gui();g.start_arm_task=MagicMock()
        g.abort_and_return();old=g.return_request_id
        g.stop_arm_task()
        g.begin_pending_return(old)
        g.start_arm_task.assert_not_called()

    def test_close_cancels_pending_return(self):
        g=self.gui();g.start_arm_task=MagicMock()
        g.abort_and_return();old=g.return_request_id
        g.close();g.begin_pending_return(old)
        g.start_arm_task.assert_not_called()
        g._destroy.assert_called_once()

    def test_unconfirmed_stop_does_not_launch_return(self):
        g=self.gui();g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        g.arm_action='touch';g.abort_return_pending=True
        g.arm_events.put((0,{'type':'error','message':'stop failed','report':{'stop_error':'timeout'}}))
        g.arm_events.put((0,{'type':'exit','code':2}))
        g.poll_arm()
        self.assertFalse(g.abort_return_pending)
        g.root.after_idle.assert_not_called()

    def test_return_uses_production_origin_adapter_and_ignores_detection_threshold(self):
        import tempfile
        g=self.gui();g.conf.get.return_value='invalid'
        with tempfile.TemporaryDirectory() as tmp, patch('run_bag_detection_gui.ROOT',Path(tmp)),patch('run_bag_detection_gui.subprocess.Popen') as p,patch('run_bag_detection_gui.threading.Thread'):
            g.start_arm_task('return')
        cmd=p.call_args.args[0]
        self.assertTrue(cmd[0].endswith('piper-handeye/bin/python'))
        self.assertTrue(any(x.endswith('bag_origin_worker.py') for x in cmd))
        self.assertIn('--gui-heartbeat',cmd)
        self.assertNotIn('--weights',cmd)

    def test_repeated_abort_return_does_not_queue_another_return(self):
        g=self.gui();g.abort_and_return();token=g.return_request_id
        g.abort_and_return()
        self.assertEqual(g.return_request_id,token)
        self.assertEqual(g.root.after_idle.call_count,1)

    def test_return_preserves_voice_environment_without_inference_conda_libraries(self):
        import tempfile
        for action in ['return','touch']:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                g=self.gui()
                with patch.dict('run_bag_detection_gui.os.environ', {'LD_LIBRARY_PATH':'/usr/local/cuda-12.6/lib64:'}, clear=True), \
                     patch('run_bag_detection_gui.ROOT',Path(tmp)), \
                     patch('run_bag_detection_gui.subprocess.Popen') as p, \
                     patch('run_bag_detection_gui.threading.Thread'):
                    g.start_arm_task(action)
                library_path=p.call_args.kwargs['env']['LD_LIBRARY_PATH']
                self.assertEqual(library_path.startswith('/env/lib:'),action=='touch')
                if action=='return':
                    self.assertEqual(library_path,'/usr/local/cuda-12.6/lib64:')


    def test_cycle_success_queues_origin_only_after_worker_exit(self):
        g=self.gui();g.arm_action='cycle';g.cycle_auto_return=True
        g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        report={'cycle_completed':True,'return_to_origin_on_success':True,'status':'grasp_place_completed_pending_origin'}
        g.arm_events.put((0,{'type':'complete','message':'cycle done','report':report}))
        g.poll_arm()
        g.root.after_idle.assert_not_called()
        g.arm_events.put((0,{'type':'exit','code':0}))
        g.poll_arm()
        self.assertTrue(g.abort_return_pending)
        self.assertEqual(g.pending_cycle_report,report)
        g.root.after_idle.assert_called_once()

    def test_plain_stop_prevents_cycle_success_from_auto_returning(self):
        g=self.gui();g.arm_action='cycle';g.cycle_auto_return=True
        g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        g.stop_arm_task()
        g.arm_events.put((0,{'type':'complete','message':'late success','report':{'cycle_completed':True,'return_to_origin_on_success':True}}))
        g.arm_events.put((0,{'type':'exit','code':0}))
        g.poll_arm()
        g.root.after_idle.assert_not_called()

    def test_cycle_failure_never_auto_returns_or_opens_gripper(self):
        g=self.gui();g.arm_action='cycle';g.cycle_auto_return=True
        g.arm_process=MagicMock();g.arm_process.poll.return_value=None
        g.arm_events.put((0,{'type':'error','message':'stopped','report':{'possibly_holding_bag':True}}))
        g.arm_events.put((0,{'type':'exit','code':2}))
        g.poll_arm()
        g.root.after_idle.assert_not_called()
        self.assertIn('夹爪可能仍夹持纸袋',g.arm_status.set.call_args.args[0])

    def test_origin_result_is_recorded_with_the_same_cycle_report(self):
        import tempfile,json
        g=self.gui();g.arm_action='return'
        g.pending_cycle_report={'cycle_completed':True,'target_base_m':[.4,0,.4],'grasp_confirmed':False}
        g.arm_events.put((0,{'type':'complete','message':'origin verified','report':{'returned_to_origin':True}}))
        with tempfile.TemporaryDirectory() as tmp,patch('run_bag_detection_gui.ROOT',Path(tmp)):
            (Path(tmp)/'artifacts/diagnostics/bag_detection_gui').mkdir(parents=True)
            g.poll_arm()
            saved=json.loads((Path(tmp)/'artifacts/diagnostics/bag_detection_gui/latest_arm_task.json').read_text())
        self.assertEqual(saved['status'],'grasp_place_completed_at_origin')
        self.assertEqual(saved['target_base_m'],[.4,0,.4])
        self.assertTrue(saved['returned_to_origin'])
        self.assertFalse(saved['grasp_confirmed'])


if __name__=='__main__':unittest.main()
