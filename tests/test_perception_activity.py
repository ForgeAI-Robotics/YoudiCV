import fcntl
from pathlib import Path
import tempfile
import threading
import time
import unittest

from piper_handeye.perception_activity import PerceptionActivity


class PerceptionActivityTests(unittest.TestCase):
    def gate(self, path):
        return PerceptionActivity(idle_interval_sec=.3, active_hold_sec=.15, motion_lock=path)

    def test_idle_wait_is_interrupted_by_request_without_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            gate = self.gate(Path(folder)/'lock')
            self.assertTrue(gate.wait_for_turn())
            result = []
            thread = threading.Thread(target=lambda: result.append(gate.wait_for_turn()))
            thread.start()
            time.sleep(.04)
            self.assertTrue(thread.is_alive())
            gate.touch()
            thread.join(.15)
            self.assertEqual(result, [True])
            self.assertEqual(gate.status()['mode'], 'active')
            time.sleep(.18)
            self.assertEqual(gate.status()['mode'], 'idle')

    def test_task_lock_keeps_active_after_rpc_lease_expires(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'lock'
            gate = self.gate(path)
            with path.open('a+') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(gate.motion_busy())
                time.sleep(.18)
                self.assertEqual(gate.status()['mode'], 'active')
                self.assertTrue(gate.wait_for_turn())
                fcntl.flock(stream, fcntl.LOCK_UN)
                self.assertFalse(gate.motion_busy())

    def test_shutdown_wakes_idle_thread(self):
        with tempfile.TemporaryDirectory() as folder:
            gate = self.gate(Path(folder)/'lock')
            gate.wait_for_turn()
            result = []
            thread = threading.Thread(target=lambda: result.append(gate.wait_for_turn()))
            thread.start()
            gate.close()
            thread.join(.15)
            self.assertEqual(result, [False])

    def test_idle_frequency_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            gate = self.gate(Path(folder)/'lock')
            gate.wait_for_turn()
            start = time.monotonic()
            gate.wait_for_turn()
            self.assertGreaterEqual(time.monotonic()-start, .28)


if __name__ == '__main__':
    unittest.main()
