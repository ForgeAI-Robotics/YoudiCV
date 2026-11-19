import json
from pathlib import Path
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from piper_handeye.realtime_planner import FixedRollFastIkPlanner
from run_online_replan_no_contact import reuse_task_preview

class TaskPreviewReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = json.loads((Path(__file__).parents[1]/'optimization_runs/20260908_sequential/ik_input.json').read_text())
        cls.result = FixedRollFastIkPlanner(roll_deg=data['roll'], stand_off_m=.08037062710599122, max_joint_delta_rad=1., max_perpendicular_deviation_deg=35., downstream_base_x_distances_m=[.01, -.005]).solve(np.array(data['button_pose']), np.array(data['seed']))

    def test_reuse_without_solver_and_copy_isolation(self):
        entry = self.result.downstream_preview_solutions[0]
        with patch('run_online_replan_no_contact.inverse_kinematics', side_effect=AssertionError('must not solve')):
            result = reuse_task_preview(self.result, entry.desired_base_to_tcp[:3,3])
        self.assertTrue(result.reused_solution)
        self.assertLess(result.position_error_m, .001)
        self.assertFalse(np.shares_memory(result.joints, entry.joints))

    def test_changed_target_is_not_reused(self):
        target = self.result.downstream_preview_solutions[0].desired_base_to_tcp[:3,3].copy()
        target[0] += .0001
        self.assertIsNone(reuse_task_preview(self.result, target))

    def test_corrupted_joint_solution_is_rejected(self):
        entry = self.result.downstream_preview_solutions[0]
        for joints in (np.full(6, np.nan), np.full(6, 100.), entry.joints + .05):
            invalid = replace(self.result, downstream_preview_solutions=(replace(entry, joints=joints),))
            self.assertIsNone(reuse_task_preview(invalid, entry.desired_base_to_tcp[:3,3]))
