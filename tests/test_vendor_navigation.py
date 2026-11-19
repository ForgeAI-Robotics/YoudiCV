import math
import unittest
from piper_handeye.vendor_navigation import ArrivalGate, NavigationPoint, confirmed_route, bind_ordered_goals


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.point = NavigationPoint("test", "jxb_105", 1., 2., 0., 1.)

    def observe(self, gate, now, **overrides):
        args = dict(now=now, pose_received=now, odom_received=now,
                    x=1., y=2., yaw=0., linear_speed=0., angular_speed=0.)
        args.update(overrides)
        return gate.observe(**args)

    def test_stationary_hold_and_stale_reset(self):
        gate = ArrivalGate(self.point)
        self.assertFalse(self.observe(gate, 0.))
        self.assertFalse(self.observe(gate, .25))
        self.assertTrue(self.observe(gate, .5))
        self.assertFalse(self.observe(gate, .6, odom_received=0.))
        self.assertFalse(self.observe(gate, .7))

    def test_moving_or_wrong_pose_never_arrives(self):
        for override in ({"linear_speed": .1}, {"angular_speed": .1},
                         {"x": 4.}, {"yaw": .3}, {"x": float("nan")}):
            gate = ArrivalGate(self.point)
            self.assertFalse(self.observe(gate, 0., **override))
            self.assertFalse(self.observe(gate, .3, **override))
            self.assertFalse(self.observe(gate, .6, **override))

    def test_no_continuous_hold_across_observation_gap(self):
        gate = ArrivalGate(self.point)
        self.assertFalse(self.observe(gate, 0.))
        self.assertFalse(self.observe(gate, 5.))
        self.assertFalse(self.observe(gate, 5.1))

    def test_yaw_wrap(self):
        gate = ArrivalGate(NavigationPoint("p", "jxb_105", 1., 2., 1., 0.))
        self.assertFalse(self.observe(gate, 0., yaw=-math.pi))
        self.assertFalse(self.observe(gate, .25, yaw=-math.pi))
        self.assertTrue(self.observe(gate, .5, yaw=-math.pi))

    def test_four_confirmed_points_and_correct_map_required(self):
        with self.assertRaises(ValueError):
            confirmed_route({"points": {}}, "jxb_105")
        entry = dict(point_id="p", map_id="jxb_105", x=1., y=2., qz=0., qw=1., verified=True)
        config = {"points": {key: dict(entry) for key in
                              ("initial", "call_panel", "floor_panel", "arrival")}}
        self.assertEqual(len(confirmed_route(config, "jxb_105")), 4)
        with self.assertRaises(ValueError):
            confirmed_route(config, "other_map")

    def test_business_goal_order_is_preserved_without_id_sorting(self):
        destinations = [NavigationPoint(name, "jxb_105", float(i), 0., 0., 1.)
                        for i, name in enumerate(("node_9", "node_2", "node_7"))]
        route = bind_ordered_goals(self.point, destinations, "jxb_105")
        self.assertEqual(route["call_panel"].point_id, "node_9")
        self.assertEqual(route["floor_panel"].point_id, "node_2")
        self.assertEqual(route["arrival"].point_id, "node_7")

    def test_ordered_route_rejects_incomplete_duplicate_or_wrong_map(self):
        p = self.point
        with self.assertRaises(ValueError):
            bind_ordered_goals(p, [p, p], "jxb_105")
        with self.assertRaises(ValueError):
            bind_ordered_goals(p, [p, p, p], "jxb_105")
        points = [NavigationPoint(str(i), "other", 0., 0., 0., 1.) for i in range(3)]
        with self.assertRaises(ValueError):
            bind_ordered_goals(p, points, "jxb_105")


if __name__ == "__main__":
    unittest.main()
