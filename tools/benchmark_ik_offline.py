"""Benchmark a recorded IK input only; never connects to robot hardware."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, default=ROOT)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--runs', type=int, default=3)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root))

import numpy as np
import piper_handeye.realtime_planner as rp
import piper_handeye.kinematics as kin

fixture = json.loads(args.input.read_text())
original_rp, original_kin = rp.least_squares, kin.least_squares
rows = []
for index in range(args.runs):
    records = []
    def wrap(kind, original):
        def solve(fun, x, *a, **kw):
            calls = [0]
            def counted(value):
                calls[0] += 1
                return fun(value)
            started = time.perf_counter()
            result = original(counted, x, *a, **kw)
            records.append(dict(kind=kind, seconds=time.perf_counter()-started,
                                nfev=result.nfev, residual_calls=calls[0]))
            return result
        return solve
    rp.least_squares = wrap('orientation', original_rp)
    kin.least_squares = wrap('fixed_ik', original_kin)
    planner = rp.FixedRollFastIkPlanner(
        roll_deg=fixture['roll'], stand_off_m=.08037062710599122,
        max_joint_delta_rad=1., max_perpendicular_deviation_deg=35.,
        downstream_base_x_distances_m=[.01, -.005],
    )
    result = planner.solve(np.array(fixture['button_pose']), np.array(fixture['seed']))
    row = dict(run=index+1, total_sec=result.solve_sec,
               position_error_m=result.position_error_m,
               orientation_error_deg=result.orientation_error_deg,
               downstream_step_rad=result.downstream_preview_max_joint_step_rad,
               joints=result.joints.tolist(), records=records)
    rows.append(row)
    print(json.dumps(dict(run=index+1, total_sec=result.solve_sec,
                          residual_calls=sum(r['residual_calls'] for r in records),
                          error_m=result.position_error_m)), flush=True)
args.output.write_text(json.dumps(rows, indent=2))
