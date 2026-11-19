# Measured-force trace button 8 milestone

Date: 2026-09-14

- Full suite: 346 passed.
- Verified run: `button_light_closed_loop_runs/20260914_140058/closed_loop_report.yaml`.
- Result: one press, two-frame ON confirmation, continuous ON observation,
  OFF transition, and successful origin return.
- Guard trigger: 13.5438 mm commanded progress, -0.7147 N estimated normal
  force, score 10.70, alignment 0.657, 100 ms persistence.
- Full trace: `guarded_press_traces/button-light-guarded-execute-1789365674178090986.jsonl`.
- The OrangePi guard evaluates current at the full CAN rate and records vector
  diagnostics at 20 Hz. The UDP result carries only eight compact scalar rows.
- The force observer learns only before the 8 mm arm point and freezes its
  free-motion model after arming, preventing both the earlier 5.85 mm false
  contact and the later absorption of a compliant contact ramp.
- Measured encoder joints are used for the Cartesian Jacobian and are recorded
  alongside desired joints and tracking error.
- A stale-current retraction continues to the post-press light check; OFF can
  therefore enter the normal fresh-target retry loop (three attempts total).
- Retry observation reuses the current reachable near pose when refreshed
  panel distance remains within 13--17 cm.
