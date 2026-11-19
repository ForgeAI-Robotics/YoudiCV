# Elevator Button Robot Milestone

Frozen on 2026-08-31 (Asia/Shanghai).

## Final behavior

- Buttons 1 through 6 completed in one sequential run.
- RGB-D obstruction monitoring is enabled during approach and press.
- A press-stage safety failure releases to pre-press and follows the
  prevalidated return through the raised observation pose back to origin.
- For a single raised observation move, return is one continuous C2 command;
  it does not stop and submit a second command at the observation pose.
- The return phase intentionally ignores visual monitoring after the guarded
  press stage.
- Pre-press distance: 0.010 m.
- Press travel: 0.025 m.
- Online IK maximum per-joint delta: 1.00 rad.
- Near-field obstacle reaction gate: less than 0.30 m from the camera.

## Runtime entry points

- `run_button_vision_server.py`: persistent Orbbec/YOLO/RGB-D safety service.
- `run_online_replan_no_contact.py`: guarded button approach, press, and exact
  inverse return.
- `run_direct_return_to_origin.py`: validated recovery-only return tool.

## Required OrangePi services and ports

- Piper UDP controller/state: UDP 5001, 5002, 5004 (5003 is controller-owned).
- `piper-timed-trajectory-bridge.service`: UDP 5010.
- `piper-prepared-splice-bridge.service`: UDP 5011.
- `piper-visual-safety-stop-bridge.service`: UDP 5013.
- SSH TCP 22 is retained for maintenance and recovery.

The obsolete rolling validation/execution services on UDP 5012/5013 and the
obsolete OrangePi vision ROS proxy are not part of this milestone.

## Final evidence

The post-cleanup successful six-button regression is stored in these report
directories:

- `online_replan_no_contact_runs/20260831_115128` (button 1)
- `online_replan_no_contact_runs/20260831_115146` (button 2)
- `online_replan_no_contact_runs/20260831_115202` (button 3)
- `online_replan_no_contact_runs/20260831_115219` (button 4)
- `online_replan_no_contact_runs/20260831_115235` (button 5)
- `online_replan_no_contact_runs/20260831_115251` (button 6)

The pre-cleanup successful six-button run is retained in these report
directories:

- `online_replan_no_contact_runs/20260828_181111` (button 1)
- `online_replan_no_contact_runs/20260828_181132` (button 2)
- `online_replan_no_contact_runs/20260828_181149` (button 3)
- `online_replan_no_contact_runs/20260828_181206` (button 4)
- `online_replan_no_contact_runs/20260828_181222` (button 5)
- `online_replan_no_contact_runs/20260828_181238` (button 6)

The press-stage failure that motivated recovery hardening is recorded in
`online_replan_no_contact_runs/20260828_165638`. The verified manual recovery is
recorded in `direct_return_runs/20260828_174706`. The post-cleanup origin
correction preceding the final regression is recorded in
`direct_return_runs/20260831_115056`.

## Safety limitation

The RGB-D system is an experimental guard, not certified personnel-safety
equipment. Tests must use a remotely operated or fixed lightweight obstacle;
people must remain outside the robot workspace.
