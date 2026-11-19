# Closed chassis collision model and permanent boundary recovery

Date: 2026-09-22

## Root cause

The calibrated URDF already places the real top-chassis surface about 5 mm
below the arm at the recorded origin.  Applying another 5 mm MoveIt link
padding put the normal origin on the numerical collision boundary.  A manual
unpowered move or small encoder/model error could therefore make all ordinary
controllers fail closed before they sent a motor command.

The vendor `top_chassis.STL` also contains eight disconnected solids.  Its
second-largest solid (47,964 faces, local bounds
`[-0.1773, 0.0810, 0.2988]..[0.1773, 0.4930, 0.6124]` m) is the slide/drawer
geometry in its open, extended state.  The physical slide is currently
closed, so that solid is absent from the collision envelope.

## Implemented policy

- Visual geometry remains the original vendor `top_chassis.STL`.
- Collision geometry uses reproducibly generated `top_chassis_closed.STL`:
  all seven closed-state solids remain and only the open extension is removed.
- MoveIt top-chassis padding is 1 mm.  This is a model-error allowance in
  addition to the approximately 5 mm physical geometry clearance.
- Both `piper_link1/top_chassis_link` and
  `piper_link2/top_chassis_link` are explicitly collision-enabled in the ACM.
- Normal jogging and trajectories remain fail-closed on every collision.
- Boundary recovery is available only when the sole reported contact pair is
  `piper_link2/top_chassis_link`.  It changes only J2 in the clearance-increase
  direction, at 0.020 rad/s and by at most 0.050 rad.  Hardware must be Active;
  live encoder feedback, non-J2 drift, final tracking and final MoveIt validity
  are checked.  The standard trajectory controller is restored in `finally`.
- Direct return and origin checks use live OrangePi ROS encoder feedback from
  `/piper_arm/joint_states`.  The legacy UDP state remains only as a hardware
  health source.

## Deployment and validation

- OrangePi backup:
  `/home/orangepi/mobile_arm_demo/backups/pre_closed_chassis_recovery_20260922_1548`
- Local snapshot:
  `version_snapshots/20260922_1542-pre-closed-chassis-recovery`
- Runtime URDF confirms `top_chassis_closed.STL`.
- ROS bringup and UDP 5010/5011/5013/5014/5015 services are active.
- MoveIt read back `moveit_link_padding_m=0.001`.
- The known button-8 origin/pointing/return path passed all 57 no-motion
  collision samples.
- Current live state is valid with no contacts.
- Calling recovery at the current safe origin returned `already_clear` and
  changed every joint by exactly 0 rad.
- Origin check uses `joint_state_source=orange_pi_ros_joint_states` and reports
  `at_origin=true`, maximum error 0.00086972 rad.
- Focused tests: 39 passed.  Production `tests/`: 466 passed in the sandbox;
  the two socket-restricted ROS1 tests passed separately outside the sandbox.

The new moving recovery branch was not tested by deliberately putting the arm
back into a collision state.  The same bounded J2 motion was physically used
to recover the preceding fault, while the deployed selector, limits,
interpolation and rejection cases are covered by unit tests.
