# Button 16 Cartesian-force and visual-resume milestone

Date: 2026-09-14

The full button 16 flow completed with status
`arrived_after_verified_press` in
`button_light_closed_loop_runs/20260914_113819/closed_loop_report.yaml`.

- one press attempt;
- legacy relative and absolute current hard stops disabled;
- Cartesian normal-force Kalman contact at 14.509 mm;
- estimated normal force 1.087 N, sigma 0.084 N, score 12.93;
- current/Jacobian alignment 0.832 and persistence 102 ms;
- button observed ON after press and later OFF at the raised waiting pose;
- visual safety explicitly disabled for outward return;
- returned to origin.

Transient camera obstruction recovery was corrected at both layers. The
visual supervisor now clears its local trigger after a stable clear view and
the OrangePi trajectory bridge resumes its remaining trajectory from encoder
state. Pointing recovery waits indefinitely while an obstruction remains and
replans from the newest encoder pose before continuing. Button tasks are run
by `piper-button16-test.service`, so terminating an interactive terminal does
not terminate the robot process.

Verification: 342 tests passed.
