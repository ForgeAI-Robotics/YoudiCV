# Button 8 second-run displacement root cause

Date: 2026-09-15

The deployed detector was active in both button-8 runs. The terminal reasons
were `sustained_cartesian_normal_force_kalman`, and the running OrangePi bridge
source matched the local source before this correction. It was nevertheless a
one-axis adaptation of Wahrburg et al. rather than the paper's complete
Cartesian wrench observer and force controller.

The first run (`20260915_105227`) accumulated valid normal-force evidence from
about 10.19 mm and stopped at 14.62 mm. At 10 mm/s, the required 400 ms
continuous evidence inherently adds about 4 mm after force first crosses the
gate.

The second run (`20260915_110205`) carried a large motor-current residual much
earlier, but that residual was poorly aligned with `J_v(q)^T n`. For example,
the saved 20 Hz trace showed normal-signature alignment around 0.11 at
17.09--18.04 mm. The old conjunctive alignment gate rejected those samples and
reset normal persistence. It accepted a continuous normal sequence only near
19.74 mm and therefore did not stop until 21.25 mm. A weighted three-axis
Cartesian replay shows that this rejected load was predominantly tangential,
consistent with the observed gripper slide.

The correction retains the scalar normal-force observer for ordinary button
actuation and adds an independent three-axis translational-force Kalman
observer. A tangential load above 0.60 N and 4 sigma for 100 ms is treated as
real contact, cancels the forward trajectory, and retracts. The normal press
gate remains 0.50 N, 4 sigma, alignment 0.20, and 400 ms. The state machine then
checks the lamp; ON completes the press, while OFF authorizes relocalization
and retry.

Offline replay of the recorded residuals leaves the first run on its normal
stop at 14.62 mm. It finds no sustained false tangential trigger there. On the
second run it finds sustained tangential contact at approximately 13.13 mm,
instead of waiting for the old normal-only stop at 21.25 mm.

The causal free-motion predictor remains frozen at the 8 mm arming boundary.
This is intentional: a compliant button can increase load gradually, and a
predictor updated after arming can absorb that force ramp and miss contact.
Previous configuration comments claiming that it continued updating until
impact were corrected.

This still does not claim the complete method from DOI
10.1109/TASE.2017.2691136. The robot lacks an identified rigid-body/friction
model and a measured six-dimensional wrench, and its current trajectory bridge
does not implement closed-loop Cartesian force control. The implementation
uses the paper's motor-current/Jacobian/Kalman observation structure with a
causal empirical free-motion model.

## Hardware verification

Run `button_light_closed_loop_runs/20260915_132501/closed_loop_report.yaml`
completed the full button-8 cycle with one attempt. The tangential observer
stopped the forward path at 9.999 mm after estimating 1.346 N tangential force
at 8.75 sigma. Two fresh near-position frames classified the lamp ON; the
waiting-position stream remained ON and later changed to OFF. The arm then
returned to origin with 0.001 rad maximum encoder error. There were no safety
events and the process exited with code zero.
