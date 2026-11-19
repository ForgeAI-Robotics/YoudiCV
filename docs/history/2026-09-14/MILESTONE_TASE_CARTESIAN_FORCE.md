# TASE Cartesian normal-force press milestone

Date: 2026-09-14

The guarded press now applies the core estimation structure from Wahrburg et
al., DOI 10.1109/TASE.2017.2691136. Piper motor currents are converted to
joint torque using the official SDK coefficients. A causal free-motion model
provides the expected motor/friction torque, and the residual is projected
through `J_v(q)^T n` into the known panel-normal force. A scalar Kalman filter
tracks force and uncertainty. Contact requires all of the following:

- estimated normal force above 0.35 N;
- force-to-sigma score above 4.0;
- current residual aligned with the panel-normal Jacobian signature above 0.60;
- continuous evidence for at least 0.10 seconds;
- forward travel armed after 5 mm.

The maximum 22 mm displacement, timeout, and stale-feedback stops remain
active. The former relative-current and absolute-current stops are disabled,
so ordinary contact is decided only by the Cartesian force observer. The full
six-dimensional generalized-momentum observer from the paper is not claimed
because this Piper installation has no identified friction and motor dynamics
model. The rolling causal predictor explicitly replaces that missing model
term.

Button 8 hardware evidence is in
`button_light_closed_loop_runs/20260914_100239/closed_loop_report.yaml`:

- contact detected at 18.154 mm forward travel;
- estimated normal force -0.63896 N;
- force sigma 0.05881 N and score 10.865;
- Jacobian-signature alignment 0.7960;
- persistence 0.10215 seconds;
- guarded bridge retracted to prepress with 0.00264 rad endpoint error;
- two independent fresh RGB observations classified button 8 as ON;
- the flow observed the button remain ON at the raised waiting pose, then
  observed OFF and recorded `arrived_after_verified_press`;
- the arm completed its return and was independently read at
  `[0.039, 0.106, 0.000, 0.052, 0.204, 0.000]`, within 0.002 rad of origin.

The `--validate-plan-only` path was corrected after the pre-test exposed a
separate journal bug: when a target first requires physical camera pointing,
it now performs a visual-safety-disabled return, records that commands were
sent, and uses the status
`guarded_plan_validated_returned_origin`.

All outward return entry points now disarm RGB/depth cancellation before the
prevalidated retreat. This covers prepress-to-15-cm, 15-cm-to-raised-waiting,
direct waiting-to-origin, validation cleanup, and failed-press recovery.

After disabling the two legacy current hard stops, button 8 was repeated in
`button_light_closed_loop_runs/20260914_101646/closed_loop_report.yaml`. The
Cartesian observer stopped at 10.899 mm with 0.567 N estimated normal force,
0.079 N sigma, score 7.18, alignment 0.859, and 106 ms persistence. The button
turned ON, later turned OFF, the return visual safety was explicitly disarmed,
and the arm returned to origin.

Verification: 340 tests passed in the `piper-handeye` Conda environment.
OrangePi pre-deployment backups: `deploy_backups/20260914_095814` and
`deploy_backups/20260914_101237`.
