# Fast post-press light verification

Date: 2026-09-15

## Failure mode

The old state machine did not inspect the lamp until the guarded bridge had
cancelled the press, retracted to prepress, returned its RPC response, and the
arm had either been accepted at prepress or completed the full retreat to the
5 cm observation pose. A low-floor trip could finish during that blind
interval. The first stationary frame would then be OFF and could authorize an
incorrect retry even though the button had briefly turned ON.

Button 9 run `20260915_133843` also showed an avoidable fallback. Its prepress
bbox ended at y=356.15 in a 360-pixel image. The crop was complete, but the old
four-pixel margin required y<=356.00, so a 0.15-pixel margin miss forced the
entire 5 cm retreat.

## New dataflow

The production executor starts a dedicated RGB monitor at the guarded-press
boundary, after the safety check and immediately before the bridge command.
It therefore does not run during moving IK or the receding approach. It
continues through forward press, guarded bridge cancellation/retraction,
and, when needed, the 5 cm retreat. A frame is eligible only when YOLO provides
a fresh, recent, confident, complete target bbox. OpenCV then classifies the
target crop. Two consecutive independent ON frames latch success.

If ON is latched before the fallback retreat begins, the state machine stays
at prepress and returns directly to the normal waiting observation. If ON is
latched during retreat, that evidence remains valid even if the first
stationary frame later sees OFF. If no ON is latched, the existing stationary
two-frame decision remains authoritative and OFF can still request a retry.
The long waiting-position ON-to-OFF observation and arrival command are
unchanged.

The complete-crop edge margin is now two pixels. The fallback 5 cm retreat is
planned at 0.25 m/s, the existing Cartesian cap, instead of 0.20 m/s. Increasing
speed is secondary: overlapping vision with motion removes the blind interval
and preserves the transient ON evidence.

## Verification

Tests cover a button that turns ON during press and is OFF by the next
stationary opportunity. The two moving ON frames prevent another press and
send the arm to its waiting observation. Tests also lock the two-frame rule,
two-pixel complete-crop margin, and 0.25 m/s fallback speed. Full verification:
379 tests passed.

The button 2 hardware run `20260915_135544` confirmed the low-floor race in
real timing. It captured two ON frames 67.65 ms apart during the press/retract
interval, but the lamp was already OFF when the arm reached the waiting pose.
The latch prevented a false retry and the task completed on its first attempt.

## Prevalidated direct return

That same run exposed a separate return pause. The guarded press completed at
Unix time 1789451761.004; generating and remotely validating a fresh return
finished at 1789451763.633, costing 2.629 seconds. Return execution did not
start until 1789451764.513, 3.509 seconds after guarded-press completion.

Each segmented plan now includes a 0.25 m/s `prepress_to_waiting` trajectory.
It is generated and MoveIt collision-validated before physical pressing. When
the two-frame monitor proves ON and the guarded bridge has retracted to its
encoder-verified prepress pose, return sends this command immediately. The
fallback dynamic planner remains available only when encoder state matches
neither the planned prepress pose nor the observation pose.

MoveIt validation is deliberately not run concurrently with the 200 Hz force
feedback loop. This moves its CPU and RPC cost before contact and avoids
increasing force-sample latency during the safety-critical pressing stage.
