"""Fresh absolute OFF evidence collected during a verified-ON return motion."""
from __future__ import annotations

from pathlib import Path
import math
import shutil
import threading
import time

SOURCE = "three_fresh_off_frames_during_verified_on_retreat"


class RetreatOffStreak:
    def __init__(self, target_label, not_before_unix, frames=3):
        self.target_label = str(target_label)
        self.not_before = not_before_unix
        self.required = frames
        self.frames = []
        self.last_sequence = -1

    def reset(self):
        self.frames = []

    def add(self, state, entry, now):
        """Missing/duplicate/stale/mismatched samples break the streak."""
        try:
            seq = int(entry["frame_sequence"])
            captured = float(entry["capture_time_unix_sec"])
            detection = entry["target_detection"]
            valid = (
                state == "OFF" and seq > self.last_sequence
                and math.isfinite(captured) and captured > self.not_before
                and -.05 <= now-captured <= .35
                and entry.get("visibility", {}).get("visible") is True
                and detection["label"] == self.target_label
                and int(detection["frame_sequence"]) == seq
                and float(detection["confidence"]) >= .70
                and bool(entry.get("image"))
            )
            self.last_sequence = max(self.last_sequence, seq)
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            self.reset()
            return False
        if self.frames and not 0 < captured-self.frames[-1]["capture_time_unix_sec"] <= .50:
            self.reset()
        self.frames.append(dict(frame_sequence=seq, capture_time_unix_sec=captured,
                                state="OFF", image=entry["image"],
                                target_detection=dict(detection)))
        self.frames = self.frames[-self.required:]
        return len(self.frames) >= self.required


def confirmed_retreat_off(result, target_label=None, frames=3):
    """Validate a finished adapter's evidence before skipping stationary wait."""
    if not isinstance(result, dict):
        return False
    samples = result.get("frames") or []
    if (result.get("source") != SOURCE or result.get("confirmed_state") != "OFF"
            or len(samples) != frames):
        return False
    target = str(target_label) if target_label is not None else result.get("target_label")
    try:
        sequences = [int(e["frame_sequence"]) for e in samples]
        times = [float(e["capture_time_unix_sec"]) for e in samples]
        return (all(e["state"] == "OFF" and e["target_detection"]["label"] == target
                    and int(e["target_detection"]["frame_sequence"]) == int(e["frame_sequence"])
                    and float(e["target_detection"]["confidence"]) >= .70 and e.get("image")
                    for e in samples)
                and all(b > a for a, b in zip(sequences, sequences[1:]))
                and all(math.isfinite(t) and t > float(result["not_before_unix_sec"]) for t in times)
                and all(0 < b-a <= .50 for a, b in zip(times, times[1:])))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class RetreatOffMonitor:
    def __init__(self, observer, evidence_directory):
        self.observer = observer
        self.directory = Path(evidence_directory)
        self.stop_event = threading.Event()
        self.thread = None
        self.result = {}

    def start(self):
        started = time.time()
        self.result = dict(source=SOURCE, target_label=str(self.observer.target_label),
                           not_before_unix_sec=started, confirmed_state=None,
                           samples=0, frames=[], errors=[])
        streak = RetreatOffStreak(self.observer.target_label, started)

        def monitor():
            while not self.stop_event.is_set():
                try:
                    state = self.observer.observe(
                        "waiting_retreat", timeout_sec=.20, emit_event=False,
                        visibility_edge_margin_px=5., not_before_unix=started)
                    entry = self.observer.audit[-1] if self.observer.audit else {}
                    self.result["samples"] += 1
                    if streak.add(state, entry, time.time()):
                        # Copy original files once before the service prunes them.
                        self.directory.mkdir(parents=True, exist_ok=True)
                        saved = []
                        for frame in streak.frames:
                            dest = self.directory / ("retreat_off_%s.jpg" % frame["frame_sequence"])
                            shutil.copy2(frame["image"], dest)
                            saved.append(dict(frame, image=str(dest)))
                        self.result.update(confirmed_state="OFF", frames=saved,
                                           confirmed_capture_unix_sec=saved[-1]["capture_time_unix_sec"])
                        return
                except Exception as exc:
                    streak.reset()
                    self.result["errors"].append("%s: %s" % (type(exc).__name__, exc))
                    self.result["errors"] = self.result["errors"][-3:]
                self.stop_event.wait(.01)

        self.thread = threading.Thread(target=monitor, name="return-off-monitor", daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=2.5)
        if self.thread.is_alive():
            raise RuntimeError("return OFF monitor did not stop; refuse concurrent observation")
        return dict(self.result)
