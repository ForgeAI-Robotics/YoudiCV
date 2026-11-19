"""Conservative initial DOWN lamp check; a highlight alone is not emission.

This is a disagreement guard, not an arrow segmentation model. It deliberately
does not turn a rejected ON into OFF or claim to distinguish every reflection.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil


class InitialLightGuard:
    def __init__(self, target: str, evidence_dir: Path):
        self.enabled = str(target).casefold() == "down"
        self.active = self.enabled
        self.evidence_dir = Path(evidence_dir)
        self.conflict_seen = False
        self.rechecks = 0
        self.initial_frames = 0
        self.off_frames = 0
        self.not_before_unix = None
        self.extended_observation = False

    def filter(self, result: dict) -> dict:
        if not self.active:
            return result
        # The local connected-component override is exactly what caused the
        # recorded no-press successes. Agreement is required here, irrespective
        # of highlight area: merely raising an area cutoff moves the failure.
        if result.get("state") == "ON" and result.get("legacy_state") != "ON":
            self.conflict_seen = True
            return {**result, "absolute_state": "ON", "state": "UNCERTAIN",
                    "initial_light_guard": "local_highlight_without_global_on",
                    "initial_light_reason": "局部高亮与整体灯态冲突，需复核"}
        return result

    def finish(self):
        self.active = False

    def save(self, snapshot: dict, result: dict) -> str | None:
        if not self.enabled:
            return None
        if self.active:
            self.initial_frames += 1
            # Flow limits three views to twelve samples each. Keep a hard
            # storage bound too, even if a future caller forgets that budget.
            if self.initial_frames > 36 and not self.extended_observation:
                raise RuntimeError("initial light evidence frame budget exceeded")
            stem = (f"recovery_{(self.initial_frames-37) % 36:02d}"
                    if self.initial_frames > 36 else
                    f"initial_{self.initial_frames:02d}_{snapshot['frame_sequence']}")
        elif result.get("state") == "OFF":
            stem = f"waiting_off_{self.off_frames % 3}"
            self.off_frames += 1
        else:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        image = self.evidence_dir / (stem + ".jpg")
        temporary = image.with_suffix(".tmp")
        shutil.copyfile(snapshot["path"], temporary)
        temporary.replace(image)
        record = self.evidence_dir / (stem + ".json")
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps({"snapshot": snapshot, "recognition": result,
                                        "image": str(image)}, ensure_ascii=False, indent=2))
        temporary.replace(record)
        return str(image)
