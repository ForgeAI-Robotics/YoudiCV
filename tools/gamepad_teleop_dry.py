#!/usr/bin/env python3
"""Read normalized JSON samples and print intents; never connects to hardware."""

from __future__ import annotations

from dataclasses import asdict
import json
import sys

from piper_handeye.gamepad_teleop import DualModeGamepad, sample_from_mapping


def main() -> int:
    controller = DualModeGamepad()
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("arm_return_complete") is not None:
            intent = controller.complete_arm_return(
                success=bool(request["arm_return_complete"])
            )
        else:
            intent = controller.update(sample_from_mapping(request))
        payload = asdict(intent)
        payload["mode"] = intent.mode.value
        print(json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
