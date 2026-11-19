"""Best-effort RGB evidence; never change perception or robot state."""
from datetime import datetime
import json
from pathlib import Path
import shutil

from .arm_report import plain_data


ROOT = Path(__file__).resolve().parents[1]


def save_missing_button_view(vision, target_label, reason, *, snapshot=None,
                             output_root=None):
    evidence = {"target_label": str(target_label), "reason": str(reason),
                "errors": {}, "note": "RGB and inference may have different sequences"}
    try:
        directory = Path(output_root or ROOT / "missing_button_captures") / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        directory.mkdir(parents=True, exist_ok=False)
        evidence["directory"] = str(directory)
        # Capture before diagnostics, especially before the caller starts return.
        try:
            frame = snapshot if snapshot is not None else vision.request("light_snapshot")
            evidence["snapshot"] = frame
            destination = directory / "camera.jpg"
            shutil.copy2(frame["path"], destination)
            evidence["image"] = str(destination)
        except Exception as exc:
            evidence["errors"]["capture"] = str(exc)
        try:
            recognition = vision.request("recognition_snapshot")
            evidence["recognition"] = recognition
            for key, name in (("input_path", "yolo_input.jpg"),
                              ("raw_yolo_path", "yolo_raw.jpg"),
                              ("enhanced_yolo_path", "yolo_enhanced.jpg"),
                              ("tile_yolo_path", "yolo_tile.jpg")):
                if recognition.get(key):
                    destination = directory / name
                    shutil.copy2(recognition[key], destination)
                    evidence[key] = str(destination)
        except Exception as exc:
            evidence["errors"]["recognition"] = str(exc)
        # Only failure evidence invokes the recognition-image RPC. It reuses
        # cached inference, never runs YOLO again. No extra context on motion.
        if snapshot is None:
            for command in ("visible_buttons", "health"):
                try:
                    evidence[command] = vision.request(command)
                except Exception as exc:
                    evidence["errors"][command] = str(exc)
        (directory / "diagnostic.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, default=plain_data), encoding="utf-8")
    except Exception as exc:
        evidence["errors"]["save"] = str(exc)
    try:
        print(json.dumps({"missing_button_capture": evidence}, ensure_ascii=False,
                         default=plain_data), flush=True)
    except Exception as exc:
        # Evidence output is secondary to the acquisition/control failure.
        evidence['errors']['emit'] = f'{type(exc).__name__}: {exc}'
    return evidence
