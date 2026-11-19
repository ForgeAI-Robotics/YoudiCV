#!/usr/bin/env python3
"""Group generated data while preserving old paths with atomic aliases."""
import argparse
import ctypes
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "runs": ["android_command_gui_runs", "button_fast_chain_runs",
             "button_light_closed_loop_runs", "button_recovery_runs",
             "chassis_simulation_runs", "clearance_return_runs", "direct_return_runs",
             "gamepad_control_runs", "guarded_press_traces", "interface_alignment_runs",
             "mission_status_runs", "navigation_gui_runs", "online_replan_no_contact_runs",
             "optimization_runs", "pointing_only_runs", "voice_gui_runs"],
    "captures": ["button_light_snapshots", "camera_snapshots", "depth_hazard_snapshots",
                 "missing_button_captures", "recognition_snapshots"],
    "trials": ["camera_exposure_trials", "force_trials", "image_enhancement_trials",
               "light_algorithm_trials"],
    "diagnostics": ["ik_diagnostics_runs", "voice_diagnostics", "vision_checks", "diagnostics"],
    "recordings": ["whole_sequence_videos"],
}


def exchange(first, second):
    # Linux atomic exchange avoids a missing path while a service writes logs.
    library = ctypes.CDLL(None, use_errno=True)
    if library.renameat2(-100, os.fsencode(first), -100, os.fsencode(second), 2):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(first))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rollback", type=Path)
    args = parser.parse_args()
    if args.rollback:
        record = json.loads(args.rollback.read_text())
        if not args.apply:
            print(json.dumps(record["moves"], ensure_ascii=False, indent=2))
            return
        with open('/tmp/youxiaomei_robot_motion.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for item in reversed(record["moves"]):
                old, new = ROOT / item['old'], ROOT / item['new']
                if not old.is_symlink() or old.resolve() != new.resolve():
                    raise RuntimeError(f"unexpected alias: {old}")
                exchange(old, new)
                new.unlink()
            hidden = ROOT / '.hidden'
            if record['hidden_before'] is None:
                hidden.unlink(missing_ok=True)
            else:
                hidden.write_text(record['hidden_before'])
        print('organization rolled back')
        return
    plan = [{"old": name, "new": f"artifacts/{group}/{name}"}
            for group, names in GROUPS.items() for name in names
            if (ROOT / name).is_dir() and not (ROOT / name).is_symlink()]
    if not args.apply:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    with open('/tmp/youxiaomei_robot_motion.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backup = ROOT / 'version_snapshots' / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_before_artifact_organization')
        backup.mkdir()
        hidden = ROOT / '.hidden'
        record = {'root': str(ROOT), 'moves': [],
                  'hidden_before': hidden.read_text() if hidden.exists() else None}
        manifest = backup / 'organization_manifest.json'
        manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        for item in plan:
            old, new = ROOT / item['old'], ROOT / item['new']
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists() or new.is_symlink():
                raise FileExistsError(new)
            inode = old.stat().st_ino
            new.symlink_to(os.path.relpath(new, old.parent), target_is_directory=True)
            try:
                exchange(old, new)
            except BaseException:
                new.unlink()
                raise
            record['moves'].append(dict(item, inode=inode))
            manifest.write_text(json.dumps(record, ensure_ascii=False, indent=2))
            assert old.is_symlink() and old.stat().st_ino == inode and new.stat().st_ino == inode
        existing = (record['hidden_before'] or '').splitlines()
        hidden.write_text('\n'.join(dict.fromkeys(existing + [i['old'] for i in plan] + ['__pycache__'])) + '\n')
    print(json.dumps({'moved': len(plan), 'manifest': str(manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
