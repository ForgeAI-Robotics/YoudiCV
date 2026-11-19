#!/usr/bin/env bash
set -euo pipefail
cd /home/qianjue/contral_test
export PYTHONPATH=/home/qianjue/contral_test${PYTHONPATH:+:$PYTHONPATH}
if [[ "${ROBOT_GUI_LOCKED:-0}" != "1" ]]; then
  lock_file="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/youxiaomei-robot-gui.lock"
  exec flock -n "$lock_file" env ROBOT_GUI_LOCKED=1 "$0" "$@"
fi
exec /usr/bin/python3 run_voice_assistant_gui.py \
  --backend-python /home/qianjue/anaconda3/envs/piper-handeye/bin/python \
  "$@"
