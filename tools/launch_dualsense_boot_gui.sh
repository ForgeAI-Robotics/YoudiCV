#!/usr/bin/env bash
set -euo pipefail

export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

# The local GDM session is normally :1, while a cold boot may allocate :0.
# Wait for either display so the service does not depend on NoMachine or SSH.
for _ in $(seq 1 180); do
  if [[ -S /tmp/.X11-unix/X1 && -r "$XAUTHORITY" ]]; then
    export DISPLAY=:1
    exec /home/qianjue/contral_test/tools/launch_dualsense_gui.sh
  fi
  if [[ -S /tmp/.X11-unix/X0 && -r "$XAUTHORITY" ]]; then
    export DISPLAY=:0
    exec /home/qianjue/contral_test/tools/launch_dualsense_gui.sh
  fi
  sleep 1
done

echo "DualSense GUI: no local X display became available" >&2
exit 1
