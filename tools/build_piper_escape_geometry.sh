#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/noetic/setup.bash
build_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
g++ -O2 -std=c++14 "$build_dir/piper_escape_geometry.cpp" \
  -o "$build_dir/piper_escape_geometry.new" \
  $(pkg-config --cflags --libs moveit_core moveit_ros_planning roscpp)
mv -- "$build_dir/piper_escape_geometry.new" "$build_dir/piper_escape_geometry"
