#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/qianjue/anaconda3/envs/elevator-yolo26/bin/python \
  "$ROOT/button_light_evaluation/evaluate.py" "$@"
