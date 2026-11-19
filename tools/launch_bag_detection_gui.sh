#!/bin/bash
set -e
cd /home/qianjue/contral_test
exec flock -n /tmp/bag-detection-gui.lock /usr/bin/python3 run_bag_detection_gui.py "$@"
