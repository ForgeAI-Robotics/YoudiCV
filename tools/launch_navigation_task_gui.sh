#!/bin/bash
set -e
cd /home/qianjue/contral_test
exec flock -n /tmp/agx-navigation-task-gui.lock /usr/bin/python3 tools/navigation_task_gui.py
