# OpenCV-only button-light runtime

Date: 2026-09-14

- Button localization: running YOLO vision service.
- Button ON/OFF decision: `tools/opencv_light_worker.py`, within-button center/ring contrast only.
- Runtime interpreter: `/home/qianjue/anaconda3/envs/piper-handeye/bin/python`.
- No vision-language model, adapter, persistent model socket, or model service is used.
- Saved-image smoke test: button 16 returned ON in 0.0207 seconds.
- Full project validation after cleanup: 343 tests passed.
- Latest physical closed-loop successes remain button 8 run `20260914_140058`
  and button 16 run `20260914_141145`.
