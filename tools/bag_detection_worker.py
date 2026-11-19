"""纸袋实时检测后端：只读取 preview，不改变视觉服务或机器人状态。"""
import argparse
import base64
import contextlib
import json
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bag_handle_v3.infer_bag import (
    DEFAULT_CONFIDENCE, DEFAULT_KEYPOINT_CONFIDENCE, DEFAULT_WEIGHTS, predict_frame,
)

PREVIEW_INFERENCE_PERIOD_SEC = 1.0


def emit(data):
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description='纸袋检测后台')
    parser.add_argument('--config', type=Path, default=ROOT / 'piper_button_pose.yaml')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument('--point-conf', type=float, default=DEFAULT_KEYPOINT_CONFIDENCE)
    parser.add_argument('--source', type=Path, help='用图片验证检测链路')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.point_conf <= 1:
        parser.error('置信度须在 0 到 1 之间')
    emit({'type': 'status', 'message': f'正在加载纸袋检测 v3 · {args.weights.name} · 置信度 {args.conf:g}…'})
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        import torch
        torch.set_num_threads(2)
        import yaml
        from ultralytics import YOLO
        from piper_handeye.vision_rpc import VisionRpcClient
        device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        model = YOLO(str(args.weights))
    if args.source:
        image = cv2.imread(str(args.source))
        if image is None:
            raise ValueError('无法读取测试图片')
        frames = None
    else:
        cfg = yaml.safe_load(args.config.read_text())['vision_service']
        client = VisionRpcClient(cfg['host'], cfg['port'], cfg['token'], timeout=2)
        frames = queue.Queue(maxsize=1)

        def publish(item):
            try:
                frames.get_nowait()
            except queue.Empty:
                pass
            frames.put_nowait(item)

        def capture():
            last_seq = None
            while True:
                try:
                    response = client.request('preview')
                    seq = response['frame_sequence']
                    captured = float(response['capture_time_unix_sec'])
                    if not -0.5 <= time.time() - captured <= 1.0:
                        raise RuntimeError('摄像头画面已过期，请检查视觉服务')
                    if seq != last_seq:
                        raw = np.frombuffer(base64.b64decode(response['jpeg_base64']), dtype=np.uint8)
                        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                        if frame is None:
                            raise RuntimeError('摄像头图像解码失败')
                        publish((frame, seq, captured, None))
                        last_seq = seq
                except Exception as exc:
                    publish((None, None, None, str(exc)))
                    time.sleep(0.5)
                time.sleep(0.08)

        threading.Thread(target=capture, daemon=True).start()
    next_inference = 0.0
    while True:
        if frames is not None:
            # Throttle only display inference; keep the newest captured frame.
            time.sleep(max(0.0, next_inference - time.monotonic()))
            image, seq, captured, error = frames.get()
            if error:
                emit({'type': 'error', 'message': '摄像头连接异常：' + error})
                continue
        else:
            seq, captured = 0, time.time()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            data, _ = predict_frame(image, model, device, args.conf, args.point_conf)
        elapsed = time.perf_counter() - started
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError('画面编码失败')
        emit({'type': 'frame', 'jpeg_base64': base64.b64encode(encoded).decode('ascii'),
              'detections': data['detections'], 'frame_sequence': seq,
              'capture_time': captured, 'inference_ms': elapsed * 1000,
              'device': device, 'image_size': data['image_size'],
              'model_release': 'v3', 'weights': str(args.weights.resolve()),
              'confidence_threshold': args.conf, 'keypoint_threshold': args.point_conf})
        next_inference = time.monotonic() + PREVIEW_INFERENCE_PERIOD_SEC
        if frames is None:
            break


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        emit({'type': 'error', 'message': '检测失败：' + str(exc)})
        print(str(exc), file=sys.stderr)
        sys.exit(1)
