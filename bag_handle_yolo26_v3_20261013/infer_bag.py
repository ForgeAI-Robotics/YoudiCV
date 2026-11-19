"""Standalone image inference. No camera, depth sensor, ROS or robot calls."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

def predict_image(image_path, weights=ROOT/'best.pt', device='cpu', conf=0.25, point_conf=0.5):
    import cv2
    from ultralytics import YOLO
    image_path = Path(image_path).resolve()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot read image: {image_path}')
    model = YOLO(str(weights))
    return predict_frame(image, model, device, conf, point_conf, str(image_path))

def predict_frame(image, model, device='cpu', conf=0.25, point_conf=0.5, source='live_camera'):
    """Infer a frame using a persistent model; preserve image inference semantics."""
    height, width = image.shape[:2]
    expected = {0: 'graspable_bag', 1: 'invalid_bag', 2: 'uncertain_bag'}
    if dict(model.names) != expected:
        raise ValueError(f'Wrong checkpoint classes: {model.names}')
    result = model.predict(image, imgsz=640, conf=conf, device=device, verbose=False)[0]
    detections = []
    statuses = {0: 'valid', 1: 'invalid', 2: 'uncertain'}
    for idx, box in enumerate(result.boxes):
        cls = int(box.cls.item())
        bbox = [float(v) for v in box.xyxy[0].tolist()]
        keypoint = result.keypoints.data[idx, 0].tolist()
        x, y, visibility = [float(v) for v in keypoint]
        usable = (cls == 0 and all(math.isfinite(v) for v in keypoint)
                  and visibility >= point_conf and 0 <= x < width and 0 <= y < height
                  and bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3])
        status = statuses[cls] if cls != 0 or usable else 'uncertain'
        detections.append({'object_index': idx+1, 'class_id': cls, 'model_status': statuses[cls],
                           'status': status, 'confidence': float(box.conf.item()),
                           'bbox_xyxy': bbox, 'handle_center_xy': [x,y] if usable else None,
                           'keypoint_confidence': visibility,
                           'reason': 'low_or_out_of_bounds_keypoint' if cls == 0 and not usable else None})
    data = {'schema_version': 1, 'source': source, 'image_size': {'width': width,'height':height},
            'coordinate_frame': 'original_image_top_left', 'units': 'pixels',
            'depth_available': False, 'robot_motion_authorized': False,
            'detections': detections}
    return data, result

def main():
    parser = argparse.ArgumentParser(description='YOLO26m-Pose bag handle image inference')
    parser.add_argument('--source', type=Path, default=ROOT/'examples/input.jpg')
    parser.add_argument('--weights', type=Path, default=ROOT/'best.pt')
    parser.add_argument('--device', default='cpu', help='cpu or 0; CUDA must be checked separately on Jetson')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--point-conf', type=float, default=0.5)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--save-overlay', type=Path)
    args = parser.parse_args()
    if not (0 <= args.conf <= 1 and 0 <= args.point_conf <= 1):
        parser.error('Confidence thresholds must be between 0 and 1.')
    for path in [args.output_json, args.save_overlay]:
        if path and path.exists():
            parser.error(f'Refusing to overwrite: {path}')
    data, result = predict_image(args.source,args.weights,args.device,args.conf,args.point_conf)
    content = json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True,exist_ok=True)
        with args.output_json.open('x') as f:
            f.write(content+'\n')
    if args.save_overlay:
        import cv2
        image = result.orig_img.copy()
        for obj in data['detections']:
            color = {'valid': (0,220,0),'invalid': (0,0,255),'uncertain': (0,165,255)}[obj['status']]
            x1,y1,x2,y2 = [int(v) for v in obj['bbox_xyxy']]
            cv2.rectangle(image,(x1,y1),(x2,y2),color,1)
            if obj['handle_center_xy']:
                x,y = [round(v) for v in obj['handle_center_xy']]
                cv2.circle(image,(x,y),3,color,-1)
        args.save_overlay.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(args.save_overlay),image):
            raise ValueError(f'Cannot save overlay: {args.save_overlay}')
    print(content)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Inference failed: {exc}',file=sys.stderr)
        sys.exit(1)
