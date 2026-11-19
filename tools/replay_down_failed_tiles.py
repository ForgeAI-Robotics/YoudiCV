#!/usr/bin/env python3
"""CPU-only JPEG regression of DOWN retry scheduling; no robot connections."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=ROOT/'tests/fixtures/down_failure_fallback')
    parser.add_argument('--model', type=Path,
                        default=ROOT.parent/'button_recognition/models/elevator_button_yolo26m_best.pt')
    args = parser.parse_args()
    import cv2
    from piper_handeye.button_pose import YoloButtonDetector, select_button
    from piper_handeye.call_button_detection import retry_call_button_tile, complete_box
    from piper_handeye.detection_enhancement import detection_payload
    from piper_handeye.recognition_image import draw_recognition
    manifest = json.loads((args.fixtures/'manifest.json').read_text())
    baseline_path = args.fixtures/'legacy_call_button_detection.py'
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == manifest['baseline_source_sha256']
    spec = importlib.util.spec_from_file_location('piper_handeye._down_legacy_cpu', baseline_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    model = YoloButtonDetector(args.model, conf=.15, iou=.5, imgsz=640, device='cpu')
    import torch
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []

    def target_ready(detections, label, image):
        try:
            d = select_button(list(detections), label, .70)
            return complete_box(d.bbox_xyxy, image.shape[1], image.shape[0])
        except RuntimeError:
            return False

    for item in manifest['frames']:
        path = args.fixtures/item['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        image = cv2.imread(str(path))
        raw = tuple(model.detect(image))
        cache = {}

        class CachedDetector:
            def detect(self, crop):
                key = (crop.shape, hashlib.sha256(crop.tobytes()).hexdigest())
                if key not in cache:
                    cache[key] = tuple(model.detect(crop))
                return cache[key]

        result = dict(name=item['name'], kind=item['kind'], target=item['target'],
                      input_sha256=item['sha256'], raw=detection_payload(raw))
        for name, fn in [('baseline', legacy.retry_call_button_tile), ('new', retry_call_button_tile)]:
            detector = CachedDetector()
            attempts = []
            for i in range(7):
                # Virtual cadence tests only the selector. Repeating a JPEG
                # does NOT prove fresh-frame confirmation or physical success.
                with patch('piper_handeye.down_button_recovery.time',
                           SimpleNamespace(monotonic=lambda: i*.15)):
                    detections, audit = fn(detector, image, raw, item['target'], .70)
                ready = target_ready(detections, item['target'], image)
                attempts.append(dict(ready=ready, detections=detection_payload(detections), audit=audit))
                if ready:
                    break
            result[name] = dict(ready=any(a['ready'] for a in attempts), attempts=attempts)
        result['original_success_preserved'] = (
            not result['baseline']['attempts'][0]['ready'] or
            result['baseline']['attempts'][0]['detections'] == result['new']['attempts'][0]['detections'])
        if item['target'] == 'up':
            assert [a['detections'] for a in result['baseline']['attempts']] == [
                a['detections'] for a in result['new']['attempts']], item['name']
        if item['kind'] == 'failed_down_search':
            last = result['new']['attempts'][-1]
            drawing = draw_recognition(image, last['detections'],
                                       caption=f'DOWN ready={result["new"]["ready"]}', target_label='down')
            cv2.imwrite(str(args.output/(item['name']+'_new.jpg')), drawing)
        cases.append(result)
        print(json.dumps(dict(name=item['name'], baseline=result['baseline']['ready'],
                              new=result['new']['ready'], attempts=len(result['new']['attempts']),
                              old_success_preserved=result['original_success_preserved']),
                         ensure_ascii=False), flush=True)
        report = dict(scope='Same-frame historical JPEG CPU inference; virtual retry cadence. '
                            'Not independent-frame confirmation, live RGB-D, accuracy or physical press validation.',
                      model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), cases=cases)
        (args.output/'historical_replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    regressions = [c['name'] for c in cases if c['baseline']['ready'] and not c['new']['ready']]
    negatives = [c['name'] for c in cases if c['kind'] == 'no_call_buttons' and c['new']['ready']]
    report.update(regressions=regressions, false_arrows=negatives,
                  legacy_first_path_changes=[c['name'] for c in cases if not c['original_success_preserved']])
    (args.output/'historical_replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if regressions or negatives or report['legacy_first_path_changes'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
