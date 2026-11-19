#!/usr/bin/env python3
"""Replay a preserved registered RGB-D bundle. No camera/RPC/robot access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from piper_handeye.button_pose import ButtonDetection,YoloButtonDetector,estimate_button_camera_pose,select_button
from piper_handeye.numeric_layout_localization import NumericLayoutLocalizer
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from run_button_vision_server import load_handeye,scaled_camera_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-detections',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    stem=ROOT/'artifacts/captures/camera_snapshots/20261026_160126_454488'
    sources=[Path(str(stem)+'.jpg'),Path(str(stem)+'_depth_mm.png')]
    hashes={}
    for source in sources:
        target=args.output/source.name;shutil.copy2(source,target)
        hashes[source.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    image=cv2.imread(str(args.output/sources[0].name));depth=cv2.imread(str(args.output/sources[1].name),cv2.IMREAD_UNCHANGED).astype(float)*.001
    config_path=ROOT/'piper_button_pose.yaml';c=yaml.safe_load(config_path.read_text())
    _,full,_=load_handeye(ROOT/c['calibration']['handeye_yaml'])
    k=scaled_camera_matrix(full,c['calibration']['rgb_calibration_size'],c['depth']['capture_size'])
    localizer=NumericLayoutLocalizer.from_config(c['numeric_layout_localization'],config_path,k,c['button_geometry'],c['detector']['target_confidence'])
    if args.reuse_detections:
        prior=json.loads(args.reuse_detections.read_text())
        if prior['inputs_sha256']!=hashes:raise ValueError('reused detections belong to different RGB-D input')
        raw=tuple(ButtonDetection(0,d['label'],d['confidence'],tuple(d['bbox_xyxy'])) for d in prior['raw_detections'])
    else:
        detector=YoloButtonDetector(ROOT/c['detector']['model_path'],conf=.15,iou=.5,imgsz=640,device='cpu')
        import torch
        torch.set_num_threads(2)
        raw=tuple(detector.detect(image))
    reference=json.loads((ROOT/'artifacts/trials/20261027_numeric_panel_layout/manifest.json').read_text())
    annotated=next(f for f in reference['frames'] if f['id']=='reference_wide')['buttons']
    verified=tuple(ButtonDetection(0,l,.95,(v['center'][0]-v['diameter']/2,v['center'][1]-v['diameter']/2,v['center'][0]+v['diameter']/2,v['center'][1]+v['diameter']/2)) for l,v in annotated.items())
    # One historical RGB-D bundle, not four new camera frames. A virtual clock
    # tests timestamp plumbing only, never proves live data freshness.
    frame=LiveRgbdFrame(1,10.,1000.,0,0,image,depth)
    results=[]
    for mode,detections in [('saved_jpeg_cpu_yolo',raw),('verified_reference_centres',verified)]:
        for label in ['2','3','4','5']:
            hidden=tuple(d for d in detections if d.label!=label)
            result=localizer.localize(frame,label,hidden,now_monotonic=10.05)
            baseline=None
            try:
                direct=select_button(list(detections),label,.70)
                pose=estimate_button_camera_pose(direct,depth,k,panel_support_detections=tuple(d for d in detections if d.confidence>=.7),**c['button_geometry'])
                baseline=dict(p_camera_m=pose.p_camera.tolist(),plane_rms_m=pose.plane_rms_m)
            except RuntimeError:pass
            entry=dict(target=label,mode=mode,localization=result,baseline_direct_same_depth=baseline)
            if baseline is not None and result.get('valid_3d'):
                entry['difference_from_direct_same_depth_m']=float(np.linalg.norm(np.array(result['pose']['p_camera_m'])-baseline['p_camera_m']))
            results.append(entry);print(mode,label,result['reason'],flush=True)
    report=dict(scope='One historical registered RGB-D bundle, target labels removed from detector results only. Virtual clock/native timestamps unknown. Current calibration; no physical ground truth or live-freshness/press test. Manual anchors are separate research controls.',inputs_sha256=hashes,raw_detections=[dict(label=d.label,confidence=d.confidence,bbox_xyxy=list(d.bbox_xyxy)) for d in raw],results=results)
    (args.output/'paired_rgbd_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
