#!/usr/bin/env python3
"""Offline RGB-D/real RPC-dispatch/stationary-transform replay, never hardware.

A preserved real RGB-D bundle tests one conversion and duplicate rejection.
Separate synthetic independent bundles test multi-frame acquisition. Historical
encoder poses are unavailable: every encoder pose below is explicitly simulated.
No claim about actual multi-frame freshness, calibration accuracy or pressing.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.button_pose import ButtonDetection
from piper_handeye.numeric_layout_localization import NumericLayoutLocalizer
from piper_handeye.numeric_layout_sampling import LayoutStationarySampler
from piper_handeye.orbbec_v1_stream import LiveRgbdFrame
from run_button_vision_server import VisionService


def dispatch_snapshot(record, now):
    """Real vision dispatch code, with only its cached sample replaced."""
    record = deepcopy(record)
    frame = SimpleNamespace(sequence=record['frame_sequence'],
                            received_unix=record['capture_time_unix_sec'])
    sample = SimpleNamespace(sequence=frame.sequence, frame=frame,
                             target_label=record['target_label'], numeric_layout_localization=record)
    service = VisionService.__new__(VisionService)
    service.args = SimpleNamespace(token='offline-only')
    service.perception = SimpleNamespace(target_label=sample.target_label, latest_sample=lambda *_: sample)
    with patch('run_button_vision_server.time.time', return_value=now):
        return service.dispatch(dict(command='layout_localization_snapshot',token='offline-only',
                                     target_label=sample.target_label,include_rgb=False))


def add_simulated_encoders(sampler, capture_mono, base):
    for t in [capture_mono-.02, capture_mono+.02]:
        sampler.observe_robot(SimpleNamespace(timestamp=t,joints=np.zeros(6),base_to_gripper=base),
                              now_unix=t+990,now_monotonic=t)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())
    handeye_path=ROOT/config['calibration']['handeye_yaml']
    handeye=np.asarray(yaml.safe_load(handeye_path.read_text())['gripper_to_camera'],float)
    base=np.eye(4);base[:3,3]=[.3,-.1,.4]
    base[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    prior_path=ROOT/'artifacts/diagnostics/20261027_same_frame_layout_localization/paired_rgbd/paired_rgbd_replay.json'
    prior=json.loads(prior_path.read_text())
    for name,digest in prior['inputs_sha256'].items():
        if hashlib.sha256((prior_path.parent/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('historical bundle SHA256 mismatch')
    historical=[]
    for result in prior['results']:
        if result['mode']!='saved_jpeg_cpu_yolo':continue
        record=result['localization'];target=record['target_label']
        s=LayoutStationarySampler(config,handeye,target,now_unix=999.9,now_monotonic=9.9)
        add_simulated_encoders(s,10.,base)
        envelope=dispatch_snapshot(record,1000.05)
        row=s.consume(envelope,now_unix=1000.05,now_monotonic=10.05)
        expected=base@handeye@np.asarray(record['pose']['camera_to_button'])
        if not row['accepted']:raise AssertionError(row)
        if not np.allclose(expected,row['base_to_button'],atol=1e-10):raise AssertionError('wrong coordinate conversion')
        duplicates=[s.consume(envelope,now_unix=1000.05,now_monotonic=10.05) for _ in range(5)]
        snapshot=s.snapshot(now_unix=1000.05)
        assert not snapshot['stable'] and snapshot['sample_count']==1
        assert all(r['reason']=='not_a_new_frame' for r in duplicates)
        historical.append(dict(target=target,accepted_one_frame=True,repeated_bundle_stays_unstable=True,
                               simulated_base_to_button=expected.tolist(),snapshot=snapshot))
    profile_path=ROOT/config['numeric_layout_localization']['profile_path']
    profile=json.loads(profile_path.read_text())
    k=np.array([[400.,0,320],[0,400.,180],[0,0,1]])
    localizer=NumericLayoutLocalizer.from_config(config['numeric_layout_localization'],ROOT/'piper_button_pose.yaml',
                                               k,config['button_geometry'],config['detector']['target_confidence'])
    synthetic=[]
    for target in '2345':
        s=LayoutStationarySampler(config,handeye,target,now_unix=999.9,now_monotonic=9.9)
        detections=[]
        for label,point in profile['coordinates'].items():
            if label==target:continue
            cx,cy=180+45*point[0],300+30*point[1]
            if 20<cy<340:
                detections.append(ButtonDetection(0,label,.95,(cx-11,cy-11,cx+11,cy+11)))
        for i,dz in enumerate([0.,.0004,-.0004,.0002]):
            t=10.+i*.1
            frame=LiveRgbdFrame(i+1,t,t+990,100+i,200+i,
                                np.full((360,640,3),150,np.uint8),np.full((360,640),.6+dz))
            record=localizer.localize(frame,target,detections,now_monotonic=t+.05)
            add_simulated_encoders(s,t,base)
            envelope=dispatch_snapshot(record,t+990+.05)
            row=s.consume(envelope,now_unix=t+990+.05,now_monotonic=t+.05)
            if not row['accepted']:raise AssertionError(row)
        snapshot=s.snapshot(now_unix=1000.35)
        assert snapshot['stable'] and snapshot['sample_count']==4 and snapshot['motion_eligible'] is False
        synthetic.append(snapshot)
    report=dict(scope=__doc__, historical_encoder_data_available=False,
                historical_bundle_inputs_sha256=prior['inputs_sha256'],
                handeye_sha256=hashlib.sha256(handeye_path.read_bytes()).hexdigest(),
                profile_sha256=config['numeric_layout_localization']['profile_sha256'],
                historical_single_bundle=historical,synthetic_four_bundle_rpc_to_base=synthetic,
                robot_commands_sent=0,service_restarts=0,images_created=0)
    destination=args.output/'sampling_replay.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(report=str(destination),historical=len(historical),synthetic=len(synthetic),
                          robot_commands_sent=0),ensure_ascii=False))


if __name__=='__main__':main()
