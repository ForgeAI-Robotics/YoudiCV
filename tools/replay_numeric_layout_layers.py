#!/usr/bin/env python3
"""Audit actual raw/local/layout layers on saved images; no hiding or robot IO."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from piper_handeye.button_pose import ButtonDetection,select_button
from piper_handeye.numeric_button_detection import retry_numeric_button_tile
from tools.offline_numeric_panel_layout import yolo_anchors,project
from tools.offline_numeric_panel_layout_v3 import infer_image


def recover_one_conflicting_anchor(template,anchors,target,image):
    """Rare last resort: four strong agreeing labels may expose one wrong label.

    Never modify raw detections or convert this result to a YOLO detection.
    Requires >=80% retained references and a unique consistent target estimate.
    """
    aa=[a for a in anchors if a['label']!=target]
    if not 5<=len(aa)<=8:return dict(accepted=False,reason='single_outlier_recovery_needs_five_to_eight',target_detected=False)
    choices=[]
    for i,bad in enumerate(aa):
        kept=aa[:i]+aa[i+1:]
        if any(a.get('confidence',0)<.85 for a in kept):continue
        result=infer_image(template,kept,target,image)
        if not result['accepted']:continue
        s=np.array([template[a['label']] for a in kept],float);d=np.array([a['center'] for a in kept],float)
        h,_=cv2.findHomography(s,d,0)
        if h is None:continue
        predicted=project(h,[template[bad['label']]])[0]
        error=float(np.linalg.norm(predicted-bad['center']))
        if error<=max(12,.75*bad['diameter']):continue
        result['excluded_conflicting_anchor']=dict(label=bad['label'],confidence=bad.get('confidence'),observed_center=bad['center'],predicted_center=predicted.tolist(),conflict_px=error)
        result['retained_fraction']=len(kept)/len(aa);choices.append(result)
    if not choices:return dict(accepted=False,reason='no_unique_single_outlier_solution',target_detected=False)
    centers=np.array([r['center'] for r in choices]);median=np.median(centers,axis=0)
    if max(np.linalg.norm(centers-median,axis=1))>min(6,.2*choices[0]['estimated_diameter_px']):return dict(accepted=False,reason='single_outlier_solutions_disagree',target_detected=False)
    result=choices[0];result['method']='four_plus_strong_references_reject_one_conflict';result['consistent_solutions']=len(choices)
    return result


def run_layers(detector,image,raw,target,template):
    try:
        d=select_button(list(raw),target,.70)
        return dict(stage='existing_raw_yolo',center=list(d.center_pixel),extra_inferences=0)
    except RuntimeError:
        pass
    # Preserve the production ambiguity rejection; this experiment concerns
    # missing/weak target recovery, not overriding two competing strong targets.
    if sum(d.label==target and d.confidence>=.70 for d in raw)>1:
        return dict(stage='unavailable',reason='raw_target_ambiguous',extra_inferences=0)
    local,audit=retry_numeric_button_tile(detector,image,raw,target,.70)
    try:
        d=select_button(list(local),target,.70)
        return dict(stage='existing_local_yolo',center=list(d.center_pixel),extra_inferences=int(audit['attempted']),local_audit=audit)
    except RuntimeError:
        pass
    payload=[dict(label=d.label,confidence=d.confidence,bbox_xyxy=list(d.bbox_xyxy)) for d in local]
    aa=yolo_anchors(payload,template,[image.shape[1],image.shape[0]])
    geometry_start=time.perf_counter()
    geometry=infer_image(template,aa,target,image)
    if not geometry['accepted']:
        recovery=recover_one_conflicting_anchor(template,aa,target,image)
        if recovery['accepted']:
            recovery['previous_failure']=geometry['reason'];geometry=recovery
    return dict(stage='layout_candidate_only' if geometry['accepted'] else 'unavailable',
                extra_inferences=int(audit['attempted']),local_audit=audit,geometry=geometry,
                center=geometry.get('center'),geometry_sec=time.perf_counter()-geometry_start)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sources=[('artifacts/trials/20261027_numeric_panel_layout/manifest.json','artifacts/trials/20261027_numeric_panel_layout/results/yolo_replay.json'),('artifacts/trials/20261027_numeric_panel_layout_v2/holdout_manifest.json','artifacts/trials/20261027_numeric_panel_layout_v2/holdout_v1/yolo_replay.json'),('artifacts/trials/20261027_numeric_panel_layout_v3/new_manifest.json','artifacts/trials/20261027_numeric_panel_layout_v3/new_v1/yolo_replay.json')]
    from piper_handeye.button_pose import YoloButtonDetector
    detector=YoloButtonDetector(ROOT.parent/'button_recognition/models/elevator_button_yolo26m_best.pt',conf=.15,iou=.5,imgsz=640,device='cpu')
    import torch
    torch.set_num_threads(2)
    rows=[]
    for manifest_path,replay_path in sources:
        mp=ROOT/manifest_path;m=json.loads(mp.read_text());replay=json.loads((ROOT/replay_path).read_text())['frames']
        for f in m['frames']:
            if f['split']!='validation':continue
            image=cv2.imread(str(mp.parent/f['image']))
            raw=[ButtonDetection(0,d['label'],d['confidence'],tuple(d['bbox_xyxy'])) for d in replay[f['id']]['detections']]
            for target in f['targets']:
                start=time.perf_counter();result=run_layers(detector,image,raw,target,m['template']['coordinates'])
                result.update(frame=f['id'],target=target,seconds=time.perf_counter()-start)
                if result.get('center'):
                    import numpy as np
                    result['error_px_against_assistant_annotation']=float(np.linalg.norm(np.array(result['center'])-f['buttons'][target]['center']))
                rows.append(result)
                if result['stage']!='existing_raw_yolo':print(f['id'],target,result['stage'],result.get('local_audit',{}).get('reason'),flush=True)
    stages={k:sum(r['stage']==k for r in rows) for k in sorted({r['stage'] for r in rows})}
    a.output.mkdir(parents=True,exist_ok=True);report=dict(scope='Actual unmasked raw detections then existing production local YOLO then experimental geometry; not physical success. Does not correct a wrong strong raw target.',cases=len(rows),stages=stages,extra_inferences=sum(r['extra_inferences'] for r in rows),results=rows)
    (a.output/'unmasked_layers.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(dict(cases=len(rows),stages=stages,extra_inferences=report['extra_inferences']),indent=2))


if __name__=='__main__':main()
