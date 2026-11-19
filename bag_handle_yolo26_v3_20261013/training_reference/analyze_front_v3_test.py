"""Measure source-pixel errors and build reference/prediction comparison gallery."""
import json, math, statistics
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'test50_predictions_v3'
report=json.loads((OUT/'report.json').read_text())
truth=json.loads((ROOT/'reannotation_v3_front/test/annotations.json').read_text())['images']
def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area+1e-9)

errors=[]; eligible_valid=0; rows=[]; invalid_as_graspable=0; unmatched_graspable=0
(OUT/'comparisons').mkdir(exist_ok=True)
for gt,pred in zip(truth,report['images']):
    idx=gt['test_index']; assert idx==pred['test_index']
    objects=pred['objects']; candidates=[o for o in objects if o['class']=='graspable_bag' and o['displayed_grasp_point'] is not None]
    eligible=gt.get('evaluation_eligible',True)
    if eligible:
        targets=gt['objects']; eligible_valid+=sum(o[1]=='valid' for o in targets)
        matched=set()
        if candidates and targets:
            overlaps=np.array([[iou(p['bbox'],t[0]) for t in targets] for p in candidates])
            pp,tt=linear_sum_assignment(-overlaps)
            for p,t in zip(pp,tt):
                if overlaps[p,t]<0.5:continue
                matched.add(int(p)); status=targets[t][1]
                if status=='valid':
                    ref=targets[t][2]; point=candidates[p]['displayed_grasp_point']
                    error=math.dist(ref,point); errors.append(error)
                    rows.append({'test_index':idx,'gt_object':int(t)+1,'reference':ref,'prediction':point,'error_pixels':error,'bbox_iou':float(overlaps[p,t])})
                else: invalid_as_graspable+=1
        unmatched_graspable+=len(candidates)-len(matched)
    original=Image.open(ROOT/'reannotation_v3_front/test/images'/f'{idx:03d}.jpg').convert('RGB')
    im=original.resize((original.width*3,original.height*3)); d=ImageDraw.Draw(im)
    for bbox,status,point in gt['objects']:
        d.rectangle(tuple(v*3 for v in bbox),outline='#2277ff',width=2)
        if point:
            x,y=[v*3 for v in point]; d.ellipse((x-5,y-5,x+5,y+5),outline='#2277ff',width=3)
    for p in objects:
        d.rectangle(tuple(v*3 for v in p['bbox']),outline='#00d840' if p['class']=='graspable_bag' else '#ff3030',width=1)
        if p['displayed_grasp_point']:
            x,y=[v*3 for v in p['displayed_grasp_point']];d.ellipse((x-3,y-3,x+3,y+3),fill='#00d840',outline='black')
    im.save(OUT/'comparisons'/f'{idx:03d}.jpg',quality=97)
summary={'prediction_confidence':0.25,'keypoint_visibility':0.5,'one_to_one_bbox_iou_threshold':0.5,
         'eligible_valid_reference_points':eligible_valid,'matched_valid_predicted_points':len(errors),
         'unmatched_or_no_point_valid_targets':eligible_valid-len(errors),
         'mean_error_pixels_matched_only':statistics.mean(errors) if errors else None,
         'median_error_pixels_matched_only':statistics.median(errors) if errors else None,
         'within_5px_including_missed_targets':sum(e<=5 for e in errors)/eligible_valid,
         'within_10px_including_missed_targets':sum(e<=10 for e in errors)/eligible_valid,
         'graspable_predictions_matched_nonvalid_targets':invalid_as_graspable,
         'unmatched_graspable_predictions':unmatched_graspable,
         'warning':'Source images have different sizes. Reference annotations are assistant-authored, not human-approved robot grasp ground truth.'}
(OUT/'point_error_report.json').write_text(json.dumps({'summary':summary,'matches':rows},ensure_ascii=False,indent=2))
gallery=''.join(f'<figure><figcaption>测试 {i:03d}</figcaption><a href="comparisons/{i:03d}.jpg"><img src="comparisons/{i:03d}.jpg"></a></figure>' for i in range(1,51))
(OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>YOLO26 v3 测试</title><style>body{font-family:sans-serif}main{display:grid;grid-template-columns:repeat(3,1fr)}figure{margin:8px}img{width:100%}</style><h1>50张测试图</h1><p>蓝色框与空心点＝参考标注；绿色框与实心点＝模型可抓取预测；红框＝模型不可抓取预测。点击图片放大。</p><p>参考标注未经过用户逐点确认；5张不参与定量评估，但都显示预测结果。</p><main>'+gallery+'</main>')
print(json.dumps(summary,ensure_ascii=False,indent=2))
