"""Train the reviewed 1000-paper/400-negative split, preserving held-out images."""
import json
import os
import time
import traceback
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'expansion_v5'
DATA=WORK/'yolo26_paper1000_negative400'
BASE=ROOT/'training_runs/yolo26m_negative_v4/weights/best.pt'
NAME='yolo26m_paper1000_negative400_v5'
RUN=ROOT/'training_runs'/NAME
STATE=WORK/'training_progress.json'
EPOCHS=40

def state(stage,**kwargs):
    payload={'stage':stage,'updated':time.strftime('%Y-%m-%d %H:%M:%S'),
             'pid':os.getpid(),'run':str(RUN),'epochs_max':EPOCHS,**kwargs}
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    temporary.replace(STATE)

def main():
    assert not RUN.exists(),'Refuse to overwrite existing training run'
    audit=json.loads((DATA/'audit.json').read_text())
    assert audit['paper_training_images']==1000 and BASE.is_file()
    counts=audit['counts']
    for row in audit['files']:
        assert Path(row['image']).is_file() and Path(row['label']).is_file()
    torch.set_num_threads(4)
    state('initializing',counts=counts,base_checkpoint=str(BASE))
    model=YOLO(str(BASE))
    def on_start(trainer):
        state('training',epoch_completed=0,counts=counts,base_checkpoint=str(BASE))
    def on_epoch(trainer):
        metrics={str(k):float(v) for k,v in trainer.metrics.items()}
        state('training',epoch_completed=int(trainer.epoch)+1,counts=counts,
              metrics=metrics,best_fitness=float(trainer.best_fitness),
              elapsed_seconds=round(time.time()-trainer.train_time_start,1))
    model.add_callback('on_train_start',on_start)
    model.add_callback('on_fit_epoch_end',on_epoch)
    model.train(data=str(DATA/'data.yaml'),epochs=EPOCHS,patience=12,imgsz=640,batch=8,
                device=0,workers=4,seed=20261015,deterministic=True,
                project=str(ROOT/'training_runs'),name=NAME,exist_ok=False,
                pretrained=True,amp=True,cache=False,optimizer='AdamW',lr0=0.0002,lrf=0.1,
                warmup_epochs=2,cos_lr=True,fliplr=0.0,flipud=0.0,degrees=0.0,shear=0.0,
                mosaic=0.5,close_mosaic=10,mixup=0.0,copy_paste=0.0,
                plots=True,save=True,save_period=10)
    best=RUN/'weights/best.pt'
    assert best.is_file()
    state('testing',checkpoint=str(best),counts=counts)
    net=YOLO(str(best))
    result=net.val(data=str(DATA/'data.yaml'),split='test',imgsz=640,batch=8,
                   device=0,workers=4,plots=True,project=str(ROOT/'training_runs'),
                   name=NAME+'_test',exist_ok=False)
    report={'checkpoint':str(best),'counts':counts,'mixed_test_metrics':
            {str(k):float(v) for k,v in result.results_dict.items()},
            'test_used_for_training_or_checkpoint_selection':False,
            'near_duplicate_group_audit_complete':False,'negative_test':{}}
    negatives=[r['image'] for r in audit['files'] if r['split']=='test' and r['kind']=='negative']
    predictions=[]
    for begin in range(0,len(negatives),8):
        for pred in net.predict(negatives[begin:begin+8],imgsz=640,conf=0.25,
                                device=0,batch=8,verbose=False):
            boxes=[]
            for j,b in enumerate(pred.boxes):
                kp=pred.keypoints.data[j,0].tolist()
                boxes.append({'class_id':int(b.cls.item()),'confidence':float(b.conf.item()),
                              'bbox':b.xyxy[0].tolist(),'keypoint':kp})
            predictions.append({'image':pred.path,'objects':boxes})
    for threshold in [0.25,0.5,0.75]:
        rows=[[o for o in r['objects'] if o['confidence']>=threshold] for r in predictions]
        grasp=[[o for o in r if o['class_id']==0 and len(o['keypoint'])>2 and o['keypoint'][2]>=0.5] for r in rows]
        report['negative_test'][str(threshold)]={
            'images':len(rows),'images_with_any_false_box':sum(bool(r) for r in rows),
            'false_boxes':sum(map(len,rows)),
            'images_with_false_grasp_point':sum(bool(r) for r in grasp)}
    report['negative_predictions']=predictions
    out=WORK/'training_test_report.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    state('completed',checkpoint=str(best),report=str(out),counts=counts)
    print('COMPLETED',str(best),flush=True)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
