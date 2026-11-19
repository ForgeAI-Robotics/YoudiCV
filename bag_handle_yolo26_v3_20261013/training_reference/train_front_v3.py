"""Train corrected v3 annotations, then evaluate held-out test images automatically."""
import collections, hashlib, json, shutil, time
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ANNOT = ROOT / 'reannotation_v3_front'
DATA = ROOT / 'yolo26_front_v3'
RUN = ROOT / 'training_runs/yolo26m_front_v3'
OUT = ROOT / 'test50_predictions_v3'
STATE = ROOT / 'front_v3_progress.json'

def state(stage, **kwargs):
    STATE.write_text(json.dumps({'stage': stage, 'updated': time.strftime('%Y-%m-%d %H:%M:%S'), **kwargs}, ensure_ascii=False, indent=2))

def prepare():
    assert not DATA.exists(), 'Refusing to overwrite an existing dataset.'
    assert not RUN.exists(), 'Refusing to overwrite an existing training run.'
    training = json.loads((ANNOT/'train/annotations.json').read_text())['images']
    tests = json.loads((ANNOT/'test/annotations.json').read_text())['images']
    previous = json.loads((ROOT/'yolo26_direct_v1/split_report.json').read_text())
    membership = {r['index']: split for split in ['train','val'] for r in previous[split]}
    test_hashes = {hashlib.sha256((ANNOT/'test/images'/f'{r["test_index"]:03d}.jpg').read_bytes()).hexdigest() for r in tests}
    seen = set(); report = {'train': [], 'val': [], 'test': [], 'excluded': [], 'test_excluded': [], 'split_policy': 'Keep original v1 train/val assignment; apply v3 training eligibility. Reserved test remains separate.', 'human_approved': False}
    for r in training:
        i = r['train_index']; src = ANNOT/'train/images'/f'{i:03d}.jpg'
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        reason = ('not_training_eligible' if not r.get('training_eligible',True) else 'test_duplicate' if digest in test_hashes else 'exact_duplicate' if digest in seen else 'no_original_split' if i not in membership else None)
        if reason:
            report['excluded'].append({'index': i, 'reason': reason}); continue
        seen.add(digest); split = membership[i]
        for kind in ['images','labels']:
            folder = DATA/kind/split; folder.mkdir(parents=True, exist_ok=True)
            source = ANNOT/'train'/kind/(f'{i:03d}.jpg' if kind=='images' else f'{i:03d}.txt')
            shutil.copy2(source, folder/source.name)
        report[split].append({'index': i, 'sha256': digest})
    for r in tests:
        i = r['test_index']
        if not r.get('evaluation_eligible',True):
            report['test_excluded'].append(i); continue
        for kind, ext in [('images','jpg'),('labels','txt')]:
            folder = DATA/kind/'test'; folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ANNOT/'test'/kind/f'{i:03d}.{ext}', folder)
        report['test'].append(i)
    for split in ['train','val']:
        hashes = {r['sha256'] for r in report[split]}
        assert hashes.isdisjoint(test_hashes)
    assert {r['sha256'] for r in report['train']}.isdisjoint(r['sha256'] for r in report['val'])
    yaml = f'path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: graspable_bag\n  1: invalid_bag\n  2: uncertain_bag\nkpt_shape: [1, 3]\nflip_idx: [0]\n'
    (DATA/'data.yaml').write_text(yaml)
    (DATA/'split_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    counts = {s: len(report[s]) for s in ['train','val','test']}
    print('PREPARED',counts,flush=True); return counts

def main():
    torch.set_num_threads(4)
    counts = prepare(); state('training', counts=counts)
    model = YOLO(str(ROOT.parent.parent/'yolo26m-pose.pt'))
    model.train(data=str(DATA/'data.yaml'), epochs=100, patience=25, imgsz=640,
                batch=8, device=0, workers=4, seed=20261013, deterministic=True,
                project=str(ROOT/'training_runs'), name='yolo26m_front_v3',
                exist_ok=False, pretrained=True, amp=True, cache=False,
                fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0,
                mosaic=0.5, close_mosaic=10, mixup=0.0, copy_paste=0.0,
                plots=True, save=True, save_period=10)
    checkpoint = RUN/'weights/best.pt'; assert checkpoint.exists()
    state('testing', counts=counts, checkpoint=str(checkpoint))
    best = YOLO(str(checkpoint))
    metrics = best.val(data=str(DATA/'data.yaml'), split='test', imgsz=640,
                       batch=8, device=0, workers=4, plots=True,
                       project=str(ROOT/'training_runs'), name='yolo26m_front_v3_test', exist_ok=False)
    OUT.mkdir(exist_ok=False)
    records = json.loads((ANNOT/'test/annotations.json').read_text())['images']; predictions=[]
    for r in records:
        i=r['test_index']; source=ANNOT/'test/images'/f'{i:03d}.jpg'
        result=best.predict(str(source),imgsz=640,conf=0.25,device=0,verbose=False)[0]
        im=Image.open(source).convert('RGB'); draw=ImageDraw.Draw(im); objects=[]
        for j,box in enumerate(result.boxes):
            cls=int(box.cls.item()); score=float(box.conf.item()); bbox=box.xyxy[0].tolist()
            kp=result.keypoints.data[j,0].tolist(); point=kp[:2] if cls==0 and kp[2]>=0.5 else None
            color={0:'#00ee45',1:'#ff3030',2:'#ffa500'}[cls]
            draw.rectangle(bbox,outline=color,width=2)
            draw.text((bbox[0],max(0,bbox[1]-10)),f'{best.names[cls]} {score:.2f}',fill=color,stroke_width=1,stroke_fill='black')
            if point:
                x,y=point; draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='black')
            objects.append({'class':best.names[cls],'confidence':score,'bbox':bbox,'raw_keypoint':kp,'displayed_grasp_point':point})
        im.save(OUT/f'{i:03d}.jpg',quality=97)
        predictions.append({'test_index':i,'evaluation_eligible':r.get('evaluation_eligible',True),'objects':objects})
    for page in range(5):
        sheet=Image.new('RGB',(1000,1500),'#eee'); draw=ImageDraw.Draw(sheet)
        for j in range(10):
            i=page*10+j+1; im=Image.open(OUT/f'{i:03d}.jpg'); im.thumbnail((470,255))
            x=j%2*500+15; y=j//2*300+30; sheet.paste(im,(x,y)); draw.text((x,y-20),f'Test {i:03d}',fill='black')
        sheet.save(OUT/f'page_{page+1}.jpg',quality=97)
    report={'checkpoint':str(checkpoint),'counts':counts,'metrics':{k:float(v) for k,v in metrics.results_dict.items()},
            'reference_labels':'Assistant visually reviewed annotations, not independently human-approved gold truth.',
            'conf':0.25,'keypoint_visibility_threshold':0.5,'images':predictions}
    old=ROOT/'training_runs/yolo26m_direct_v1/weights/best.pt'
    if old.exists():
        state('baseline_test_comparison', counts=counts)
        baseline=YOLO(str(old)).val(data=str(DATA/'data.yaml'),split='test',imgsz=640,batch=8,device=0,workers=4,plots=False,
                                   project=str(ROOT/'training_runs'),name='yolo26m_v1_on_v3_test',exist_ok=False)
        report['old_model_same_test_metrics']={k:float(v) for k,v in baseline.results_dict.items()}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    state('completed',counts=counts,checkpoint=str(checkpoint),metrics=report['metrics'],report=str(OUT/'report.json'))
    print('COMPLETED',json.dumps({k:v for k,v in report.items() if k!='images'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        state('failed',error=repr(e)); raise
