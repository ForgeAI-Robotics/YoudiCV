#!/usr/bin/env python3
"""Render saved offline evidence only; no model, camera or robot connections."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/trials/20261027_numeric_panel_layout'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size): return ImageFont.truetype(FONT,size)


def main():
    m=json.loads((BASE/'manifest.json').read_text())
    report=json.loads((BASE/'results/evaluation.json').read_text())
    out=BASE/'images';out.mkdir(exist_ok=True)
    canvas=Image.new('RGB',(960,900),'white');d=ImageDraw.Draw(canvas)
    d.text((30,20),'内呼面板相对布局模板（离线试验）',fill='#152c48',font=font(30))
    d.text((30,70),'行列关系已核对原图；间距是相对单位，并非毫米标定。',fill='black',font=font(21))
    for label,(x,y) in m['template']['coordinates'].items():
        px,py=260+x*260,770+y*75
        d.ellipse((px-26,py-26,px+26,py+26),fill='#ddeafb',outline='#194a80',width=2)
        d.text((px-18,py-20),label,fill='#123a69',font=font(26))
    d.text((610,210),'可用邻键：9～14、6～7等',fill='black',font=font(19))
    d.text((610,255),'每张图重新匹配几何位置',fill='black',font=font(19))
    d.text((610,300),'不沿用上一张图的像素坐标',fill='black',font=font(19))
    d.text((610,345),'不据此推断按钮亮灭',fill='black',font=font(19))
    d.text((30,845),'当前未接入正式识别或按压；B1、开门、关门不在本模板中。',fill='#923333',font=font(21))
    canvas.save(out/'01_面板布局模板.png')
    examples=[('wide_lower_26_1','1'),('close_upper_23_5','5'),('close_bottom_26_2','2'),('wide_reflection_wrong3','3')]
    canvas=Image.new('RGB',(1280,1000),'white');d=ImageDraw.Draw(canvas)
    d.text((20,10),'隐藏目标框后，用其他数字按钮推算（原图离线回放）',font=font(27),fill='#183850')
    for idx,(fid,target) in enumerate(examples):
        f=next(f for f in m['frames'] if f['id']==fid)
        rr=next(r for r in report['results'] if r['frame']==fid and r['target']==target and r['mode']=='raw_yolo_cpu_jpeg')
        im=Image.open(BASE/f['image']).convert('RGB');di=ImageDraw.Draw(im)
        replay=json.loads((BASE/'results/yolo_replay.json').read_text())['frames'][fid]['detections']
        # Show only anchors actually supplied to the estimator.
        for dd in replay:
            if dd['label'] not in rr.get('anchors',[]): continue
            x1,y1,x2,y2=dd['bbox_xyxy'];di.rectangle((x1,y1,x2,y2),outline='#39baff',width=2)
            di.text((x1,y2),dd['label'],font=font(14),fill='#39baff')
        x,y=f['buttons'][target]['center'];di.ellipse((x-7,y-7,x+7,y+7),outline='#29ff50',width=2)
        if rr['accepted']:
            x,y=rr['center'];di.line((x-10,y,x+10,y),fill='#ff3939',width=2);di.line((x,y-10,x,y+10),fill='#ff3939',width=2)
        xoff,yoff=(idx%2)*640,70+(idx//2)*450
        canvas.paste(im,(xoff,yoff))
        text=f"隐藏{target}："+(f"通过，中心误差 {rr['error_px']:.2f} 像素" if rr['accepted'] else '拒绝，参考按钮不足')
        d.text((xoff+10,yoff+365),text,font=font(22),fill='#223344')
        d.text((xoff+10,yoff+398),fid,font=font(15),fill='#555555')
    d.text((20,975),'蓝框：YOLO参考；绿圈：目视标注中心；红十字：推算中心。并非实机按压验证。',font=font(19),fill='#333333')
    canvas.save(out/'02_隐藏目标离线验证.png')
    print(out)


if __name__=='__main__': main()
