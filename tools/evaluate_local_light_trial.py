#!/usr/bin/env python3
"""Three inspected stationary frames; manual ROIs, excludes moving/occluded frames."""
import json
from pathlib import Path
from opencv_light_worker import RelativeLuminance
root=Path(__file__).resolve().parents[1]
boxes={'1':[282,143,354,216],'2':[277,46,354,119],'9':[398,45,476,120],'B1':[280,228,357,306]}
results=[]
for name in ('20261009_133837_363062.jpg','20261009_133837_448696.jpg','20261009_133837_507238.jpg'):
    image=root/'button_light_snapshots'/name
    for target,box in boxes.items():
        r=RelativeLuminance().predict(image,target,box)
        results.append(dict(image=str(image),**r))
out=root/'light_algorithm_trials/20261009_local_regions/offline_results.json'
out.write_text(json.dumps(results,ensure_ascii=False,indent=2))
for target in boxes:
    rows=[r for r in results if r['target']==target]
    print(target,'frames',len(rows),'legacy', {s:sum(r['legacy_state']==s for r in rows) for s in ['ON','OFF','UNCERTAIN']},'local',{s:sum(r['state']==s for r in rows) for s in ['ON','OFF','UNCERTAIN']})
