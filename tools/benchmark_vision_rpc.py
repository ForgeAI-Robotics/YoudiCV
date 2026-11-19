"""Read-only interleaved RPC benchmark against the running vision service."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import statistics
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.vision_rpc import VisionRpcClient
spec=importlib.util.spec_from_file_location('baseline_rpc', '/home/qianjue/snapshots/20260908_110636_button15_near_depth_pause/source/contral_test/piper_handeye/vision_rpc.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
cfg=yaml.safe_load((ROOT/'piper_button_pose.yaml').read_text())['vision_service']
clients={'unbuffered':old.VisionRpcClient(cfg['host'],cfg['port'],cfg['token'],timeout=2.), 'buffered':VisionRpcClient(cfg['host'],cfg['port'],cfg['token'],timeout=2.)}
rows={k:[] for k in clients}
for index in range(30):
 for name in (list(clients) if index%2==0 else list(reversed(clients))):
  start=time.perf_counter()
  result=clients[name].request('latest',target_label='15',include_realtime=False,after_sequence=0,max_batch_age_sec=.35)
  rows[name].append(time.perf_counter()-start)
summary={name:{'median_ms':statistics.median(values)*1000,'mean_ms':statistics.mean(values)*1000,'max_ms':max(values)*1000} for name,values in rows.items()}
path=ROOT/'optimization_runs/20260908_sequential/rpc_buffering.json'
path.write_text(json.dumps({'summary':summary,'seconds':rows},indent=2)); print(json.dumps(summary))
