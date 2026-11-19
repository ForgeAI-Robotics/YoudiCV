"""Lazy JSON bridge to the existing arm dispatcher under its production Python."""
import json
import subprocess
import threading
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/qianjue/anaconda3/envs/piper-handeye/bin/python')


class NavigationArmClient:
    def __init__(self, events, run):
        self.events=events;self.run=Path(run);self.process=None;self.log=None
        self.lock=threading.RLock();self.pending={};self.closed=False

    @property
    def busy(self):
        with self.lock:return bool(self.pending)

    def request(self, operation, mission_id, **args):
        with self.lock:
            if self.closed:raise RuntimeError('机械臂后端已关闭')
            if self.process is None:
                self.log=(self.run/'arm_backend.log').open('ab')
                self.process=subprocess.Popen([str(PYTHON),str(ROOT/'tools/navigation_arm_worker.py'),'--run-dir',str(self.run)],
                    cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                    text=True,bufsize=1,start_new_session=True)
                threading.Thread(target=self._read,daemon=True).start()
            if self.process.poll() is not None:raise RuntimeError('机械臂后端已退出，请检查日志')
            request_id=str(uuid.uuid4());packet=dict(operation=operation,mission_id=mission_id,request_id=request_id,**args)
            self.pending[request_id]=packet
            try:self.process.stdin.write(json.dumps(packet,ensure_ascii=False)+'\n');self.process.stdin.flush()
            except Exception:
                self.pending.pop(request_id,None);raise
            return request_id

    def _read(self):
        for line in self.process.stdout:
            try:m=json.loads(line)
            except ValueError:continue
            if m.get('channel')!='navigation_arm_worker':continue
            with self.lock:
                expected=self.pending.get(m.get('request_id'))
                if expected is None or m.get('mission_id')!=expected['mission_id']:continue
                if m.get('event') in {'completed','error'}:self.pending.pop(m['request_id'],None)
            self.events.put(('arm_worker',m))
        code=self.process.wait()
        with self.lock:
            pending=list(self.pending.values());self.pending.clear()
        if not self.closed:
            self.events.put(('arm_worker_exit',dict(code=code,pending=pending)))
        if self.log:self.log.close()

    def close(self):
        with self.lock:
            self.closed=True
            if self.process is not None and self.process.poll() is None:
                # EOF makes the worker invoke the dispatcher's independent safe stop.
                try:self.process.stdin.close()
                except OSError:pass
