import sys

from piper_handeye.button_light_process import ButtonLightProcess


def test_persistent_worker_protocol_reuses_one_process(tmp_path):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """import json,sys
P='BUTTON_LIGHT_RESPONSE '
print(P+json.dumps({'ready':True,'loads':1}),flush=True)
count=0
for line in sys.stdin:
 r=json.loads(line)
 if r.get('command')=='shutdown': break
 count+=1
 print(P+json.dumps({'id':r['id'],'ok':True,'state':'OFF','count':count}),flush=True)
"""
    )
    with ButtonLightProcess([sys.executable, str(worker)], 2.0) as client:
        first = client.observe("one.jpg", "15")
        second = client.observe("two.jpg", "15")
        assert client.ready["loads"] == 1
        assert first["count"] == 1
        assert second["count"] == 2
        assert first["state"] == second["state"] == "OFF"
