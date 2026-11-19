import stat
import pytest
from tools.piper_escape_geometry import EscapeGeometry

def executable(tmp_path,body):
    path=tmp_path/'worker'
    path.write_text('#!/usr/bin/python3\n'+body)
    path.chmod(path.stat().st_mode|stat.S_IXUSR)
    return path

def test_resident_protocol_and_cleanup(tmp_path):
    path=executable(tmp_path,"import sys,json\nprint('ESCAPE_GEOMETRY {\"ready\":true}',flush=True)\nfor line in sys.stdin:\n print('ESCAPE_GEOMETRY '+json.dumps(dict(ok=True,overlap=False,upward_exit_m=0.,clearance_m=.007,panel_padding_m=.001)),flush=True)\n")
    worker=EscapeGeometry(path)
    assert worker.measure([0.]*6)['clearance_m']==.007
    process=worker._process
    assert worker.measure([.1]*6)['clearance_m']==.007
    assert worker._process is process
    worker.close()
    assert process.poll() is not None

def test_malformed_reply_fails_closed(tmp_path):
    path=executable(tmp_path,"import sys\nprint('ESCAPE_GEOMETRY {\"ready\":true}',flush=True)\nfor line in sys.stdin:\n print('ESCAPE_GEOMETRY invalid',flush=True)\n")
    worker=EscapeGeometry(path)
    with pytest.raises(RuntimeError,match='invalid escape geometry response'):
        worker.measure([0.]*6)
    assert worker._process is None
