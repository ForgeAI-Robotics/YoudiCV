"""Exercise preservation and refusal boundaries without touching robot paths."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('runtime_reset',ROOT/'tools/reset_runtime_data.py')
resetter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resetter)


def test_verified_reset_retains_binary_content_and_can_restore(tmp_path):
    run=tmp_path/'logs';run.mkdir();(run/'sub').mkdir()
    content=bytes(range(256))*31
    (run/'sub/trace.bin').write_bytes(content)
    (run/'alias').symlink_to('sub/trace.bin')
    manifest,archive=resetter.backup([run],tmp_path/'backup')
    resetter.reset([run],tmp_path/'backup',manifest,True)
    assert list(run.iterdir())==[]
    restore=tmp_path/'restored'
    subprocess.run([sys.executable,str(ROOT/'tools/restore_runtime_backup.py'),
        str(archive),'--output',str(restore)],check=True,capture_output=True)
    assert (restore/'sub/trace.bin').read_bytes()==content
    assert (restore/'alias').is_symlink()
    assert (restore/'alias').read_bytes()==content


@pytest.mark.parametrize('change',['edit_existing','add_file'])
def test_data_changed_after_backup_refuses_reset(tmp_path,change):
    run=tmp_path/'logs';run.mkdir();f=run/'log';f.write_text('original')
    manifest,_=resetter.backup([run],tmp_path/'backup')
    if change=='edit_existing':f.write_text('new unbacked data')
    else:(run/'newfile').write_text('new unbacked data')
    with pytest.raises(RuntimeError):resetter.reset([run],tmp_path/'backup',manifest,True)
    assert f.exists()
    if change=='add_file':assert (run/'newfile').read_text()=='new unbacked data'


def test_open_writable_log_is_preserved(tmp_path):
    run=tmp_path/'logs';run.mkdir();f=run/'log';f.write_text('original')
    manifest,_=resetter.backup([run],tmp_path/'backup')
    with f.open('a'):
        with pytest.raises(RuntimeError,match='Writable open files'):
            resetter.reset([run],tmp_path/'backup',manifest,True)
    assert f.read_text()=='original'
