import importlib.util
from pathlib import Path
import zipfile,json

spec=importlib.util.spec_from_file_location('diagnostic_logs',Path(__file__).parents[1]/'tools/maintain_diagnostic_logs.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_archive_before_prune_and_keep_pinned(tmp_path):
    root=tmp_path/'captures';root.mkdir();archive=tmp_path/'zip'
    for n in range(6):
        d=root/str(n);d.mkdir();(d/'camera.jpg').write_bytes(bytes([n])*50)
    r=mod.archive_old_groups(root,archive,keep=2,protected=[root/'0'],apply=True)
    assert sorted(p.name for p in root.iterdir())==['0','4','5']
    assert r['archived_groups']==3
    with zipfile.ZipFile(r['archive']) as z:
        assert z.read('1/camera.jpg')==bytes([1])*50
        assert len(json.loads(z.read('_archive_manifest.json'))['files'])==3


def test_dry_run_and_retention_marker(tmp_path):
    root=tmp_path/'captures';root.mkdir()
    for n in range(4):
        d=root/str(n);d.mkdir();(d/'log').write_text(str(n))
    (root/'0/.retain').touch()
    r=mod.archive_old_groups(root,tmp_path/'zip',keep=1,apply=False)
    assert r['candidate_groups']==2 and len(list(root.iterdir()))==4
    assert not (tmp_path/'zip').exists()
