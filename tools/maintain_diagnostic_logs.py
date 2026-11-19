#!/usr/bin/env python3
"""Idle-only bounded diagnostic-cache archiving, no control/network imports."""
from pathlib import Path
from datetime import datetime
import argparse,fcntl,hashlib,json,shutil,zipfile

ROOT=Path(__file__).resolve().parents[1]


def archive_old_groups(root, archive_root, *, keep=256, protected=(), limit=512, apply=False):
    root=Path(root);archive_root=Path(archive_root)
    assert keep>=1 and limit>=1
    if not root.exists():return {'archived_groups':0,'remaining_unpinned':0}
    pinned={Path(p).resolve() for p in protected}
    groups=sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
        and p.resolve() not in pinned and not (p/'.retain').exists()),key=lambda p:p.name,reverse=True)
    chosen=groups[keep:keep+limit]
    result={'candidate_groups':len(chosen),'unpinned_groups':len(groups),
        'pinned_groups':sum(p.is_dir() and (p.resolve() in pinned or (p/'.retain').exists()) for p in root.iterdir()),
        'keep_unpinned':keep,'archived_groups':0}
    if not apply or not chosen:return result
    archive_root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');dest=archive_root/('诊断缓存_'+stamp+'.zip')
    entries=[]
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for group in chosen:
            for path in sorted(group.rglob('*')):
                if path.is_symlink():raise RuntimeError('Diagnostic symlink requires manual review: '+str(path))
                if not path.is_file():continue
                s=path.stat();raw=path.read_bytes();name=str(path.relative_to(root))
                sha=hashlib.sha256(raw).hexdigest()
                z.writestr(name,raw);entries.append({'name':name,'sha256':sha,'size':s.st_size,'mtime_ns':s.st_mtime_ns})
        z.writestr('_archive_manifest.json',json.dumps({'original_root':str(root),'files':entries},ensure_ascii=False))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for r in entries:
            assert hashlib.sha256(z.read(r['name'])).hexdigest()==r['sha256']
            s=(root/r['name']).stat();assert (s.st_size,s.st_mtime_ns)==(r['size'],r['mtime_ns'])
    for group in chosen:shutil.rmtree(group)
    result.update(archive=str(dest),archived_groups=len(chosen),remaining_unpinned=len(groups)-len(chosen),
        verified_files=len(entries),crc_and_sha256_verified=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--all-old',action='store_true',help='One supervised initial cleanup, still idle-only')
    args=parser.parse_args();cfg=json.loads((ROOT/'config/diagnostic_retention.json').read_text())
    protected=[]
    for name in cfg['protected_manifests']:
        path=ROOT/name
        if path.exists():
            for row in json.loads(path.read_text()):protected.append(Path(row['source_image']).parent)
    lock=open('/tmp/youxiaomei_robot_motion.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({'skipped':'active_robot_task'}));return
    try:
        result=archive_old_groups(ROOT/cfg['root'],ROOT/cfg['archive_root'],
            keep=cfg['keep_unpinned_groups'],protected=protected,
            limit=10000 if args.all_old else cfg['max_groups_per_pass'],apply=args.apply)
    finally:lock.close()
    result['time']=datetime.now().astimezone().isoformat()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
