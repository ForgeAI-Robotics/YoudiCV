#!/usr/bin/env python3
"""Explicit offline log/cache reset, after a verified external ZIP backup.

Does not import robot clients, change configuration, clear stop locks or move
hardware. Never selects model weights, calibration, task code or system caches.
"""
from pathlib import Path
from datetime import datetime
import argparse,ctypes,fcntl,hashlib,json,os,shutil,stat,zipfile

ROOT=Path(__file__).resolve().parents[1]


def targets():
    paths=[ROOT/'artifacts'/x for x in ('runs','captures','trials','recordings')]
    paths += [Path.home()/'.cache/pip',Path.home()/'.ros/log',ROOT/'__pycache__',
              ROOT/'.pytest_cache',ROOT/'Log']
    for name in ('piper_handeye','tools','tests','button_light_evaluation',
                 'bag_handle_v2','bag_handle_v3','bag_handle_yolo26_v3_20261013'):
        for here,dirs,_ in os.walk(ROOT/name,followlinks=False):
            dirs[:]=[d for d in dirs if d not in ('apk_analysis_dependencies','.git')]
            for d in list(dirs):
                if d in ('__pycache__','.pytest_cache'):
                    paths.append(Path(here)/d);dirs.remove(d)
    paths += list(ROOT.glob('*.log'))+list(ROOT.glob('*.jsonl'))
    paths += list(ROOT.glob('.bag_handle_v3_upload.*'))
    return sorted(set(p for p in paths if p.exists() and not p.is_symlink()))


def inventory(paths):
    entries=[]
    for i,top in enumerate(paths):
        members=[top]
        if top.is_dir():
            for here,dirs,files in os.walk(top,followlinks=False):
                members.extend(Path(here)/n for n in dirs+files)
        for p in sorted(members):
            s=p.lstat();kind=('symlink' if p.is_symlink() else 'dir' if p.is_dir() else 'file')
            if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode)):
                raise RuntimeError('Unsupported filesystem object: '+str(p))
            rel=p.relative_to(top) if p!=top else Path('.')
            entry={'top':i,'relative':str(rel),'kind':kind,'size':s.st_size,
                   'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),
                   'inode':s.st_ino,'source':str(p),'archive':f'data/{i:03d}/'+str(rel)}
            if kind=='symlink':entry['link_target']=os.readlink(p)
            entries.append(entry)
    return entries


def check_inventory(entries):
    for e in entries:
        s=Path(e['source']).lstat()
        if e['kind']=='dir':continue
        if (s.st_size,s.st_mtime_ns,s.st_ino)!=(e['size'],e['mtime_ns'],e['inode']):
            raise RuntimeError('Changed during backup; do not reset: '+e['source'])


def writable_open_files(paths):
    found=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    raw=os.readlink(fd)
                    if not raw.startswith('/'):continue
                    p=Path(raw.removesuffix(' (deleted)'))
                    if not any(p==t or t in p.parents for t in paths):continue
                    info=(proc/'fdinfo'/fd.name).read_text()
                    flags=next(l.split()[1] for l in info.splitlines() if l.startswith('flags:'))
                    if int(flags,8)&3:found.append({'pid':int(proc.name),'file':str(p)})
                except (OSError,StopIteration):pass
        except OSError:pass
    return found


def backup(paths,dest):
    dest.mkdir(parents=True,exist_ok=False);os.chmod(dest,0o700)
    entries=inventory(paths);arc=dest/'日志与缓存_完整备份.zip';done=0
    print(json.dumps({'phase':'backup','entries':len(entries),
        'source_bytes':sum(e['size'] for e in entries if e['kind']=='file')},ensure_ascii=False),flush=True)
    with zipfile.ZipFile(arc,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for i,e in enumerate(entries):
            if e['kind']=='file':
                h=hashlib.sha256()
                with Path(e['source']).open('rb') as f,z.open(e['archive'],'w',force_zip64=True) as w:
                    for b in iter(lambda:f.read(1024*1024),b''):h.update(b);w.write(b)
                e['sha256']=h.hexdigest();done+=e['size']
            if i%2000==0:print(json.dumps({'phase':'backup','entry':i,'processed_bytes':done}),flush=True)
        manifest={'created':datetime.now().astimezone().isoformat(),'roots':[str(p) for p in paths],'entries':entries}
        z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
    with zipfile.ZipFile(arc) as z:
        for i,e in enumerate(entries):
            if e['kind']!='file':continue
            h=hashlib.sha256()
            with z.open(e['archive']) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=e['sha256']:raise RuntimeError('Backup SHA256 mismatch: '+e['source'])
            if i%4000==0:print(json.dumps({'phase':'verify','entry':i}),flush=True)
    check_inventory(entries)
    current=inventory(paths)
    if {e['source'] for e in entries}!={e['source'] for e in current}:
        raise RuntimeError('New/removed files during backup; no reset performed')
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'phase':'backup_verified','zip_bytes':arc.stat().st_size}),flush=True)
    return manifest,arc


def exchange(a,b):
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.renameat2(-100,os.fsencode(a),-100,os.fsencode(b),2):
        number=ctypes.get_errno();raise OSError(number,os.strerror(number))


def reset(paths,dest,manifest,remove_originals=False):
    check_inventory(manifest['entries'])
    expected={(e['top'],e['relative'],e['kind']) for e in manifest['entries']}
    if expected!={(e['top'],e['relative'],e['kind']) for e in inventory(paths)}:
        raise RuntimeError('New/removed files after backup; no reset performed')
    active=writable_open_files(paths)
    if active:raise RuntimeError('Writable open files; backup kept, no reset: '+json.dumps(active))
    staging=dest/'verified_originals';staging.mkdir();operations=[]
    for i,p in enumerate(paths):
        parked=staging/f'{i:03d}'
        original=p.stat()
        if p.is_dir():
            parked.mkdir();os.chmod(parked,stat.S_IMODE(original.st_mode))
            exchange(p,parked)
            if p.parent==ROOT/'artifacts':
                # Preserve empty run-type folders so legacy root aliases do
                # not become dangling links after the history is reset.
                for child in parked.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        (p/child.name).mkdir(mode=stat.S_IMODE(child.stat().st_mode))
        else:
            # Root logs remain available as empty files for existing units.
            parked.touch();os.chmod(parked,stat.S_IMODE(original.st_mode))
            exchange(p,parked)
        assert parked.stat().st_ino==original.st_ino
        operations.append({'original':str(p),'parked':str(parked),'archive_top':i})
        (dest/'reset_operations.json').write_text(json.dumps(operations,ensure_ascii=False,indent=2))
    # Source data remain as raw originals until all swaps complete and the
    # already verified ZIP is available. Only the precise parked entries may
    # be removed; any changed file aborts removal and stays backed up raw.
    if remove_originals:
        parked_paths=[Path(op['parked']) for op in operations]
        if expected!={(e['top'],e['relative'],e['kind']) for e in inventory(parked_paths)}:
            raise RuntimeError('Parked originals changed; verified ZIP and raw originals preserved')
        for e in manifest['entries']:
            if e['kind']=='dir':continue
            p=staging/f"{e['top']:03d}"/e['relative']
            if e['relative']=='.':p=staging/f"{e['top']:03d}"
            s=p.lstat()
            assert (s.st_size,s.st_mtime_ns,s.st_ino)==(e['size'],e['mtime_ns'],e['inode']),str(p)
        for op in operations:
            p=Path(op['parked'])
            if p.is_dir():shutil.rmtree(p)
            else:p.unlink()
        staging.rmdir()
    return operations


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--backup-dir',type=Path)
    parser.add_argument('--remove-verified-originals',action='store_true')
    args=parser.parse_args();paths=targets()
    if not args.apply:
        print(json.dumps({'roots':[str(p) for p in paths]},ensure_ascii=False,indent=2));return
    if args.backup_dir is None or ROOT==args.backup_dir.resolve() or ROOT in args.backup_dir.resolve().parents:
        raise RuntimeError('Backup must be outside the project')
    with open('/tmp/youxiaomei_robot_motion.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=writable_open_files(paths)
        if active:raise RuntimeError('Writable active logs: '+json.dumps(active))
        manifest,arc=backup(paths,args.backup_dir)
        operations=reset(paths,args.backup_dir,manifest,args.remove_verified_originals)
        result={'completed':datetime.now().astimezone().isoformat(),'archive':str(arc),
                'roots_reset':len(operations),'files_verified':sum(e['kind']=='file' for e in manifest['entries']),
                'source_bytes':sum(e['size'] for e in manifest['entries'] if e['kind']=='file'),
                'archive_bytes':arc.stat().st_size,'crc_and_each_file_sha256_verified':True,
                'verified_originals_removed':args.remove_verified_originals,'motion_performed':False}
        (args.backup_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
