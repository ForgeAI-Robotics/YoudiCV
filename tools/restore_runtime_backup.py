#!/usr/bin/env python3
"""List or restore an archived log/cache snapshot into a separate directory."""
from pathlib import Path
import argparse,hashlib,json,os,zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,help='Restore into a separate folder; never overwrite different data')
    p.add_argument('--prefix',help='Optional original absolute path, e.g. one historical cycle-test group')
    args=p.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        m=json.loads(z.read('manifest.json'))
        if args.output is None:
            print(json.dumps({'created':m['created'],'roots':m['roots'],
                'files':sum(e['kind']=='file' for e in m['entries'])},ensure_ascii=False,indent=2));return
        common=Path(os.path.commonpath(m['roots']));out=args.output.resolve()
        if out==common:raise RuntimeError('Choose a separate restore folder, not the original source tree')
        out.mkdir(parents=True,exist_ok=True);count=0
        for e in m['entries']:
            source=Path(e['source'])
            if args.prefix and source!=Path(args.prefix) and Path(args.prefix) not in source.parents:continue
            target=out/source.relative_to(common)
            if target!=out and out not in target.resolve().parents:raise RuntimeError('Unsafe restore path')
            if e['kind']=='dir':target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if e['kind']=='symlink':
                if target.exists() or target.is_symlink():raise RuntimeError('Refuse overwrite: '+str(target))
                target.symlink_to(e['link_target']);continue
            h=hashlib.sha256()
            with z.open(e['archive']) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=e['sha256']:raise RuntimeError('Archive checksum failure: '+e['source'])
            if target.exists():
                existing=hashlib.sha256()
                with target.open('rb') as f:
                    for b in iter(lambda:f.read(1024*1024),b''):existing.update(b)
                if existing.hexdigest()!=e['sha256']:raise RuntimeError('Refuse overwrite: '+str(target))
            else:
                with z.open(e['archive']) as f,target.open('xb') as w:
                    for b in iter(lambda:f.read(1024*1024),b''):w.write(b)
                os.chmod(target,e['mode'])
                os.utime(target,ns=(e['mtime_ns'],e['mtime_ns']))
            count+=1
        print(json.dumps({'restored_files':count,'output':str(out),'original_data_overwritten':False},ensure_ascii=False))


if __name__=='__main__':main()
