"""User-authorized local cleanup, with byte-for-byte verification before removal."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('out')/'b47_storage_cleanup'


def digest(f):
    h=hashlib.sha256()
    while True:
        b=f.read(4*1024*1024)
        if not b:break
        h.update(b)
    return h.hexdigest()


def status(obj):
    OUT.mkdir(parents=True,exist_ok=True)
    obj=dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),free_bytes=shutil.disk_usage(ROOT).free,**obj)
    tmp=OUT/'status.json.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(OUT/'status.json')
    print(json.dumps(obj),flush=True)


def main():
    changes=[];errors=[];initial=shutil.disk_usage(ROOT).free
    # Only our newly downloaded, inactive cache. Do not touch live-recording
    # directories, model weights, original server recordings or results.
    files=sorted((ROOT/'data/sports_live/event_replay_inputs').rglob('books_ws.jsonl'))
    for p in files:
        before=p.stat();dst=Path(str(p)+'.gz');tmp=Path(str(dst)+'.cleanup.tmp')
        status(dict(state='compressing_and_verifying',file=str(p),completed=len(changes)))
        try:
            if dst.exists():
                with p.open('rb') as f:a=digest(f)
                with gzip.open(dst,'rb') as f:b=digest(f)
                assert a==b,'Existing gzip differs: retain both files'
            else:
                h=hashlib.sha256()
                with p.open('rb') as f,gzip.open(tmp,'wb',compresslevel=1) as z:
                    while True:
                        block=f.read(4*1024*1024)
                        if not block:break
                        h.update(block);z.write(block)
                a=h.hexdigest()
                with gzip.open(tmp,'rb') as f:b=digest(f)
                assert a==b,'Compressed round-trip checksum differs'
                now=p.stat();assert (now.st_size,now.st_mtime_ns)==(before.st_size,before.st_mtime_ns),'Source changed during compression'
                tmp.replace(dst)
            now=p.stat();assert (now.st_size,now.st_mtime_ns)==(before.st_size,before.st_mtime_ns),'Source changed before removal'
            p.unlink()
            changes.append(dict(action='lossless_gzip_verified_then_removed_raw',source=str(p),retained=str(dst),sha256=a,
                                old_bytes=before.st_size,new_bytes=dst.stat().st_size,reclaimed_bytes=before.st_size-dst.stat().st_size))
        except Exception as ex:errors.append(dict(file=str(p),error=repr(ex)))
        (OUT/'changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    # These are completed older-job transfer containers. Verify every archived
    # file against the extracted destination before removing the container.
    for src,dst in [(Path('/tmp/b42_all_books.tar.gz'),ROOT/'data/sports_live/book_l1'),
                    (Path('/tmp/b42_extra_inputs.tar.gz'),ROOT/'data/sports_live/replay_inputs')]:
        if not src.exists():continue
        status(dict(state='verifying_old_transfer',file=str(src),completed=len(changes)))
        try:
            checked=0
            with tarfile.open(src) as tar:
                for m in tar:
                    assert not m.name.startswith('/') and '..' not in Path(m.name).parts
                    if m.isdir():continue
                    assert m.isfile(),'Unexpected archive entry'
                    p=dst/m.name;assert p.is_file() and p.stat().st_size==m.size,('Missing/different extracted file',str(p))
                    with tar.extractfile(m) as f,p.open('rb') as g:assert digest(f)==digest(g),('Checksum mismatch',str(p))
                    checked+=1
            n=src.stat().st_size;src.unlink()
            changes.append(dict(action='removed_verified_transfer_container',source=str(src),verified_files=checked,reclaimed_bytes=n))
        except Exception as ex:errors.append(dict(file=str(src),error=repr(ex)))
    (OUT/'changes.json').write_text(json.dumps(changes,indent=2)+'\n')
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
    status(dict(state='completed' if not errors else 'completed_with_skips',changes=len(changes),errors=errors,
                reclaimed_bytes=sum(x['reclaimed_bytes'] for x in changes),free_increase_bytes=shutil.disk_usage(ROOT).free-initial))


if __name__=='__main__':main()
