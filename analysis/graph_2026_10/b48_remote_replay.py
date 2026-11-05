"""Storage-safe remote replay supervisor: stage server-to-server, return results.

Only code and compact results touch local disk. Raw recording streams are
relayed between SSH pipes and extracted in the isolated remote workspace.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).with_name('out')/'b46_event_replay'
REMOTE='/tmp/event-pred-b46-20261030'
SAN=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','31223','gw@lum.id']
VM=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','fyp-vm']


def status(state,**kw):
    obj=dict(state=state,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),remote_workspace=REMOTE,**kw)
    p=OUT/'status.json.tmp';p.write_text(json.dumps(obj,indent=2)+'\n');p.replace(OUT/'status.json');print(json.dumps(obj),flush=True)


def py(ssh,code,**kw):
    return subprocess.run(ssh+['python3 -c '+shlex.quote(code)],check=True,**kw)


def inventory(ssh,source):
    code='''import pathlib,json
base=pathlib.Path.home()/'fyp/sports_live' if SOURCE=='vm' else pathlib.Path('/home/casperngeen/sports_live')
rs=[]
for p in sorted(base.glob('2026*/*/game.json')):
 try:g=json.loads(p.read_text())
 except Exception:continue
 if g.get('league') not in ('MLB','NFL','NBA'):continue
 if g['league']=='NBA' and p.parent.parent.name<'2026-10-20':continue
 rs.append(dict(day=p.parent.parent.name,name=p.parent.name,root=str(p.parent),feed_versions=len(list((p.parent/'feed').glob('*.json.gz')))))
print(json.dumps(rs))
'''.replace('SOURCE',repr(source))
    return json.loads(py(ssh,code,capture_output=True,text=True,timeout=90).stdout)


def main():
    import fcntl
    lock=(OUT/'remote_supervisor.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    status('remote_inventory')
    sandbox=inventory(SAN,'sandbox');vm=inventory(VM,'vm')
    present={(r['day'],r['name']):r for r in sandbox}
    needed=[r for r in vm if (r['day'],r['name']) not in present or r['feed_versions']>present[r['day'],r['name']]['feed_versions']]
    (OUT/'remote_stage_inventory.json').write_text(json.dumps(dict(sandbox=sandbox,vm=vm,stage_from_vm=needed),indent=2)+'\n')
    for i,r in enumerate(needed,1):
        status('staging_vm_to_sandbox_without_local_files',done=i-1,total=len(needed),day=r['day'],game=r['name'])
        src='''import pathlib,sys,tarfile
root=pathlib.Path(ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
 for name in ['game.json','feed_arrivals.jsonl','trades.jsonl','candles.jsonl','feed','books_ws.jsonl.gz','books_ws.jsonl']:
  p=root/name
  if not p.exists():p=root/(name+'.gz')
  if p.exists():tar.add(p,arcname=p.name)
'''.replace('ROOT',repr(r['root']))
        target=REMOTE+'/data/sports_live/vm/'+r['day']+'/'+r['name']
        dst='''import pathlib,tarfile,sys,shutil
root=pathlib.Path(ROOT);root.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(root).free>20*1024**3,'Remote reserve below 20 GiB'
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:
 for m in tar:
  assert not m.name.startswith('/') and '..' not in pathlib.Path(m.name).parts and (m.isfile() or m.isdir())
  tar.extract(m,root,filter='data')
'''.replace('ROOT',repr(target))
        with (OUT/'remote_transfer.log').open('a') as log:
            producer=subprocess.Popen(VM+['python3 -c '+shlex.quote(src)],stdout=subprocess.PIPE,stderr=log)
            consumer=subprocess.Popen(SAN+['python3 -c '+shlex.quote(dst)],stdin=producer.stdout,stdout=log,stderr=log)
            producer.stdout.close();rc2=consumer.wait();rc1=producer.wait()
            assert rc1==0 and rc2==0,('remote_stream_failed',r,rc1,rc2)
    status('uploading_replay_code_only')
    extract="import pathlib,tarfile,sys; root=pathlib.Path("+repr(REMOTE)+"); t=tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz'); t.extractall(root,filter='data')"
    with (OUT/'remote_transfer.log').open('a') as log:
        child=subprocess.Popen(SAN+['python3 -c '+shlex.quote(extract)],stdin=subprocess.PIPE,stdout=log,stderr=log)
        with tarfile.open(fileobj=child.stdin,mode='w|gz',compresslevel=1) as tar:
            for rel in ['scripts/event_execution.py','scripts/test_event_execution.py','analysis/graph_2026_10/b46_event_replay.py']:
                tar.add(ROOT/rel,arcname=rel)
        child.stdin.close();assert child.wait()==0
    # A detached remote wrapper survives SSH closure. The local observer waits
    # on a small completion file; it never downloads raw data or polls in chat.
    launch='''import pathlib,subprocess,sys,json
root=pathlib.Path(ROOT);done=root/'supervisor_done.json'
if done.exists():done.unlink()
code="import subprocess,json,pathlib; root=pathlib.Path("+repr(str(root))+"); rc=subprocess.run(['python3','-u','analysis/graph_2026_10/b46_event_replay.py','--monitor','--local-only'],cwd=root).returncode; (root/'supervisor_done.json').write_text(json.dumps(dict(exit_code=rc)))"
log=(root/'remote_supervisor.log').open('a')
p=subprocess.Popen([sys.executable,'-u','-c',code],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,root=str(root))))
'''.replace('ROOT',repr(REMOTE))
    launched=json.loads(py(SAN,launch,capture_output=True,text=True,timeout=30).stdout)
    status('remote_replay_running',**launched)
    wait='''import pathlib,time,json,sys,tarfile
root=pathlib.Path(ROOT);done=root/'supervisor_done.json';end=time.monotonic()+48*3600
while not done.exists() and time.monotonic()<end:time.sleep(20)
assert done.exists(),'Remote job timeout; original remote job may still run'
base=root/'analysis/graph_2026_10/out/b46_event_replay'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p in base.rglob('*'):
  if p.is_file() and (p.name in ['status.json','report.txt','groups.json','errors.json','rule.json','inventory.json','result.json','lots.json']):
   tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
'''.replace('ROOT',repr(REMOTE))
    with (OUT/'remote_transfer.log').open('a') as log:
        child=subprocess.Popen(SAN+['python3 -c '+shlex.quote(wait)],stdout=subprocess.PIPE,stderr=log)
        with tarfile.open(fileobj=child.stdout,mode='r|gz') as tar:
            for m in tar:
                assert not m.name.startswith('/') and '..' not in Path(m.name).parts and m.isfile()
                assert m.size<20*1024**2,'Unexpectedly large result file'
                tar.extract(m,OUT,filter='data')
        assert child.wait()==0
    print('Remote replay finished; compact reports retrieved automatically.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as ex:
        status('remote_supervisor_failed',error=repr(ex));raise
