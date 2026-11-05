"""Background completion of newly found early NBA recordings and reports.

Downloads only archived inputs, writes isolated analysis outputs, never deploys
or trades. Run with SSH/network permission once; no assistant polling needed.
"""
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import b37_standard_replay as B

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','31223','gw@lum.id']
STATUS=B.OUT/'archive_status.json'
CONFIG=B.OUT/'extra_archive_targets.json'


def status(**kw):
    STATUS.write_text(json.dumps(dict(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw),indent=2)+'\n')
    print(json.dumps(kw),flush=True)


def extract(src,dst,prefix=None):
    with tarfile.open(src) as t:
        for m in t.getmembers():
            assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()),m
            if prefix:assert m.name.startswith(prefix),m
        t.extractall(dst,filter='data')


def main():
    targets=json.loads(CONFIG.read_text())
    status(status='downloading earlier recorded NBA inputs',additional_games=len(targets))
    code='''from pathlib import Path
import sys,tarfile
base=Path('/home/casperngeen/sports_live')
targets=TARGETS
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
 for day,game in targets:
  root=base/day/game
  files=[]
  for name in ['game.json','feed_arrivals.jsonl','trades.jsonl','candles.jsonl']:
   p=root/name
   if not p.exists():p=Path(str(p)+'.gz')
   if p.exists():files.append(p)
  files+=sorted((root/'feed').glob('*.json.gz'))
  for p in files:tar.add(p,arcname=str(p.relative_to(base)),recursive=False)
'''.replace('targets=TARGETS','targets='+repr(targets))
    with Path('/tmp/b42_extra_inputs.tar.gz').open('wb') as f:
        subprocess.run(SSH+['python3','-'],input=code.encode(),stdout=f,check=True)
    extract('/tmp/b42_extra_inputs.tar.gz',B.ROOT/'data/sports_live/replay_inputs')
    entries,excluded=B.discover()
    (B.OUT/'inventory.json').write_text(json.dumps(dict(entries=entries,excluded=excluded),indent=2)+'\n')
    selected=[e for e in entries if e['mode']=='recorded' and [e['date'],e['league']+'_'+e['game']] in targets]
    rule=json.loads((B.OUT/'rule.json').read_text());h=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()
    # Extend compact quote reconstruction on the remote host in parallel.
    core=(Path(__file__).with_name('b38_extract_remote_l1.py')).read_text().rsplit('\nOUT.mkdir',1)[0]
    book_code=core+'\nimport os\nTARGETS='+repr(targets)+'''\nOUT.mkdir(parents=True,exist_ok=True)
pids=[]
for worker in range(2):
 pid=os.fork()
 if pid==0:
  for day,game in TARGETS[worker::2]:
   try:print(json.dumps(extract(day,game)),flush=True)
   except Exception as ex:print(json.dumps(dict(day=day,game=game,error=repr(ex))),flush=True)
  os._exit(0)
 pids.append(pid)
for pid in pids:os.waitpid(pid,0)
'''
    booklog=Path('/tmp/b42_extra_books.log').open('wb')
    bookproc=subprocess.Popen(SSH+['python3','-'],stdin=subprocess.PIPE,stdout=booklog,stderr=subprocess.STDOUT)
    bookproc.stdin.write(book_code.encode());bookproc.stdin.close()
    # The earlier watch process finishes its original cohort before this restart.
    studylog=Path('/tmp/b42_study.log').open('w')
    study=subprocess.Popen([sys.executable,str(Path(__file__).with_name('b38_strategy_study.py')),'--watch'],cwd=B.ROOT,stdout=studylog,stderr=subprocess.STDOUT)
    status(status='replaying earlier recorded NBA games',additional_games=len(selected),expected_scenarios=len(entries)+5)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(B.run_one,e,h) for e in selected]
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            r=f.result();status(status='replaying earlier recorded NBA games',additional_done=i,additional_games=len(selected),last_game=r['game'])
    study.wait()
    if study.returncode:raise RuntimeError('Strategy study failed; see /tmp/b42_study.log')
    # Fill in execution checks for older/local recordings; completed GCE cases reuse their checkpoints.
    with Path('/tmp/b42_execution.log').open('w') as f:
        subprocess.run([sys.executable,str(Path(__file__).with_name('b39_execution_checks.py')),'--workers','2'],cwd=B.ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    bookproc.wait();booklog.close()
    expected=len(entries)+5
    while sum(json.loads(p.read_text()).get('manifest_hash')==h for p in B.OUT.glob('*/*/*/result.json'))<expected:
        status(status='waiting for remaining original replay jobs',expected_scenarios=expected)
        time.sleep(30)
    for name in ['b37_summarise.py','b38_summarise.py']:
        with (B.OUT/(name+'_background.log')).open('w') as f:
            subprocess.run([sys.executable,str(Path(__file__).with_name(name))],cwd=B.ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    import b41_finalize_background as F
    F.comparison()
    # Wait for the first quote-extraction batch and then retrieve all compact artifacts.
    expected_books=sum(e['mode']=='recorded' and e['phase']!='preseason' for e in entries)
    remote="import pathlib,time,subprocess; p=pathlib.Path('/tmp/event-pred-b38-l1'); end=time.time()+3600\nwhile len(list(p.glob('*/*/stats.json')))<"+str(expected_books)+" and time.time()<end: time.sleep(15)\nsubprocess.run(['tar','-czf','-','-C',str(p),'.'],check=True)"
    with Path('/tmp/b42_all_books.tar.gz').open('wb') as f:
        subprocess.run(SSH+['python3','-'],input=remote.encode(),stdout=f,check=True)
    bookdst=B.ROOT/'data/sports_live/book_l1';bookdst.mkdir(exist_ok=True)
    extract('/tmp/b42_all_books.tar.gz',bookdst)
    status(status='reports complete; compact books downloaded',scenarios=expected,
           forward_report=str(B.OUT/'backtest_comparison.txt'),strategy_report=str(B.OUT.parent/'b38_strategy_study/report.txt'),
           book_artifacts=str(bookdst),note='No live deployment or trades changed. Full quote-based strategy validation is separate from print-based replay.')


if __name__=='__main__':main()
