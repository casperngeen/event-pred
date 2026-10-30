"""v2: exchange-time event replay with per-host latency, Kalshi fee model and fill-rule fixes (no live orders).

Superseded b46 harness; see scripts/event_execution.py for what changed.

Run --monitor for background inventory/download, validation, replay and report.
Signals are generated afresh by live_trade.Engine, never copied from old orders.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import heapq
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

import b37_standard_replay as B
import event_execution as X
import live_exec as LE

LIVE=os.environ.get('LIVE_PATH')=='1'     # drive the engine through live_exec.LiveExecution (+ the print-fill book as shadow) instead of the batch Execution
OUT=Path(__file__).with_name('out')/os.environ.get('OUT_NAME','b49_event_replay')
INPUT=B.ROOT/'data/sports_live/event_replay_inputs'
# Decision -> exchange acceptance, seconds. Measured 2026-10-30: keep-alive HTTPS round trip to the API is 7 ms from the
# VM and 145 ms from the Lumid sandbox (both clocks NTP-synced); 'host' = half round trip + ~10 ms gateway/processing +
# ~5 ms to build the order; 'slow' and 'stress' are sensitivities.
LATENCY={'vm':{'host':.015,'slow':.1,'stress':.5},'sandbox':{'host':.09,'slow':.25,'stress':.5}}
SSH={'sandbox':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','31223','gw@lum.id'],
     'fyp-vm':['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','fyp-vm']}
REMOTE={'sandbox':'/home/casperngeen/sports_live','fyp-vm':'/home/casper/fyp/sports_live'}


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix(path.suffix+'.tmp')
    p.write_text(json.dumps(obj,indent=2,default=float)+'\n');p.replace(path)


def status(state,**kw):
    obj=dict(state=state,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    write(OUT/'status.json',obj);print(json.dumps(obj),flush=True)


# arm -> (latency key, Execution options). vol = fills limited to print volume; strict = invalid until snapshot after any book inconsistency.
ARM_SPECS={'host':('host',{}),'slow':('slow',{}),'stress':('stress',{}),'vol':('host',dict(through='volume')),'strict':('host',dict(repair='strict'))}
ARMS=tuple(os.environ.get('ARMS','host,slow,stress').split(','))
SIZING=os.environ.get('SIZING','5x')       # 5x = round(5*weight,.01) (user rule); 1x = round(weight,.01) (b37 'latest' arm)


def raw_rows(path,diagnostics):
    op=gzip.open if path.suffix=='.gz' else open
    try:
        with op(path,'rt') as f:
            for i,line in enumerate(f):
                try:r=json.loads(line)
                except ValueError:
                    diagnostics['bad_json_lines']+=1;continue
                if r.get('received_at'):yield B.RP.ts(r['received_at']),i,r
    except (EOFError,OSError) as ex:
        diagnostics['truncated_stream']=repr(ex)


def book_index():
    found=collections.defaultdict(list)
    base=B.ROOT/'data/sports_live'
    # Path.rglob does not descend directory symlinks. Remote workspaces link
    # the original archives here so replay needs no duplicate raw recordings.
    bases=[base]+[p for p in base.iterdir() if p.is_symlink() and p.is_dir()]
    for archive in bases:
        for p in archive.rglob('books_ws.jsonl*'):
            if p.is_file() and not p.name.endswith('.tmp'):
                found[p.parent.name].append(p)
    return found


def inventory(fetch=True):
    books=book_index(); roots={};excluded=[];remote_errors=[]
    for source in ['vm','replay_inputs','sandbox','local','repaired','event_replay_inputs']:
        for p in sorted((B.ROOT/'data/sports_live'/source).rglob('game.json')):
            if 'feed' in p.parts:continue
            try:g=json.loads(p.read_text())
            except ValueError:continue
            day=next((x for x in reversed(p.parts[:-1]) if x.startswith('2026-')),None)
            if day and g.get('league') in B.MODELS:
                key=day,p.parent.name;old=roots.get(key)
                if old is None or len(list((p.parent/'feed').glob('*.json.gz')))>len(list((old/'feed').glob('*.json.gz'))):roots[key]=p.parent
    if fetch:
        for host in SSH:
            status('inventory_remote',host=host)
            code='''import pathlib,json
base=pathlib.Path.home()/'fyp/sports_live' if HOST=='fyp-vm' else pathlib.Path('/home/casperngeen/sports_live')
for p in sorted(base.glob('2026*/*/game.json')):
 try:g=json.loads(p.read_text())
 except Exception:continue
 root=p.parent;book=next((str(x) for x in [root/'books_ws.jsonl.gz',root/'books_ws.jsonl'] if x.exists()),None)
 print(json.dumps(dict(day=root.parent.name,name=root.name,root=str(root),book=book,g=g,feed_versions=len(list((root/'feed').glob('*.json.gz'))))))
'''.replace('HOST',repr(host))
            try:
                proc=subprocess.run(SSH[host]+['python3','-'],input=code,text=True,capture_output=True,check=True,timeout=90)
                remote=[json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
            except Exception as ex:
                remote_errors.append(dict(host=host,error=repr(ex)));continue
            write(OUT/(host+'_inventory.json'),remote)
            for r in remote:
                day,name=r['day'],r['name'];key=day,name;league=r['g'].get('league')
                if league not in B.MODELS or (league=='NBA' and day<'2026-10-20'):continue
                root=roots.get(key);have_inputs=root is not None and all(B.present(root,n) for n in ['feed_arrivals.jsonl','trades.jsonl','candles.jsonl'])
                if have_inputs and r.get('feed_versions',0)>len(list((root/'feed').glob('*.json.gz'))):have_inputs=False
                have_book=any(day in str(p) or day[5:] in str(p) for p in books.get(name,[]))
                if have_inputs and (have_book or not r['book']):continue
                dest=INPUT/host/day/name;dest.mkdir(parents=True,exist_ok=True)
                select=['game.json','feed_arrivals.jsonl','trades.jsonl','candles.jsonl','feed'] if not have_inputs else []
                if r['book'] and not have_book:select.append(Path(r['book']).name)
                if not select:continue
                free=shutil.disk_usage(B.ROOT).free
                if free<10*1024**3:
                    status('insufficient_local_space',free_bytes=free,minimum_free_bytes=10*1024**3,
                           note='No further downloads. Run against server-side originals or free verified cache space.')
                    raise RuntimeError('Local free space below 10 GiB reserve; download stopped before writing.')
                status('downloading_recording',host=host,day=day,game=name,files=select)
                remote_code='''import pathlib,tarfile,sys
root=pathlib.Path(ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as tar:
 for name in NAMES:
  p=root/name
  if not p.exists():p=root/(name+'.gz')
  if p.exists():tar.add(p,arcname=p.name)
'''.replace('ROOT',repr(r['root'])).replace('NAMES',repr(select))
                archive=OUT/'transfer.tar.gz'
                try:
                    with archive.open('wb') as f:subprocess.run(SSH[host]+['python3','-'],input=remote_code.encode(),stdout=f,check=True,timeout=3600)
                    with tarfile.open(archive) as tar:
                        for m in tar.getmembers():assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
                        required=sum(m.size for m in tar.getmembers() if m.isfile())+2*1024**3
                        if shutil.disk_usage(B.ROOT).free<required:
                            raise RuntimeError('Insufficient extraction space including 2 GiB reserve; archive not extracted.')
                        tar.extractall(dest,filter='data')
                    archive.unlink()
                    if not have_inputs:roots[key]=dest
                    for p in dest.glob('books_ws.jsonl*'):books[name].append(p)
                except Exception as ex:remote_errors.append(dict(host=host,game=name,error=repr(ex)))
    entries=[]
    for (day,name),root in sorted(roots.items()):
        g=json.loads((root/'game.json').read_text());league=g['league']
        why=None
        if league=='NBA' and day<'2026-10-20':why='preseason'
        if league=='MLB' and day=='2026-10-17':why='wrong-game contracts after rainout'
        missing=[n for n in ['feed_arrivals.jsonl','trades.jsonl','candles.jsonl'] if not B.present(root,n)]
        if missing:why='missing received-time inputs: '+','.join(missing)
        candidates=[p for p in books.get(name,[]) if day in str(p) or day[5:] in str(p)]
        if not candidates:why=why or 'no raw WebSocket book'
        e=dict(date=day,league=league,game=str(g['game_id']),root=str(root.relative_to(B.ROOT)),
               book=str(max(candidates,key=lambda p:p.stat().st_size).relative_to(B.ROOT)) if candidates else None)
        if why:excluded.append(dict(e,reason=why))
        else:entries.append(e)
    write(OUT/'inventory.json',dict(entries=entries,excluded=excluded,remote_errors=remote_errors))
    return entries,remote_errors


def run(e,rule_hash):
    dst=OUT/'games'/e['date']/(e['league']+'_'+e['game']);done=dst/'result.json'
    if done.exists() and json.loads(done.read_text()).get('rule_hash')==rule_hash:return json.loads(done.read_text())
    dst.mkdir(parents=True,exist_ok=True)
    g,arrivals,doc,final,hs,aws,trades,candles,missing=B.load(e)
    result=dict(e,rule_hash=rule_hash,final_score_available=bool(final),missing_feed_files=missing)
    if not final:
        result['excluded']='unfinished final feed';write(done,result);return result
    first_doc=json.loads(gzip.decompress(arrivals[0][1].read_bytes()))
    if e['league']=='MLB':
        n=sum(bool(x.get('isPitch')) for pa in first_doc['feed']['liveData']['plays']['allPlays'] for x in pa['playEvents'])
    else:
        n=len(first_doc.get('plays') or [])
        if not n:
            drives=first_doc.get('drives') or {};n=sum(len(x.get('plays') or []) for x in drives.get('previous',[]))+len((drives.get('current') or {}).get('plays') or [])
    result['initial_feed_plays']=n;result['coverage']='starts_midgame' if n>5 else 'recorded_from_start'
    if e['league'] not in B.FORECASTERS:B.FORECASTERS[e['league']]=B.SharedForecaster(e['league'])
    # Explicit user rule: round(5 * normalized existing weight, .01), half up.
    sizer=(lambda weight,k:X.size_contracts(weight)) if SIZING=='5x' else (lambda weight,k:math.floor(weight*100+.5)/100)
    # b37_replay_lookup replaces Engine._decide with an exec'd copy that holds a snapshot of the module globals, so patching
    # live_trade.contracts_of alone never reaches it (every earlier replay silently ran 1x sizing): patch both.
    B.T.contracts_of=sizer;B.T.Engine._decide.__globals__['contracts_of']=sizer
    assert B.T.Engine._decide.__globals__['contracts_of'] is sizer
    arms=[]
    for tag in ARMS:
        out=dst/tag;out.mkdir(exist_ok=True)
        eng=B.T.Engine(g,B.FORECASTERS[e['league']],out,B.T.KINDS_DEFAULT,10. if e['league']=='MLB' else 60.,False)
        eng.log=B.CompactLog(out);eng.pub=B.NoPublish();eng.shadow=False;eng.k=None
        eng.gate_true=e['league']=='NBA';eng.late_block=e['league']=='NBA'
        if LIVE:
            ex=LE.LiveExecution(eng.log,None,eng.mk.tickers,float(os.environ.get('SIM_LATENCY','.09')),shadow=B.T.PaperBook(eng.log),**ARM_SPECS[tag][1])
        else:ex=X.Execution(eng.log,0.,**ARM_SPECS[tag][1])
        eng.book=ex
        def ref(tk,t,eng=eng):
            eng.mk.build();p=eng.mk.p[tk];i=int(B.LM.np.searchsorted(p['t'],t,side='right'))
            return float(p['px'][i-1]) if i else None
        ex.ref=ref;arms.append((tag,eng,ex))
    diagnostics=collections.Counter()
    if LIVE:
        # raw messages in receive order, each known FLUSH s after it arrived (the recorder's flush), fed lazily
        FLUSH=float(os.environ.get('FLUSH_S','.25'));wanted=set(arms[0][1].mk.tickers);opn=gzip.open if str(e['book']).endswith('.gz') else open
        fh=opn(B.ROOT/e['book'],'rt')
        def nxt():
            for line in fh:
                if '"market_ticker"' not in line:continue
                m=json.loads(line)
                if m.get('type') in ('orderbook_snapshot','orderbook_delta','trade') and (m.get('msg') or {}).get('market_ticker') in wanted:
                    return dt.datetime.fromisoformat(m['received_at'].replace('Z','+00:00')).timestamp()+FLUSH,m
            return None
        pend=[nxt()]
        def pull(at):
            while pend[0] is not None and pend[0][0]<=at:
                for _,_,ex in arms:ex.feed(X.book_ts(pend[0][1]),pend[0][1])
                pend[0]=nxt()
        host='live_path';result['host']=host;result['sizing']=SIZING;result['book_lag_median_s']=None;result['latency_s']=float(os.environ.get('SIM_LATENCY','.09'))
        stream=[]
    else:
        stream=X.load_book_stream(B.ROOT/e['book'],set(arms[0][1].mk.tickers),diagnostics)
        lag=stream.lag_median;host='vm' if lag<.05 else 'sandbox'
        for tag,_,ex in arms:ex.latency=LATENCY[host][ARM_SPECS[tag][0]];ex.attach(stream)
        result['host']=host;result['book_lag_median_s']=lag;result['latency_s']=LATENCY[host];result['sizing']=SIZING
    # Signals run at exact five-second model boundaries; missed boundaries are
    # not backfilled. Feed rebuilds are independent of the forecast timer.
    q=[];serial=0
    for t,p in arrivals:serial+=1;q.append((t,serial,'feed',p))
    for r in trades:serial+=1;q.append((B.RP.ts(r['received_at']),serial,'trade',r))
    for r in candles:serial+=1;q.append((B.RP.ts(r['received_at']),serial,'candle',r))
    heapq.heapify(q)
    end=arrivals[-1][0]+600;next_tick=None;last_at=-math.inf
    while q or next_tick is not None:
        at=min(q[0][0] if q else math.inf,next_tick if next_tick is not None else math.inf)
        if at>end:break
        if at<last_at-1e-6:
            diagnostics['arrival_clock_regressions']+=1
            raise ValueError('raw receive-time order reversed; needs stable external sort')
        last_at=at
        if LIVE:
            pull(at)
            for _,eng_,ex in arms:ex.update(eng_.mk,at)
        else:
            for _,_,ex in arms:ex.advance(at)
        while q and q[0][0]<=at:
            _,_,kind,r=heapq.heappop(q)
            for _,eng,ex in arms:
                if kind=='feed':
                    eng.on_feed(json.loads(gzip.decompress(r.read_bytes())),dt.datetime.fromtimestamp(at,dt.timezone.utc));eng._rebuild_plays()
                elif kind=='trade':eng.on_trades([r])
                else:eng.on_candles([r])
            if kind=='feed' and next_tick is None and arms[0][1].first_play is not None:
                first=arms[0][1].first_play;next_tick=first+math.ceil(max(0.,at-first)/B.LM.STEP)*B.LM.STEP
        if next_tick is not None and next_tick<=at+1e-7:
            for _,eng,ex in arms:eng.step(at)
            next_tick+=B.LM.STEP
    if LIVE:
        pull(end+5)
        for _,eng_,ex in arms:ex.update(eng_.mk,end+5)
    else:
        for _,_,ex in arms:ex.advance(end)
    meta={m['ticker']:m for m in g['contracts']}
    def payoff(tk):return B.A.payoff(meta[tk],g['league'],g['home_k'],g['away_k'],hs,aws)
    result['results']={}
    for tag,eng,ex in arms:
        result['results'][tag]=ex.results(payoff);write(dst/tag/'lots.json',ex.export_lots());eng.log.close()
    result['source_diagnostics']=dict(diagnostics);result['raw_events']=len(stream) if not LIVE else None
    if diagnostics.get('truncated_stream'):result['coverage']+=';truncated_book'
    if any(ex.stats.get('negative_level') or ex.stats.get('delta_without_snapshot') for _,_,ex in arms):
        result['coverage']+=';book_state_errors'
    # Number of filled entry orders, distinct from fill slices/partial lots.
    for tag,eng,ex in arms:
        for kind,a in result['results'][tag]['by_kind'].items():
            a['filled_orders']=len({lot['parent'] for lot in ex.lots if lot['kind']==kind})
    write(done,result);return result


def summarize(errors,remote_errors):
    groups=collections.defaultdict(lambda:dict(games=0,orders=0,fills=0,filled_orders=0,entry_contracts=0.,gross_c=0.,pnl_c=0.,fees_c=0.,pnl_1c_c=0.,fees_1c_c=0.,settled_contracts=0.))
    games=[json.loads(p.read_text()) for p in sorted(OUT.glob('games/*/*/result.json'))]
    for r in games:
        for arm,s in r.get('results',{}).items():
            for kind,a in s['by_kind'].items():
                key='|'.join((r['coverage'],arm,r['league'],kind));v=groups[key];v['games']+=1;v['orders']+=s['orders']
                for k in ['fills','filled_orders','entry_contracts','gross_c','pnl_c','fees_c','pnl_1c_c','fees_1c_c','settled_contracts']:v[k]+=a[k]
    for v in groups.values():
        n=v['entry_contracts']
        for k in ('gross','pnl','pnl_1c'):v[k+'_per_contract_c']=v[k+'_c']/n if n else None
        v['net_per_contract_c']=v['pnl_per_contract_c']
    write(OUT/'groups.json',dict(groups));write(OUT/'errors.json',errors)
    lines=['EVENT-DRIVEN RECORDED LIVE REPLAY (v2, exchange clock)','',f'Processed {len(games)} games; job errors {len(errors)}; remote inventory/download errors {len(remote_errors)}.',
           'Sizing: round_half_up(5 * existing normalized signal/width weight, 0.01 contracts); fractional sizes down to 0.01 are valid on Kalshi.',
           'Fresh inference and decisions on received-time feeds/trades/candles; no copied historical orders.',
           'Book replayed in EXCHANGE time (delta/trade ts_ms, snapshot sending_ts_ms): an order accepted at T sees the book at T.',
           'Latency = decision -> exchange acceptance, per recording host (arm host: VM 15 ms, sandbox 90 ms from measured round trips; slow / stress are sensitivities).',
           'Limits snapped to the whole-cent grid (buys down, sells up). Price/time priority; cancellations never advance our queue (conservative).',
           'A print strictly through our limit fills us in full (the level was exhausted); prints at our limit are consumed once, queue ahead first.',
           'Liquidity we take stays removed from the real level, clamped to the level when it shrinks. No market impact or participant response.',
           'Fees: per-series type/multiplier from GET /series (MLB x0.5, MLB spread/total taker-only), taker 0.07 / maker 0.0175 of p(1-p), per-order cent rounding + rebate accumulator, $0.01 balance precision.',
           'ASSUMPTION (decided 2026-10-30): the account is a Kalshi direct member ($0.0001 balance precision), so headline net uses that precision. Not confirmed with Kalshi; net$0.01 (FCM-cleared) is reported alongside as a sensitivity and is far worse at these sizes.',
           'Taker residuals retried once/second for 120s, then held to settlement. Cancel and market exit have no race (cancel lands first).',
           'Not modelled: exchange rejections (halts, rate limits, position caps), message loss (seq gaps unchecked), WS reconnect gaps beyond snapshot replacement.',
           'Main groups below require recorded-from-start feeds and no detected truncated book; partial coverage is separate in groups.json.',
           'A completed job is not proof of exact historical decision parity. Original-run forecast differences remain separately unresolved.','',
           'Gross = before fees. Net = after fees for a direct-member account ($0.0001 balance precision, headline); net$0.01 = FCM-cleared ($0.01).','',
           'coverage | arm | league | kind | games | entry contracts | gross c | gross c/contract | net c | net c/contract | net$0.01 c/contract | fees c']
    for k,v in sorted(groups.items()):
        c,arm,league,kind=k.split('|');f=lambda x:f'{x:.4f}' if v['entry_contracts'] else 'n/a'
        lines.append(f'{c} | {arm} | {league} | {kind} | {v["games"]} | {v["entry_contracts"]:.2f} | {v["gross_c"]:.2f} | {f(v["gross_per_contract_c"])} | {v["pnl_c"]:.2f} | {f(v["pnl_per_contract_c"])} | {f(v["pnl_1c_per_contract_c"])} | {v["fees_c"]:.2f}')
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    return dict(games=len(games),errors=len(errors),remote_errors=len(remote_errors),report=str(OUT/'report.txt'))


def batch(fetch):
    entries,remote_errors=inventory(fetch)
    if os.environ.get('GAMES'):entries=[e for e in entries if e['game'] in os.environ['GAMES'].split(',')]
    rule=dict(arms=ARMS,sizing_mode=SIZING,sizing='round_half_up(5 * live_trade.size_of(signal,width),0.01)',models=B.MODELS,latency_seconds=LATENCY,
              queue='exchange-time book; no advancement from cancellations; through prints fill in full; prints at limit consumed once; whole-cent snap; Kalshi fee ledger',
              sources={str(p.relative_to(B.ROOT)):B.digest(p) for p in [Path(__file__),B.ROOT/'scripts/event_execution.py',B.ROOT/'scripts/live_trade.py',B.ROOT/'scripts/live_model.py']})
    rule['model_weights']={str(p.relative_to(B.ROOT)):B.digest(p) for p in (B.ROOT/'results/ingame/deploy').glob('*.pt') if any(name in p.name for name in B.MODELS.values())}
    rule_hash=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest();write(OUT/'rule.json',rule)
    errors=[]
    # Fail before the full batch if end-to-end integration fails in any league.
    for league in B.MODELS:
        sample=next((e for e in entries if e['league']==league),None)
        if sample:
            status('end_to_end_preflight',league=league,game=sample['game'])
            try:run(sample,rule_hash)
            except Exception as ex:
                errors.append(dict(entry=sample,error=repr(ex),traceback=traceback.format_exc()))
                summary=summarize(errors,remote_errors);status('preflight_failed',**summary);return 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.environ.get('WORKERS',2))) as pool:
        fs={pool.submit(run,e,rule_hash):e for e in entries}
        for i,f in enumerate(concurrent.futures.as_completed(fs),1):
            try:r=f.result()
            except Exception as ex:errors.append(dict(entry=fs[f],error=repr(ex),traceback=traceback.format_exc()))
            status('replaying',done=i,total=len(entries),errors=len(errors),last_game=fs[f]['game'])
            write(OUT/'errors.json',errors)
    summary=summarize(errors,remote_errors)
    status('completed' if not errors and not remote_errors else 'completed_with_gaps',**summary)
    return bool(errors)


def main():
    p=argparse.ArgumentParser();p.add_argument('--monitor',action='store_true');p.add_argument('--local-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not a.monitor:return batch(not a.local_only)
    import fcntl
    lock=(OUT/'monitor.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Monitor already running',flush=True);return 0
    status('validating',monitor_pid=os.getpid())
    with (OUT/'worker.log').open('a') as log:
        check=subprocess.run([sys.executable,str(B.ROOT/'scripts/test_event_execution.py')],cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT)
        if check.returncode:status('validation_failed',log=str(OUT/'worker.log'));return check.returncode
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())]+(['--local-only'] if a.local_only else []),cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'monitor.json',dict(monitor_pid=os.getpid(),worker_pid=child.pid,completion='blocking process wait; automatic report; no chat polling'))
        rc=child.wait()
    if rc:status('failed_or_incomplete',exit_code=rc,log=str(OUT/'worker.log'))
    return rc


if __name__=='__main__':sys.exit(main())
