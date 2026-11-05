"""Detached replay supervisor. Child completion, not chat polling, triggers reports.

Uses the already computed, frozen b37 forecasts and recorded feed arrivals.
Book execution is a one-second, top-of-book sensitivity, not a queue simulation.
Original recordings and running traders are never modified.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import types

import b37_standard_replay as B
import b38_strategy_study as S
import b37_summarise as SUM

OUT = Path(__file__).with_name('out')/'b45_book_forward'
BOOKS = B.ROOT/'data/sports_live/book_l1'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=float)+'\n')
    tmp.replace(path)


def status(state, **kw):
    write(OUT/'status.json', dict(state=state, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw))
    print(json.dumps(dict(state=state, **kw)), flush=True)


class Quotes:
    def __init__(self, path):
        self.head = {}
        self.f = gzip.open(path, 'rt')
        self.reader = csv.DictReader(self.f)
        self.next = next(self.reader, None)
        self.rows = 0

    def advance(self, now):
        # Extraction emits each ticker's completed buckets as it changes, so
        # CSV rows are not globally sorted. Load/sort once in the worker below.
        while self.i < len(self.data) and self.data[self.i][0] <= now:
            at, r = self.data[self.i]
            self.head[r['ticker']] = r
            self.i += 1

    def prepare(self):
        self.data = []
        if self.next is not None:
            self.data.append((float(self.next['ts']), self.next))
        self.data.extend((float(r['ts']), r) for r in self.reader)
        self.data.sort(key=lambda r:r[0])
        self.rows = len(self.data)
        self.i = 0
        self.f.close()


class QuoteBook(B.T.PaperBook):
    """Full-size fills at observed quotes, with causal exits and visible capacity.

On submission a marketable order takes the available best quote. A resting
limit fills at its limit only if the opposite best subsequently moves strictly
through it. No at-limit queue fills are assumed. Capacity at a given published
quote is shared across this arm's orders; stale/invalid quotes cannot fill.
"""
    def __init__(self, quotes, max_age):
        super().__init__(lambda *args:None)
        self.quotes, self.max_age = quotes, max_age
        self.used = collections.defaultdict(float)
        self.rejected = collections.Counter()

    def take(self, o, side, limit, now, strict=False):
        q = self.quotes.head.get(o['ticker'])
        if q is None or q['valid'] != '1':
            self.rejected['missing_or_invalid'] += 1
            return None
        if not 0 <= now-float(q['last_msg']) <= self.max_age:
            self.rejected['stale'] += 1
            return None
        field, qty = ('ask','ask_qty') if side==1 else ('bid','bid_qty')
        if not q[field]:
            return None
        px = float(q[field])
        if limit is not None:
            crossed = (px < limit if side==1 else px > limit) if strict else (px <= limit if side==1 else px >= limit)
            if not crossed:
                return None
        key = (o['ticker'], field, q['last_msg'])
        if float(q[qty])-self.used[key]+1e-9 < o['size']:
            self.rejected['insufficient_best_size'] += 1
            return None
        self.used[key] += o['size']
        return px

    def place(self, d):
        super().place(d)
        o = self.orders[-1]
        px = self.take(o, o['side'], o['limit'], self.now)
        if px is not None:
            o.update(stage='hold', t_fill=self.now, entry=px, entry_fee=B.T.fee(B.T.TAKER, px))

    def update(self, mk, now):
        self.now = now
        mk.build()
        for o in list(self.busy.values()):
            if o is None or o['stage'] in ('closed','expired','settle'):
                continue
            side = o['side']
            if o['stage']=='entry':
                if now > o['t0']+B.T.ENTRY_S:
                    o['stage']='expired'; self.busy[o['ticker']]=None
                    continue
                if now > o['t0'] and self.take(o,side,o['limit'],now,strict=True) is not None:
                    o.update(stage='hold',t_fill=now,entry_fee=B.T.fee(B.T.MAKER,o['limit']))
            if o['stage']=='hold' and now >= o['t_fill']+B.T.HOLD_S:
                p=mk.p[o['ticker']]
                n=int(B.LM.np.searchsorted(p['t'],now,side='right'))
                ref=float(p['px'][n-1]) if n else o.get('entry',o['limit'])
                o.update(stage='exit',te=now,exit_limit=min(99.,max(1.,ref+B.T.OFFSET*side)))
                px=self.take(o,-side,o['exit_limit'],now)
                if px is not None:
                    self._close(o,px,'book crossed',now,B.T.fee(B.T.TAKER,px))
            if o['stage']=='exit':
                if o['te'] < now <= o['te']+B.T.EXIT_S:
                    px=self.take(o,-side,o['exit_limit'],now,strict=True)
                    if px is not None:
                        self._close(o,o['exit_limit'],'limit filled',now,B.T.fee(B.T.MAKER,o['exit_limit']))
                elif now > o['te']+B.T.EXIT_S:
                    px=self.take(o,-side,None,now)
                    if px is not None:
                        self._close(o,px,'book taken',now,B.T.fee(B.T.TAKER,px))
                    elif now > o['te']+2*B.T.EXIT_S:
                        o['stage']='settle'


def decide(book, rows, now, league, k, drop_totals):
    for d in rows:
        if 'ref' not in d or d.get('action','').startswith('skip:'):
            continue
        if drop_totals and league=='NBA' and d['kind']=='total':
            continue
        sig=d['level']-d['ref']
        if abs(sig)<B.T.MIN_SIGNAL or book.is_busy(d['ticker']):
            continue
        side=1 if sig>0 else -1
        limit=d['ref']-B.T.OFFSET*side
        size=B.T.contracts_of(B.T.size_of(sig,d.get('width')),k)
        if size>0 and 0<limit<100:
            book.place(dict(d,side=side,limit=limit,size=round(size,4),signal=round(sig,3)))


def live_parity(r, base, decisions):
    """Compare frozen standard-rule decisions to real orders; never force parity.

    Standardization intentionally changes historical guards/sizing. This test
    separates matching model values from matching the actual order path.
    """
    root=B.ROOT/r['root']
    placed=[e for e in B.A.rows(root/'orders.jsonl') if e.get('event')=='placed']
    originals={}
    keys={(e['t0'],e['ticker']) for e in placed}
    for d in B.A.rows(root/'decisions.jsonl'):
        key=(d.get('t'),d.get('ticker'))
        if key in keys:
            originals[key]=d
    current={(d['t'],d['ticker']):d for ds in decisions.values() for d in ds if (d['t'],d['ticker']) in keys}
    same_level=0; comparable=0; actions=collections.Counter(); differences=[]
    for e in placed:
        key=e['t0'],e['ticker']; a=originals.get(key); b=current.get(key)
        if a and b:
            comparable+=1
            same_level+=abs(a['level']-b['level'])<1e-4
        actions[b.get('action','unknown') if b else 'no_matching_decision_tick']+=1
        if not b or b.get('action')!='order' or any(abs(float(e.get(k,0))-float(b.get(k,0)))>1e-6 for k in ('side','size','limit')):
            differences.append(dict(t=e['t0'],ticker=e['ticker'],original_size=e['size'],standard=b))
    try:
        fills, audit=B.A.audit(root)
        actual=dict(audit=audit,fills=len(fills),pnl_c=sum(x['pnl_c'] for x in fills))
    except Exception as ex:
        actual=dict(audit_error=repr(ex))
    return dict(original_orders=len(placed),comparable_model_decisions=comparable,model_levels_within_0_0001c=same_level,
                actions_at_original_entries=dict(actions),different_original_entry_count=len(differences),
                entry_examples=differences[:10],actual=actual,
                status='not_testable_no_original_orders' if not placed else 'entry_mismatch' if differences else 'entries_match_only',
                caveat='Entry/model comparison only; exact original fill/exit parity is not established by this check.')


def run(path, digest):
    r=json.loads(Path(path).read_text()); name=r['league']+'_'+r['game']
    dst=OUT/'games'/r['date']/name
    done=dst/'result.json'
    if done.exists() and json.loads(done.read_text()).get('study_hash')==digest:
        return json.loads(done.read_text())
    info_path=BOOKS/r['date']/name/'stats.json'
    info=json.loads(info_path.read_text()) if info_path.exists() else {}
    result=dict(date=r['date'],league=r['league'],game=r['game'],phase=r['phase'],study_hash=digest,book=info,**SUM.coverage(r))
    if not info.get('available') or not (info_path.parent/'bbo.csv.gz').exists():
        result['excluded']='no available recorded book';write(done,result);return result
    if not r['final_score_available']:
        result['excluded']='no final score';write(done,result);return result
    g,arrivals,doc,final,hs,aws,trades,candles,missing=B.load(r)
    ds=collections.defaultdict(list)
    for d in B.A.rows(Path(path).parent/'whole5/decisions.jsonl'):
        if 'ticker' in d: ds[d['t']].append(d)
    result['live_parity']=live_parity(r,None,ds) if r['date']>='2026-10-24' else None
    clock0=arrivals[0][0]-1; end=arrivals[-1][0]+600
    last=max(ds) if ds else None
    end_regular=clock0+math.floor(end-clock0); end_final=end_regular+1+B.LM.STEP
    schedule={round(end_final if t==last else clock0+math.ceil(t-clock0),6):rows for t,rows in ds.items()}
    quotes=Quotes(info_path.parent/'bbo.csv.gz');quotes.prepare()
    meta=B.LM.contract_meta(g['league'],g['contracts'],g['home_k'],g['away_k'])
    mk=B.LM.Market([c['ticker'] for c in meta if c['kind'] in ('spread','total')])
    tape=sorted((B.RP.ts(x['received_at']),i,x) for i,x in enumerate(trades) if x['ticker'] in mk.tickers)
    arms=[('strict_whole5',S.Book({}),5,False)]
    originals=[]
    if r['date']>='2026-10-24':
        originals=[e for e in B.A.rows(B.ROOT/r['root']/'orders.jsonl') if e.get('event')=='placed']
    # A second control holds the actual entries fixed. It diagnoses execution
    # separately from the deliberate changes to guards and sizing in b37.
    probe=S.Book({})
    placed_at=collections.defaultdict(list)
    for e in originals:
        at=round(clock0+math.ceil(e['t0']-clock0),6)
        placed_at[at].append(e)
    for k,tag in [(None,'fractional'),(5,'whole5')]:
        for drop,suffix in [(False,'all'),(True,'no_nba_totals')]:
            for age in (5,60):
                arms.append((f'book_{tag}_{suffix}_age{age}',QuoteBook(quotes,age),k,drop))
    clock=clock0;idx=0
    while clock<=end_final+1e-6:
        quotes.advance(clock)
        if clock<=end_regular+1e-6:
            incoming=[]
            while idx<len(tape) and tape[idx][0]<=clock:
                x=tape[idx][2];idx+=1
                incoming.append(dict(ticker=x['ticker'],trade_id=x['trade_id'],taker_side=x['taker_side'],t=B.RP.ts(x['created_time']),
                                     price=float(x['yes_price_dollars'])*100 if x.get('yes_price_dollars') is not None else float(x['yes_price']),
                                     count=float(x.get('count_fp',x.get('count')))))
            if incoming:mk.add_trades(incoming)
        rows=schedule.get(round(clock,6))
        for tag,book,k,drop in arms:
            book.update(mk,clock)
            if rows:decide(book,rows,clock,g['league'],k,drop)
        if originals:
            probe.update(mk,clock)
            for e in placed_at.get(round(clock,6),[]):
                probe.place(dict(e,t=e['t0']))
        clock+=1
    result['results']={}
    for tag,book,k,drop in arms:
        proxy=types.SimpleNamespace(book=book,first_play=r['results']['whole5']['first_play'],L=r['results']['whole5']['pregame_total'],lags=[])
        stats,fills=B.score(proxy,g,final,hs,aws)
        if tag=='strict_whole5':
            expected=json.loads((Path(path).parent/'whole5/fills.json').read_text())
            assert S.baseline_signature(fills)==S.baseline_signature(expected), ('cached baseline mismatch',name)
        stats['quote_rejections']=dict(getattr(book,'rejected',{}))
        result['results'][tag]=stats
        write(dst/(tag+'_fills.json'),fills)
    if originals:
        proxy=types.SimpleNamespace(book=probe,first_play=r['results']['whole5']['first_play'],L=r['results']['whole5']['pregame_total'],lags=[])
        stats,replayed=B.score(proxy,g,final,hs,aws)
        actual,audit=B.A.audit(B.ROOT/r['root'])
        def execution_key(o):return (o['ticker'],o['t0'],o['side'],o['size'])
        expected={execution_key(o):o for o in actual}
        observed={execution_key(o):o for o in replayed}
        mismatches=[]
        for key in expected.keys() | observed.keys():
            a,b=expected.get(key),observed.get(key)
            reason=[]
            if a is None or b is None:reason.append('fill_presence')
            else:
                for field,old,new in [('fill_time',a['t_fill'],b['t_fill']),('entry',a['entry'],b.get('entry',b['limit'])),
                                      ('net',a['net_c'],b['net']),('exit_time',a['t_exit'],b.get('t_exit'))]:
                    if (old is None)!=(new is None) or (old is not None and abs(old-new)>1e-5):reason.append(field)
            if reason:mismatches.append(dict(order=list(key),reasons=reason))
        result['execution_parity']=dict(status='matches' if not mismatches else 'mismatch',
                                        actual_fills=len(actual),replayed_fills=len(replayed),
                                        mismatch_count=len(mismatches),examples=mismatches[:20],
                                        actual_pnl_c=sum(o['pnl_c'] for o in actual),replay_pnl_c=stats['pnl_c'],
                                        scope='Recorded entries held fixed; received-time print replay. Model/entry selection validated separately.',
                                        caveat='One-second update clock; original restarts and historical execution versions may differ.')
        write(dst/'actual_entries_replayed_fills.json',replayed)
    result['strict_cache_parity']=True
    result['book_rows']=quotes.rows
    write(done,result)
    return result


def refresh_missing_books():
    targets=[]
    for p in sorted(B.OUT.glob('recorded/*/*/result.json')):
        if not (BOOKS/p.parent.parent.name/p.parent.name/'stats.json').exists():
            targets.append((p.parent.parent.name,p.parent.name))
    if not targets:return
    status('extracting missing archived books',games=len(targets))
    core=Path(__file__).with_name('b38_extract_remote_l1.py').read_text().rsplit('\nOUT.mkdir',1)[0]
    code=core+'\nTARGETS='+repr(targets)+'''\nOUT.mkdir(parents=True,exist_ok=True)
for day,game in TARGETS:
 try:
  extract(day,game)
 except Exception as ex:
  dst=OUT/day/game;dst.mkdir(parents=True,exist_ok=True)
  (dst/'stats.json').write_text(json.dumps(dict(day=day,game=game,available=False,error=repr(ex))))
import sys,tarfile
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for day,game in TARGETS:
  root=OUT/day/game
  for p in root.glob('*'):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
'''
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','31223','gw@lum.id','python3','-']
    archive=OUT/'additional_books.tar.gz'
    with archive.open('wb') as f:
        subprocess.run(ssh,input=code.encode(),stdout=f,check=True,timeout=3600)
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            assert not m.name.startswith('/') and '..' not in Path(m.name).parts and (m.isfile() or m.isdir())
        tar.extractall(BOOKS,filter='data')


def report(errors):
    games=[json.loads(p.read_text()) for p in sorted(OUT.glob('games/*/*/result.json'))]
    groups=collections.defaultdict(lambda:dict(games=0,fills=0,contracts=0.,pnl_c=0.))
    for r in games:
        for arm,stats in r.get('results',{}).items():
            integrity='partial_book' if r['book'].get('coverage') or r['book'].get('source_integrity') else 'book_integrity_not_flagged'
            for kind,s in stats['by_kind'].items():
                key='|'.join((r['phase'],r['coverage'],integrity,arm,r['league'],kind))
                a=groups[key];a['games']+=1
                for k in ('fills','contracts','pnl_c'):a[k]+=s[k]
    for a in groups.values():a['net_per_contract_c']=a['pnl_c']/a['contracts'] if a['contracts'] else None
    write(OUT/'groups.json',dict(groups))
    parity=[dict(date=r['date'],league=r['league'],game=r['game'],**r['live_parity']) for r in games if r.get('live_parity')]
    write(OUT/'live_parity.json',parity)
    executions=[dict(date=r['date'],league=r['league'],game=r['game'],**r['execution_parity']) for r in games if r.get('execution_parity')]
    write(OUT/'execution_parity.json',executions)
    lines=['Recorded-feed and recorded-book replay','',f'Games processed: {len(games)}; excluded: {sum("excluded" in r for r in games)}; errors: {len(errors)}.',
           'Feed-delay inputs and forecasts are the frozen b37 received-time replay; strict K5 cache parity is asserted for every scored game.',
           'This additional execution sensitivity uses received-time one-second best quotes, visible best size and causal exits.',
           'It does not establish queue position, full-depth impact, subsecond fills or global WS sequence continuity.',
           'Both 5s and 60s maximum quote ages are reported. Empty, invalid, stale or insufficient-size quotes cannot fill.',
           'Live parity is an entry/model comparison at actual entry ticks, not a proof of matching original fills/exits.',
           'Historical rule/sizing changes and restart behavior can make standardized entries differ; inspect live_parity.json.',
           'Any entry mismatch means original-live parity has NOT been established. Do not represent this replay as matching live trading.',
           'Preseason, midgame recordings, unfinished feeds and truncated book prefixes are excluded or reported separately.',
           'Reconstructed final-feed histories are excluded from this job. These retrospective results cannot guarantee profit.','',
           f'Oct24+ games with original orders: {sum(p["original_orders"]>0 for p in parity)}.',
           f'Games with different original entry decisions: {sum(p["different_original_entry_count"]>0 for p in parity)}.','',
           f'Actual-entry execution controls: {len(executions)} games; exact fill/exit/P&L matches: {sum(p["status"]=="matches" for p in executions)}.',
           'Execution mismatches and examples: execution_parity.json. A mismatch blocks any claim of complete live parity.','',
           'Comparable groups (regular/postseason recordings from start; book not flagged partial):']
    for key,a in sorted(groups.items()):
        phase,coverage,integrity,arm,league,kind=key.split('|')
        if phase!='regular_or_postseason' or coverage!='recorded_from_start' or integrity!='book_integrity_not_flagged':continue
        value=f'{a["net_per_contract_c"]:.3f}' if a['contracts'] else 'n/a'
        lines.append(f'{arm:44s} {league} {kind:6s}: {a["fills"]:4d} fills; {a["contracts"]:.2f} contracts; {a["pnl_c"]:.3f} cents net; {value} c/contract')
    lines.extend(['','Full coverage-separated results: groups.json. Game fills and exclusions: games/.',
                  'Previous backtest comparison: ../b37_standard_replay/backtest_comparison.txt.',
                  'Strategy comparisons: ../b38_strategy_study/report.txt. Errors: errors.json.'])
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n')
    write(OUT/'errors.json',errors)
    return dict(games=len(games),scored=sum('results' in r for r in games),excluded=sum('excluded' in r for r in games),errors=len(errors),
                live_parity_established=False,report=str(OUT/'report.txt'))


def worker():
    refresh_missing_books()
    rule=dict(base_rule=json.loads((B.OUT/'rule.json').read_text()),source_sha256=B.digest(Path(__file__)),
              forecast_reuse='b37 received-time frozen decisions; guards unchanged; separate fractional and K5 sizing',
              execution='received-time 1s BBO strict-cross full-size sensitivity; causal exits; no at-limit queue fills',quote_max_age_seconds=[5,60])
    digest=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest();write(OUT/'rule.json',rule)
    paths=sorted(B.OUT.glob('recorded/*/*/result.json'))
    status('replaying',total=len(paths),done=0)
    errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(run,str(p),digest):p for p in paths}
        for count,f in enumerate(concurrent.futures.as_completed(fs),1):
            try:
                r=f.result();last=r['league']+'_'+r['game']
            except Exception as ex:
                last=fs[f].parent.name
                errors.append(dict(game=last,error=repr(ex),traceback=traceback.format_exc()))
            status('replaying',total=len(paths),done=count,last_game=last,errors=len(errors))
    result=report(errors)
    status('completed_with_errors' if errors else 'completed',**result)
    return bool(errors)


def monitor():
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(OUT/'monitor.lock').open('w')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('A replay monitor already owns this job.',flush=True)
        return 0
    status('starting',monitor_pid=os.getpid())
    with (OUT/'worker.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(OUT/'monitor.json',dict(monitor_pid=os.getpid(),worker_pid=child.pid,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                      completion='blocking child wait; worker writes report automatically; no assistant polling'))
        rc=child.wait()
    if rc:
        status('failed',exit_code=rc,log=str(OUT/'worker.log'),note='Incomplete results must not be represented as a finished audit.')
        return rc
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    sys.exit(worker() if args.worker else monitor())
