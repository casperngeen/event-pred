"""Full-path strategy comparisons using b37's cached observational decisions.

Forecasts do not depend on our paper positions. Reuse their exact levels, print
references and guard outcomes, while rerunning every entry, fill, exit and busy
contract transition. Test K=5, actual recorded regular/postseason segments only.
The baseline must reproduce b37's fills and P&L before variants are accepted.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import time
import types
from pathlib import Path

import b37_standard_replay as B

OUT = Path(__file__).with_name('out')/'b38_strategy_study'
# Fixed candidate families, not an outcome-driven parameter grid. Keep the
# observations and all existing deployment guards unchanged in every arm.
CANDIDATES = [
    dict(name='baseline'),
    dict(name='no_nba_totals',no_nba_totals=True),
    dict(name='spreads_only',spreads_only=True),
    dict(name='no_short_totals',no_short_totals=True),
    dict(name='signal10',min_signal=10),
    dict(name='signal12',min_signal=12),
    dict(name='offset2',offset=2),
    dict(name='entry30',entry_s=30),
    dict(name='hold60',hold_s=60),
    dict(name='exit60',exit_s=60),
    dict(name='hold60_exit60',hold_s=60,exit_s=60),
    dict(name='no_missing_width',no_missing_width=True),
    dict(name='width10',min_width=10),
    dict(name='one_direction',one_direction=True,rank=True),
    dict(name='cap250',risk_cap=250,rank=True),
    dict(name='cap500',risk_cap=500,rank=True),
    dict(name='cancel_invalid',cancel_invalid=True),
    dict(name='cancel_guard',cancel_guard=True),
    dict(name='conservative',no_nba_totals=True,risk_cap=500,rank=True),
    dict(name='guarded',min_signal=10,risk_cap=500,cancel_invalid=True,rank=True),
]


class Book(B.T.PaperBook):
    def __init__(self,cfg):
        super().__init__(lambda *args:None)
        env = dict(B.T.PaperBook.update.__globals__,ENTRY_S=cfg.get('entry_s',60.0),
                   HOLD_S=cfg.get('hold_s',120.0),EXIT_S=cfg.get('exit_s',120.0))
        self.raw_update = types.FunctionType(B.T.PaperBook.update.__code__,env)
        self.peak_risk = 0.0

    def active(self):
        return [o for o in self.busy.values() if o is not None]

    def risk(self):
        return sum(o['size']*(o.get('entry',o['limit']) if o['side']==1 else 100-o.get('entry',o['limit'])) for o in self.active())

    def update(self,mk,now):
        history = self.orders
        self.orders = sorted((o for o in self.active() if o['stage'] not in ('closed','expired','settle')),key=lambda o:o['id'])
        try:
            self.raw_update(self,mk,now)
        finally:
            self.orders = history
        self.peak_risk = max(self.peak_risk,self.risk())


def fingerprint():
    base = json.loads((B.OUT/'rule.json').read_text())
    rule = dict(base_rule_sha256=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),
                candidates=CANDIDATES,contracts_k=5,cancel_latency_s=.5,
                design_end='2026-10-25',validation_start='2026-10-26',
                selection='largest design P&L after 1c per contract per executed leg stress; no validation outcomes in selection',
                caveat='retrospective chronological validation; these dates have been viewed before',
                study_source_sha256=B.digest(Path(__file__)))
    return rule,hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()


def direction(o,meta):
    if o['kind']=='total':
        return o['side']
    return o['side']*(1 if meta[o['ticker']]['side']=='home' else -1)


def decide(book,cfg,ds,now,league,meta):
    for d in ds:
        o = book.busy.get(d['ticker'])
        if o is None or o['stage']!='entry' or 'cancel_at' in o:
            continue
        invalid = cfg.get('cancel_invalid') and o['side']*(d['level']-o['limit'])<=0
        guard = cfg.get('cancel_guard') and d.get('action','').startswith('skip:')
        if invalid or guard:
            book.busy[d['ticker']]['cancel_at'] = now+.5
    if cfg.get('rank'):
        ds = sorted(ds,key=lambda d:abs(d['level']-d.get('ref',d['level'])),reverse=True)
    for d in ds:
        if 'ref' not in d or d.get('action','').startswith('skip:'):
            continue
        kind,tk = d['kind'],d['ticker']
        if cfg.get('spreads_only') and kind!='spread':
            continue
        if cfg.get('no_nba_totals') and league=='NBA' and kind=='total':
            continue
        sig = d['level']-d['ref']
        if abs(sig)<cfg.get('min_signal',5) or book.is_busy(tk):
            continue
        side = 1 if sig>0 else -1
        if cfg.get('no_short_totals') and kind=='total' and side==-1:
            continue
        width = d.get('width')
        if cfg.get('no_missing_width') and width is None:
            continue
        if cfg.get('min_width') is not None and (width is None or width<cfg['min_width']):
            continue
        lim = d['ref']-cfg.get('offset',1)*side
        if not 0<lim<100:
            continue
        size = B.T.contracts_of(B.T.size_of(sig,width),5)
        candidate = dict(kind=kind,ticker=tk,side=side)
        if cfg.get('one_direction') and any(o['kind']==kind and direction(o,meta)==direction(candidate,meta) for o in book.active()):
            continue
        if cfg.get('risk_cap'):
            risk = lim if side==1 else 100-lim
            size = min(size,max(0,math.floor((cfg['risk_cap']-book.risk()+1e-9)/risk)))
        if size<=0:
            continue
        book.place(dict(d,side=side,limit=lim,size=size,signal=round(sig,3)))
        book.peak_risk = max(book.peak_risk,book.risk())


def baseline_signature(fills):
    return [(o['ticker'],o['side'],o['size'],o['t0'],o['t_fill'],o.get('entry',o['limit']),o['stage'],o.get('t_exit'),o['net']) for o in fills]


def run(result_file,rule_hash):
    r = json.loads(result_file.read_text())
    name = r['league']+'_'+r['game']
    dst = OUT/'games'/r['date']/name
    dst.mkdir(parents=True,exist_ok=True)
    done = dst/'result.json'
    if done.exists() and json.loads(done.read_text()).get('study_hash')==rule_hash:
        return json.loads(done.read_text())
    wall = time.time()
    g,arrivals,doc,final,hs,aws,trades,candles,missing = B.load(r)
    if not final:
        out = dict(date=r['date'],league=r['league'],game=r['game'],study_hash=rule_hash,excluded='unfinished final feed')
        done.write_text(json.dumps(out,indent=2)+'\n')
        return out
    clock0 = arrivals[0][0]-1
    end = arrivals[-1][0]+600
    end_regular = clock0+math.floor(end-clock0)
    end_final = end_regular+1+B.LM.STEP
    ds = collections.defaultdict(list)
    with gzip.open(result_file.parent/'whole5/decisions.jsonl.gz','rt') as f:
        for line in f:
            d = json.loads(line)
            if 'ticker' in d:
                ds[d['t']].append(d)
    last_t = max(ds) if ds else None
    schedule = {round(end_final if t==last_t else clock0+math.ceil(t-clock0),6):rows for t,rows in ds.items()}
    contracts = B.LM.contract_meta(g['league'],g['contracts'],g['home_k'],g['away_k'])
    meta = {c['ticker']:c for c in contracts}
    mk = B.LM.Market([c['ticker'] for c in contracts if c['kind'] in ('spread','total')])
    tape = sorted([(B.RP.ts(x['received_at']),x) for x in trades if x['ticker'] in mk.tickers],key=lambda x:x[0])
    books = [(cfg,Book(cfg)) for cfg in CANDIDATES]
    clock,idx = clock0,0
    while clock<=end_final+1e-6:
        if clock<=end_regular+1e-6:
            trs=[]
            while idx<len(tape) and tape[idx][0]<=clock:
                x=tape[idx][1];idx+=1
                trs.append(dict(ticker=x['ticker'],trade_id=x['trade_id'],taker_side=x['taker_side'],t=B.RP.ts(x['created_time']),
                                price=float(x['yes_price_dollars'])*100 if x.get('yes_price_dollars') is not None else float(x['yes_price']),
                                count=float(x.get('count_fp',x.get('count')))))
            if trs:
                mk.add_trades(trs)
        rows=schedule.get(round(clock,6))
        for cfg,book in books:
            book.update(mk,clock)
            if rows:
                decide(book,cfg,rows,clock,g['league'],meta)
        clock+=1.0
    results={}
    for cfg,book in books:
        proxy=types.SimpleNamespace(book=book)
        # B.score also reads feed metadata; supply it without constructing an engine.
        proxy.first_play=r['results']['whole5']['first_play'];proxy.L=r['results']['whole5']['pregame_total'];proxy.lags=[]
        stats,fills=B.score(proxy,g,final,hs,aws)
        stats['peak_principal_risk_c']=book.peak_risk
        legs=sum(o['size']*(2 if o['stage']=='closed' else 1) for o in fills)
        stats['executed_contract_legs']=legs
        stats['pnl_stress_c']=stats['pnl_c']-legs
        stats['pnl_stress_half_c']=stats['pnl_c']-.5*legs
        results[cfg['name']]=stats
        if cfg['name']=='baseline':
            expected=json.loads((result_file.parent/'whole5/fills.json').read_text())
            assert baseline_signature(fills)==baseline_signature(expected), ('baseline path mismatch',g['game_id'],baseline_signature(fills),baseline_signature(expected))
            assert abs(stats['pnl_c']-r['results']['whole5']['pnl_c'])<1e-7
        (dst/(cfg['name']+'_fills.json')).write_text(json.dumps(fills,default=float)+'\n')
    out=dict(date=r['date'],league=r['league'],game=r['game'],matchup=g['away']+'@'+g['home'],source=r['root'],
             study_hash=rule_hash,wall_s=round(time.time()-wall,2),results=results,missing_feed_files=missing)
    done.write_text(json.dumps(out,indent=2)+'\n')
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--game',action='append');ap.add_argument('--watch',action='store_true');a=ap.parse_args()
    rule,h=fingerprint();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'study_rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    errors=[]
    while True:
        files=[]
        for p in sorted(B.OUT.glob('recorded/*/*/result.json')):
            r=json.loads(p.read_text())
            if r['phase']=='preseason' or r.get('manifest_hash')!=rule['base_rule_sha256']:
                continue
            if a.game and r['game'] not in a.game:
                continue
            done=OUT/'games'/r['date']/(r['league']+'_'+r['game'])/'result.json'
            if done.exists() and json.loads(done.read_text()).get('study_hash')==h:
                continue
            files.append(p)
        for p in files:
            try:
                r=run(p,h);print(json.dumps({k:r[k] for k in ('date','league','game')} | {'wall_s':r.get('wall_s'),'baseline':r.get('results',{}).get('baseline',{}).get('pnl_c'),'excluded':r.get('excluded')}),flush=True)
            except Exception as ex:
                import traceback
                e=dict(game=p.parent.name,error=repr(ex),traceback=traceback.format_exc());errors.append(e);print(json.dumps(e),flush=True)
        (OUT/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')
        if not a.watch:
            break
        expected=sum(e['mode']=='recorded' and e['phase']!='preseason' for e in json.loads((B.OUT/'inventory.json').read_text())['entries'])
        done=[p for p in OUT.glob('games/*/*/result.json') if json.loads(p.read_text()).get('study_hash')==h]
        if len(done)>=expected:
            break
        time.sleep(10)


if __name__=='__main__':
    main()
