"""Resimulate recorded games using the current deployed rule and model weights.

Primary sizing is the current fractional default; K=5 is a separate companion.
Actual feed/trade/candle arrival times are retained. Reconstructed historical
feeds have separate results. Outputs never overwrite original recordings.

venv/bin/python -u -W ignore analysis/graph_2026_10/b37_standard_replay.py --workers 3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import heapq
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault('POLARS_MAX_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import live_model as LM
import live_record as R
import live_trade as T
import live_replay as RP
import b36_live_loss_audit as A
import b37_replay_lookup as LOOKUP

LOOKUP.install(T.Engine)

OUT = Path(__file__).with_name('out') / 'b37_standard_replay'
MODELS = {'MLB': 'gru_d10_h30-60-120-1800_tp_pitch',
          'NBA': 'gru_d60_h30-60-120-1800_tp', 'NFL': 'gru_d60_h30-60-120-1800_tp'}
FORECASTERS = {}
OBSERVATION_FIELDS = ('doc','first_seen','plays','px','pt','pstate','px_t','pt_t','pstate_t',
                      'first_play','L','p_home','in_play','final','lags','review_type',
                      'review_pending','review_hold','placeholders')
_step_inputs = LM.step_inputs
_step_cache = {}


def shared_step_inputs(t, first_play, contracts, mk, p_home_prev):
    # Both arms see the same observations at the same tick. The primary's
    # returned home price is also the companion's carry, so the result is exact.
    key = (id(mk),first_play,t)
    if _step_cache.get('key')!=key:
        _step_cache.update(key=key,value=_step_inputs(t,first_play,contracts,mk,p_home_prev))
    return _step_cache['value']


LM.step_inputs = shared_step_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present(root, name):
    return (root / name).exists() or (root / (name + '.gz')).exists()


class CompactLog:
    def __init__(self, root):
        self.root, self.files = root, {}
        self.counts = {}

    def __call__(self, name, rs):
        if name in ('forecasts.jsonl', 'published.jsonl'):
            return
        rs = rs if isinstance(rs, list) else [rs]
        if name not in self.files:
            self.files[name] = gzip.open(self.root / (name + '.gz'), 'wt', compresslevel=1)
        f = self.files[name]
        for r in rs:
            f.write(json.dumps(r, separators=(',', ':'), default=float) + '\n')
            if name == 'decisions.jsonl':
                a = r.get('action', r.get('event', 'unknown'))
                self.counts[a] = self.counts.get(a, 0) + 1

    def close(self):
        for f in self.files.values():
            f.close()


class SharedForecaster:
    """Memoize identical inference across sizing arms; weights/math are unchanged."""
    def __init__(self, league):
        self.inner = LM.Forecaster(league, ROOT / 'results/ingame/deploy', name=MODELS[league])
        self.horizons = self.inner.horizons
        self.at, self.cache = None, {}
        # The GRU encoding depends only on the received play tensor, not on the
        # current market step. Keep four exact encodings, including true stamps.
        for model in self.inner.models:
            original = model.play_states
            cache = {}

            def states(plays, original=original, cache=cache):
                key = hashlib.blake2b(plays.cpu().numpy().tobytes(), digest_size=16).digest()
                if key not in cache:
                    if len(cache) >= 4:
                        cache.pop(next(iter(cache)))
                    cache[key] = original(plays)
                return cache[key]
            model.play_states = states

    def forecast(self, steps, px, pt, pstate, L, delay_s=0.0):
        at = tuple(s['t'] for s in steps)
        if at != self.at:
            self.at, self.cache = at, {}
        key = hashlib.blake2b(pickle.dumps((steps, px, pt, pstate, L, delay_s), protocol=5), digest_size=16).digest()
        if key not in self.cache:
            self.cache[key] = self.inner.forecast(steps, px, pt, pstate, L, delay_s)
        return self.cache[key]


class NoPublish:
    def send(self, *args):
        pass


def discover():
    entries, seen, excluded = [], set(), []
    for source in ('vm', 'replay_inputs', 'sandbox', 'local'):
        for p in sorted((ROOT / 'data/sports_live' / source).glob('2026*/*/game.json')):
            g = json.loads(p.read_text())
            if g.get('league') not in MODELS:
                continue
            root, day = p.parent, p.parent.parent.name
            key = (g['league'], g['game_id'])
            if key in seen:
                continue
            if g['league'] == 'MLB' and day == '2026-10-17':
                excluded.append(dict(root=str(root.relative_to(ROOT)), reason='wrong-game contracts after rainout'))
                continue
            if not any('SPREAD' in m['ticker'] or 'TOTAL' in m['ticker'] for m in g.get('contracts', [])):
                excluded.append(dict(root=str(root.relative_to(ROOT)), reason='no spread/total contracts recorded'))
                continue
            missing = [n for n in ('feed_arrivals.jsonl','trades.jsonl','candles.jsonl') if not present(root,n)]
            if missing:
                excluded.append(dict(root=str(root.relative_to(ROOT)), reason='missing inputs: ' + ','.join(missing)))
                continue
            seen.add(key)
            entries.append(dict(root=str(root.relative_to(ROOT)), league=g['league'], game=g['game_id'],
                                date=day, mode='recorded', phase='preseason' if g['league']=='NBA' and day<'2026-10-20' else 'regular_or_postseason'))
    for p in sorted((ROOT / 'data/sports_live/replay_hist').glob('2026*/*/game.json')):
        g = json.loads(p.read_text())
        entries.append(dict(root=str(p.parent.relative_to(ROOT)), league=g['league'], game=g['game_id'],
                            date=p.parent.parent.name, mode='reconstructed', phase='regular_or_postseason'))
    return entries, excluded


def load(entry):
    root = ROOT / entry['root']
    g = json.loads((root / 'game.json').read_text())
    arrivals, missing = {}, 0
    for a in R.rows(root / 'feed_arrivals.jsonl'):
        p = root / 'feed' / (a['stamp'] + '.json.gz')
        if p.exists():
            arrivals[p] = min(arrivals.get(p, math.inf), RP.ts(a['received_at']))
        else:
            missing += 1
    if not arrivals:
        raise ValueError('no readable feed arrivals')
    arrivals = sorted((t,p) for p,t in arrivals.items())
    doc = json.loads(gzip.decompress(sorted((root/'feed').glob('*.json.gz'))[-1].read_bytes()))
    final, hs, aws = A.scores(doc,g['league'])
    trades = {}
    for r in R.rows(root / 'trades.jsonl'):
        old = trades.get(r['trade_id'])
        if old is None or RP.ts(r['received_at']) < RP.ts(old['received_at']):
            trades[r['trade_id']] = r
    # Repeat candle polls contain identical rows; retain their earliest arrival.
    raw_candles = []
    for c in R.rows(root / 'candles.jsonl'):
        at = RP.ts(c['received_at'])
        if c['end_period_ts'] > at:
            continue
        raw_candles.append((at,c))
    candles,last = [],{}
    for _,c in sorted(raw_candles,key=lambda x:x[0]):
        key = (c['ticker'],c['end_period_ts'])
        value = (json.dumps(c.get('yes_bid'),sort_keys=True),json.dumps(c.get('yes_ask'),sort_keys=True))
        if last.get(key)!=value:
            candles.append(c)
            last[key] = value
    return g, arrivals, doc, final, hs, aws, list(trades.values()), candles, missing


def score(eng, g, final, hs, aws):
    meta = {m['ticker']:m for m in g['contracts']}
    fills = []
    for o in eng.book.orders:
        if 't_fill' not in o or o['size'] <= 0:
            continue
        resolved = o['stage']=='closed' or final
        net = o.get('net') if o['stage']=='closed' else (o['side'] * (A.payoff(meta[o['ticker']],g['league'],g['home_k'],g['away_k'],hs,aws)-o.get('entry',o['limit']))-o['entry_fee'] if final else None)
        fills.append(dict(o, resolved=resolved, net=net, pnl_c=net*o['size'] if net is not None else None))
    resolved = [o for o in fills if o['resolved']]
    size = sum(o['size'] for o in resolved)
    pnl = sum(o['pnl_c'] for o in resolved)
    by_kind = {}
    for kind in T.KINDS_DEFAULT:
        xs = [o for o in resolved if o['kind']==kind]
        n = sum(o['size'] for o in xs)
        by_kind[kind] = dict(fills=len(xs), contracts=n, pnl_c=sum(o['pnl_c'] for o in xs),
                             net_per_contract_c=sum(o['pnl_c'] for o in xs)/n if n else None)
    return dict(orders=len(eng.book.orders), fills=len(fills), resolved_fills=len(resolved), unresolved_fills=len(fills)-len(resolved),
                contracts=size, pnl_c=pnl, net_per_contract_c=pnl/size if size else None, by_kind=by_kind,
                closed_c=sum(o['pnl_c'] for o in resolved if o['stage']=='closed'),
                settlement_c=sum(o['pnl_c'] for o in resolved if o['stage']!='closed'),
                first_play=eng.first_play, pregame_total=eng.L, feed_delay_median_s=float(LM.np.median(eng.lags)) if eng.lags else None), fills


def run_one(entry, manifest_hash):
    wall = time.time()
    target = OUT / entry['mode'] / entry['date'] / (entry['league']+'_'+entry['game'])
    done = target / 'result.json'
    if done.exists():
        old = json.loads(done.read_text())
        if old.get('manifest_hash') == manifest_hash:
            return old
    g, arrivals, doc, final, hs, aws, trades, candles, missing = load(entry)
    league = g['league']
    if league not in FORECASTERS:
        FORECASTERS[league] = SharedForecaster(league)
    fc = FORECASTERS[league]
    delay = 10.0 if league=='MLB' else 60.0
    engines = []
    for name,k in [('latest',None),('whole5',5)]:
        dest = target/name
        eng = T.Engine(g,fc,dest,T.KINDS_DEFAULT,delay,False)
        eng.log = CompactLog(dest)
        eng.book.log = eng.log
        eng.pub, eng.shadow, eng.k = NoPublish(), False, k
        eng.gate_true = league=='NBA'
        eng.late_block = league=='NBA'
        engines.append((name,eng))
    # Market observations are independent of our paper orders. Sharing the
    # market avoids ingesting/building the same tape twice without sharing books.
    engines[1][1].mk = engines[0][1].mk
    q = []
    live = entry['mode']=='recorded'
    if live:
        for t,p in arrivals:
            q.append((t,len(q),'feed',p))
        clock = arrivals[0][0]-1
    else:
        first = LM.nba_live_plays(doc,dt.datetime.fromisoformat(g['start'].replace('Z','+00:00')),None,delay)[1]
        clock = first-30
        q.append((clock,len(q),'feed',arrivals[-1][1]))
    for r in trades:
        q.append((RP.ts(r['received_at'] if live else r['created_time']),len(q),'trade',r))
    for c in candles:
        q.append((RP.ts(c['received_at']) if live else float(c['end_period_ts']),len(q),'candle',c))
    heapq.heapify(q)
    end = arrivals[-1][0]+600
    try:
        while clock <= end:
            trs,cds,feed = [],[],None
            while q and q[0][0] <= clock:
                at,_,kind,x = heapq.heappop(q)
                if kind=='feed':
                    feed = x
                elif kind=='trade':
                    trs.append(x)
                else:
                    cds.append(x | {'received_at':dt.datetime.fromtimestamp(min(at,clock),dt.timezone.utc).isoformat()})
            if trs:
                engines[0][1].on_trades(trs)
            if cds:
                engines[0][1].on_candles(cds)
            if feed is not None:
                d = json.loads(gzip.decompress(feed.read_bytes()))
                got = dt.datetime.fromtimestamp(clock,dt.timezone.utc)
                engines[0][1].on_feed(d,got)
            primary, companion = engines[0][1],engines[1][1]
            primary.step(clock)
            first = companion.first_play is None and primary.first_play is not None
            for name in OBSERVATION_FIELDS:
                setattr(companion,name,getattr(primary,name))
            companion.doc_new = False
            if first:
                companion.log('decisions.jsonl',dict(event='first play',t=companion.first_play,L=companion.L,
                                                     late_block=companion.late_block,
                                                     late_kinds=list(T.LATE_KINDS) if companion.late_block else []))
            companion.step(clock)
            clock += 1.0
        results = {}
        for name,eng in engines:
            eng.final = True
            eng.step(clock+LM.STEP)
            stats, fills = score(eng,g,final,hs,aws)
            stats['actions'] = eng.log.counts
            (target/name/'fills.json').write_text(json.dumps(fills,indent=1,default=float)+'\n')
            results[name] = stats
        result = dict(entry, manifest_hash=manifest_hash, wall_s=round(time.time()-wall,2),
                      final_score_available=final, score=f'{aws:g}-{hs:g}', missing_feed_files=missing,
                      feed_versions=len(arrivals), trade_rows=len(trades), candle_rows=len(candles), results=results)
        done.write_text(json.dumps(result,indent=2,default=float)+'\n')
        return result
    finally:
        for _,eng in engines:
            eng.log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--game',action='append')
    ap.add_argument('--league',choices=list(MODELS))
    ap.add_argument('--limit',type=int)
    args = ap.parse_args()
    entries, excluded = discover()
    if args.game:
        entries = [e for e in entries if e['game'] in args.game]
    if args.league:
        entries = [e for e in entries if e['league']==args.league]
    if args.limit:
        entries = entries[:args.limit]
    OUT.mkdir(parents=True,exist_ok=True)
    sources = [ROOT/'scripts'/n for n in ('live_model.py','live_trade.py','live_record.py','fetch_pbp.py')]
    sources += [ROOT/'stg_infra/stg/models/ingame.py',Path(__file__),Path(LOOKUP.__file__)]
    weights = sorted((ROOT/'results/ingame/deploy').glob('*_final_*.pt'))
    manifest = dict(models=MODELS, delay_s={'MLB':10,'NFL':60,'NBA':60}, kinds=list(T.KINDS_DEFAULT),
                    primary_contracts_k=None, companion_contracts_k=5, nba_gate_true=True,nba_late_block=True,
                    fill_rule='strict through-print',cancel_rule=None,hold_s=T.HOLD_S,entry_s=T.ENTRY_S,exit_s=T.EXIT_S,
                    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources+weights})
    manifest_hash = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    (OUT/'rule.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'inventory.json').write_text(json.dumps(dict(entries=entries,excluded=excluded),indent=2)+'\n')
    print(json.dumps(dict(games=len(entries),workers=args.workers,manifest_hash=manifest_hash)),flush=True)
    results, errors = [], []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(run_one,e,manifest_hash):e for e in entries}
        for task in concurrent.futures.as_completed(tasks):
            e = tasks[task]
            try:
                r = task.result()
                results.append(r)
                print(json.dumps(dict(done=len(results),game=e['game'],league=e['league'],mode=e['mode'],wall_s=r['wall_s'],
                                      latest=r['results']['latest']['pnl_c'],whole5=r['results']['whole5']['pnl_c'])),flush=True)
            except Exception as ex:
                import traceback
                err = dict(e,error=repr(ex),traceback=''.join(traceback.format_exception(ex)))
                errors.append(err)
                print(json.dumps(err),flush=True)
            (OUT/'progress.json').write_text(json.dumps(dict(completed=len(results),total=len(entries),errors=errors),indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')


if __name__=='__main__':
    main()
