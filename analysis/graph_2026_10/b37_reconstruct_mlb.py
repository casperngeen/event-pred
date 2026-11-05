"""Companion to b37: resimulate five MLB games with reconstructed archive feeds.

These games were missed live or only partly recorded. They are kept separate
from simulations using actual feed versions and arrival times.
"""
from __future__ import annotations

import datetime as dt
import csv
import gzip
import hashlib
import json
import time
from pathlib import Path

import b37_standard_replay as B


def run(root, fc, manifest_hash):
    g = json.loads((root/'game.json').read_text())
    day = root.parent.parent.name
    target = B.OUT/'reconstructed'/day/('MLB_'+g['game_id'])
    done = target/'result.json'
    if done.exists():
        old = json.loads(done.read_text())
        if old.get('manifest_hash')==manifest_hash and old.get('extension_sha256')==B.digest(Path(__file__)):
            return old
    wall = time.time()
    doc = json.loads(gzip.decompress((root/'doc.json.gz').read_bytes()))
    final,hs,aws = B.A.scores(doc,'MLB')
    tr,seen = [],set()
    for r in ([] if (root/'ws_trades.csv.gz').exists() else B.R.rows(root/'trades.jsonl')):
        if r['trade_id'] in seen:
            continue
        seen.add(r['trade_id'])
        tr.append(dict(ticker=r['ticker'],trade_id=r['trade_id'],taker_side=r['taker_side'],t=B.RP.ts(r['created_time']),
                       price=float(r['yes_price_dollars'])*100 if r.get('yes_price_dollars') is not None else float(r['yes_price']),
                       count=float(r.get('count_fp',r.get('count'))),arr=B.RP.ts(r['created_time'])))
    if (root/'ws_trades.csv.gz').exists():
        with gzip.open(root/'ws_trades.csv.gz','rt') as f:
            for i,(ticker,ms,price,count,side) in enumerate(csv.reader(f)):
                tr.append(dict(ticker=ticker,trade_id=f'ws{i}',taker_side=side,t=float(ms)/1000,
                               price=float(price)*100,count=float(count),arr=float(ms)/1000))
    assert tr, f'No trade tape available: {root}'
    cd = {}
    for c in B.R.rows(root/'candles.jsonl'):
        if c['end_period_ts'] > B.RP.ts(c['received_at']):
            continue
        key = (c['ticker'],c['end_period_ts'])
        if key in cd:
            continue
        def close(side):
            v = (c.get(side) or {}).get('close_dollars')
            return None if v is None else float(v)*100
        cd[key] = dict(ticker=c['ticker'],ts=float(c['end_period_ts']),arr=float(c['end_period_ts']),
                       bid=close('yes_bid'),ask=close('yes_ask'))
    results = {}
    for name,k in [('latest',None),('whole5',5)]:
        eng = B.T.Engine(g,fc,target/name,B.T.KINDS_DEFAULT,10.0,False)
        eng.log = B.CompactLog(target/name)
        eng.book.log = eng.log
        eng.pub,eng.shadow,eng.k,eng.live_clock = B.NoPublish(),False,k,False
        try:
            B.RP.replay(eng,doc,tr,list(cd.values()),10.0,0.0,0.0)
            stats,fills = B.score(eng,g,final,hs,aws)
            stats['actions'] = eng.log.counts
            (target/name/'fills.json').write_text(json.dumps(fills,indent=1,default=float)+'\n')
            results[name] = stats
        finally:
            eng.log.close()
    r = dict(root=str(root.relative_to(B.ROOT)),league='MLB',game=g['game_id'],date=day,mode='reconstructed',phase='regular_or_postseason',
             manifest_hash=manifest_hash,wall_s=round(time.time()-wall,2),final_score_available=final,score=f'{aws:g}-{hs:g}',
             extension_sha256=B.digest(Path(__file__)),
             missing_feed_files=0,feed_versions=0,trade_rows=len(tr),candle_rows=len(cd),
             feed_basis='archive cut at pitch end +10s; trade and candle event times',results=results)
    done.write_text(json.dumps(r,indent=2,default=float)+'\n')
    return r


def main():
    manifest = json.loads((B.OUT/'rule.json').read_text())
    manifest_hash = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    extension = dict(source_sha256=B.digest(Path(__file__)),base_manifest_hash=manifest_hash)
    (B.OUT/'mlb_reconstruction_rule.json').write_text(json.dumps(extension,indent=2)+'\n')
    fc = B.SharedForecaster('MLB')
    for p in sorted((B.ROOT/'data/sports_live/sandbox').glob('2026*/missed/*/game.json')):
        r = run(p.parent,fc,manifest_hash)
        print(json.dumps(dict(game=r['game'],wall_s=r['wall_s'],latest=r['results']['latest']['pnl_c'],whole5=r['results']['whole5']['pnl_c'])),flush=True)


if __name__=='__main__':
    main()
