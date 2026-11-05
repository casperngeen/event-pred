"""Stream selected archived WS books into conservative one-second BBO samples.

Run on the sandbox via stdin. Original recordings are read only; compact
artifacts go to /tmp/event-pred-b38-l1. Snapshots REPLACE each market's book.
Per-game files partition a global subscription, so sequence gaps in one game
alone are not evidence of dropped messages; global continuity is not asserted.
"""
import csv
import datetime as dt
import gzip
import heapq
import json
import math
import time
from pathlib import Path

BASE=Path('/home/casperngeen/sports_live')
OUT=Path('/tmp/event-pred-b38-l1')
TARGETS=[('2026-10-24','NBA_401909854'),('2026-10-25','NBA_401909863'),
         ('2026-10-26','MLB_849804'),('2026-10-18','NFL_401873003'),
         ('2026-10-25','NFL_401873016')]


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()


def extract(day,game):
    root=BASE/day/game;src=root/'books_ws.jsonl.gz'
    if not src.exists():src=root/'books_ws.jsonl'
    dst=OUT/day/game;dst.mkdir(parents=True,exist_ok=True)
    if not src.exists():
        info=dict(day=day,game=game,available=False)
        (dst/'stats.json').write_text(json.dumps(info,indent=2)+'\n');return info
    wall=time.time();state={};pending={};bad=0;types={};written=0;first=None;last=None
    with (gzip.open(src,'rt') if src.suffix=='.gz' else src.open()) as f,gzip.open(dst/'bbo.csv.gz','wt',compresslevel=2,newline='') as z:
        w=csv.writer(z);w.writerow(['ts','ticker','bid','ask','bid_qty','ask_qty','last_msg','valid'])
        for line in f:
            if 'SPREAD' not in line and 'TOTAL' not in line:continue
            try:m=json.loads(line)
            except ValueError:bad+=1;continue
            kind=m.get('type');d=m.get('msg') or {};tk=d.get('market_ticker','')
            if kind not in ('orderbook_snapshot','orderbook_delta') or ('SPREAD' not in tk and 'TOTAL' not in tk):continue
            at=ts(m['received_at']);bucket=math.ceil(at)
            first=at if first is None else min(first,at);last=at if last is None else max(last,at)
            types[kind]=types.get(kind,0)+1
            if tk in pending and bucket>pending[tk][0]:
                w.writerow(pending.pop(tk));written+=1
            if kind=='orderbook_snapshot':
                b=dict(yes={},no={},hy=[],hn=[],valid=True,negative_deltas=0)
                for side,h in [('yes','hy'),('no','hn')]:
                    dollars=d.get(side+'_dollars_fp')
                    for p,q in dollars if dollars is not None else (d.get(side) or []):
                        pk=round(float(p)*(10000 if dollars is not None else 100))
                        q=float(q)
                        if q>0:b[side][pk]=q;b[h].append(-pk)
                    heapq.heapify(b[h])
                state[tk]=b
            else:
                if tk not in state:state[tk]=dict(yes={},no={},hy=[],hn=[],valid=False,negative_deltas=0)
                b=state[tk];side=d['side'];h='hy' if side=='yes' else 'hn'
                pk=round(float(d['price_dollars'])*10000) if d.get('price_dollars') is not None else round(float(d['price'])*100)
                before=b[side].get(pk,0);after=before+float(d.get('delta_fp',d.get('delta',0)))
                if after<-.000001:b['valid']=False;b['negative_deltas']+=1
                if after>1e-9:
                    b[side][pk]=after
                    if before<=0:heapq.heappush(b[h],-pk)
                else:b[side].pop(pk,None)
            b=state[tk]
            for side,h in [('yes','hy'),('no','hn')]:
                while b[h] and -b[h][0] not in b[side]:heapq.heappop(b[h])
            y=-b['hy'][0] if b['hy'] else None;n=-b['hn'][0] if b['hn'] else None
            bid=y/100 if y is not None else None;ask=100-n/100 if n is not None else None
            valid=b['valid'] and (bid is None or ask is None or bid<=ask+1e-8)
            pending[tk]=[bucket,tk,bid,ask,b['yes'].get(y,0),b['no'].get(n,0),at,int(valid)]
        for row in pending.values():w.writerow(row);written+=1
    info=dict(day=day,game=game,available=True,input=str(src),input_bytes=src.stat().st_size,
              snapshots=types.get('orderbook_snapshot',0),deltas=types.get('orderbook_delta',0),rows=written,
              bad_json_lines=bad,negative_deltas=sum(b['negative_deltas'] for b in state.values()),
              first_received=first,last_received=last,wall_s=round(time.time()-wall,2),
              continuity='global subscription is partitioned among game files; continuity not verified',
              sampling='state at ceiling(received_at), using only messages received at/before that second')
    (dst/'stats.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


OUT.mkdir(parents=True,exist_ok=True)
for day,game in TARGETS:
    print(json.dumps(extract(day,game)),flush=True)
