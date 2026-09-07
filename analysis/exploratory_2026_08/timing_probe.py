import httpx, time, json, sys
B="https://api.elections.kalshi.com/trade-api/v2/historical/trades"
tickers=json.load(open(sys.argv[1]))
import random
random.seed(0)
sample=random.sample(tickers, min(25,len(tickers)))
c=httpx.Client(timeout=30)
t0=time.time()
total_trades=0; total_calls=0
for t in sample:
    cur=None
    while True:
        p={"ticker":t,"limit":1000}
        if cur: p["cursor"]=cur
        r=c.get(B, params=p); total_calls+=1
        if r.status_code==429:
            time.sleep(int(r.headers.get("retry-after",3))); continue
        d=r.json()
        tr=d.get("trades",[])
        total_trades+=len(tr)
        cur=d.get("cursor")
        if not cur or not tr: break
elapsed=time.time()-t0
print(f"sample={len(sample)} tickers, calls={total_calls}, trades={total_trades}, elapsed={elapsed:.1f}s")
print(f"avg calls/ticker={total_calls/len(sample):.2f}  avg trades/ticker={total_trades/len(sample):.1f}  sec/ticker={elapsed/len(sample):.2f}")
