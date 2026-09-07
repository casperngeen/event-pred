import httpx, time, json, datetime as dt
B="https://api.elections.kalshi.com/trade-api/v2"
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"
CORE=["KXCPI","KXCPIYOY","KXCPICORE","KXCPICOREYOY","KXFEDDECISION","KXFED","KXPAYROLLS",
      "KXU3","KXGDP","KXPCECORE","KXJOBLESSCLAIMS","KXISMPMI","KXADP","KXUSPPIYOY","KXUSPPI",
      "KXUSNFP","KXUSRETAIL","KXSHELTERCPI","KXUSGASCPI","KXUSEDCARCPI","KXAIRFARECPI",
      "KXCPINDEX","KXUSDURABLE","KXUSMICHCSP","KXUSISMSERV"]  # excl. KXWTI: not core to this harvest

c=httpx.Client(timeout=60, follow_redirects=True)

def get(url, params, tries=4):
    for i in range(tries):
        r=c.get(url, params=params)
        if r.status_code==429:
            time.sleep(int(r.headers.get("retry-after",3))); continue
        if r.status_code==200: return r.json()
        if r.status_code in (500,502,503,504): time.sleep(2); continue
        return None
    return None

# 1) currently-listed tickers (candlesticks only work for still-listed markets)
live=[]
for s in CORE:
    cur=None
    while True:
        p={"series_ticker":s,"limit":1000}
        if cur: p["cursor"]=cur
        d=get(f"{B}/markets", p)
        if not d: break
        live+=[(s,m["ticker"]) for m in d.get("markets",[])]
        cur=d.get("cursor")
        if not cur: break
print(f"live tickers: {len(live)}", flush=True)

now=int(dt.datetime.now(dt.timezone.utc).timestamp())
start=now-120*86400  # 120 days back; retention will clip further if shorter

out=f"{SC}/candles.jsonl"
done=set()
import os
if os.path.exists(out):
    for line in open(out):
        try: done.add(json.loads(line)["ticker"])
        except: pass
print(f"already done: {len(done)}", flush=True)

with open(out,"a") as f:
    for i,(s,t) in enumerate(live):
        if t in done: continue
        d=get(f"{B}/markets/candlesticks", {"market_tickers":t,"start_ts":start,"end_ts":now,"period_interval":1440})
        arr=(d or {}).get("markets") or []
        cs = arr[0]["candlesticks"] if arr and "candlesticks" in arr[0] else []
        for row in cs:
            row["_ticker"]=t; row["_series"]=s
            f.write(json.dumps(row)+"\n")
        if not cs:
            f.write(json.dumps({"_ticker":t,"_series":s,"_empty":True})+"\n")
        if i%100==0:
            f.flush(); print(f"[{i}/{len(live)}] {t}: {len(cs)} candles", flush=True)
        time.sleep(0.05)
print("DONE", flush=True)
