import httpx, json, time, os, sys
B="https://api.elections.kalshi.com/trade-api/v2"
OUT="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"
CORE=["KXCPI","KXCPIYOY","KXCPICORE","KXCPICOREYOY","KXFEDDECISION","KXFED","KXPAYROLLS",
      "KXU3","KXGDP","KXPCECORE","KXJOBLESSCLAIMS","KXISMPMI","KXADP","KXUSPPIYOY","KXUSPPI",
      "KXUSNFP","KXUSRETAIL","KXSHELTERCPI","KXUSGASCPI","KXUSEDCARCPI","KXAIRFARECPI",
      "KXCPINDEX","KXUSDURABLE","KXUSMICHCSP","KXUSISMSERV","KXWTI"]
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

# 1) events
evfile=f"{OUT}/events.json"
if os.path.exists(evfile):
    events=json.load(open(evfile))
else:
    events={}
    for s in CORE:
        got=set()
        for st in ["settled","closed","open"]:
            cur=None
            while True:
                p={"series_ticker":s,"limit":200,"status":st}
                if cur: p["cursor"]=cur
                d=get(f"{B}/events",p)
                if not d: break
                got|={e["event_ticker"] for e in d.get("events",[])}
                cur=d.get("cursor")
                if not cur: break
        events[s]=sorted(got)
        print(f"{s}: {len(got)} events", flush=True)
    json.dump(events,open(evfile,"w"))

allev=[(s,e) for s,evs in events.items() for e in evs]
print(f"TOTAL EVENTS: {len(allev)}", flush=True)

# 2) historical/markets per event -> jsonl cache
mkfile=f"{OUT}/markets_api.jsonl"
done=set()
if os.path.exists(mkfile):
    for line in open(mkfile):
        try: done.add(json.loads(line)["event_ticker"])
        except: pass
print(f"already cached: {len(done)} events", flush=True)

with open(mkfile,"a") as f:
    for i,(s,ev) in enumerate(allev):
        if ev in done: continue
        d=get(f"{B}/historical/markets", {"event_ticker":ev,"limit":500})
        ms=(d or {}).get("markets",[])
        for m in ms:
            m["_series"]=s; m["_event"]=ev
            f.write(json.dumps(m)+"\n")
        if not ms:
            f.write(json.dumps({"_series":s,"_event":ev,"event_ticker":ev,"_empty":True})+"\n")
        if i%100==0:
            f.flush(); print(f"[{i}/{len(allev)}] {ev}: {len(ms)}", flush=True)
        time.sleep(0.08)
print("DONE", flush=True)
