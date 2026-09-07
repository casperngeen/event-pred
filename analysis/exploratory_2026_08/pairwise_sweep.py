import polars as pl, re, math, datetime as dt

IS_CUTOFF = dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)  # hard OOS wall, explicit regardless of source file

NUM=re.compile(r'(-?[\d,]+\.?\d*)')
def parse_thr(s):
    if s is None: return None
    m=NUM.search(s.replace(',',''))
    return float(m.group(1)) if m else None

mkts = pl.read_parquet("data/markets/*.parquet").sort("_fetched_at",descending=True).unique(subset=["ticker"],keep="first")
mkts = mkts.filter(pl.col("close_time")<IS_CUTOFF)  # explicit OOS wall
mkts = mkts.with_columns(
    pl.col("ticker").str.replace(r"^KX","").str.split("-").list.first().alias("series"),
    pl.col("yes_sub_title").map_elements(parse_thr, return_dtype=pl.Float64).alias("threshold"),
)
trades_lf = pl.scan_parquet("data/trades/*.parquet").filter(pl.col("created_time")<IS_CUTOFF)

def build_surprise(series):
    ev = mkts.filter(pl.col("series")==series).filter(pl.col("status")=="finalized").filter(pl.col("threshold").is_not_null())
    events = ev["event_ticker"].unique().to_list()
    rows=[]
    for e in events:
        sub = ev.filter(pl.col("event_ticker")==e)
        close_time = sub["close_time"].min()
        if close_time is None: continue
        tks = sub["ticker"].to_list()
        pre = (trades_lf.filter(pl.col("ticker").is_in(tks) & (pl.col("created_time")<close_time))
               .sort("created_time").group_by("ticker").agg(pl.col("yes_price").last().alias("last_px")).collect())
        if pre.height==0: continue
        pre = pre.join(sub.select("ticker","threshold"), on="ticker").with_columns((pl.col("last_px")-50).abs().alias("dist"))
        implied_guess = pre.sort("dist").row(0,named=True)["threshold"]
        yes_thr = sub.filter(pl.col("result")=="yes")["threshold"]
        no_thr  = sub.filter(pl.col("result")=="no")["threshold"]
        if yes_thr.len() and no_thr.len(): resolved=(yes_thr.max()+no_thr.min())/2
        elif yes_thr.len(): resolved=yes_thr.max()+0.05
        elif no_thr.len(): resolved=no_thr.min()-0.05
        else: continue
        rows.append({"event":e,"close_time":close_time,"surprise":resolved-implied_guess})
    return pl.DataFrame(rows).sort("close_time") if rows else pl.DataFrame()

def representative_tickers(series):
    tev = mkts.filter(pl.col("series")==series).select("event_ticker","ticker","close_time").filter(pl.col("close_time").is_not_null())
    tks = tev["ticker"].unique().to_list()
    tc = trades_lf.filter(pl.col("ticker").is_in(tks)).group_by("ticker").agg(pl.len().alias("n")).collect()
    tev = tev.join(tc, on="ticker", how="left").with_columns(pl.col("n").fill_null(0))
    rep = (tev.sort("n",descending=True).group_by("event_ticker")
           .agg(pl.col("ticker").first().alias("rep_ticker"), pl.col("close_time").first().alias("meeting_time"), pl.col("n").first()))
    return rep.filter(pl.col("n")>0).sort("meeting_time")

def run_pair(trigger, target, max_gap_days=45, horizons_tr=(1,3,5,10)):
    surp = build_surprise(trigger)
    if surp.height==0: return None
    rep = representative_tickers(target)
    if rep.height==0: return None
    meeting_map = list(zip(rep["meeting_time"].to_list(), rep["rep_ticker"].to_list()))
    def next_meeting(t):
        cands=[(m,tk) for m,tk in meeting_map if m>t]
        return min(cands, key=lambda x:x[0]) if cands else (None,None)
    recs=[]
    for row in surp.iter_rows(named=True):
        m,tk = next_meeting(row["close_time"])
        recs.append({**row,"target_meeting":m,"target_ticker":tk})
    pairs = pl.DataFrame(recs).filter(pl.col("target_ticker").is_not_null())
    if pairs.height==0: return None
    pairs = pairs.with_columns(((pl.col("target_meeting")-pl.col("close_time")).dt.total_hours()/24).alias("gap"))
    pairs = pairs.filter((pl.col("gap")<=max_gap_days)&(pl.col("gap")>=0))
    if pairs.height<8: return {"trigger":trigger,"target":target,"n":pairs.height,"note":"too few pairs"}

    target_tickers = pairs["target_ticker"].unique().to_list()
    tt = (trades_lf.filter(pl.col("ticker").is_in(target_tickers)).select("ticker","yes_price","created_time")
          .sort(["ticker","created_time"]).collect())
    results=[]
    for row in pairs.iter_rows(named=True):
        tk=row["target_ticker"]; t0=row["close_time"]; surprise=row["surprise"]
        sub = tt.filter(pl.col("ticker")==tk)
        pre = sub.filter(pl.col("created_time")<=t0)
        if pre.height==0: continue
        p0 = pre["yes_price"][-1]
        post = sub.filter(pl.col("created_time")>t0)
        rec={"surprise":surprise}
        for k in horizons_tr:
            rec[f"tt_{k}"] = (post["yes_price"][k-1]-p0) if post.height>=k else None
        results.append(rec)
    res = pl.DataFrame(results).with_columns(pl.col("surprise").abs().alias("abs_surprise"))
    out = {"trigger":trigger,"target":target,"n_pairs":pairs.height,"n_resp":res.height,"horizons":{}}
    for k in horizons_tr:
        c=f"tt_{k}"
        sub = res.filter(pl.col(c).is_not_null()).with_columns(pl.col(c).abs().alias("abs_resp"))
        n=sub.height
        if n<8: out["horizons"][c]=None; continue
        r = sub.select(pl.corr("abs_surprise","abs_resp")).item()
        t = r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r*r)) if r is not None else None
        out["horizons"][c] = {"n":n,"r":r,"t":t}
    return out

PAIRS = [
    ("CPI","FEDDECISION"), ("CPIYOY","FEDDECISION"), ("U3","FEDDECISION"), ("PAYROLLS","FEDDECISION"),
    ("PAYROLLS","GDP"), ("U3","GDP"), ("WTI","CPI"), ("WTI","FED"), ("CPI","FED"),
]

print(f"{'trigger->target':<20}{'n_pairs':>9}", "".join(f"{'r@tt'+str(k):>12}" for k in (1,3,5,10)))
all_out=[]
for trig,tgt in PAIRS:
    o = run_pair(trig,tgt)
    all_out.append(o)
    if o is None:
        print(f"{trig+'->'+tgt:<20}{'--':>9}  (no data)")
        continue
    if "note" in o:
        print(f"{trig+'->'+tgt:<20}{o['n']:>9}  ({o['note']})")
        continue
    line = f"{trig+'->'+tgt:<20}{o['n_pairs']:>9}"
    for k in (1,3,5,10):
        h = o["horizons"].get(f"tt_{k}")
        line += f"{h['r']:>12.3f}" if h else f"{'--':>12}"
    print(line)

import json
with open("/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2/pairwise_results.json","w") as f:
    json.dump([o for o in all_out if o], f, default=str, indent=1)
