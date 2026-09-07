import polars as pl, json, re, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"

# ---- combined trades: original archive + the 2026 backfill pulled earlier this session ----
trades_lf = pl.concat([
    pl.scan_parquet("data/trades/*.parquet").select("ticker","yes_price","taker_side","created_time"),
    pl.scan_parquet(f"{SC}/new_trades_2026.parquet").select("ticker","yes_price","taker_side","created_time"),
]).unique(subset=["ticker","created_time","yes_price","taker_side"])

# ---- combined market metadata: local parquet + A2 raw pull (covers into 2026) ----
mkts_local = pl.read_parquet("data/markets/*.parquet").sort("_fetched_at",descending=True).unique(subset=["ticker"],keep="first")
local_tickers = set(mkts_local["ticker"].to_list())

extra=[]
for line in open(f"{SC}/markets_api.jsonl"):
    d=json.loads(line)
    if d.get("_empty") or d.get("ticker") in local_tickers: continue
    extra.append(d)
print(f"extra tickers from A2 pull not already local: {len(extra)}")

def parse_close(s):
    if not s: return None
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc) if "." not in s else \
           dt.datetime.strptime(s.split(".")[0]+"Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)

extra_rows=[{"ticker":d["ticker"],"event_ticker":d["event_ticker"],"yes_sub_title":d.get("yes_sub_title"),
             "status":d.get("status"),"result":d.get("result"),"close_time":parse_close(d.get("close_time"))}
            for d in extra]
extra_df = pl.DataFrame(extra_rows) if extra_rows else None

NUM=re.compile(r'(-?[\d,]+\.?\d*)')
def parse_thr(s):
    if s is None: return None
    m=NUM.search(s.replace(',',''))
    return float(m.group(1)) if m else None

base = mkts_local.select("ticker","event_ticker","yes_sub_title","status","result","close_time").with_columns(pl.col("close_time").cast(pl.Datetime("us","UTC")))
if extra_df is not None:
    extra_df = extra_df.with_columns(pl.col("close_time").cast(pl.Datetime("us","UTC")))
mkts = pl.concat([base, extra_df], how="diagonal") if extra_df is not None else base
mkts = mkts.with_columns(
    pl.col("ticker").str.replace(r"^KX","").str.split("-").list.first().alias("series"),
    pl.col("yes_sub_title").map_elements(parse_thr, return_dtype=pl.Float64).alias("threshold"),
)
print(f"combined market rows: {mkts.height}")

expv={}
for line in open(f"{SC}/markets_api.jsonl"):
    d=json.loads(line)
    if d.get("_empty"): continue
    if d.get("expiration_value") not in (None,""):
        try: expv[d["ticker"]] = float(str(d["expiration_value"]).replace(",",""))
        except: pass

cpi = mkts.filter(pl.col("series")=="CPI").filter(pl.col("status")=="finalized").filter(pl.col("threshold").is_not_null())
cpi_events = cpi["event_ticker"].unique().to_list()
print(f"CPI finalized events (pooled): {len(cpi_events)}")

trades = trades_lf
rows=[]
for ev in cpi_events:
    sub = cpi.filter(pl.col("event_ticker")==ev)
    close_time = sub["close_time"].min()
    if close_time is None: continue
    tks = sub["ticker"].to_list()
    pre = (trades.filter(pl.col("ticker").is_in(tks) & (pl.col("created_time") < close_time))
           .sort("created_time").group_by("ticker").agg(pl.col("yes_price").last().alias("last_px"))
           .collect())
    if pre.height==0: continue
    pre = pre.join(sub.select("ticker","threshold"), on="ticker")
    pre = pre.with_columns((pl.col("last_px")-50).abs().alias("dist"))
    atm = pre.sort("dist").row(0, named=True)
    implied_guess = atm["threshold"]
    resolved=None
    for t in tks:
        if t in expv: resolved=expv[t]; break
    if resolved is None:
        yes_thr = sub.filter(pl.col("result")=="yes")["threshold"]
        no_thr  = sub.filter(pl.col("result")=="no")["threshold"]
        if yes_thr.len() and no_thr.len(): resolved = (yes_thr.max()+no_thr.min())/2
        elif yes_thr.len(): resolved = yes_thr.max()+0.05
        elif no_thr.len(): resolved = no_thr.min()-0.05
    if resolved is None: continue
    rows.append({"event":ev,"close_time":close_time,"implied_guess":implied_guess,"resolved":resolved,"surprise":resolved-implied_guess})

surp = pl.DataFrame(rows).sort("close_time")
print(f"CPI events with surprise computed (pooled): {surp.height}")

# ---- pair to next FEDDECISION meeting ----
fed_events = mkts.filter(pl.col("series")=="FEDDECISION").select("event_ticker","ticker","close_time").filter(pl.col("close_time").is_not_null())
fed_tickers = fed_events["ticker"].unique().to_list()
tcounts = trades.filter(pl.col("ticker").is_in(fed_tickers)).group_by("ticker").agg(pl.len().alias("n")).collect()
fed_events = fed_events.join(tcounts, on="ticker", how="left").with_columns(pl.col("n").fill_null(0))
rep = (fed_events.sort("n", descending=True).group_by("event_ticker")
       .agg(pl.col("ticker").first().alias("rep_ticker"), pl.col("close_time").first().alias("meeting_time"), pl.col("n").first()))
rep = rep.filter(pl.col("n")>0).sort("meeting_time")
meeting_map = list(zip(rep["meeting_time"].to_list(), rep["rep_ticker"].to_list()))
def next_meeting(t):
    cands=[(m,tk) for m,tk in meeting_map if m>t]
    return min(cands, key=lambda x:x[0]) if cands else (None,None)

recs=[]
for row in surp.iter_rows(named=True):
    m,tk = next_meeting(row["close_time"])
    recs.append({**row, "target_meeting":m, "target_ticker":tk})
pairs = pl.DataFrame(recs).filter(pl.col("target_ticker").is_not_null())
pairs = pairs.with_columns(((pl.col("target_meeting")-pl.col("close_time")).dt.total_hours()/24).alias("days_to_meeting"))
tight = pairs.filter(pl.col("days_to_meeting")<=45)
print(f"pooled pairs within 45-day next-meeting window: {tight.height} (was 33 before pooling)")
tight.write_parquet(f"{SC}/pooled_pairs.parquet")

# ---- response ----
target_tickers = tight["target_ticker"].unique().to_list()
tt = (trades.filter(pl.col("ticker").is_in(target_tickers)).select("ticker","yes_price","created_time")
      .sort(["ticker","created_time"]).collect())

results=[]
for row in tight.iter_rows(named=True):
    tk=row["target_ticker"]; t0=row["close_time"]; surprise=row["surprise"]
    sub = tt.filter(pl.col("ticker")==tk)
    pre = sub.filter(pl.col("created_time")<=t0)
    if pre.height==0: continue
    p0 = pre["yes_price"][-1]
    post = sub.filter(pl.col("created_time")>t0)
    rec={"event":row["event"],"surprise":surprise}
    for days in (1,3,7,14):
        h=t0+dt.timedelta(days=days)
        w=sub.filter(pl.col("created_time")<=h)
        rec[f"cal_{days}d"]=(w["yes_price"][-1]-p0) if w.height>0 else None
    for k in (1,3,5,10):
        rec[f"tt_{k}tr"]=(post["yes_price"][k-1]-p0) if post.height>=k else None
    results.append(rec)

res = pl.DataFrame(results).with_columns(pl.col("surprise").abs().alias("abs_surprise"))
print(f"\npooled pairs with computable response: {res.height}")
cols=["cal_1d","cal_3d","cal_7d","cal_14d","tt_1tr","tt_3tr","tt_5tr","tt_10tr"]
print(f"{'horizon':<10}{'n':>4}{'corr(|surp|,|resp|)':>22}{'t-stat':>10}")
import math
for c in cols:
    sub = res.filter(pl.col(c).is_not_null()).with_columns(pl.col(c).abs().alias("abs_resp"))
    n=sub.height
    if n<5: continue
    r = sub.select(pl.corr("abs_surprise","abs_resp")).item()
    t = r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r*r))
    print(f"{c:<10}{n:>4}{r:>22.3f}{t:>10.2f}")
res.write_parquet(f"{SC}/pooled_response.parquet")
