import polars as pl, json, re, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"

# ---- load local market metadata (already verified: 99% yes_sub_title parseable) ----
mkts = pl.read_parquet("data/markets/*.parquet").sort("_fetched_at",descending=True).unique(subset=["ticker"],keep="first")

NUM=re.compile(r'(-?[\d,]+\.?\d*)')
def parse_thr(s):
    if s is None: return None
    m=NUM.search(s.replace(',',''))
    return float(m.group(1)) if m else None

mkts = mkts.with_columns(
    pl.col("ticker").str.replace(r"^KX","").str.split("-").list.first().alias("series"),
    pl.col("ticker").str.replace(r"^KX","").str.splitn("-",3).struct.field("field_1").alias("event_period"),
    pl.col("yes_sub_title").map_elements(parse_thr, return_dtype=pl.Float64).alias("threshold"),
)

# expiration_value from A2 raw pull, where available
expv={}
for line in open(f"{SC}/markets_api.jsonl"):
    d=json.loads(line)
    if d.get("_empty"): continue
    if d.get("expiration_value") not in (None,""):
        try: expv[d["ticker"]] = float(str(d["expiration_value"]).replace(",",""))
        except: pass

# ---- CPI trigger events: resolution date + surprise ----
cpi = mkts.filter(pl.col("series")=="CPI").filter(pl.col("status")=="finalized").filter(pl.col("threshold").is_not_null())
cpi_events = cpi["event_ticker"].unique().to_list()
print(f"CPI finalized events: {len(cpi_events)}")

trades = pl.scan_parquet("data/trades/*.parquet")

rows=[]
for ev in cpi_events:
    sub = cpi.filter(pl.col("event_ticker")==ev)
    close_time = sub["close_time"].min()
    if close_time is None: continue
    tks = sub["ticker"].to_list()
    pre = (trades.filter(pl.col("ticker").is_in(tks) & (pl.col("created_time") < close_time))
           .sort("created_time").group_by("ticker").agg(pl.col("yes_price").last().alias("last_px"), pl.col("created_time").last().alias("t"))
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
        if yes_thr.len() and no_thr.len():
            resolved = (yes_thr.max()+no_thr.min())/2
        elif yes_thr.len(): resolved = yes_thr.max()+0.05
        elif no_thr.len(): resolved = no_thr.min()-0.05
    if resolved is None: continue

    rows.append({"event":ev,"close_time":close_time,"implied_guess":implied_guess,"resolved":resolved,
                 "surprise": resolved-implied_guess})

surp = pl.DataFrame(rows).sort("close_time")
print(f"CPI events with surprise computed: {surp.height}")
surp.write_parquet(f"{SC}/cpi_surprise.parquet")
with pl.Config(tbl_rows=60):
    print(surp.select("event","close_time","implied_guess","resolved","surprise"))
