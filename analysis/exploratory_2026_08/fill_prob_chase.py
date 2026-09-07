import json, polars as pl, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"
TK="KXFEDDECISION-26JAN-H0"

# ---- full-resolution order book: best yes bid + resting size, every snapshot ----
obs=[]
with open(f"{SC}/feddecision_ob.jsonl") as f:
    for line in f:
        if TK not in line: continue
        d=json.loads(line)
        if d["ticker"]!=TK: continue
        yes = d["orderbook"].get("yes",[])
        if not yes: continue
        yb, ybsize = max(yes, key=lambda x:x[0])
        obs.append({"ts":d["timestamp"], "P":yb, "resting":ybsize})
ob = pl.DataFrame(obs).sort("ts")

# collapse to touch-shift segments: each time best_yes_bid changes, that's a new cancel-replace
ob = ob.with_columns((pl.col("P")!=pl.col("P").shift(1)).fill_null(True).alias("is_new"))
ob = ob.with_columns(pl.col("is_new").cum_sum().alias("segment_id"))
segs = (ob.group_by("segment_id").agg(
    pl.col("P").first().alias("P"),
    pl.col("resting").first().alias("ahead"),   # depth already resting when we join (back of queue)
    pl.col("ts").min().alias("t0_ms"),
    pl.col("ts").max().alias("t_last_ms"),
).sort("t0_ms"))
# segment end = start of NEXT segment (that's when we'd cancel/replace again)
segs = segs.with_columns(pl.col("t0_ms").shift(-1).alias("t1_ms"))
segs = segs.filter(pl.col("t1_ms").is_not_null())  # drop the still-open final segment

print(f"total touch-shift segments: {segs.height}")
dur_s = (segs["t1_ms"]-segs["t0_ms"])/1000
print(f"segment duration (sec): median={dur_s.median():.1f}  mean={dur_s.mean():.1f}  p90={dur_s.quantile(0.9):.1f}")

# ---- trades (fills come from 'no'-taker prints at/through our current price) ----
trades = (pl.scan_parquet("data/trades/*.parquet")
          .filter(pl.col("ticker")==TK)
          .select("yes_price","count","taker_side","created_time")
          .filter(pl.col("taker_side")=="no")
          .sort("created_time").collect())
trades = trades.with_columns((pl.col("created_time").dt.epoch("ms")).alias("ts_ms"))
tr_ts = trades["ts_ms"].to_list(); tr_px = trades["yes_price"].to_list(); tr_ct = trades["count"].to_list()

opt_fills=0; cons_fills=0; n=segs.height
for row in segs.iter_rows(named=True):
    t0,t1,P,ahead = row["t0_ms"],row["t1_ms"],row["P"],row["ahead"]
    vol=0; filled_opt=False
    for ts,px,ct in zip(tr_ts,tr_px,tr_ct):
        if ts<=t0 or ts>t1: continue
        if px<=P:
            filled_opt=True
            vol+=ct
    if filled_opt: opt_fills+=1
    if vol>=ahead and ahead>0: cons_fills+=1
    elif ahead==0 and filled_opt: cons_fills+=1  # nothing ahead of us -> any fill counts

print(f"\n'chase the touch' (cancel-replace on every shift), per-segment fill rate:")
print(f"  segments: {n}")
print(f"  optimistic P(fill within segment):   {opt_fills/n:.1%}")
print(f"  conservative P(fill within segment): {cons_fills/n:.1%}")

# time-weighted version: fraction of TOTAL RESTING TIME during which a fill occurred in that segment
seg_dur = (segs["t1_ms"]-segs["t0_ms"]).to_list()
total_time = sum(seg_dur)
opt_time=0; cons_time=0
for row,d_ in zip(segs.iter_rows(named=True), seg_dur):
    t0,t1,P,ahead = row["t0_ms"],row["t1_ms"],row["P"],row["ahead"]
    vol=0; filled_opt=False
    for ts,px,ct in zip(tr_ts,tr_px,tr_ct):
        if ts<=t0 or ts>t1: continue
        if px<=P:
            filled_opt=True; vol+=ct
    if filled_opt: opt_time+=d_
    if (ahead>0 and vol>=ahead) or (ahead==0 and filled_opt): cons_time+=d_
print(f"\n  time-weighted optimistic:   {opt_time/total_time:.1%} of resting time ends in a fill during that segment")
print(f"  time-weighted conservative: {cons_time/total_time:.1%}")
