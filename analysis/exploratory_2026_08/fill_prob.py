import json, polars as pl, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"
TK="KXFEDDECISION-26JAN-H0"

# ---- order book snapshots: best yes bid + resting size at that price ----
obs=[]
with open(f"{SC}/feddecision_ob.jsonl") as f:
    for line in f:
        if TK not in line: continue
        d=json.loads(line)
        if d["ticker"]!=TK: continue
        ob=d["orderbook"]
        yes = ob.get("yes",[])
        if not yes: continue
        yb, ybsize = max(yes, key=lambda x:x[0])
        obs.append({"ts":d["timestamp"], "best_yes_bid":yb, "resting_at_bid":ybsize})
ob_df = pl.DataFrame(obs).with_columns(pl.from_epoch("ts",time_unit="ms").alias("t")).sort("t")

# downsample to hourly candidate "I place my order now" moments
ob_df = ob_df.with_columns(pl.col("t").dt.truncate("1h").alias("hour")).unique(subset=["hour"],keep="first").sort("t")
print(f"candidate hourly order-placement moments: {ob_df.height}")

# ---- matched trades ----
trades = (pl.scan_parquet("data/trades/*.parquet")
          .filter(pl.col("ticker")==TK)
          .select("yes_price","count","taker_side","created_time")
          .sort("created_time").collect())
trades = trades.with_columns((pl.col("created_time").dt.epoch("ms")).alias("ts_ms"))
print(f"trades on this ticker: {trades.height}")

# a resting BUY-YES order gets filled by a taker who is effectively selling yes,
# i.e. taker_side == 'no', executing AT OR BELOW our limit price
fills = trades.filter(pl.col("taker_side")=="no")
print(f"'no'-taker trades (fill candidates): {fills.height}")

HORIZONS = {"1h":1, "1d":24, "3d":72, "7d":168, "end_of_window":None}
results=[]
for row in ob_df.iter_rows(named=True):
    t0 = row["t"]; P = row["best_yes_bid"]; ahead = row["resting_at_bid"]
    t0_ms = int(t0.timestamp()*1000)
    for hname,hh in HORIZONS.items():
        t1_ms = t0_ms + int(hh*3600*1000) if hh else int(dt.datetime(2100,1,1,tzinfo=dt.timezone.utc).timestamp()*1000)
        w = fills.filter((pl.col("ts_ms")>t0_ms)&(pl.col("ts_ms")<=t1_ms)&(pl.col("yes_price")<=P))
        optimistic = w.height>0
        cum_vol = w["count"].sum() if w.height else 0
        conservative = cum_vol >= ahead
        results.append({"t0":t0,"P":P,"ahead":ahead,"horizon":hname,"optimistic_fill":optimistic,"conservative_fill":conservative})

res = pl.DataFrame(results)
print("\nFill probability by horizon (join-the-touch order, hourly candidate placements):")
print(f"{'horizon':<15}{'n':>6}{'optimistic P(fill)':>20}{'conservative P(fill)':>22}")
for h in HORIZONS:
    sub = res.filter(pl.col("horizon")==h)
    print(f"{h:<15}{sub.height:>6}{sub['optimistic_fill'].mean():>20.1%}{sub['conservative_fill'].mean():>22.1%}")

res.write_parquet(f"{SC}/fill_prob_results.parquet")
print("\nmedian resting size at touch when placing:", ob_df["resting_at_bid"].median())
print("median 'no'-taker trade size:", fills["count"].median())
