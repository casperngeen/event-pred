import polars as pl, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"

pairs = pl.read_parquet(f"{SC}/cpi_fed_pairs_tight.parquet")
trades_lf = pl.scan_parquet("data/trades/*.parquet")

target_tickers = pairs["target_ticker"].unique().to_list()
tt = (trades_lf.filter(pl.col("ticker").is_in(target_tickers))
      .select("ticker","yes_price","created_time")
      .sort(["ticker","created_time"]).collect())

results=[]
for row in pairs.iter_rows(named=True):
    tk = row["target_ticker"]; t0 = row["close_time"]; surprise = row["surprise"]
    sub = tt.filter(pl.col("ticker")==tk)
    if sub.height < 2: continue
    pre = sub.filter(pl.col("created_time")<=t0)
    if pre.height==0: continue
    p0 = pre["yes_price"][-1]
    post = sub.filter(pl.col("created_time")>t0)

    rec = {"event":row["event"],"target":tk,"surprise":surprise,"p0":p0,"n_post_trades":post.height}
    # calendar-time (last trade at/before horizon) -- what naive forward-fill approach would use
    for days in (1,3,7,14):
        h = t0 + dt.timedelta(days=days)
        w = sub.filter(pl.col("created_time")<=h)
        rec[f"cal_{days}d"] = (w["yes_price"][-1]-p0) if w.height>0 else None
    # trade-time (Kth trade strictly after t0)
    for k in (1,3,5,10):
        rec[f"tt_{k}tr"] = (post["yes_price"][k-1]-p0) if post.height>=k else None
    results.append(rec)

res = pl.DataFrame(results)
res.write_parquet(f"{SC}/cpi_fed_response.parquet")
print(f"pairs with computable response: {res.height}")
with pl.Config(tbl_rows=40, tbl_cols=20):
    print(res)
