import polars as pl, datetime as dt
SC="/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2"

surp = pl.read_parquet(f"{SC}/cpi_surprise.parquet")
mkts = pl.read_parquet("data/markets/*.parquet").sort("_fetched_at",descending=True).unique(subset=["ticker"],keep="first")
mkts = mkts.with_columns(pl.col("ticker").str.replace(r"^KX","").str.split("-").list.first().alias("series"))
fed_events = mkts.filter(pl.col("series")=="FEDDECISION").select("event_ticker","ticker","volume","open_time","close_time").filter(pl.col("close_time").is_not_null())

# representative ticker per FEDDECISION event = highest local trade count
trades = pl.scan_parquet("data/trades/*.parquet")
fed_tickers = fed_events["ticker"].unique().to_list()
tcounts = (trades.filter(pl.col("ticker").is_in(fed_tickers))
           .group_by("ticker").agg(pl.len().alias("n"))
           .collect())
fed_events = fed_events.join(tcounts, on="ticker", how="left").with_columns(pl.col("n").fill_null(0))
rep = (fed_events.sort("n", descending=True).group_by("event_ticker")
       .agg(pl.col("ticker").first().alias("rep_ticker"), pl.col("close_time").first().alias("meeting_time"), pl.col("n").first()))
rep = rep.filter(pl.col("n")>0).sort("meeting_time")
print(f"FEDDECISION events with a representative liquid ticker: {rep.height}")

meeting_times = rep.select("meeting_time").to_series().to_list()
rep_tickers = rep.select("rep_ticker").to_series().to_list()
meeting_map = list(zip(meeting_times, rep_tickers))

def next_meeting(t):
    cands = [(m,tk) for m,tk in meeting_map if m > t]
    return min(cands, key=lambda x:x[0]) if cands else (None,None)

recs=[]
for row in surp.iter_rows(named=True):
    m,tk = next_meeting(row["close_time"])
    recs.append({**row, "target_meeting": m, "target_ticker": tk})
surp = pl.DataFrame(recs).filter(pl.col("target_ticker").is_not_null())
print(f"CPI events with a valid next-meeting FEDDECISION target: {surp.height}")
surp.write_parquet(f"{SC}/cpi_fed_pairs.parquet")
with pl.Config(tbl_rows=60, fmt_str_lengths=20):
    print(surp.select("event","close_time","surprise","target_ticker","target_meeting"))
