"""Is the maker strategy's profit driven by liquid contracts, illiquid ones, or neither? From the
per-trade logs of b7_maker_trades.py (fixed exit, offset 1c, strictly-through fills, |signal| >= M).

Liquidity of the contract at the decision time t, from trade prints known by then only:
    n_5m       prints in (t - 300 s, t]
    vol_5m     contracts traded in (t - 300 s, t]
    age_s      seconds since the last print
    width      last YES-taker price - last NO-taker price, each within 120 s (a stand-in for the
               bid-ask spread; null if either side is missing)
Each strategy is split into quintiles of each measure (pooled over delays, so the cut points are the
same at every delay), with net cents per contract, t clustered by game, and the share of trades
stuck to settlement. Then: net when only trading contracts above each n_5m cut-off.

    venv/bin/python analysis/graph_2026_10/b7_maker_liquidity.py
    -> out/b7_maker_liquidity.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.argv = sys.argv[:1] + ["10"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b6_maker_exit import sweep_sorted                               # noqa: E402
from b7_maker_robust import with_net, summ, OUT                     # noqa: E402
from stg.sports.loaders import scan_trades                          # noqa: E402

TAG = "_h30-60-120-1800_tpf"
STRATS = [("NBA", "spread", "120s"), ("NBA", "total", "120s"), ("MLB", "total", "60s")]
DELAYS = [0, 5, 10]
M = 8.0
MEASURES = ["n_5m", "vol_5m", "age_s", "width"]
N_CUTS = [0, 2, 5, 10, 20]


def liquidity(f: pl.DataFrame) -> pl.DataFrame:
    tk = f["ticker"].unique()
    series = f.select(pl.col("ticker").str.extract(r"^([^-]+)")).to_series().unique().to_list()
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "count", "taker_side",
                  "trade_id")
          .collect().pipe(sweep_sorted)   # same-timestamp prints in sweep order: the last one per side is the sweep's end
          .with_columns(pl.int_range(1, pl.len() + 1).over("ticker").alias("ci"), pl.col("count").cum_sum().over("ticker").alias("cv")))
    now = tr.select("ticker", "t", "ci", "cv", pl.col("t").alias("t_last"))
    ago = tr.select("ticker", (pl.col("t") + 300).alias("t"), pl.col("ci").alias("ci5"), pl.col("cv").alias("cv5"))
    yes = tr.filter(pl.col("taker_side") == "yes").select("ticker", "t", pl.col("yes_price").alias("p_yes"))
    no = tr.filter(pl.col("taker_side") == "no").select("ticker", "t", pl.col("yes_price").alias("p_no"))
    f = f.sort("t", maintain_order=True)
    f = f.join_asof(now.sort("t", maintain_order=True), on="t", by="ticker", strategy="backward")
    f = f.join_asof(ago.sort("t", maintain_order=True), on="t", by="ticker", strategy="backward")
    f = f.join_asof(yes.sort("t", maintain_order=True), on="t", by="ticker", strategy="backward", tolerance=120.0)
    f = f.join_asof(no.sort("t", maintain_order=True), on="t", by="ticker", strategy="backward", tolerance=120.0)
    return f.with_columns((pl.col("ci").fill_null(0) - pl.col("ci5").fill_null(0)).alias("n_5m"),
                          (pl.col("cv").fill_null(0) - pl.col("cv5").fill_null(0)).alias("vol_5m"),
                          (pl.col("t") - pl.col("t_last")).alias("age_s"),
                          (pl.col("p_yes") - pl.col("p_no")).alias("width"))


def main() -> None:
    pl.Config.set_tbl_rows(400)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    lines = []
    out = lambda s: (print(s), lines.append(str(s)))   # noqa: E731
    for L, kind, hold in STRATS:
        parts = []
        for d in DELAYS:
            f = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d{d}{TAG}.parquet").filter(
                pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == kind, pl.col("hold") == hold, pl.col("signal").abs() >= M)
            parts.append(with_net(f).with_columns(pl.lit(d).alias("delay")))
        f = liquidity(pl.concat(parts, how="diagonal_relaxed"))
        out(f"\n================ {L} {kind}, hold {hold}, |signal| >= {M:g}c")
        out(f.select([pl.col(c).quantile(q).round(1).alias(f"{c}_q{int(q * 100)}") for c in MEASURES for q in (0.2, 0.5, 0.8)]))
        for meas in MEASURES:
            g0 = f.filter(pl.col(meas).is_not_null())
            cuts = sorted(set(g0[meas].quantile(q) for q in (0.2, 0.4, 0.6, 0.8)))
            edges = [g0[meas].min()] + cuts + [g0[meas].max()]
            labels = [f"{i + 1}: {edges[i]:g}..{edges[i + 1]:g}" for i in range(len(cuts) + 1)]
            g0 = g0.with_columns(pl.col(meas).cut(cuts, labels=labels, left_closed=True).cast(pl.String).alias("bucket"))
            rows = []
            for d in DELAYS:
                for (b,), g in sorted(g0.filter(pl.col("delay") == d).group_by("bucket"), key=lambda x: int(x[0][0].split(":")[0])):
                    rows.append(summ(g, delay=d, **{meas: str(b)}) | dict(stuck=round((g["how"] == "settled").mean(), 2),
                                                                         net_if_stuck_cost_taken=round(
                                                                             g.with_columns(pl.when(pl.col("how") == "settled").then(-3.5).otherwise(pl.col("net")).alias("x"))["x"].mean(), 2)))
            out(f"\n######## by {meas} quintile (stuck = share held to the end of the game; last column: stuck trades charged -3.5c instead)")
            out(pl.DataFrame(rows))
        rows = []
        for d in DELAYS:
            for c in N_CUTS:
                g = f.filter(pl.col("delay") == d, pl.col("n_5m") >= c)
                rows.append(summ(g, delay=d, min_prints_5m=c) | dict(stuck=round((g["how"] == "settled").mean(), 2)))
        out("\n######## filter: only contracts with at least N prints in the last 5 min")
        out(pl.DataFrame(rows))
    (OUT / "b7_maker_liquidity.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
