"""Are the profits of positions stuck to settlement statistically real, and where do they come from?
For trades whose exit never filled (held to the game's result), split side x (100 y - entry) into
    drift     side x (p_stuck - entry): entry to the last print when we gave up exiting (fill + hold +
              2 x EXIT_S, as in b7_maker_trades)
    result    side x (100 y - p_stuck): from that market price to the result
If the market prices the final result fairly, "result" averages zero; t clustered by game.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_stuck_test.py
    -> out/b9_stuck_test.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                 # noqa: E402
from b6_maker_exit import EXIT_S                     # noqa: E402
from b7_maker_robust import with_net, OUT            # noqa: E402
from stg.sports.loaders import scan_trades           # noqa: E402

HOLDS = {"60s": 60, "120s": 120}
RUNS = [("current, all folds", "NBA", d, "_h30-60-120-1800_tpf", None) for d in (0, 5, 10)] + \
       [("current, folds 3-5", "NBA", 10, "_h30-60-120-1800_tpf", [3, 4, 5]),
        ("mid Huber 5 [vs last print], folds 3-5", "NBA", 10, "_h30-60-120-1800_mid", [3, 4, 5])]


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    rows = []
    for name, L, d, tag, folds in RUNS:
        f = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d{d}{tag}.parquet").filter(
            pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s")
        if folds:
            f = f.filter(pl.col("fold").is_in(folds))
        if "mid" in tag:      # the cut-off giving the current model's trade count on these folds
            n = pl.read_parquet(OUT / f"b7_maker_trades_{L}_d{d}_h30-60-120-1800_tpf.parquet").filter(
                pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s",
                pl.col("fold").is_in(folds), pl.col("signal").abs() >= 8).height
            f = f.sort(pl.col("signal").abs(), descending=True).head(n)
        else:
            f = f.filter(pl.col("signal").abs() >= 8)
        f = with_net(f)
        s = f.filter(pl.col("how") == "settled").with_columns(
            (pl.col("t_fill") + pl.col("hold").replace_strict(HOLDS) + 2 * EXIT_S).alias("t_stuck"))
        series = s.select(pl.col("ticker").str.extract(r"^([^-]+)")).to_series().unique().to_list()
        tr = (scan_trades(series).filter(pl.col("ticker").is_in(s["ticker"].unique().implode()))
              .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t_stuck"),
                      pl.col("yes_price").alias("p_stuck")).collect().sort("t_stuck"))
        s = s.sort("t_stuck").join_asof(tr, on="t_stuck", by="ticker", strategy="backward")
        s = s.with_columns((pl.col("side") * (pl.col("p_stuck") - pl.col("entry"))).alias("drift"),
                           (pl.col("side") * (100 * pl.col("y") - pl.col("p_stuck"))).alias("result"))
        g = s["game_id"].to_numpy()
        t = lambda c: round(clustered_t(s[c].to_numpy(), g), 1)   # noqa: E731
        rows.append(dict(run=name, delay=d, all_trades=f.height, stuck=s.height, stuck_net=round(s["net"].mean(), 2), t_net=t("net"),
                         drift=round(s["drift"].mean(), 2), t_drift=t("drift"), result=round(s["result"].mean(), 2), t_result=t("result"),
                         won=round((s["result"] + s["drift"] > 0).mean(), 2), mean_entry=round(s["entry"].mean(), 1)))
    text = ("NBA spread, 120 s hold, >= 8c (mid: matched count): positions stuck to settlement; cents per contract\n"
            + str(pl.DataFrame(rows)))
    (OUT / "b9_stuck_test.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
