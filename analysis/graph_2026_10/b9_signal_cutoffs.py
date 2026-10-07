"""Does net per trade rise with the signal cut-off for both models? NBA spreads, 120 s hold, feed delay
10 s, test folds 3-5, fixed exit, offset 1c, strictly-through fills. Current model (trade-price targets)
vs mid Huber 5 (signal = predicted future mid - last print); entry filters all / fresh mid / width >= 9c.
Cumulative (|signal| >= cut) and in bands (cut <= |signal| < next cut).

    venv/bin/python -W ignore analysis/graph_2026_10/b9_signal_cutoffs.py
    -> out/b9_signal_cutoffs.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_backtest import clustered_t                     # noqa: E402
from b7_maker_robust import with_net, OUT                # noqa: E402
from b9_candidates_by_model import entry_state          # noqa: E402
from stg.sports.loaders import scan_trades               # noqa: E402

MODELS = {"current": "_h30-60-120-1800_tpf", "mid Huber 5": "_h30-60-120-1800_mid"}
FILTERS = {"all": pl.lit(True), "fresh mid": pl.col("older_s") <= 30, "width >= 9c": pl.col("width") >= 9}
CUTS = [2, 3, 5, 8, 11, 15]


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_rows(80)
    raw = {m: pl.read_parquet(OUT / f"b7_maker_trades_NBA_d10{t}.parquet").filter(
        pl.col("k") == 1, pl.col("through") == 0, pl.col("kind") == "spread", pl.col("hold") == "120s",
        pl.col("fold").is_in([3, 4, 5]), pl.col("signal").abs() >= 2) for m, t in MODELS.items()}
    tk = pl.concat([f.select("ticker") for f in raw.values()]).unique()["ticker"]
    tr = (scan_trades(tk.str.extract(r"^([^-]+)").unique().to_list()).filter(pl.col("ticker").is_in(tk.implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side").collect())
    rows = []
    for m, f in raw.items():
        f = entry_state(with_net(f), tr)
        for fn, cond in FILTERS.items():
            g0 = f.filter(cond)
            for i, c in enumerate(CUTS):
                hi = CUTS[i + 1] if i + 1 < len(CUTS) else 1e9
                r = dict(model=m, entry=fn, cut=c)
                for lab, g in (("cum", g0.filter(pl.col("signal").abs() >= c)),
                               ("band", g0.filter(pl.col("signal").abs() >= c, pl.col("signal").abs() < hi))):
                    ok = g.height >= 20
                    r |= {f"{lab}_trades": g.height, f"{lab}_net": round(g["net"].mean(), 2) if ok else None,
                          f"{lab}_t": round(clustered_t(g["net"].to_numpy(), g["game_id"].to_numpy()), 1) if ok else None}
                    if lab == "band":
                        r["band_stuck"] = round((g["how"] == "settled").mean(), 2) if ok else None
                rows.append(r)
    t = pl.DataFrame(rows)
    text = ("NBA spreads, 120 s hold, 10 s delay, folds 3-5: net cents per contract by signal cut-off. cum = |signal| >= cut; "
            "band = cut <= |signal| < next cut\n" + str(t))
    (OUT / "b9_signal_cutoffs.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
