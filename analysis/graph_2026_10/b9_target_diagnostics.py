"""Do the fresh-mid-target models show the same structure as the current model? NBA spread, 120 s hold,
feed delay 10 s, folds 3-5. Each model on its own cut-off giving the current model's trade count at
>= 8c (all widths); then the same splits as b7_maker_robust / b7_maker_liquidity: market width, trades
in the last 5 min, how the exit happened, fill strictness and order offset.

    venv/bin/python -W ignore analysis/graph_2026_10/b9_target_diagnostics.py
    -> out/b9_target_diagnostics.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity            # noqa: E402
from b7_maker_robust import with_net, summ, OUT     # noqa: E402

MODELS = {"current": "b7_maker_trades_NBA_d10_h30-60-120-1800_tpf.parquet",
          "mid Huber 5": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid.parquet",
          "mid ~squared": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000.parquet",
          "mid ~sq no result": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000_ly0.parquet"}
KIND, HOLD = "spread", "120s"


def main() -> None:
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    raw = {m: pl.read_parquet(OUT / f).filter(pl.col("fold").is_in([3, 4, 5]), pl.col("kind") == KIND, pl.col("hold") == HOLD)
           for m, f in MODELS.items()}
    base = {m: f.filter(pl.col("k") == 1, pl.col("through") == 0) for m, f in raw.items()}
    n_ref = base["current"].filter(pl.col("signal").abs() >= 8).height
    cut = {m: float(f["signal"].abs().sort(descending=True)[n_ref - 1]) for m, f in base.items()}
    sel = {m: liquidity(with_net(f.filter(pl.col("signal").abs() >= cut[m]))) for m, f in raw.items()}
    lines = [f"NBA {KIND}, hold {HOLD}, 10 s delay, folds 3-5; cut-offs giving {n_ref} trades: "
             + ", ".join(f"{m} {c:.1f}c" for m, c in cut.items())]
    out = lambda title, rows: lines.extend([f"\n######## {title}", str(pl.DataFrame(rows))])   # noqa: E731
    b = {m: f.filter(pl.col("k") == 1, pl.col("through") == 0) for m, f in sel.items()}
    rows = []
    for m, f in b.items():
        for lab, c in [("< 4c", pl.col("width") < 4), ("4-9c", (pl.col("width") >= 4) & (pl.col("width") < 9)), (">= 9c", pl.col("width") >= 9)]:
            g = f.filter(c)
            rows.append(summ(g, model=m, width=lab) | dict(share=round(g.height / f.height, 2), stuck=round((g["how"] == "settled").mean(), 2)))
    out("by market width (last buyer-started minus last seller-started price)", rows)
    rows = []
    for m, f in b.items():
        for lab, c in [("<= 21", pl.col("n_5m") <= 21), ("22-202", (pl.col("n_5m") > 21) & (pl.col("n_5m") < 202)), (">= 202", pl.col("n_5m") >= 202)]:
            g = f.filter(c)
            x = g.with_columns(pl.when(pl.col("how") == "settled").then(-3.5).otherwise(pl.col("net")).alias("x"))
            rows.append(summ(g, model=m, trades_last_5m=lab) | dict(stuck=round((g["how"] == "settled").mean(), 2),
                                                                   net_if_stuck_cost_3_5c=round(x["x"].mean(), 2)))
    out("by trading activity (same bands as the current model's least / middle / most active fifths)", rows)
    rows = []
    for m, f in b.items():
        for (how,), g in sorted(f.group_by("how"), key=lambda x: x[0]):
            rows.append(summ(g, model=m, exit=how) | dict(share=round(g.height / f.height, 2)))
    out("how the exit happened", rows)
    rows = []
    for m, f in sel.items():
        for k, thr, lab in [(1, 0, "1c offset, strictly through"), (1, 1, "1c offset, 2c+ through"), (2, 0, "2c offset, strictly through")]:
            rows.append(summ(f.filter(pl.col("k") == k, pl.col("through") == thr), model=m, rule=lab))
    out("order offset and fill strictness", rows)
    rows = []
    for m, f in b.items():
        by = f.group_by("game_id").agg(pl.col("net").sum().alias("s")).sort("s", descending=True)
        top = by.head(by.height // 20)["game_id"]
        rows.append(dict(model=m, median_trade=round(f["net"].median(), 2), share_from_top10_games=round(by.head(10)["s"].sum() / by["s"].sum(), 2),
                         net_best_5pct_games_dropped=round(f.filter(~pl.col("game_id").is_in(top.implode()))["net"].mean(), 2)))
    out("concentration", rows)
    (OUT / "b9_target_diagnostics.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
