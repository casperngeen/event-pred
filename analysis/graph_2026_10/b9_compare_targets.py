"""Training-target test: the maker strategy with the current model (trade-price targets, zero loss inside
the range of later prints; signal = predicted level - last print) vs models trained on fresh mids
(panel_v4m; signal = predicted change of the mid, SIGNAL_FROM=anchor, or [vs last print] = predicted
future mid - last print, the current model's rule). NBA, feed delay 10 s, test
folds 3-5. Same trade logic (b7_maker_trades.py): fixed exit, offset 1c, strictly-through fills.

The mid models forecast smaller moves (no bid-ask bounce in them), so fixed cut-offs are not
comparable: each model is also shown on its own top-N signals, N = the current model's trade count
at >= 8c in that group ("matched").

    venv/bin/python -W ignore analysis/graph_2026_10/b9_compare_targets.py
    -> out/b9_compare_targets.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b7_maker_liquidity import liquidity            # noqa: E402
from b7_maker_robust import with_net, summ, OUT     # noqa: E402

MODELS = {"current (trade-price range)": "b7_maker_trades_NBA_d10_h30-60-120-1800_tpf.parquet",
          "mid, Huber 5": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_sa.parquet",
          "mid, ~squared": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000_sa.parquet",
          "mid, ~squared, no result loss": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000_ly0_sa.parquet",
          # the same models, signal = predicted future mid - last print (what a buy at the last print would gain)
          "mid, Huber 5 [vs last print]": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid.parquet",
          "mid, ~squared [vs last print]": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000.parquet",
          "mid, ~sq, no result [vs last print]": "b7_maker_trades_NBA_d10_h30-60-120-1800_mid_hub1000_ly0.parquet"}
STRATS = [("spread", "120s"), ("spread", "60s"), ("total", "120s"), ("total", "60s")]
CUTS = [3.0, 5.0, 8.0]
REF = "current (trade-price range)"


def main() -> None:
    pl.Config.set_tbl_rows(500)
    pl.Config.set_tbl_width_chars(230)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    logs = {}
    for name, fn in MODELS.items():
        f = pl.read_parquet(OUT / fn).filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("fold").is_in([3, 4, 5]))
        logs[name] = liquidity(with_net(f))
    rows = []
    for kind, hold in STRATS:
        for grp, cond in [("all", pl.lit(True)), ("width >= 9c", pl.col("width") >= 9)]:
            base = {name: f.filter(pl.col("kind") == kind, pl.col("hold") == hold, cond) for name, f in logs.items()}
            n_ref = base[REF].filter(pl.col("signal").abs() >= 8).height
            for name, g0 in base.items():
                r = dict(kind=kind, hold=hold, group=grp, model=name)
                for c in CUTS:
                    s = summ(g0.filter(pl.col("signal").abs() >= c), x=0)
                    r |= {f">={c:g}c trades": s["trades"], f">={c:g}c net": s.get("net"), f">={c:g}c t": s.get("t")}
                top = g0.sort(pl.col("signal").abs(), descending=True).head(n_ref)
                s = summ(top, x=0)
                r |= {"matched trades": s["trades"], "matched net": s.get("net"), "matched t": s.get("t"),
                      "matched cut": round(top["signal"].abs().min(), 1) if top.height else None,
                      "matched stuck": round((top["how"] == "settled").mean(), 2) if top.height else None}
                rows.append(r)
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "b9_compare_targets.parquet")
    text = ("Maker backtest, NBA, 10 s delay, folds 3-5: current model vs fresh-mid-target models (net cents per contract)\n"
            + str(t))
    (OUT / "b9_compare_targets.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
