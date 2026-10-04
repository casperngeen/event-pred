"""Tradeability of the trading model's out-of-sample predictions, 4a (constant weights) vs 4b
(state-dependent): trade when |predicted rem_h| > spread at t; buy YES at the ask at t and sell at
the bid at t+h (reverse for a predicted fall). Cents per contract, gross; net subtracts Kalshi's
taker fee on both legs, ~7 p (1 - p) cents per contract per side. Single-leg live contracts.

    venv/bin/python -W ignore analysis/graph_2026_10/trade_4b.py > analysis/graph_2026_10/out/trade_4b.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stg.sports.asof import SportsData                          # noqa: E402
from stg.sports.walkforward import walk_forward                 # noqa: E402
from checks_4a import exits                                     # noqa: E402
from linear_weights import load                                 # noqa: E402
from linear_weights_4b import GRID, specs, with_state           # noqa: E402

FEE = 7.0          # cents x p(1-p), per contract per side (Kalshi taker fee, rounded per order in practice)


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    d = SportsData()
    s = with_state(load()).filter(pl.col("kind") != "wins")
    rows = []
    for h in ("12h", "24h"):
        for spec in ("4a", "4b"):
            preds = []
            for (lg, kind), g in s.group_by("league", "kind"):
                p, _ = walk_forward(g, specs(g, kind, True)[spec], f"rem_{h}", f"label_end_{h}", lam=GRID)
                preds.append(p)
            p = exits(d, pl.concat(preds, how="diagonal_relaxed"), h)
            mid_h = pl.col("value_t") + pl.col(f"rem_{h}")
            ask_t, bid_t = pl.col("value_t") + pl.col("spread_t") / 2, pl.col("value_t") - pl.col("spread_t") / 2
            bid_h, ask_h = mid_h - pl.col("spread_exit") / 2, mid_h + pl.col("spread_exit") / 2
            up = pl.col("pred") > 0
            fee = lambda x: FEE * (x / 100) * (1 - x / 100)                      # noqa: E731
            p = (p.with_columns(pl.when(up).then(bid_h - ask_t).otherwise(bid_t - ask_h).alias("gross"),
                                (pl.col("pred").sign() * pl.col(f"rem_{h}")).alias("mid"),
                                (fee(pl.col("value_t")) + fee(mid_h)).alias("fees"))
                 .filter(pl.col("pred").abs() > pl.col("spread_t"),
                         pl.col("spread_exit").is_not_null(), ~pl.col("one_sided_exit").fill_null(True)))
            for (lg, kind), g in p.group_by("league", "kind"):
                rows.append({"horizon": h, "league": lg, "kind": kind, "spec": spec, "trades": g.height,
                             "nights": g["night"].n_unique(), "mid": round(g["mid"].mean(), 2),
                             "gross": round(g["gross"].mean(), 2), "hit": round((g["gross"] > 0).mean(), 2),
                             "net": round((g["gross"] - g["fees"]).mean(), 2),
                             "gross_total": round(g["gross"].sum(), 1)})
    r = pl.DataFrame(rows)
    print(__doc__.split("\n\n")[0], "\n")
    print(r.pivot(on="spec", index=["horizon", "league", "kind"], values=["trades", "gross", "hit", "net", "gross_total"])
          .sort("horizon", "league", "kind"))
    tot = r.group_by("horizon", "spec").agg(pl.col("trades").sum(), pl.col("gross_total").sum().round(1),
                                            (pl.col("gross_total").sum() / pl.col("trades").sum()).round(2).alias("gross_per_trade"))
    print("\nall kinds together:")
    print(tot.sort("horizon", "spec"))


if __name__ == "__main__":
    main()
