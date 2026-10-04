"""Two checks on chunk 4a's trading model (linear_weights.py).

1. Shared noise. rem_h = v(t+h) - v(t) and moved = v(t) - v_pre share v(t) with opposite signs, so
   noise in the decision-time mid alone makes b_moved negative. Skip an hour: regress
   rem_skip_h = v(t+h) - v(t+1h) (= total_h - total_1h) on the same inputs. If b_moved < 0 and
   b_own > 0 survive, the reversal and the under-reaction are real.

2. Tradeability. From the out-of-sample predictions of rem_h, trade when |prediction| exceeds the
   round-trip spread at t: buy YES at the ask at t and sell at the bid at t+h (or the reverse for a
   predicted fall). Gross P&L in cents per contract, before fees, against the mid-to-mid move.
   Single-leg live contracts; exits need a two-sided quote at t+h.

    venv/bin/python -W ignore analysis/graph_2026_10/checks_4a.py > analysis/graph_2026_10/out/checks_4a.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))

from stg.sports.asof import SportsData                          # noqa: E402
from stg.sports.graph import between_game_graph, node_values    # noqa: E402
from stg.sports.walkforward import oos_scores, walk_forward     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_weights import LAM, load                            # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def features(g: pl.DataFrame, extra: list[str]) -> list[str]:
    rival = ["x_rival"] if abs(np.corrcoef(g["z_own"], g["x_rival"])[0, 1]) < 0.999 else []
    return ["z_own", *rival, *extra]


def check_skip(s: pl.DataFrame) -> pl.DataFrame:
    s = s.with_columns(*[(pl.col(f"total_{h}") - pl.col("total_1h")).alias(f"remskip_{h}") for h in ("12h", "24h", "48h")])
    rows = []
    for (lg, kind), g in s.group_by("league", "kind"):
        feats = features(g, ["moved"])
        for label, h in (("rem", "12h"), ("remskip", "12h"), ("rem", "24h"), ("remskip", "24h")):
            pred, fits = walk_forward(g, feats, f"{label}_{h}", f"label_end_{h}", lam=LAM)
            sc = oos_scores(pred, f"{label}_{h}")
            c = pl.DataFrame([f.coef for f in fits]) if fits else None
            rows.append({"league": lg, "kind": kind, "label": f"{label}_{h}", "n": sc.get("n"),
                         "r2": round(sc.get("r2_vs_zero", np.nan), 3),
                         **{f"b_{k}": (f"{c[k].mean():.2f} ({c[k].std():.2f})" if c is not None and k in c.columns else None)
                            for k in ("z_own", "x_rival", "moved")}})
    return pl.DataFrame(rows).sort("league", "kind", "label")


def exits(d: SportsData, preds: pl.DataFrame, h: str) -> pl.DataFrame:
    """Two-sided spread at t+h for each prediction row (for the exit price)."""
    edges = pl.read_parquet("data/sports/derived/edges.parquet")
    out = []
    for (lg, season), p in preds.group_by("league", "season"):
        gr = between_game_graph(d, edges, lg, season)
        at = p.select("node", pl.col(f"label_end_{h}").alias("t")).unique()
        v = node_values(d, gr, at).select("node", pl.col("t").alias(f"label_end_{h}"),
                                          pl.col("spread").alias("spread_exit"), pl.col("one_sided").alias("one_sided_exit"))
        out.append(p.join(v, on=["node", f"label_end_{h}"], how="left"))
    return pl.concat(out)


def check_trade(s: pl.DataFrame, d: SportsData) -> pl.DataFrame:
    rows = []
    for h in ("12h", "24h"):
        preds = []
        for (lg, kind), g in s.filter(pl.col("kind") != "wins").group_by("league", "kind"):
            p, _ = walk_forward(g, features(g, ["moved"]), f"rem_{h}", f"label_end_{h}", lam=LAM)
            preds.append(p)
        p = exits(d, pl.concat(preds, how="diagonal_relaxed"), h)
        mid_h = pl.col("value_t") + pl.col(f"rem_{h}")
        bid_t, ask_t = pl.col("value_t") - pl.col("spread_t") / 2, pl.col("value_t") + pl.col("spread_t") / 2
        bid_h, ask_h = mid_h - pl.col("spread_exit") / 2, mid_h + pl.col("spread_exit") / 2
        side = pl.col("pred").sign()
        p = p.with_columns(
            (pl.col("pred").abs() > pl.col("spread_t")).alias("trade"),
            pl.when(side > 0).then(bid_h - ask_t).otherwise(bid_t - ask_h).alias("pnl_exec"),
            (side * pl.col(f"rem_{h}")).alias("pnl_mid"),
            (pl.col("spread_exit").is_not_null() & ~pl.col("one_sided_exit").fill_null(True)).alias("exit_ok"))
        t = p.filter(pl.col("trade"))
        for (lg, kind), g in t.group_by("league", "kind"):
            ok = g.filter(pl.col("exit_ok"))
            rows.append({"horizon": h, "league": lg, "kind": kind, "oos_rows": p.filter((pl.col("league") == lg) & (pl.col("kind") == kind)).height,
                         "trades": g.height, "exit_ok": ok.height,
                         "mean_spread_t": round(g["spread_t"].mean(), 2),
                         "pnl_mid": round(ok["pnl_mid"].mean(), 2) if ok.height else None,
                         "pnl_exec": round(ok["pnl_exec"].mean(), 2) if ok.height else None,
                         "hit_exec": round((ok["pnl_exec"] > 0).mean(), 2) if ok.height else None,
                         "pnl_exec_total": round(ok["pnl_exec"].sum(), 1) if ok.height else None})
    return pl.DataFrame(rows).sort("horizon", "league", "kind")


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(230)
    s = load()
    d = SportsData()
    print("1. shared noise: rem_h (from t) vs remskip_h (from t + 1 h); OOS R^2; coefficients mean (sd over folds)")
    print(check_skip(s))
    print("\n2. tradeability: trade when |predicted rem_h| > spread at t; executable = buy at ask t, sell at bid t+h "
          "(reverse for a fall); cents per contract, gross of fees")
    print(check_trade(s, d))


if __name__ == "__main__":
    main()
