"""NFL holdout follow-up (diagnostic only; nothing here changes the frozen spec):

1. Why totals lost more forecasting skill than spreads. The "stale print" rule: the next 60 s of the traded
   price moves toward the middle of the bid and ask (mid_gap = mid - last print). On rows with both a bid and
   an ask, the 60 s move is regressed on mid_gap alone, on the model's forecast alone, and on both:
       R2_mid      best-fit stale-print rule alone
       R2_model    the model's forecast, as is
       R2_both     both together (best-fit weights)
       beyond_mid  R2_both - R2_mid: what the model adds on top of the stale-print rule
       corr        correlation between the model's forecast and mid_gap
   Plus the model's R^2 on rows with no stale gap (|mid_gap| < 1c) and with a large one (>= 3c).
2. Early season. Model-free market measures for every spread / total row (no forecasts, so the training-only
   fold 0 can be used): 2025 weeks 1-5, 2025 weeks 6-8 (both fold 0), 2025-26 test folds (from week 9), 2026
   holdout weeks 1-5. Then, within the holdout, the model and the stale-print rule week by week, and the
   trades (every 5 s, one at a time, fixed exit) week by week.

    venv/bin/python -W ignore analysis/graph_2026_10/b29_nfl_stale_season.py test
    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b29_nfl_stale_season.py holdout
    -> out/b29_nfl_stale_season_<test|holdout>.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

MODE = sys.argv[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b27_nfl_total_diag import forecasts                  # noqa: E402  (reads MODE from argv, then sets it to NFL)
from b17_holdout_report import exits, load as load_holdout   # noqa: E402
from b7_maker_robust import OUT                           # noqa: E402
from stg.models.ingame import Panel                      # noqa: E402
from stg.sports.loaders import derived_dir               # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

assert MODE in ("test", "holdout") and (MODE == "test") != sports_holdout_unsealed()
KINDS = {1: "spread", 2: "total"}
D = derived_dir() / "ingame" / "NFL"
H60 = "60s"


def ols_r2(y: np.ndarray, *xs: np.ndarray) -> float:
    X = np.column_stack(xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 100 * (1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum())


def r2(y: np.ndarray, d: np.ndarray) -> float:
    return 100 * (1 - ((y - d) ** 2).sum() / (y ** 2).sum())


def weeks(pn: Panel) -> np.ndarray:
    """NFL week of each game (1 = the season's first game's week), from the game's first step."""
    t0 = pn.steps_t[pn.game_step_ptr[:-1]].astype("datetime64[s]")
    day = (t0 - np.timedelta64(4 * 3600, "s")).astype("datetime64[D]")          # US evening games stay on their day
    season = np.where(day >= np.datetime64("2026-08-01"), np.datetime64("2026-09-10"), np.datetime64("2025-09-04"))
    return ((day - season).astype(int) // 7 + 1).astype(int)


def market(pn: Panel) -> pl.DataFrame:
    """Model-free measures on every spread / total row of the panel, by period."""
    feats = pn.manifest["node_market_features"]
    hi = pn.manifest["horizons"].index(H60)
    n_nodes = np.diff(pn.step_node_ptr)
    step_game = np.repeat(np.arange(pn.n_games), np.diff(pn.game_step_ptr))
    node_game = np.repeat(step_game, n_nodes)
    wk = weeks(pn)[node_game]
    fold = np.asarray(pn.game_fold)[node_game]
    period = np.where(fold == 99, np.where(wk <= 5, "2026 weeks 1-5 (holdout)", "2026 later"),
                      np.where(fold >= 1, "2025-26 test folds (week 9 on)",
                               np.where(wk <= 5, "2025 weeks 1-5 (fold 0)", "2025 weeks 6-8 (fold 0)")))
    nx = lambda c: np.asarray(pn.nodes_x[:, feats.index(c)], dtype=np.float64)   # noqa: E731
    value = np.asarray(pn.value, dtype=np.float64)
    y = np.asarray(pn.value_h[:, hi], dtype=np.float64) - value
    both = (nx("has_bid") > 0) & (nx("has_ask") > 0)
    mid = 50 * (nx("bid") + nx("ask")) - value
    act = np.repeat(np.asarray(pn.steps_x[:, pn.manifest["step_market_features"].index("log_activity")], dtype=np.float64), n_nodes)
    rows = []
    kind = np.asarray(pn.nodes_kind)
    for kc, kn in KINDS.items():
        for p in np.unique(period):
            m = (kind == kc) & (period == p) & np.isfinite(y)
            mb = m & both
            if m.sum() < 1000:
                continue
            rows.append(dict(kind=kn, period=p, games=len(np.unique(node_game[m])), rows=int(m.sum()),
                             move_sd_60s_c=round(float(y[m].std()), 2), moved_pct=round(100 * float((np.abs(y[m]) >= 1).mean()), 1),
                             stale_gap_sd_c=round(float(mid[mb].std()), 2), stale_rule_R2_60s=round(ols_r2(y[mb], mid[mb]), 2),
                             has_print_pct=round(100 * float((nx("has_print")[m] > 0).mean()), 1),
                             log_activity=round(float(act[m].mean()), 3)))
    return pl.DataFrame(rows).sort("kind", "period")


def decompose(z: dict, pn: Panel, by: np.ndarray | None = None) -> pl.DataFrame:
    step = z["step"]
    first = np.r_[True, step[1:] != step[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(step)), 0))
    node = pn.step_node_ptr[step] + (np.arange(len(step)) - start)
    feats = pn.manifest["node_market_features"]
    nx = lambda c: np.asarray(pn.nodes_x[node, feats.index(c)], dtype=np.float64)   # noqa: E731
    mid = np.where((nx("has_bid") > 0) & (nx("has_ask") > 0), 50 * (nx("bid") + nx("ask")) - z["value"], np.nan)
    y = z["value_h"][:, 1] - z["value"]               # forecast files keep 30 / 60 / 120 / 1800 s: column 1 = 60 s
    d = z["delta"][:, 1].astype(np.float64)
    rows = []
    groups = [("all", np.ones(len(y), bool))] if by is None else [(f"week {w}", by == w) for w in np.unique(by)]
    for kc, kn in KINDS.items():
        for lab, g in groups:
            m = (z["kind"] == kc) & g & np.isfinite(y) & np.isfinite(mid)
            if m.sum() < 1000:
                continue
            yy, dd, mm = y[m], d[m], mid[m]
            r_mid, r_both = ols_r2(yy, mm), ols_r2(yy, mm, dd)
            small, large = np.abs(mm) < 1, np.abs(mm) >= 3
            rows.append(dict(kind=kn, group=lab, rows=int(m.sum()), R2_model=round(r2(yy, dd), 2), R2_mid=round(r_mid, 2),
                             R2_both=round(r_both, 2), beyond_mid=round(r_both - r_mid, 2),
                             corr_forecast_mid=round(float(np.corrcoef(dd, mm)[0, 1]), 2),
                             large_gap_pct=round(100 * float(large.mean()), 1),
                             R2_model_no_gap=round(r2(yy[small], dd[small]), 2), R2_model_large_gap=round(r2(yy[large], dd[large]), 2)))
    return pl.DataFrame(rows)


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_str_lengths(40)
    pl.Config.set_tbl_hide_column_data_types(True)
    z, pn = forecasts()
    lines = [__doc__.split("\n\n")[0], f"\n######## {MODE}",
             "-- 1. model vs the stale-print rule, 60 s, rows with a bid and an ask", str(decompose(z, pn)),
             "", "-- 2. model-free market measures by period (60 s)", str(market(pn))]
    if MODE == "holdout":
        wk = weeks(pn)[z["game"]]
        lines += ["", "-- 2b. holdout week by week: model vs stale-print rule", str(decompose(z, pn, by=wk))]
        t = load_holdout(D.parent / "results", "NFL", "_ens5")
        gw = dict(zip(__import__("json").loads((D / "panel_v3" / "manifest.json").read_text())["games"], weeks(pn).tolist()))
        rows = []
        for kn in KINDS.values():
            g = exits(t.filter(pl.col("kind") == kn))["fixed exit"].with_columns(pl.col("game_id").replace_strict(gw).alias("week"))
            for (w,), x in sorted(g.group_by("week"), key=lambda p: p[0][0]):
                c = x["size"].sum()
                rows.append(dict(kind=kn, week=w, games=x["game_id"].n_unique(), contracts=round(c, 1), net_c=round(x["pnl"].sum() / c, 2)))
        lines += ["", "-- 2c. holdout trades week by week (fixed exit)", str(pl.DataFrame(rows))]
    (OUT / f"b29_nfl_stale_season_{MODE}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
