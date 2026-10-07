"""Which trades of the in-game model pay after fees? Splits the taker trades of b5_backtest by conditions
known at entry, then checks any filter chosen on earlier test folds against the later ones.

Trades: hold 60 s (or HOLD), entry quotes <= 30 s old, predicted edge >= 3c, one entry per contract per
holding period (as b5_backtest). Conditions, each in quantile-free fixed buckets:

    edge          predicted level minus the entry price (cents)
    price         entry price (fees peak at 50c: 7 x p (1 - p) = 1.75c)
    gap           ask - bid at entry (cents; one-sided = no gap known)
    age           entry quote age (s)
    since_play    seconds since the last play the model knows of (plays reach it after the feed delay)
    tau           share of regulation left (1 = start)
    abs_margin    |score margin| after the last known play (league units)
    activity      log trading activity of the game's winner contract (steps_x[:, 2])
    agree         the model's forecasts at the next-longer horizon point the same way (1) or not (0)

Selection check: for each condition, the best bucket by net P&L on test folds 1-3, and its net P&L on
folds 4-5 (out of sample for the selection).

    venv/bin/python -W ignore analysis/graph_2026_10/b5_condition.py results/ingame NBA 10
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stg.models.ingame import KIND_NAMES, Panel           # noqa: E402
from stg.sports.loaders import derived_dir                 # noqa: E402
from b5_backtest import HOLDS, clustered_t, fee            # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ingame")
LEAGUE = sys.argv[2] if len(sys.argv) > 2 else "NBA"
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
HOLD = sys.argv[4] if len(sys.argv) > 4 else "60s"
MARGIN, MAX_AGE = 3.0, 30.0
BUCKETS = {
    "edge": [3, 4, 5, 7, 10, np.inf],
    "price": [0, 10, 25, 40, 60, 75, 90, 100.1],
    "gap": [0, 1.5, 2.5, 4.5, 8.5, np.inf],
    "age": [0, 5, 10, 20, 30.1],
    "since_play": [0, 10, 20, 40, 80, np.inf],
    "tau": [-np.inf, 0.1, 0.25, 0.5, 0.75, np.inf],
    "abs_margin": [0, 3, 6, 10, 15, np.inf],
    "activity": [-np.inf, 1, 2, 3, 4, np.inf],
    "agree": [-0.5, 0.5, 1.5],
}


def trade_rows(pn: Panel, pr: dict, hz: list[str]) -> pl.DataFrame:
    h, every = hz.index(HOLD), HOLDS[HOLD]
    step_k = pr["step"] - pn.game_step_ptr[pr["game"]]
    lvl = pr["value"] + pr["delta"][:, h]
    bid, ask = pr["entry_bid"], pr["entry_ask"]
    ok = (step_k % every == 0) & np.isfinite(pr["y"]) & (pr["quote_age_s"] <= MAX_AGE)
    le = np.where(pr["has_ask"], lvl - ask, -np.inf)
    se = np.where(pr["has_bid"], bid - lvl, -np.inf)
    side = np.where((le >= MARGIN) & (le >= se), 1, np.where(se >= MARGIN, -1, 0))
    m = np.flatnonzero(ok & (side != 0))
    s = side[m]
    entry = np.where(s == 1, ask[m], bid[m])
    exit_px = np.where(s == 1, pr["lo"][m, h], pr["hi"][m, h]).astype(float)
    quoted = np.isfinite(exit_px) & np.where(s == 1, exit_px > 0, exit_px < 100)
    exit_val = np.where(quoted, exit_px, 100 * pr["y"][m])
    gross = s * (exit_val - entry)
    net = gross - fee(entry) - np.where(quoted, fee(exit_val), 0)
    # state the model knew at entry
    steps, games = pr["step"][m], pr["game"][m]
    st_t = pn.steps_t[steps].astype(float)
    since = np.full(len(m), 3600.0)
    tau = np.ones(len(m))
    marg = np.zeros(len(m))
    for g in np.unique(games):
        i = np.flatnonzero(games == g)
        a, b = pn.game_play_ptr[g], pn.game_play_ptr[g + 1]
        pt = pn.plays_t[a:b]
        k = np.searchsorted(pt, st_t[i] - DELAY, side="right")
        has = k > 0
        j = a + np.maximum(k - 1, 0)
        since[i] = np.where(has, st_t[i] - pt[np.maximum(k - 1, 0)], 3600.0) if len(pt) else 3600.0
        tau[i] = np.where(has, pn.plays_state[j, 0], 1.0)
        marg[i] = np.where(has, pn.plays_state[j, 1], 0.0)
    nxt = min(h + 1, len(hz) - 1)
    agree = (np.sign(pr["delta"][m, nxt]) == s).astype(float)
    gap = np.where(pr["has_bid"][m] & pr["has_ask"][m], ask[m] - bid[m], np.inf)
    return pl.DataFrame({"kind": pr["kind"][m], "game": games, "fold": pr["fold"][m], "gross": gross, "net": net,
                         "edge": np.where(s == 1, le[m], se[m]), "price": entry, "gap": gap, "age": pr["quote_age_s"][m],
                         "since_play": since, "tau": tau, "abs_margin": np.abs(marg),
                         "activity": pn.steps_x[steps, 2].astype(float), "agree": agree})


def table(t: pl.DataFrame, col: str) -> pl.DataFrame:
    edges = BUCKETS[col]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = t.filter((pl.col(col) >= lo) & (pl.col(col) < hi))
        if b.height < 30:
            continue
        g = b["game"].to_numpy()
        rows.append(dict(bucket=f"[{lo:g}, {hi:g})", trades=b.height, games=b["game"].n_unique(),
                         gross=round(b["gross"].mean(), 2), net=round(b["net"].mean(), 2),
                         t_net=round(clustered_t(b["net"].to_numpy(), g), 1), lo=lo, hi=hi))
    return pl.DataFrame(rows)


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(200)
    pn = Panel(derived_dir() / "ingame" / LEAGUE / "panel_v2")
    files = [f for f in sorted(RES.glob(f"{LEAGUE}_fold*_gru_d{DELAY:g}_seed0.npz"))]
    prs, hz = [], None
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        hz = meta["horizons"]
        p = dict(np.load(f))
        p["fold"] = np.full(len(p["value"]), meta["test_fold"])
        prs.append(p)
    pr = {k: np.concatenate([p[k] for p in prs]) for k in prs[0]}
    t = trade_rows(pn, pr, hz)
    print(f"{LEAGUE}, feed delay {DELAY:g} s, hold {HOLD}, edge >= {MARGIN:g}c, quotes <= {MAX_AGE:g} s: "
          f"{t.height:,} trades over {len(files)} folds")
    for kind in (1, 2):
        tk = t.filter(pl.col("kind") == kind)
        print(f"\n================ {KIND_NAMES[kind]}: {tk.height:,} trades, gross {tk['gross'].mean():.2f}, net {tk['net'].mean():.2f}")
        early, late = tk.filter(pl.col("fold") <= 3), tk.filter(pl.col("fold") >= 4)
        for col in BUCKETS:
            print(f"--- {col}")
            print(table(tk, col).drop("lo", "hi"))
            te = table(early, col)
            if te.height:
                best = te.sort("net", descending=True).row(0, named=True)
                lb = late.filter((pl.col(col) >= best["lo"]) & (pl.col(col) < best["hi"]))
                if lb.height:
                    print(f"    chosen on folds 1-3: {best['bucket']} (net {best['net']}, {best['trades']} trades) -> folds 4-5: "
                          f"net {lb['net'].mean():.2f}, t {clustered_t(lb['net'].to_numpy(), lb['game'].to_numpy()):.1f}, {lb.height} trades")


if __name__ == "__main__":
    main()
