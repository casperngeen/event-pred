#!/usr/bin/env python
"""Can a learned graph trade the post-game lag as well as the mechanical graph?

    venv/bin/python -W ignore analysis/sports_2026_09/predict.py > analysis/sports_2026_09/out/predict.txt

Needs ``build_mlb.py`` and imports ``graphs.py`` (its graph learners). The lag
(``diffusion.py``): a team's division contract keeps moving for ~24 h after its
game ends, and so do its division rivals' (drift slope +1.42c and −0.33c per
unit surprise). This script asks whether each graph captures that well enough
to trade it.

The trade, per team-game (team a, surprise z_a) and target team b: take b's
division contract at its first print after the game ends, on side
sign(W[a, b]·z_a), and exit at its last print before +24 h (a trade after the
end is required for both). Gross cents per contract, no costs. Only edges in
the graph's trading set are traded: its 150 largest |W[a, b]| (the size of the
true edge set), 30 self + 120 rival for the mechanical graph.

Walk-forward: each graph is fitted on slate days up to a cut and trades the
games of the following month (cuts 2025-05-15 … 2025-08-15; the last test
block runs to the end of the regular season). Graphs: the mechanical graph
(A), title co-occurrence (C2, the best text graph, signed −1 off the diagonal
and +1 on it, as a "rivals move against you" reading of the schedule), and the
learned pairwise, ridge, rank-6 and AGCRN-signed graphs. Controls: always-long
and random side with the same long/short mix. CIs: bootstrap over games.
"""
from __future__ import annotations

import sys

import os

import numpy as np
import polars as pl

sys.path.insert(0, "analysis/sports_2026_09")
import graphs as G  # noqa: E402

OUT = os.environ.get("SPORTS_PILOT_OUT", "analysis/sports_2026_09/out")
H = np.timedelta64(1, "h")
N_BOOT = 2000
rng = np.random.default_rng(0)
CUTS = ["2025-05-15", "2025-06-15", "2025-07-15", "2025-08-15", G.SEASON_END]
teams, N = G.teams, G.N
ix = {t: i for i, t in enumerate(teams)}

fut = pl.read_parquet(f"{OUT}/mlb_futures.parquet").filter(pl.col("kind") == "division")
book = {r[0]: (np.array(r[1], dtype="datetime64[us]"), np.array(r[2], dtype=float))
        for r in fut.group_by("team").agg(pl.col("t"), pl.col("yes_price")).iter_rows()}


def drift(team, e):
    """first print after e (≤ 24 h) → last print at or before e + 24 h."""
    if team not in book:
        return np.nan
    ts, px = book[team]
    i = np.searchsorted(ts, e, side="right")
    j = np.searchsorted(ts, e + 24 * H, side="right") - 1
    if i >= len(ts) or ts[i] - e > 24 * H or j < i:
        return np.nan
    return px[j] - px[i]


games = (pl.read_parquet(f"{OUT}/mlb_games.parquet")
         .filter(pl.col("z").is_finite() & (pl.col("slate") <= pl.lit(G.SEASON_END).str.to_date())))
rows = []
for g in games.iter_rows(named=True):
    e = np.datetime64(g["end"], "us")
    for b in teams:
        d = drift(b, e)
        if np.isfinite(d):
            rows.append(dict(event=g["event"], slate=g["slate"], a=ix[g["team"]], b=ix[b], z=g["z"], drift=d))
D = pl.DataFrame(rows)
print(f"team-game × target pairs with a post-game division drift: {D.height} "
      f"(own {D.filter(pl.col('a') == pl.col('b')).height})")


def rival_schedule(cut):
    W = -G.cooccur(cut)
    np.fill_diagonal(W, 1.0)
    return W


def learned(cut):
    m = G.days <= np.datetime64(cut)
    out = {"(A) mechanical": G.W_A, "(C2) schedule from titles, rival-signed": rival_schedule(cut)}
    W, _ = G.pairwise(m)
    out["(E) pairwise Spearman"] = W
    out["(E) free linear graph (ridge)"] = G.ridge_graph(m)
    out["(E) diagonal + rank-6 graph"] = G.torch_graph("lowrank", m, 0)
    out["(E) AGCRN signed"] = np.mean([G.torch_graph("signed", m, s) for s in G.SEEDS], 0)
    out["(E) AGCRN signed, surprise-only"] = np.mean([G.torch_graph("signed", m, s, cols=[0, 1])
                                                      for s in G.SEEDS], 0)
    return out


def trading_set(W, k=150):
    flat = np.abs(np.nan_to_num(W)).ravel()
    keep = np.zeros(N * N, bool)
    keep[np.argsort(-flat, kind="stable")[:k]] = flat[np.argsort(-flat, kind="stable")[:k]] > 0
    return keep.reshape(N, N)


TR = []
for i in range(len(CUTS) - 1):
    lo, hi = np.datetime64(CUTS[i]), np.datetime64(CUTS[i + 1])
    test = D.filter((pl.col("slate") > pl.lit(CUTS[i]).str.to_date())
                    & (pl.col("slate") <= pl.lit(CUTS[i + 1]).str.to_date()))
    a, b, z, dr = (test[c].to_numpy() for c in ("a", "b", "z", "drift"))
    ev = test["event"].to_numpy()
    for name, W in learned(CUTS[i]).items():
        keep = trading_set(W)[a, b]
        side = np.sign(W[a, b] * z)
        t = keep & (side != 0)
        TR.append(pl.DataFrame({"graph": name, "fold": i, "event": ev[t], "own": (a == b)[t],
                                "true_edge": G.TRUE[a, b][t], "side": side[t], "drift": dr[t]}))
    print(f"fold {i}: fitted on slates ≤ {CUTS[i]}, traded {test['event'].n_unique()} games", flush=True)
TR = pl.concat(TR)


def summarise(d):
    g, s, y = d["event"].to_numpy(), d["side"].to_numpy(), d["drift"].to_numpy()
    pnl = s * y
    q = (s < 0).mean()
    ex = pnl - (1 - 2 * q) * y
    u, inv = np.unique(g, return_inverse=True)
    P, E, C = (np.bincount(inv, v, len(u)) for v in (pnl, ex, np.ones_like(pnl)))
    bs = np.array([(P[k].sum() / C[k].sum(), E[k].sum() / C[k].sum())
                   for k in (rng.integers(0, len(u), len(u)) for _ in range(N_BOOT))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return (len(pnl), float(d["true_edge"].mean()), float((pnl > 0).mean()), pnl.mean(), lo[0], hi[0],
            ex.mean(), lo[1], hi[1])


print("\nTrading the post-game lag of division contracts (first print after the game → +24 h),\n"
      "walk-forward, gross ¢/contract, 95% CI over games. 'true' = share of traded edges that\n"
      "are true edges of (A). 'vs random' = P&L minus a random side with the same short share.\n")
print(f"{'graph':42} {'trades':>6} {'true':>5} {'hit':>5} {'gross ¢ [95% CI]':>24} {'vs random ¢ [95% CI]':>24}")
for name in TR["graph"].unique(maintain_order=True):
    r = summarise(TR.filter(pl.col("graph") == name))
    print(f"{name:42} {r[0]:>6} {r[1]:>5.2f} {r[2]:>5.3f} {r[3]:>+7.2f} [{r[4]:+6.2f},{r[5]:+6.2f}] "
          f"{r[6]:>+7.2f} [{r[7]:+6.2f},{r[8]:+6.2f}]")
print("\nsplit by edge type, mechanical graph (A):")
for own, lab in ((True, "own contract"), (False, "division rivals")):
    r = summarise(TR.filter((pl.col("graph") == "(A) mechanical") & (pl.col("own") == own)))
    print(f"  {lab:40} {r[0]:>6} {'':>5} {r[2]:>5.3f} {r[3]:>+7.2f} [{r[4]:+6.2f},{r[5]:+6.2f}] "
          f"{r[6]:>+7.2f} [{r[7]:+6.2f},{r[8]:+6.2f}]")
TR.write_parquet(f"{OUT}/predict_trades.parquet")
