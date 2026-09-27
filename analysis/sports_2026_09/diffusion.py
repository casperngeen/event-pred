#!/usr/bin/env python
"""How fast does a game result reach the futures? (plan step 1, risk 3)

    venv/bin/python analysis/sports_2026_09/diffusion.py > analysis/sports_2026_09/out/diffusion.txt

Needs ``build_mlb.py``. For each team-game with surprise z = 1[won] − p_pre,
the price path of the affected futures, read as the last trade at or before
each checkpoint (≤ 7 days old; the post-game checkpoints also require a trade
after the game ended, so a stale price is never read as "no response"):

  start    first pitch (detected)
  end⁻     last trade before the game ended  → start→end⁻ is in-game repricing
  first⁺   first trade after the end (≤ 24 h)  → the jump at resolution
  +1h, +3h, +12h, +24h after the end        → post-game drift

The response at each checkpoint is regressed on z (cents per unit of surprise,
through the origin, SE clustered by game). If the futures have fully absorbed
the result by first⁺, the later slopes equal first⁺'s and there is no
post-resolution lag to learn: the question becomes one of in-game diffusion.

Targets: the team's own championship, pennant and division contracts (mechanical
sign +), and its division rivals' division and championship contracts (−).
"""
from __future__ import annotations

import numpy as np
import polars as pl

OUT = "analysis/sports_2026_09/out"
H = np.timedelta64(1, "h")
STALE = np.timedelta64(7, "D")

games = pl.read_parquet(f"{OUT}/mlb_games.parquet").filter(pl.col("z").is_finite())
fut = pl.read_parquet(f"{OUT}/mlb_futures.parquet")
teams = pl.read_parquet(f"{OUT}/mlb_teams.parquet")
div = dict(teams.select("team", "division").iter_rows())

# per contract: sorted trade times and prices
book = {(r[0], r[1]): (np.array(r[2], dtype="datetime64[us]"), np.array(r[3], dtype=float))
        for r in fut.group_by("team", "kind").agg(pl.col("t"), pl.col("yes_price")).iter_rows()}


def asof(key, t, after=None):
    """Last price at or before t (≤ 7 d old); if ``after`` is given, only if a
    trade happened in (after, t]. NaN otherwise."""
    if key not in book:
        return np.nan
    ts, px = book[key]
    i = np.searchsorted(ts, t, side="right") - 1
    if i < 0 or t - ts[i] > STALE:
        return np.nan
    if after is not None and ts[i] <= after:
        return np.nan
    return float(px[i])


def first_after(key, t, within=24 * H):
    if key not in book:
        return np.nan
    ts, px = book[key]
    i = np.searchsorted(ts, t, side="right")
    return float(px[i]) if i < len(ts) and ts[i] - t <= within else np.nan


CHECK = ["end⁻", "first⁺", "+1h", "+3h", "+12h", "+24h"]
rows = []
for g in games.iter_rows(named=True):
    st, e = np.datetime64(g["start"], "us"), np.datetime64(g["end"], "us")
    rivals = [t for t, d in div.items() if d == div[g["team"]] and t not in (g["team"], g["opp"])]
    targets = [(g["team"], k, "own", +1) for k in ("champ", "pennant", "division")]
    targets += [(r, k, "division rival", -1) for r in rivals for k in ("champ", "division")]
    for team, kind, rel, sign in targets:
        key = (team, kind)
        p0 = asof(key, st)
        if not np.isfinite(p0):
            continue
        path = {"end⁻": asof(key, e - np.timedelta64(1, "us")),
                "first⁺": first_after(key, e)}
        for lab, h in (("+1h", 1), ("+3h", 3), ("+12h", 12), ("+24h", 24)):
            path[lab] = asof(key, e + h * H, after=e)
        rows.append(dict(event=g["event"], team=team, kind=kind, rel=rel, z=g["z"], p0=p0,
                         **{c: path[c] - p0 for c in CHECK}))
R = pl.DataFrame(rows)


def slope(d, col):
    d = d.filter(pl.col(col).is_finite())
    if d.height < 20:
        return d.height, np.nan, np.nan
    z, y, g = d["z"].to_numpy(), d[col].to_numpy(), d["event"].to_numpy()
    b = (z * y).sum() / (z * z).sum()
    e = y - b * z
    u, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, z * e)
    se = np.sqrt((s ** 2).sum()) / (z * z).sum()
    return d.height, b, b / se


print("response of futures price (cents) to the game surprise z, by checkpoint after the game;\n"
      "slope through the origin, t clustered by game. n = team-games with a price at that checkpoint.\n")
for rel in ("own", "division rival"):
    for kind in ("champ", "pennant", "division"):
        d = R.filter((pl.col("rel") == rel) & (pl.col("kind") == kind))
        if d.height == 0:
            continue
        print(f"{rel:15} {kind:9} (p0 median {d['p0'].median():.0f}c)")
        out = [(c, *slope(d, c)) for c in CHECK]
        ref = next((b for c, n, b, t in out if c == "+24h"), np.nan)
        for c, n, b, t in out:
            frac = b / ref if np.isfinite(ref) and ref != 0 else np.nan
            print(f"   {c:7} n {n:>5}  slope {b:+7.2f}c  t {t:+6.1f}  share of +24h response {frac:6.2f}")
        # the post-resolution drift, on the same games: first⁺ → +24h
        dd = d.filter(pl.col("first⁺").is_finite() & pl.col("+24h").is_finite()).with_columns(
            (pl.col("+24h") - pl.col("first⁺")).alias("drift"))
        n, b, t = slope(dd, "drift")
        print(f"   drift first⁺→+24h on the same {n} games: slope {b:+.2f}c, t {t:+.1f}\n")
R.write_parquet(f"{OUT}/mlb_diffusion.parquet")
