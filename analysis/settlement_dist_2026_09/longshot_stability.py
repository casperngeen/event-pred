#!/usr/bin/env python
"""Is the longshot bias stable across years, and is it profitable?

    venv/bin/python analysis/settlement_dist_2026_09/longshot_stability.py

Two questions that are not the same one:

* **Is the BIAS stable?** Measured in percentage points, independent of costs.
  This is the behavioural claim.
* **Is the TRADE profitable?** The bias net of fee and half-spread. A stable
  2pp bias is a real phenomenon and an unprofitable strategy.

Run on both populations, because they disagree in magnitude:

* bucket ladders (WTI/WTIW) from ``wing_legs.parquet`` -- folded to the
  longshot side, entry on the pre-resolution snapshot day;
* macro threshold targets from ``leadlag_legs.parquet`` -- raw YES prices,
  entry at the first print after a trigger resolved. Kept unfolded so the
  cheap-YES and expensive-YES halves of the bias stay visible separately.

Everything clusters on the event; year cells are small, so per-year CIs are
wide by construction and the trend matters more than any single year.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

WING = Path("analysis/settlement_dist_2026_09/out/wing_legs.parquet")
LEAD = Path("analysis/leadlag_2026_09/out/leadlag_legs.parquet")
N_BOOT = 10000
FEE_RATE, CONTRACTS = 0.07, 100


def fee_cents(p):
    p = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p):
    out = np.full(np.shape(p), 2.0)
    out[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cboot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    k = len(u)
    s = np.bincount(inv, weights=v, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float((b <= 0).mean()))


def trend(years, vals):
    """Spearman of the per-year statistic on the year."""
    if len(years) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(years, float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(vals, float))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    # ---------------------------------------------------------- bucket / WTI
    w = pl.read_parquet(WING).filter(pl.col("kind") == "bucket")
    w = w.filter(pl.col("p_long").is_between(3, 20, closed="left"))
    fav = 100.0 - w["p_long"].to_numpy()
    gross = 100.0 * (1 - w["long_win"].to_numpy()) - fav
    w = w.with_columns(
        pl.Series("net", gross - (fee_cents(fav) + spread_cents(fav) / 2.0)),
        pl.Series("edge_pp", 100.0 * w["long_win"].to_numpy() - w["p_long"].to_numpy()))

    print("=== WTI/WTIW bucket ladders, longshot legs priced 3-20c ===")
    print("edge_pp < 0 = the longshot was overpriced (the bias).")
    print("net = selling it, held to settlement, after fee + half spread.\n")
    rows = []
    for y in sorted(w["yr"].unique().to_list()):
        s = w.filter(pl.col("yr") == y)
        if s.height < 20:
            continue
        e, elo, ehi, _ = cboot(s["edge_pp"].to_numpy(), s["event_ticker"].to_numpy())
        n, nlo, nhi, pneg = cboot(s["net"].to_numpy(), s["event_ticker"].to_numpy())
        rows.append(dict(year=int(y), legs=s.height,
                         events=s["event_ticker"].n_unique(),
                         price=float(s["p_long"].mean()),
                         edge_pp=e, e_lo=elo, e_hi=ehi,
                         net=n, n_lo=nlo, n_hi=nhi, p_le0=pneg))
    t = pl.DataFrame(rows)
    with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=230):
        print(t)
    print(f"\ntrend(edge_pp vs year) rho = {trend(t['year'], t['edge_pp']):+.2f}   "
          f"trend(net vs year) rho = {trend(t['year'], t['net']):+.2f}")
    print("(rho = +1 means the bias is shrinking toward zero year on year)")

    # ------------------------------------------------- macro threshold legs
    d = pl.read_parquet(LEAD)
    print("\n\n=== macro threshold targets, raw YES prices (unfolded) ===")
    print("Cheap-YES and expensive-YES halves reported separately: under")
    print("longshot bias the first is negative and the second positive.\n")
    for lo, hi, label in [(3, 10, "cheap YES 3-10c"), (88, 97, "dear YES 88-97c")]:
        s = d.filter(pl.col("p_entry").is_between(lo, hi, closed="left"))
        print(f"--- {label}")
        rows = []
        for y in sorted(s["yr"].unique().to_list()):
            x = s.filter(pl.col("yr") == y)
            if x.height < 40:
                continue
            e = 100.0 * x["win"].to_numpy() - x["p_entry"].to_numpy()
            obs, elo, ehi, _ = cboot(e, x["target_event"].to_numpy())
            rows.append(dict(year=int(y), legs=x.height,
                             events=x["target_event"].n_unique(),
                             paid=float(x["p_entry"].mean()),
                             realised=100.0 * float(x["win"].mean()),
                             edge_pp=obs, ci_lo=elo, ci_hi=ehi))
        tt = pl.DataFrame(rows)
        with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=210):
            print(tt)
        if tt.height >= 3:
            print(f"trend rho = {trend(tt['year'], tt['edge_pp']):+.2f}\n")

    # ---------------------------------------------- the macro trade, by year
    print("\n=== would trading the macro bias have paid? ===")
    print("Sell the longshot side of any leg priced 3-20c or 80-97c, hold.\n")
    m = d.with_columns(
        pl.min_horizontal("p_entry", 100.0 - pl.col("p_entry")).alias("p_long"),
        pl.when(pl.col("p_entry") <= 50).then(pl.col("win"))
          .otherwise(1 - pl.col("win")).alias("long_win"))
    m = m.filter(pl.col("p_long").is_between(3, 20, closed="left"))
    favm = 100.0 - m["p_long"].to_numpy()
    grossm = 100.0 * (1 - m["long_win"].to_numpy()) - favm
    m = m.with_columns(
        pl.Series("net", grossm - (fee_cents(favm) + spread_cents(favm) / 2.0)),
        pl.Series("edge_pp", 100.0 * m["long_win"].to_numpy() - m["p_long"].to_numpy()))
    rows = []
    for y in sorted(m["yr"].unique().to_list()):
        s = m.filter(pl.col("yr") == y)
        if s.height < 40:
            continue
        e, _, _, _ = cboot(s["edge_pp"].to_numpy(), s["target_event"].to_numpy())
        n, nlo, nhi, pneg = cboot(s["net"].to_numpy(), s["target_event"].to_numpy())
        rows.append(dict(year=int(y), legs=s.height,
                         events=s["target_event"].n_unique(),
                         edge_pp=e, net=n, ci_lo=nlo, ci_hi=nhi, p_le0=pneg))
    tm = pl.DataFrame(rows)
    with pl.Config(tbl_rows=12, float_precision=2, tbl_width_chars=210):
        print(tm)
    a, alo, ahi, apneg = cboot(m["net"].to_numpy(), m["target_event"].to_numpy())
    print(f"\npooled macro: net {a:+.2f}c  CI [{alo:+.2f}, {ahi:+.2f}]  "
          f"P(<=0) = {apneg:.3f}   legs {m.height}")
    if tm.height >= 3:
        print(f"trend(net vs year) rho = {trend(tm['year'], tm['net']):+.2f}")


if __name__ == "__main__":
    main()
