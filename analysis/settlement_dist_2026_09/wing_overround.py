#!/usr/bin/env python
"""Is the bucket wing edge longshot bias, or is it just ladder overround?

    venv/bin/python analysis/settlement_dist_2026_09/wing_overround.py
    (needs wing_calibration.py to have written out/wing_legs.parquet)

``wing_calibration.py`` found the wings fair on threshold ladders (-0.13c,
P(<=0) = 0.55) and richly profitable on bucket ladders (+4.96c, clustered CI
[+4.06, +5.79]). Taken at face value that is longshot bias exactly where the
plan predicted it.

It is not, and the check is arithmetic. Bucket contracts partition the outcome
space: they are mutually exclusive and exhaustive, so a complete ladder's prices
must sum to 100c. The observed median is **124c**. Selling every leg of a ladder
that sums to 124 returns 24c per ladder no matter what settles -- no forecast,
no bias, no skill. Since the wing trade sells the cheap side of every leg, it
harvests precisely that.

Threshold legs price P(X > k) cumulatively, so their sum has no such constraint
and is not comparable -- which is why the threshold result needs no correction
here.

Three tests
-----------
1. Does the per-event wing P&L track the ladder's overround? If the edge is
   overround, `net` should be ~ (mass - 1) * something and vanish at mass = 1.
2. Re-run the wing bands on **mass-normalised** prices, p / mass, which is the
   ladder renormalised to a real probability distribution. If longshot bias is
   present on top of the overround, it survives normalisation.
3. Restrict to near-complete, near-coherent ladders (mass in 0.95-1.10) and
   re-run the trade. Small n, but no normalisation assumption.

The honest prior is that (1) and (2) kill it. The reason to run it anyway is
that "the venue's bucket ladders carry a ~24% overround at last-trade prices"
is itself a finding worth recording, and it has a second reading: last trades
on different legs happen at different moments, so part of that 24% is stale
non-synchronous pricing rather than a spread a seller could actually cross.
Test 3 is the one that separates those, and it is the one with the least power.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.events.implied import BUCKET, THRESHOLD

OUT = Path("analysis/settlement_dist_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100
N_BOOT = 10000


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(price_cents, mult: float = 1.0) -> np.ndarray:
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out * mult


def cluster_boot(df: pl.DataFrame, col: str, by: str = "event_ticker", seed: int = 0):
    groups = [g[col].to_numpy() for _, g in df.group_by(by)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    boot = np.array([
        np.concatenate([groups[i] for i in rng.choice(idx, len(idx), replace=True)]).mean()
        for _ in range(N_BOOT)])
    obs = float(np.concatenate(groups).mean())
    return obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), float((boot <= 0).mean())


def wing_trade(d: pl.DataFrame, price_col: str = "p_long", spread_mult: float = 1.0) -> pl.DataFrame:
    """Sell the longshot at 3-20c on `price_col`, hold to settlement."""
    w = d.filter(pl.col(price_col).is_between(3, 20, closed="left"))
    if w.height == 0:
        return w
    fav_price = 100.0 - w[price_col].to_numpy()
    gross = 100.0 * (1 - w["long_win"].to_numpy()) - fav_price
    cost = fee_cents(fav_price) + spread_cents(fav_price, spread_mult) / 2.0
    return w.with_columns(pl.Series("gross", gross), pl.Series("net", gross - cost))


def main() -> None:
    legs = pl.read_parquet(OUT / "wing_legs.parquet")

    mass = (legs.group_by("event_ticker")
            .agg((pl.col("price").sum() / 100.0).alias("mass"),
                 pl.len().alias("n_legs_ladder")))
    legs = legs.join(mass, on="event_ticker", how="left")
    buck = legs.filter(pl.col("kind") == BUCKET)

    print("=== the arithmetic ===")
    print("Bucket ladders partition the outcome space: prices MUST sum to 100c.")
    q = buck.group_by("event_ticker").agg(pl.col("mass").first()).to_series(1).to_numpy()
    print(f"bucket events: {len(q)}   mass p10/p50/p90 = "
          f"{np.percentile(q,10):.2f} / {np.percentile(q,50):.2f} / {np.percentile(q,90):.2f}")
    print(f"share of bucket ladders with mass > 1.05: {(q > 1.05).mean():.1%}")
    thr_q = (legs.filter(pl.col("kind") == THRESHOLD).group_by("event_ticker")
             .agg(pl.col("mass").first()).to_series(1).to_numpy())
    print(f"(threshold ladders are cumulative, sum unconstrained: median {np.median(thr_q):.2f} "
          f"-- shown only to make clear it is not comparable)")

    # ---------------------------------------------------------------- test 1
    print("\n=== test 1: does the wing P&L track the overround? ===")
    w = wing_trade(buck)
    bands = [(0.0, 1.00), (1.00, 1.15), (1.15, 1.30), (1.30, 1.50), (1.50, 9.9)]
    rows = []
    for lo, hi in bands:
        b = w.filter(pl.col("mass").is_between(lo, hi, closed="left"))
        if b.height < 20:
            continue
        obs, clo, chi, pneg = cluster_boot(b, "net")
        rows.append(dict(mass_band=f"{lo:.2f}-{hi:.2f}", n_legs=b.height,
                         n_events=b["event_ticker"].n_unique(),
                         mean_mass=float(b["mass"].mean()),
                         net=obs, ci_lo=clo, ci_hi=chi, p_le0=pneg))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    ev = (w.group_by("event_ticker").agg(pl.col("mass").first(),
                                         pl.col("net").mean().alias("net")))
    r = np.corrcoef(ev["mass"].to_numpy(), ev["net"].to_numpy())[0, 1]
    print(f"\nper-event corr(ladder mass, wing net) = {r:+.3f}   (n = {ev.height} events)")

    # ---------------------------------------------------------------- test 2
    print("\n=== test 2: mass-normalised prices (renormalise the ladder to 1.0) ===")
    print("If longshot bias exists on top of the overround, it survives this.\n")
    nb = buck.with_columns((pl.col("price") / pl.col("mass")).alias("price_n"))
    nb = nb.with_columns(
        pl.min_horizontal("price_n", 100.0 - pl.col("price_n")).alias("p_long_n"))
    rows = []
    for lo, hi in [(3, 10), (10, 20), (20, 35), (35, 50)]:
        b = nb.filter(pl.col("p_long_n").is_between(lo, hi, closed="left"))
        if b.height < 10:
            continue
        # which side is the longshot can flip under normalisation; recompute
        b = b.with_columns(
            pl.when(pl.col("price_n") <= 50.0).then(pl.col("win"))
              .otherwise(1 - pl.col("win")).alias("lw_n"))
        e = b.with_columns((100.0 * pl.col("lw_n") - pl.col("p_long_n")).alias("e"))
        obs, clo, chi, pneg = cluster_boot(e, "e")
        rows.append(dict(band=f"{lo}-{hi}c", n_legs=b.height,
                         n_events=b["event_ticker"].n_unique(),
                         mean_price=float(b["p_long_n"].mean()),
                         realised=float(b["lw_n"].mean()) * 100.0,
                         edge_pp=obs, ci_lo=clo, ci_hi=chi, p_ge0=1.0 - pneg))
    with pl.Config(tbl_rows=20, float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    # ---------------------------------------------------------------- test 3
    print("\n=== test 3: coherent ladders only (mass 0.95-1.10), no normalisation ===")
    coh = buck.filter(pl.col("mass").is_between(0.95, 1.10))
    print(f"bucket events surviving: {coh['event_ticker'].n_unique()} "
          f"of {buck['event_ticker'].n_unique()}")
    cw = wing_trade(coh)
    if cw.height >= 20:
        obs, clo, chi, pneg = cluster_boot(cw, "net")
        print(f"sell the longshot 3-20c: {obs:+.2f}c   95% CI [{clo:+.2f}, {chi:+.2f}]   "
              f"P(mean<=0) = {pneg:.3f}   (n = {cw.height} legs, "
              f"{cw['event_ticker'].n_unique()} events)")
        e = cw.with_columns((100.0 * pl.col("long_win") - pl.col("p_long")).alias("e"))
        obs, clo, chi, pneg = cluster_boot(e, "e")
        print(f"longshot edge:            {obs:+.2f}pp  95% CI [{clo:+.2f}, {chi:+.2f}]   "
              f"P(mean>=0) = {1-pneg:.3f}")
    else:
        print("too few legs to test")

    # ---------------------------------------------------------- spread sens.
    print("\n=== spread sensitivity on the headline bucket trade ===")
    print("research_log §11.2's 1-2c effective spread was measured on liquid legs;")
    print("a far bucket that trades a handful of times a day is optimistically priced there.\n")
    rows = []
    for mult in (1.0, 2.0, 3.0, 4.0):
        t = wing_trade(buck, spread_mult=mult)
        obs, clo, chi, pneg = cluster_boot(t, "net")
        rows.append(dict(spread_x=mult, net=obs, ci_lo=clo, ci_hi=chi, p_le0=pneg))
    with pl.Config(float_precision=2):
        print(pl.DataFrame(rows))

    OUT.mkdir(parents=True, exist_ok=True)
    legs.write_parquet(OUT / "wing_legs_mass.parquet")
    print(f"\nwrote {OUT / 'wing_legs_mass.parquet'}")


if __name__ == "__main__":
    main()
