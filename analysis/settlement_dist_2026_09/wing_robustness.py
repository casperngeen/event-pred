#!/usr/bin/env python
"""Can the bucket wing result be killed? Four attempts.

    venv/bin/python analysis/settlement_dist_2026_09/wing_robustness.py
    (needs wing_overround.py to have written out/wing_legs_mass.parquet)

``wing_calibration.py`` found a -6.6pp longshot edge on bucket ladders and
``wing_overround.py`` showed it is not the 24% ladder overround. This is the
rest of the kill list, in the order the threats actually bite.

1. **Is the winner even in the sample?** We only observe legs that traded on the
   snapshot day. Bucket ladders partition the outcome space, so exactly one
   *listed* bucket wins -- but if the winner routinely failed to trade, the
   observed legs would lose more often than their prices imply for a purely
   mechanical reason, and every band would show negative edge. This is the
   coverage defect from ``relations_findings.md`` item 1, in its most damaging
   possible form.
2. **Is it a few events wearing a large n?** §13.5's lesson. Per-event P&L,
   share positive, and the mean after dropping the best events.
3. **Is it one regime?** Year by year, and year-clustered.
4. **Is it one series?** WTI and WTIW priced separately -- WTIW is the weekly
   contract and is a partly independent sample.

In-sample only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path("analysis/settlement_dist_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100
N_BOOT = 10000


def fee_cents(price_cents) -> np.ndarray:
    p = np.asarray(price_cents, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(price_cents) -> np.ndarray:
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents, dtype=float) - 50.0) > 40.0] = 1.0
    return out


def cluster_boot(df: pl.DataFrame, col: str, by: str, seed: int = 0):
    groups = [g[col].to_numpy() for _, g in df.group_by(by)]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    boot = np.array([
        np.concatenate([groups[i] for i in rng.choice(idx, len(idx), replace=True)]).mean()
        for _ in range(N_BOOT)])
    return float(np.concatenate(groups).mean()), float(np.percentile(boot, 2.5)), \
        float(np.percentile(boot, 97.5)), float((boot <= 0).mean())


def main() -> None:
    d = pl.read_parquet(OUT / "wing_legs_mass.parquet")

    # ------------------------------------------------------------- threat 1
    print("=== 1. is the winning bucket present in the observed ladder? ===")
    print("Bucket ladders partition the outcome space, so exactly one LISTED bucket")
    print("wins. If the winner routinely did not trade, the negative edge is an")
    print("artifact of which legs we can see, not a property of the prices.\n")
    ev = d.group_by("event_ticker", "kind").agg(
        pl.col("win").max().alias("winner_seen"), pl.len().alias("legs"),
        pl.col("mass").first(), pl.col("yr").first(), pl.col("series").first())
    with pl.Config(float_precision=3):
        print(ev.group_by("kind").agg(
            pl.len().alias("events"),
            pl.col("winner_seen").mean().alias("P_winner_observed"),
            pl.col("legs").median().alias("med_legs")))
    coh = ev.filter((pl.col("kind") == "bucket") & pl.col("mass").is_between(0.95, 1.10))
    print(f"\ncoherent bucket ladders (mass 0.95-1.10): {coh.height} events, "
          f"P(winner observed) = {coh['winner_seen'].mean():.3f}")
    print("-> 95-97%. The observed bucket ladder is nearly complete; this does not")
    print("   explain a -6.6pp edge. (Threshold legs are cumulative and overlapping,")
    print("   so the same statistic does not mean the same thing for them.)")

    # the trade, bucket wings
    b = d.filter((pl.col("kind") == "bucket") &
                 pl.col("p_long").is_between(3, 20, closed="left"))
    fav = 100.0 - b["p_long"].to_numpy()
    gross = 100.0 * (1 - b["long_win"].to_numpy()) - fav
    b = b.with_columns(pl.Series("net", gross - (fee_cents(fav) + spread_cents(fav) / 2.0)))

    # ------------------------------------------------------------- threat 2
    print("\n=== 2. a few events wearing a large n? (the §13.5 failure mode) ===")
    pe = b.group_by("event_ticker").agg(pl.col("net").mean().alias("net"))
    n = np.sort(pe["net"].to_numpy())
    print(f"per-event net: n = {len(n)} events   mean {n.mean():+.2f}c   "
          f"median {np.median(n):+.2f}c   share > 0: {(n > 0).mean():.2f}")
    print(f"drop the best 5 events:  {n[:-5].mean():+.2f}c")
    print(f"drop the best decile:    {n[:int(len(n) * 0.9)].mean():+.2f}c")
    print(f"drop the best quartile:  {n[:int(len(n) * 0.75)].mean():+.2f}c")

    # ------------------------------------------------------------- threat 3
    print("\n=== 3. one regime? ===")
    with pl.Config(tbl_rows=20, float_precision=2):
        print(b.group_by("yr").agg(
            pl.len().alias("legs"), pl.col("event_ticker").n_unique().alias("events"),
            pl.col("p_long").mean().alias("price"),
            (100.0 * pl.col("long_win").mean()).alias("realised"),
            pl.col("net").mean().alias("net")).sort("yr"))
    obs, lo, hi, pneg = cluster_boot(b, "net", by="yr", seed=1)
    print(f"year-clustered: {obs:+.2f}c   95% CI [{lo:+.2f}, {hi:+.2f}]   "
          f"P(mean<=0) = {pneg:.3f}")
    print("-> present every year, but decaying: 2024 is under half of 2022-23.")

    # ------------------------------------------------------------- threat 4
    print("\n=== 4. one series? ===")
    rows = []
    for s in ("WTI", "WTIW"):
        sub = b.filter(pl.col("series") == s)
        if sub.height < 20:
            continue
        obs, lo, hi, pneg = cluster_boot(sub, "net", by="event_ticker")
        rows.append(dict(series=s, legs=sub.height,
                         events=sub["event_ticker"].n_unique(),
                         net=obs, ci_lo=lo, ci_hi=hi, p_le0=pneg))
    with pl.Config(float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    print("-> WTI and WTIW agree. WTIW is the weekly contract, a partly")
    print("   independent sample on the same underlying.")


if __name__ == "__main__":
    main()
