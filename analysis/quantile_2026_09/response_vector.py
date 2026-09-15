#!/usr/bin/env python
"""A resolves; which moments of B's belief distribution move?

    venv/bin/python analysis/quantile_2026_09/response_vector.py
    (needs build_ladder_panel.py)

Every previous study collapsed B's response to a scalar -- one representative
leg's price change, or one leg's settlement. B is a distribution, so A's
resolution acts on a vector. Using the reconstruction-free moments:

    d_q50      change in the ladder's median        -- LOCATION
    d_log_iqr  change in log interquartile range    -- WIDTH
    d_skew     change in Bowley's quartile skew     -- ASYMMETRY

Location is scaled by B's own IQR before the trigger, so series in different
units pool.

The control that matters
------------------------
``vol_term.py`` §3b established that B's uncertainty shrinks as B nears its own
close regardless of anything A does, so a window straddling A's resolution also
advances B's own clock. Every test below therefore compares straddling steps
against non-straddling steps **matched on elapsed days and B's own horizon**.
Without that control the term structure fakes a cross-event effect.

Sign restriction as in the lead-lag study: ``direction = HAWKISH[A]*HAWKISH[B]``,
zero fitted parameters, with U3 and JOBLESSCLAIMS carrying -1 so the
falsification cell is live.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel.registry import is_same_release
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/quantile_2026_09/out")
N_BOOT = 10000
MAXGAP = 7

HAWKISH = {"CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1,
           "PCECORE": +1, "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1,
           "CPIFOOD": +1, "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1,
           "U3": -1, "JOBLESSCLAIMS": -1, "GDP": +1, "ISMPMI": +1, "FED": +1}


def cluster_boot(v, g, seed=0):
    u, inv = np.unique(g, return_inverse=True)
    k = len(u)
    s = np.bincount(inv, weights=v, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    b = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1e-9)
    return (float(v.mean()), float(np.percentile(b, 2.5)),
            float(np.percentile(b, 97.5)), float((b <= 0).mean()))


def cluster_boot_diff(v_a, v_b, g_a, g_b, seed=0):
    """Clustered bootstrap of mean(a) - mean(b), resampling whole events.

    Replaces an earlier concat([a, -b]) shortcut, whose bootstrap statistic is
    (sum_a - sum_b)/(n_a + n_b) and therefore does NOT estimate the reported
    difference once the two groups have unequal sizes. Here both sides share one
    event universe and an event enters the resample carrying whatever rows it
    has on each side.
    """
    uniq = np.unique(np.concatenate([g_a, g_b]))
    k = len(uniq)
    ia = np.searchsorted(uniq, g_a)
    ib = np.searchsorted(uniq, g_b)
    sa = np.bincount(ia, weights=v_a, minlength=k)
    ca = np.bincount(ia, minlength=k).astype(float)
    sb = np.bincount(ib, weights=v_b, minlength=k)
    cb = np.bincount(ib, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    ma = sa[pick].sum(1) / np.maximum(ca[pick].sum(1), 1e-9)
    mb = sb[pick].sum(1) / np.maximum(cb[pick].sum(1), 1e-9)
    boot = ma - mb
    obs = float(v_a.mean() - v_b.mean())
    return (obs, float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


def main() -> None:
    q = pl.read_parquet(OUT / "ladder_panel.parquet")
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    sp = sp.with_columns(
        (pl.col("surprise") / pl.col("implied_std")).alias("z"),
        pl.col("close_time").dt.date().alias("tdate"))
    sp = sp.filter(pl.col("z").is_finite() & pl.col("series").is_in(list(HAWKISH)))

    q = q.filter(pl.col("series").is_in(list(HAWKISH)))
    print(f"ladder-days with q50: {q['q50'].is_not_null().sum()}   "
          f"with iqr: {q['iqr'].is_not_null().sum()}")

    # consecutive steps within a target event, both endpoints defined
    q = q.sort("event_ticker", "date")
    st = (q.with_columns([
        pl.col("q50").shift(1).over("event_ticker").alias("q50_0"),
        pl.col("iqr").shift(1).over("event_ticker").alias("iqr_0"),
        pl.col("skew_q").shift(1).over("event_ticker").alias("skew_0"),
        pl.col("date").shift(1).over("event_ticker").alias("d0"),
    ]).drop_nulls(["q50", "q50_0", "iqr", "iqr_0", "d0"])
        .with_columns([
            (pl.col("date") - pl.col("d0")).dt.total_days().alias("elapsed"),
            (pl.col("q50") - pl.col("q50_0")).alias("d_q50_raw"),
            (pl.col("iqr").log() - pl.col("iqr_0").log()).alias("d_log_iqr"),
            (pl.col("skew_q") - pl.col("skew_0")).alias("d_skew"),
        ])
        .filter(pl.col("elapsed").is_between(1, MAXGAP)
                & pl.col("d_log_iqr").is_finite()))
    st = st.with_columns((pl.col("d_q50_raw") / pl.col("iqr_0")).alias("d_q50"))
    print(f"usable consecutive steps: {st.height}   "
          f"target events: {st['event_ticker'].n_unique()}")

    # which foreign series resolved inside each step, and with what signed z
    trig: dict = {}
    for r in sp.iter_rows(named=True):
        trig.setdefault(r["series"], []).append((r["tdate"], r["z"]))

    rows = []
    for r in st.iter_rows(named=True):
        tgt = r["series"]
        hits = []
        for s, lst in trig.items():
            if s == tgt or is_same_release(s, tgt) or s not in HAWKISH:
                continue
            for dd, z in lst:
                if r["d0"] < dd <= r["date"]:
                    hits.append(HAWKISH[s] * HAWKISH[tgt] * z)
        rows.append(dict(event_ticker=r["event_ticker"], series=tgt,
                         date=r["date"], elapsed=r["elapsed"],
                         days_to_close=r["days_to_close"],
                         d_q50=r["d_q50"], d_log_iqr=r["d_log_iqr"],
                         d_skew=r["d_skew"],
                         n_trig=len(hits),
                         signal=float(np.sum(hits)) if hits else 0.0))
    x = pl.DataFrame(rows).filter(pl.col("d_q50").is_finite()
                                  & pl.col("days_to_close").is_not_null())
    x = x.with_columns(
        pl.when(pl.col("days_to_close") < 7).then(pl.lit("0-6d"))
          .when(pl.col("days_to_close") < 21).then(pl.lit("7-20d"))
          .otherwise(pl.lit("21d+")).alias("h"),
        (pl.col("n_trig") > 0).alias("strad"))
    print(f"steps straddling a foreign resolution: {int(x['strad'].sum())} "
          f"of {x.height}\n")

    # -------------------------------------------------------- 1. WIDTH
    print("=== 1. WIDTH: does A's resolution shrink B's IQR beyond normal decay? ===")
    print("Matched on elapsed days and B's own horizon (the vol_term §3b lesson).\n")
    rows = []
    for h in ("0-6d", "7-20d", "21d+"):
        for el in ((1, 1), (2, 3), (4, 7)):
            s = x.filter((pl.col("h") == h) & pl.col("elapsed").is_between(*el))
            a, b = s.filter(pl.col("strad")), s.filter(~pl.col("strad"))
            if a.height < 20 or b.height < 20:
                continue
            d, lo, hi, _ = cluster_boot_diff(
                a["d_log_iqr"].to_numpy(), b["d_log_iqr"].to_numpy(),
                a["event_ticker"].to_numpy(), b["event_ticker"].to_numpy())
            rows.append(dict(horizon=h, elapsed=f"{el[0]}-{el[1]}d",
                             n_s=a.height, n_n=b.height,
                             diff=d, ci_lo=lo, ci_hi=hi))
    with pl.Config(tbl_rows=12, float_precision=4, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    # -------------------------------------------------------- 2. LOCATION
    print("\n=== 2. LOCATION: does B's median move with the aligned signal? ===")
    print("d_q50 is in units of B's own prior IQR. Straddling steps only,")
    print("split by the sign of the aligned signal. Clustered on B's event.\n")
    s = x.filter(pl.col("strad") & (pl.col("signal") != 0))
    pos = s.filter(pl.col("signal") > 0)
    neg = s.filter(pl.col("signal") < 0)
    print(f"steps with a signed signal: {s.height}  "
          f"(+{pos.height} / -{neg.height})   events {s['event_ticker'].n_unique()}")
    if pos.height >= 20 and neg.height >= 20:
        obs, lo, hi, pneg = cluster_boot_diff(
            pos["d_q50"].to_numpy(), neg["d_q50"].to_numpy(),
            pos["event_ticker"].to_numpy(), neg["event_ticker"].to_numpy())
        print(f"\nmean d_q50 | signal>0 : {pos['d_q50'].mean():+.4f} IQR")
        print(f"mean d_q50 | signal<0 : {neg['d_q50'].mean():+.4f} IQR")
        print(f"aligned difference     : {obs:+.4f} IQR   "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]   P(<=0) = {pneg:.3f}")
        r = np.corrcoef(s["signal"].to_numpy(), s["d_q50"].to_numpy())[0, 1]
        print(f"corr(signal, d_q50) = {r:+.4f}   n = {s.height}")

    # -------------------------------------------------------- 3. SHAPE
    print("\n=== 3. ASYMMETRY: does the quartile skew move? ===")
    sk = s.drop_nulls("d_skew")
    if sk.height >= 40:
        p2 = sk.filter(pl.col("signal") > 0)
        n2 = sk.filter(pl.col("signal") < 0)
        if p2.height >= 15 and n2.height >= 15:
            obs, lo, hi, pneg = cluster_boot_diff(
                p2["d_skew"].to_numpy(), n2["d_skew"].to_numpy(),
                p2["event_ticker"].to_numpy(), n2["event_ticker"].to_numpy())
            print(f"aligned d_skew: {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"   P(<=0) = {pneg:.3f}   n = {sk.height}")
    else:
        print(f"only {sk.height} steps carry a defined skew change - not tested")

    x.write_parquet(OUT / "response_vector.parquet")
    print(f"\nwrote {OUT / 'response_vector.parquet'}")


if __name__ == "__main__":
    main()
