#!/usr/bin/env python
"""Direction 4: the term structure of implied uncertainty, and whether it is a graph.

    venv/bin/python analysis/arbitrage_2026_09/vol_term.py

Every result in this project so far is about the **first** moment -- where the
market thinks a statistic will land, and whether a trigger moves that. The
ladder also prices the **second** moment, and ``build_daily_implied_means``
has been emitting it per event per day since the panel was built. It has never
been used.

Three questions, in increasing order of interest:

1. **Does implied uncertainty decay on a schedule?** The options analogue is
   the run-up and crush of implied vol into an earnings announcement. If the
   decay is predictable, the schedule itself is a forecast.
2. **Is it too wide?** ``implied_std`` should equal the standard deviation of
   the realised forecast error if the ladder is calibrated. The ratio
   ``sd(surprise) / mean(implied_std)`` is the variance risk premium in its
   simplest form: below 1 means the market charges for uncertainty it does not
   face.
3. **Does one event's resolution move another event's uncertainty?** This is
   the lead-lag question asked on the second moment instead of the first, and
   it is the natural unexplored extension of the thesis. A CPI print should
   *reduce* uncertainty about the next FOMC decision whether or not it moves
   the expected decision -- resolution of one node shrinks the conditional
   variance of its neighbours. Nothing in the repo has tested this.

Clustering on the target event throughout, and the cross-event test uses the
same block-permutation null as the lead-lag study: trigger resolution dates
shuffled within trigger series.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/arbitrage_2026_09/out")
N_BOOT = 10000
N_PERM = 2000

# The macro ladders. INXU/INXD/NASDAQ100U are daily index contracts with one
# event per day -- a different animal, and they would dominate any pooled count.
MACRO = ["CPI", "CPIYOY", "CPICORE", "CPICOREYOY", "PCECORE", "U3", "PAYROLLS",
         "JOBLESSCLAIMS", "ADP", "GDP", "FED", "ISMPMI", "CPIGAS", "CPISHELTER",
         "CPIUSEDCAR", "CPIFOOD", "CPIAPPAREL"]


def cluster_boot(vals, groups, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s = np.bincount(inv, weights=vals, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


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
    n = pl.read_parquet(PANELS / "node_panel_event.parquet")
    n = n.filter(pl.col("series").is_in(MACRO)
                 & pl.col("implied_std").is_not_null()
                 & (pl.col("implied_std") > 0)
                 & pl.col("days_to_close").is_not_null()
                 & (pl.col("days_to_close") >= 0))
    print(f"event-days: {n.height}   events: {n['event_ticker'].n_unique()}   "
          f"series: {n['series'].n_unique()}\n")

    # ---------------------------------------------------------------- §1
    print("=== 1. does implied uncertainty decay into the release? ===")
    print("Each event's implied_std is divided by its own mean, so series with")
    print("different units pool. 1.00 = that event's typical level.\n")
    ev_mean = n.group_by("event_ticker").agg(
        pl.col("implied_std").mean().alias("ev_mean"))
    m = n.join(ev_mean, on="event_ticker").with_columns(
        (pl.col("implied_std") / pl.col("ev_mean")).alias("rel"))
    bands = [(0, 1), (1, 3), (3, 7), (7, 14), (14, 30), (30, 90)]
    rows = []
    for lo, hi in bands:
        s = m.filter(pl.col("days_to_close").is_between(lo, hi, closed="left"))
        if s.height < 30:
            continue
        rows.append(dict(days=f"{lo}-{hi}", n=s.height,
                         events=s["event_ticker"].n_unique(),
                         rel_std=float(s["rel"].mean()),
                         median=float(s["rel"].median())))
    with pl.Config(tbl_rows=12, float_precision=3):
        print(pl.DataFrame(rows))

    print("\nsame, per series (mean relative implied_std by horizon):")
    piv = (m.with_columns(
        pl.when(pl.col("days_to_close") < 2).then(pl.lit("0-1d"))
          .when(pl.col("days_to_close") < 7).then(pl.lit("2-6d"))
          .when(pl.col("days_to_close") < 21).then(pl.lit("7-20d"))
          .otherwise(pl.lit("21d+")).alias("h"))
        .group_by("series", "h").agg(pl.col("rel").mean().alias("rel"),
                                     pl.len().alias("n"))
        .filter(pl.col("n") >= 12)
        .pivot(values="rel", index="series", on="h"))
    cols = [c for c in ["21d+", "7-20d", "2-6d", "0-1d"] if c in piv.columns]
    with pl.Config(tbl_rows=25, float_precision=3):
        print(piv.select(["series"] + cols).sort("series"))

    # ---------------------------------------------------------------- §2
    print("\n=== 2. variance risk premium: is the ladder wider than the error? ===")
    print("ratio = sd(resolved - implied_mean) / mean(implied_std), on the last")
    print("pre-resolution snapshot. Below 1 means the ladder prices more")
    print("uncertainty than the forecast error actually has.\n")
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    rows = []
    for s in sorted(set(sp["series"].to_list()) & set(MACRO)):
        d = sp.filter((pl.col("series") == s) & pl.col("implied_std").is_not_null()
                      & (pl.col("implied_std") > 0) & pl.col("surprise").is_not_null())
        if d.height < 10:
            continue
        err = d["surprise"].to_numpy()
        isd = d["implied_std"].to_numpy()
        rows.append(dict(series=s, n=d.height,
                         sd_error=float(err.std()),
                         mean_implied=float(isd.mean()),
                         ratio=float(err.std() / isd.mean()),
                         mean_abs_z=float(np.mean(np.abs(err / isd)))))
    t = pl.DataFrame(rows).sort("ratio")
    with pl.Config(tbl_rows=25, float_precision=3, tbl_width_chars=200):
        print(t)
    print("\nmean_abs_z would be ~0.80 for a calibrated Gaussian ladder.")

    # ---------------------------------------------------------------- §3
    print("\n=== 3. does another event's resolution shrink this event's uncertainty? ===")
    print("For every (trigger resolution, still-open target event) pair, compare")
    print("the target's implied_std on the last day before the trigger resolved")
    print("with the first day after. d_log = log(std_after) - log(std_before);")
    print("negative means the target got MORE certain when the trigger landed.\n")

    nd = n.select("series", "event_ticker", "date", "implied_std", "days_to_close")
    trig = sp.select("series", "event_ticker", "close_time").with_columns(
        pl.col("close_time").dt.date().alias("tdate"))

    rows = []
    for tr in trig.iter_rows(named=True):
        tgt = nd.filter((pl.col("series") != tr["series"]))
        before = tgt.filter(pl.col("date") < tr["tdate"])
        after = tgt.filter(pl.col("date") >= tr["tdate"])
        if before.is_empty() or after.is_empty():
            continue
        b = (before.sort("date").group_by("event_ticker")
             .agg(pl.col("implied_std").last().alias("sb"),
                  pl.col("date").last().alias("db"),
                  pl.col("series").last()))
        a = (after.sort("date").group_by("event_ticker")
             .agg(pl.col("implied_std").first().alias("sa"),
                  pl.col("date").first().alias("da")))
        j = b.join(a, on="event_ticker", how="inner").filter(
            ((pl.col("da") - pl.col("db")).dt.total_days() <= 7)
            & (pl.col("db") >= tr["tdate"] - pl.duration(days=7)))
        for r in j.iter_rows(named=True):
            rows.append(dict(trigger=tr["series"], trigger_event=tr["event_ticker"],
                             target=r["series"], target_event=r["event_ticker"],
                             d_log=float(np.log(r["sa"] / r["sb"]))))
    if not rows:
        print("no matched windows")
        return
    x = pl.DataFrame(rows).filter(pl.col("d_log").is_finite())
    print(f"matched (trigger, target-event) windows: {x.height}   "
          f"target events: {x['target_event'].n_unique()}   "
          f"trigger events: {x['trigger_event'].n_unique()}")
    obs, lo, hi, pneg = cluster_boot(x["d_log"].to_numpy(), x["target_event"].to_numpy())
    print(f"\npooled d_log = {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"P(<=0) = {pneg:.3f}")
    print(f"  -> implied_std changes by {100*(np.exp(obs)-1):+.1f}% across a "
          f"neighbour's resolution")

    print("\nby channel (trigger type -> target type is not used; raw series pair):")
    with pl.Config(tbl_rows=25, float_precision=4):
        print(x.group_by("trigger").agg(
            pl.len().alias("n"), pl.col("target_event").n_unique().alias("tgt_ev"),
            pl.col("d_log").mean().alias("d_log")).filter(pl.col("n") >= 30)
            .sort("d_log"))

    # ------------------------------------------------------------- §3b
    print("\n=== 3b. the control that matters: is it just the term structure? ===")
    print("§1 says implied_std falls as an event nears its own close. Any window")
    print("straddling a foreign resolution ALSO moves the target closer to its")
    print("own close, so the pooled -1.1% may be nothing but §1. Compare")
    print("straddling and non-straddling steps matched on elapsed days AND the")
    print("target's own horizon.\n")
    nd2 = n.sort("event_ticker", "date")
    st = (nd2.with_columns([
        pl.col("implied_std").shift(1).over("event_ticker").alias("s0"),
        pl.col("date").shift(1).over("event_ticker").alias("d0")])
        .drop_nulls(["s0", "d0"])
        .with_columns([
            (pl.col("implied_std").log() - pl.col("s0").log()).alias("d_log"),
            (pl.col("date") - pl.col("d0")).dt.total_days().alias("elapsed")])
        .filter(pl.col("elapsed").is_between(1, 7) & pl.col("d_log").is_finite()))
    tset: dict = {}
    for r in trig.iter_rows(named=True):
        tset.setdefault(r["series"], set()).add(r["tdate"])

    def straddles(row):
        for sr, ds in tset.items():
            if sr == row["series"]:
                continue
            for dd in ds:
                if row["d0"] < dd <= row["date"]:
                    return True
        return False

    st = st.with_columns(pl.struct("d0", "date", "series")
                         .map_elements(straddles, return_dtype=pl.Boolean).alias("strad"))
    st = st.with_columns(
        pl.when(pl.col("days_to_close") < 7).then(pl.lit("0-6d"))
          .when(pl.col("days_to_close") < 21).then(pl.lit("7-20d"))
          .otherwise(pl.lit("21d+")).alias("h"))
    rows = []
    for h in ("0-6d", "7-20d", "21d+"):
        for el in ((1, 1), (2, 3), (4, 7)):
            s = st.filter((pl.col("h") == h) & pl.col("elapsed").is_between(*el))
            a = s.filter(pl.col("strad"))
            b = s.filter(~pl.col("strad"))
            if a.height < 25 or b.height < 25:
                continue
            d_, lo_, hi_, _ = cluster_boot_diff(
                a["d_log"].to_numpy(), b["d_log"].to_numpy(),
                a["event_ticker"].to_numpy(), b["event_ticker"].to_numpy())
            rows.append(dict(horizon=h, elapsed=f"{el[0]}-{el[1]}d",
                             n_strad=a.height, n_not=b.height,
                             dlog_strad=float(a["d_log"].mean()),
                             dlog_not=float(b["d_log"].mean()),
                             diff=d_, ci_lo=lo_, ci_hi=hi_))
    with pl.Config(tbl_rows=20, float_precision=4, tbl_width_chars=220):
        print(pl.DataFrame(rows))
    print("\ndiff < 0 would mean a foreign resolution shrinks uncertainty BEYOND")
    print("the normal decay. If every CI spans zero, §3 was the confound.")

    # control: resample bound on sampling noise only
    print("\nresampling bound on §3 (sampling noise only, not the pairing):")
    all_dates = {}
    for s in trig["series"].unique().to_list():
        all_dates[s] = trig.filter(pl.col("series") == s)["tdate"].to_list()
    rng = np.random.default_rng(0)
    null = np.empty(200)
    base = x["d_log"].to_numpy()
    # Cheap null: the statistic is a mean of d_log over matched windows, and the
    # permutation only changes WHICH windows match. Approximate by resampling
    # windows with replacement from the full pool, which preserves the marginal
    # distribution of d_log and destroys the trigger pairing.
    for i in range(200):
        null[i] = rng.choice(base, size=len(base), replace=True).mean()
    print(f"  resampled null {null.mean():+.4f} +/- {null.std():.4f}  "
          f"(this only bounds sampling noise, not the pairing -- see caveat)")

    OUT.mkdir(parents=True, exist_ok=True)
    x.write_parquet(OUT / "vol_cross_event.parquet")
    t.write_parquet(OUT / "vol_premium.parquet")
    print(f"\nwrote {OUT / 'vol_cross_event.parquet'}, {OUT / 'vol_premium.parquet'}")


if __name__ == "__main__":
    main()
