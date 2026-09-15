#!/usr/bin/env python
"""Direction 1: two separately-traded ladders that price the same unknown.

    venv/bin/python analysis/arbitrage_2026_09/identity_cpi.py

Kalshi lists, for the *same reference month*, both

    CPI      "Will inflation rise more than 0.7% in April?"   -> MoM
    CPIYOY   "Will the rate of CPI inflation be above 2.5%
              for the year ending in April?"                  -> YoY

and the same pair for core (CPICORE / CPICOREYOY). These are not two related
quantities. They are **one unknown under an affine map**::

    (1 + YoY_t) = (1 + YoY_{t-1}) * (1 + MoM_t) / (1 + MoM_{t-12})
    =>  YoY_t ~= YoY_{t-1} + MoM_t - MoM_{t-12}  =  c_t + MoM_t

where ``c_t`` is a constant *already published* when both ladders trade: it
needs last month's YoY print and the year-ago MoM print, nothing about the
future. So the two ladders must imply the **same distribution, translated**:

    implied_std(YoY)  ==  implied_std(MoM)              <- needs no history at all
    implied_mean(YoY) -  implied_mean(MoM)  ==  c_t     <- needs the two prints

The first test is the strong one. It is parameter-free, needs no external data,
and any wedge is a pricing inconsistency rather than a forecast disagreement --
two ladders cannot honestly disagree about the *width* of a distribution over a
quantity they both settle on.

Three things this has to establish before any wedge means anything
------------------------------------------------------------------
1. **The identity's own noise floor.** BLS publishes MoM and YoY each rounded
   to 0.1pp, and ``resolved_value`` carries ladder granularity on top. The
   additive approximation also drops a second-order term. §1 measures the
   residual on resolved values, and nothing smaller than that is a finding.
2. **Reconstruction noise.** ``implied_std`` comes from ``recover_pdf`` on a
   discretised ladder with open tails, and the MoM and YoY ladders have
   different strike spacings and different coverage. §2 gates on coverage and
   reports the wedge against a matched-quality subset.
3. **Simultaneity.** Prices are last trades, not quotes. A wedge between two
   ladders last traded hours apart is staleness, not arbitrage. §3 restricts to
   days where both sides actually traded.

In-sample only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/arbitrage_2026_09/out")

PAIRS = [("CPI", "CPIYOY"), ("CPICORE", "CPICOREYOY")]
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_month(ticker: str):
    """'CPIYOY-23APR' -> date(2023, 4, 1). None if it does not parse."""
    tail = ticker.split("-", 1)[1] if "-" in ticker else ""
    if len(tail) < 5:
        return None
    yy, mmm = tail[:2], tail[2:5].upper()
    if not yy.isdigit() or mmm not in MONTHS:
        return None
    return dt.date(2000 + int(yy), MONTHS[mmm], 1)


def shift_months(d: dt.date, k: int) -> dt.date:
    m = d.month - 1 + k
    return dt.date(d.year + m // 12, m % 12 + 1, 1)


def main() -> None:
    n = pl.read_parquet(PANELS / "node_panel_event.parquet")
    n = n.with_columns(
        pl.col("event_ticker").map_elements(parse_month, return_dtype=pl.Date).alias("ref"))
    n = n.filter(pl.col("ref").is_not_null())

    resolved = (n.group_by("series", "ref")
                .agg(pl.col("resolved_value").first())
                .drop_nulls("resolved_value"))
    rmap = {(r["series"], r["ref"]): r["resolved_value"]
            for r in resolved.iter_rows(named=True)}

    # ---------------------------------------------------------------- §1
    print("=== 1. the identity's own noise floor, on resolved values ===")
    print("residual = YoY_t - (YoY_{t-1} + MoM_t - MoM_{t-12}), in pp.")
    print("Nothing smaller than this residual can be called a mispricing.\n")
    floors = {}
    for mom_s, yoy_s in PAIRS:
        rows = []
        for (s, ref), v in rmap.items():
            if s != yoy_s:
                continue
            prev = rmap.get((yoy_s, shift_months(ref, -1)))
            back = rmap.get((mom_s, shift_months(ref, -12)))
            cur = rmap.get((mom_s, ref))
            if prev is None or back is None or cur is None:
                continue
            rows.append(dict(ref=ref, resid=v - (prev + cur - back)))
        if not rows:
            print(f"  {mom_s}/{yoy_s}: no months with all four prints")
            continue
        r = np.array([x["resid"] for x in rows])
        floors[(mom_s, yoy_s)] = float(np.std(r))
        print(f"  {mom_s}/{yoy_s}: n = {len(r):2d}   mean {r.mean():+.3f}pp   "
              f"sd {r.std():.3f}pp   |resid| p50 {np.median(np.abs(r)):.3f}   "
              f"p90 {np.percentile(np.abs(r), 90):.3f}")

    # ---------------------------------------------------------------- §2
    print("\n=== 2. do the two ladders imply the same WIDTH? ===")
    print("implied_std(YoY) vs implied_std(MoM), same reference month, same day.")
    print("They price one unknown under an affine map, so these must be equal.")
    print("No history, no weights, no external data enters this test.\n")

    all_rows = []
    for mom_s, yoy_s in PAIRS:
        a = n.filter(pl.col("series") == mom_s).select(
            "ref", "date", "days_to_close",
            pl.col("implied_std").alias("sd_mom"),
            pl.col("implied_mean").alias("mu_mom"),
            pl.col("n_fresh_legs").alias("legs_mom"),
            pl.col("n_submarkets").alias("sub_mom"))
        b = n.filter(pl.col("series") == yoy_s).select(
            "ref", "date",
            pl.col("implied_std").alias("sd_yoy"),
            pl.col("implied_mean").alias("mu_yoy"),
            pl.col("n_fresh_legs").alias("legs_yoy"),
            pl.col("n_submarkets").alias("sub_yoy"))
        j = a.join(b, on=["ref", "date"], how="inner").drop_nulls(
            ["sd_mom", "sd_yoy", "mu_mom", "mu_yoy"])
        if j.is_empty():
            continue
        j = j.with_columns(
            pl.lit(f"{mom_s}/{yoy_s}").alias("pair"),
            (pl.col("sd_yoy") - pl.col("sd_mom")).alias("sd_gap"),
            (pl.col("sd_yoy") / pl.col("sd_mom")).alias("sd_ratio"),
            (pl.col("mu_yoy") - pl.col("mu_mom")).alias("mu_gap"),
        )
        all_rows.append(j)
    if not all_rows:
        print("no overlapping days")
        return
    j = pl.concat(all_rows, how="diagonal")
    assert_no_oos(j.with_columns(
        pl.col("date").cast(pl.Datetime).dt.replace_time_zone("UTC").alias("t")),
        time_col="t")

    def width_table(d: pl.DataFrame, label: str):
        rows = []
        for pair in d["pair"].unique().to_list():
            s = d.filter(pl.col("pair") == pair)
            if s.height < 10:
                continue
            rt = s["sd_ratio"].to_numpy()
            rows.append(dict(pair=pair, n_days=s.height,
                             months=s["ref"].n_unique(),
                             sd_mom=float(s["sd_mom"].mean()),
                             sd_yoy=float(s["sd_yoy"].mean()),
                             ratio_p50=float(np.median(rt)),
                             ratio_p10=float(np.percentile(rt, 10)),
                             ratio_p90=float(np.percentile(rt, 90)),
                             frac_yoy_wider=float((rt > 1).mean())))
        print(f"--- {label}")
        with pl.Config(tbl_rows=10, float_precision=3, tbl_width_chars=210):
            print(pl.DataFrame(rows))

    width_table(j, "all overlapping days")
    gated = j.filter((pl.col("legs_mom") >= 4) & (pl.col("legs_yoy") >= 4))
    width_table(gated, "both ladders carrying >= 4 freshly traded legs")

    # ---------------------------------------------------------------- §3
    print("\n=== 3. does the LOCATION gap equal the published constant? ===")
    print("mu_yoy - mu_mom should equal c_t = YoY_{t-1} - MoM_{t-12}, a number")
    print("already printed when both ladders trade. wedge = (gap - c), in pp.\n")
    cvals = []
    for r in j.iter_rows(named=True):
        mom_s, yoy_s = r["pair"].split("/")
        prev = rmap.get((yoy_s, shift_months(r["ref"], -1)))
        back = rmap.get((mom_s, shift_months(r["ref"], -12)))
        cvals.append(None if prev is None or back is None else prev - back)
    j = j.with_columns(pl.Series("c_t", cvals, dtype=pl.Float64))
    jc = j.drop_nulls("c_t").with_columns(
        (pl.col("mu_gap") - pl.col("c_t")).alias("wedge"))
    rows = []
    for pair in jc["pair"].unique().to_list():
        s = jc.filter(pl.col("pair") == pair)
        if s.height < 10:
            continue
        w = s["wedge"].to_numpy()
        rows.append(dict(pair=pair, n_days=s.height, months=s["ref"].n_unique(),
                         mean_wedge=float(w.mean()), sd=float(w.std()),
                         p50_abs=float(np.median(np.abs(w))),
                         p90_abs=float(np.percentile(np.abs(w), 90)),
                         floor=floors.get(tuple(pair.split("/")), np.nan)))
    with pl.Config(tbl_rows=10, float_precision=3, tbl_width_chars=210):
        print(pl.DataFrame(rows))
    print("\n`floor` is §1's residual sd — the wedge has to clear it to mean anything.")

    # ------------------------------------------------------- by horizon
    print("\n=== 4. does the inconsistency shrink as the print approaches? ===")
    with pl.Config(tbl_rows=20, float_precision=3):
        print(jc.with_columns(
            pl.when(pl.col("days_to_close") <= 2).then(pl.lit("0-2d"))
              .when(pl.col("days_to_close") <= 7).then(pl.lit("3-7d"))
              .when(pl.col("days_to_close") <= 20).then(pl.lit("8-20d"))
              .otherwise(pl.lit("21d+")).alias("h")
        ).group_by("pair", "h").agg(
            pl.len().alias("n"),
            pl.col("sd_ratio").median().alias("sd_ratio_p50"),
            pl.col("wedge").abs().median().alias("abs_wedge_p50"),
        ).sort("pair", "h"))

    OUT.mkdir(parents=True, exist_ok=True)
    jc.write_parquet(OUT / "identity_cpi.parquet")
    print(f"\nwrote {OUT / 'identity_cpi.parquet'}   rows {jc.height}")


if __name__ == "__main__":
    main()
