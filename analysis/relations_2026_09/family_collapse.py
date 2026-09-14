#!/usr/bin/env python
"""Item 3 — collapse the CPI family, and build the labour index.
(relations_study_plan §3.1)

    venv/bin/python analysis/relations_2026_09/family_collapse.py

Two opposite readings of the same fact — several trigger ladders resolve at the
same instant from one government print:

(a) The CPI family is **one factor wearing five names**.  If the members
    correlate at ~0.9 there is no "core vs headline" contrast to estimate; they
    are five copies of one number each spending a slot of the multiple-testing
    budget.  Collapsing them raises events per node and cuts the grid.

(b) The Employment Situation is a **genuine double trigger**.  PAYROLLS and U3
    resolve from the same BLS print but carry near-orthogonal information, so
    the zero-parameter index

        z_labour = z_PAYROLLS - z_U3      (hawkish-positive by construction)

    should *dominate either component alone* if both carry signal.  That is a
    sharp test with nothing fitted, and it rehabilitates U3, the project's
    weakest falsification cell since §5, by testing it as the second component
    of a two-dimensional release rather than in isolation.

§3.1 measured the correlations on ``surprise / implied_std``.  This re-measures
them on ``s_pit`` too, and adds the two subcomponents (CPIGAS, CPIUSEDCAR) that
§3.1 explicitly left untested before folding anything in — they may carry
separate information and should not be collapsed on assumption.

The collapse is then *tested*, not asserted: the pooled factor is run through
the same ``response_panel`` -> Spearman machinery as its components, on the same
releases, so "does collapsing help" has an answer rather than a rationale.

In-sample only.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.panel.targets import response_panel, target_frames
from stg.splits import assert_no_oos
from stg.structure.stats import permutation_p, spearman, spearman_p

PANELS = Path("artifacts/panels")

CPI_FAMILY = ["CPI", "CPICORE", "CPIYOY", "CPICOREYOY"]
# §3.1(a) proposes collapsing all four.  The s_pit correlations below do not
# support that: headline and core agree at ~0.93 *within* a tenor, but the
# month-over-month and year-over-year ladders agree at only 0.18-0.48, so they
# are not one factor.  These are the two partial collapses the measurement
# actually licenses, tested beside the full one rather than instead of it.
CPI_MOM = ["CPI", "CPICORE"]
CPI_YOY = ["CPIYOY", "CPICOREYOY"]
CPI_SUBCOMPONENTS = ["CPIGAS", "CPIUSEDCAR", "CPISHELTER", "CPIFOOD", "CPIAPPAREL"]
LABOUR = ["PAYROLLS", "U3"]
TARGETS = [("FED", "any"), ("FEDDECISION", "hike"), ("FEDDECISION", "cut")]


def with_measures(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(
        (pl.col("surprise") / pl.col("implied_std")).alias("z"))


def pairwise_table(panel: pl.DataFrame, members: list[str],
                   col: str) -> pl.DataFrame:
    """Matched-release correlation between every pair of members."""
    rows = []
    for a, b in itertools.combinations(members, 2):
        la = panel.filter(pl.col("series") == a).select("close_time", col)
        lb = panel.filter(pl.col("series") == b).select("close_time", col)
        m = la.join(lb, on="close_time", how="inner", suffix="_b").drop_nulls()
        if m.height < 8:
            continue
        x = m[col].to_numpy().astype(float)
        y = m[f"{col}_b"].to_numpy().astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        x, y = x[ok], y[ok]
        r, p = stats.pearsonr(x, y)
        rows.append(dict(measure=col, a=a, b=b, n=int(x.size), r=float(r),
                         p=float(p), sign_agree=float((np.sign(x) == np.sign(y)).mean())))
    return pl.DataFrame(rows)


def synthetic(panel: pl.DataFrame, name: str, expr: pl.Expr,
              members: list[str], require_all: bool = False) -> pl.DataFrame:
    """One synthetic trigger row per release instant.

    ``expr`` is evaluated per release over the member rows present at that
    instant.  ``require_all=False`` keeps releases where only some members
    printed — the practical case, since family members are not all present on
    every date — at the cost of the factor meaning slightly different things
    across rows; ``require_all=True`` is the strict version, reported beside it.
    """
    sub = panel.filter(pl.col("series").is_in(members))
    g = sub.group_by("close_time").agg(
        expr.alias("surprise"),
        pl.len().alias("k"),
        pl.col("event_ticker").sort().first().alias("event_ticker"),
        pl.col("snap_date").max().alias("snap_date"),
    )
    if require_all:
        g = g.filter(pl.col("k") == len(members))
    return (g.filter(pl.col("surprise").is_finite())
            .with_columns(pl.lit(name).alias("series"))
            .select("series", "event_ticker", "close_time", "snap_date", "surprise")
            .sort("close_time"))


def edge(s_panel: pl.DataFrame, target: str, side: str, frames, mk, tr,
         restrict: set | None = None) -> dict:
    """One trigger->target cell: the Stage-1 statistic, nothing new."""
    rp = response_panel(s_panel, target, side, horizon="dormant",
                        markets=mk, trades=tr, frames=frames)
    if rp.height == 0:
        return dict(n=0)
    if restrict is not None:
        rp = rp.filter(pl.col("target_event").is_in(list(restrict)))
    if rp.height < 8:
        return dict(n=int(rp.height))
    S = rp["surprise"].to_numpy().astype(float)
    R = rp["response"].to_numpy().astype(float)
    rho = spearman(S, R)
    rng = np.random.default_rng(0)
    return dict(n=int(len(S)), n_targets=int(rp["target_event"].n_unique()),
                rho=float(rho), p_asym=float(spearman_p(rho, len(S))),
                p_perm=float(permutation_p(S, R, 2000, rng=rng)),
                events=set(rp["target_event"].to_list()))


def main() -> None:
    panel = with_measures(pl.read_parquet(PANELS / "surprise_panel.parquet"))
    assert_no_oos(panel, time_col="close_time")
    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)

    # ---------------------------------------------------------------- (a)
    print("=== simultaneity: how many trigger series share a close_time ===")
    k = (panel.group_by("close_time").agg(pl.len().alias("k"))["k"]
         .value_counts().sort("k"))
    print(k)

    print("\n=== CPI family, matched-release correlation ===")
    for col in ("z", "s_pit"):
        print(f"\n-- {col} --")
        with pl.Config(tbl_rows=30, float_precision=3):
            print(pairwise_table(panel, CPI_FAMILY, col).sort("r", descending=True))

    print("\n=== the subcomponents §3.1 left untested, vs the family ===")
    for col in ("z", "s_pit"):
        print(f"\n-- {col} --")
        tab = pairwise_table(panel, CPI_FAMILY + CPI_SUBCOMPONENTS, col)
        tab = tab.filter(pl.col("a").is_in(CPI_SUBCOMPONENTS)
                         | pl.col("b").is_in(CPI_SUBCOMPONENTS))
        with pl.Config(tbl_rows=40, float_precision=3):
            print(tab.sort("r", descending=True))

    print("\n=== the labour release: PAYROLLS vs U3 ===")
    for col in ("z", "s_pit"):
        with pl.Config(float_precision=3):
            print(pairwise_table(panel, LABOUR, col))

    # ---------------------------------------------------------------- (b)(c)
    # The "same releases" control below is partly a *date* control, and saying
    # so is the difference between a finding and an artefact: U3's events that
    # have no PAYROLLS partner are almost all from 2022 and early 2023, before
    # the payrolls ladder was trading.  So "U3 measured on joint releases" and
    # "U3 measured after 2023-04" are nearly the same subsample, and this data
    # cannot separate "U3 reads correctly alongside payrolls" from "U3 reads
    # correctly in the later period".
    print("\n=== is the joint-release subset also a date subset? ===")
    for a, b in (("U3", "PAYROLLS"), ("PAYROLLS", "U3")):
        partner = set(panel.filter(pl.col("series") == b)["close_time"].to_list())
        d = (panel.filter(pl.col("series") == a)
             .with_columns(pl.col("close_time").is_in(list(partner)).alias("joint")))
        print(d.group_by("joint").agg(
            pl.len().alias("n"),
            pl.col("close_time").min().alias("first"),
            pl.col("close_time").max().alias("last")).sort("joint"),
            f"  <- {a}, partnered with {b}")

    print("\n=== collapsed factors ===")
    infl_pit = synthetic(panel, "INFL_PIT", pl.col("s_pit").mean(), CPI_FAMILY)
    infl_z = synthetic(panel, "INFL_Z", pl.col("z").mean(), CPI_FAMILY)
    mom_pit = synthetic(panel, "INFL_MOM_PIT", pl.col("s_pit").mean(), CPI_MOM)
    yoy_pit = synthetic(panel, "INFL_YOY_PIT", pl.col("s_pit").mean(), CPI_YOY)
    # hawkish-positive: payrolls up is hawkish, unemployment up is dovish
    lab_z = synthetic(
        panel, "LABOUR_Z",
        (pl.col("z").filter(pl.col("series") == "PAYROLLS").sum()
         - pl.col("z").filter(pl.col("series") == "U3").sum()),
        LABOUR, require_all=True)
    lab_pit = synthetic(
        panel, "LABOUR_PIT",
        (pl.col("s_pit").filter(pl.col("series") == "PAYROLLS").sum()
         - pl.col("s_pit").filter(pl.col("series") == "U3").sum()),
        LABOUR, require_all=True)
    for f in (infl_pit, infl_z, mom_pit, yoy_pit, lab_z, lab_pit):
        print(f"{f['series'][0]:11s} releases={f.height}")

    components = {
        "INFL": [("CPI", "surprise"), ("CPI", "s_pit"), ("CPICORE", "surprise"),
                 ("CPIYOY", "surprise"), ("CPICOREYOY", "surprise")],
        "LABOUR": [("PAYROLLS", "surprise"), ("PAYROLLS", "s_pit"),
                   ("U3", "surprise"), ("U3", "s_pit")],
    }
    factors = {"INFL": [infl_pit, infl_z, mom_pit, yoy_pit],
               "LABOUR": [lab_z, lab_pit]}

    for target, side in TARGETS:
        frames = target_frames(target, side, markets=mk, trades=tr)
        if frames[0].height == 0:
            continue
        print(f"\n######## target {target}/{side} ########")
        for block in ("INFL", "LABOUR"):
            rows = []
            # The factors run first: the *intersection* of the target events
            # they match becomes the common subset each component is then
            # re-scored on, so "(same releases)" means "every factor and every
            # component measured on identical target events".  It is therefore
            # set by the narrowest factor in the block, which is the honest
            # control but not a like-for-like comparison of row counts — the
            # unrestricted rows above it are the ones to read for power.
            for f in factors[block]:
                r = edge(f, target, side, frames, mk, tr)
                r.update(trigger=f["series"][0], measure="factor")
                rows.append(r)
            common = set.intersection(*[r["events"] for r in rows
                                        if r.get("events")]) if rows else set()
            for series, col in components[block]:
                sp = (panel.filter(pl.col("series") == series)
                      .with_columns(pl.col(col).alias("surprise")))
                r = edge(sp, target, side, frames, mk, tr)
                r.update(trigger=series, measure=col)
                rows.append(r)
                if common:
                    r2 = edge(sp, target, side, frames, mk, tr, restrict=common)
                    r2.update(trigger=series, measure=f"{col} (same releases)")
                    rows.append(r2)
            tab = pl.DataFrame([{k: v for k, v in r.items() if k != "events"}
                                for r in rows])
            with pl.Config(tbl_rows=40, float_precision=4):
                print(tab.select([c for c in ("trigger", "measure", "n", "n_targets",
                                              "rho", "p_asym", "p_perm")
                                  if c in tab.columns]))


if __name__ == "__main__":
    main()
