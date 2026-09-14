#!/usr/bin/env python
"""Item 2 — is magnitude really unrecoverable, or was the estimator bad?
(relations_study_plan §1.5)

    venv/bin/python analysis/relations_2026_09/magnitude_xval.py

``research_log.md`` §2 is one of the load-bearing design decisions of the whole
project: CPI and CPIYOY resolve from the *same* BLS print, so their surprises
must agree if the measure is real.  Matched on release date, n = 37, it found

    signed surprise   r = 0.686
    |surprise|        r = 0.242

and concluded "the signal is in the direction, not the size".  That conclusion
is why Stage 1 is rank-based, why Stage 2 is a direction classifier, and why the
AGCRN regression target was abandoned.

But ``|resolved - implied_mean|`` is a *bad* magnitude estimator: it inherits
every bit of the estimated mean's error, and §14.2 left that mean dragged toward
the ladder centre by two open tails.  This re-runs the identical check with
magnitudes that do not depend on the mean's placement:

    surprisal   -log p(bin that printed) — Shannon surprise
    |s_pit|     distance of the PIT from the centre of the distribution

Binary outcome.  If these cross-validate near the signed measure's 0.686, the
§2 conclusion is a statement about an estimator rather than about the world, and
a large amount of design space reopens.  If they also land near 0.24, the
original call is confirmed on a much stronger measure and the thesis can say so.

Spearman is reported beside Pearson throughout, because §2's own diagnosis was
that Pearson is magnitude-weighted — so judging a *magnitude* measure by Pearson
alone would repeat the mistake it identified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, "stg_infra")

from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")

# Every measure is declared once as a function of the join suffix, so the a-side
# and the b-side of the cross-check cannot drift apart — they are the same
# expression evaluated against ``col`` and ``col_b``.
def _m(sfx: str) -> dict:
    sp = pl.col(f"surprise{sfx}")
    sd = pl.col(f"implied_std{sfx}")
    pit_ = pl.col(f"s_pit{sfx}")
    return {
        "surprise (signed)":   sp,
        "s_pit (signed)":      pit_,
        "z_surprise (signed)": sp / sd,
        "|surprise|":          sp.abs(),
        "|z_surprise|":        (sp / sd).abs(),
        "|s_pit|":             pit_.abs(),
        "surprisal":           pl.col(f"surprisal{sfx}"),
        "implied_std":         sd,
        "implied_entropy":     pl.col(f"implied_entropy{sfx}"),
    }


MEASURE_NAMES = list(_m("").keys())

PAIRS = [("CPI", "CPIYOY"), ("CPI", "CPICORE"), ("CPICORE", "CPICOREYOY"),
         ("CPI", "CPICOREYOY"), ("CPIYOY", "CPICOREYOY")]


def matched(panel: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    """The two ladders' rows for the same release instant, one row per release.

    Matched on ``close_time``, not on calendar date: §1.2 established these
    ladders close at the same instant five minutes before the BLS print, and a
    date match would also pair a ladder with a *different* release that happens
    to fall on the same day.
    """
    cols = ["close_time"] + [c for c in panel.columns
                             if c not in ("series", "event_ticker", "close_time")]
    la = panel.filter(pl.col("series") == a).select(cols)
    lb = panel.filter(pl.col("series") == b).select(cols)
    return la.join(lb, on="close_time", how="inner", suffix="_b")


def crossvalidate(panel: pl.DataFrame, a: str, b: str) -> pl.DataFrame:
    m = matched(panel, a, b)
    if m.height < 8:
        return pl.DataFrame()
    left, right = _m(""), _m("_b")
    rows = []
    for name in MEASURE_NAMES:
        x = m.select(left[name].alias("v"))["v"].to_numpy().astype(float)
        y = m.select(right[name].alias("v"))["v"].to_numpy().astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            continue
        xi, yi = x[ok], y[ok]
        r, p_r = stats.pearsonr(xi, yi)
        rho, p_s = stats.spearmanr(xi, yi)
        n = int(ok.sum())
        rows.append(dict(pair=f"{a}~{b}", measure=name, n=n,
                         pearson_r=float(r),
                         t=float(r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-12))),
                         p_pearson=float(p_r), spearman=float(rho),
                         p_spearman=float(p_s)))
    return pl.DataFrame(rows)


def main() -> None:
    gated = pl.read_parquet(PANELS / "surprise_panel.parquet")
    raw = pl.read_parquet(PANELS / "surprise_panel_ungated.parquet")
    for p in (gated, raw):
        assert_no_oos(p, time_col="close_time")

    for panel, label in ((raw, "ungated"), (gated, "gated")):
        print(f"\n############ {label} panel ############")
        print("\n=== the §2 headline cell: CPI vs CPIYOY, matched on close_time ===")
        head = crossvalidate(panel, "CPI", "CPIYOY")
        with pl.Config(tbl_rows=20, float_precision=3):
            print(head)

        print("\n=== every CPI-family sibling pair ===")
        allp = pl.concat([c for a, b in PAIRS
                          if (c := crossvalidate(panel, a, b)).height])
        with pl.Config(tbl_rows=60, float_precision=3):
            print(allp.sort("measure", "pair"))

        print("\n=== mean |r| across sibling pairs, by measure "
              "(the summary the §2 conclusion rests on) ===")
        with pl.Config(tbl_rows=20, float_precision=3):
            print(allp.group_by("measure").agg(
                pl.len().alias("pairs"),
                pl.col("n").mean().alias("mean_n"),
                pl.col("pearson_r").mean().alias("mean_pearson"),
                pl.col("spearman").mean().alias("mean_spearman"),
            ).sort("mean_spearman", descending=True))


if __name__ == "__main__":
    main()
