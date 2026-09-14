#!/usr/bin/env python
"""Item 1 — is the Kalshi macro ladder calibrated?  (relations_study_plan §1.4)

    venv/bin/python analysis/relations_2026_09/pit_calibration.py

Under a calibrated market the realised value's position in the market's own
implied distribution is uniform:  u = F_implied(resolved) ~ U(0, 1).  This is
the distributional generalisation of the §5B unbiasedness result, which tests
only the first moment — a market can be unbiased in the mean and still be
systematically over- or under-confident about the spread, and only the PIT sees
that.

Read on the **ungated** panel.  ``gate_panel`` keeps rows whose ladder mass is
already close to 1, which is most of the way to assuming the answer; the gated
panel is reported beside it only to show what the gates do to the verdict.

A standalone finding either way: it needs no edge, no target, and no horizon.
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
MIN_N = 10


def ks_uniform(u: np.ndarray) -> tuple[float, float]:
    """KS distance of ``u`` from U(0,1), and its p-value.

    ``mode="exact"`` because n is 10-200 here, where the asymptotic
    Kolmogorov form is noticeably anti-conservative.
    """
    r = stats.kstest(u, "uniform", mode="exact")
    return float(r.statistic), float(r.pvalue)


def summarise(panel: pl.DataFrame, label: str) -> pl.DataFrame:
    rows = []
    for series in sorted(panel["series"].unique().to_list()):
        u = panel.filter(pl.col("series") == series)["pit"].to_numpy()
        u = u[np.isfinite(u)]
        if u.size < MIN_N:
            continue
        d, p = ks_uniform(u)
        rows.append(dict(
            panel=label, series=series, n=int(u.size),
            mean_u=float(u.mean()),
            # tail mass: the overconfidence diagnostic.  Under U(0,1) exactly
            # 20% of draws land in the outer decile pair; more than that means
            # the ladder's distribution is too narrow for the outcomes it sees.
            tail20=float(((u < 0.10) | (u > 0.90)).mean()),
            centre50=float(((u > 0.25) & (u < 0.75)).mean()),
            ks_d=d, ks_p=p,
        ))
    return pl.DataFrame(rows)


def histogram(u: np.ndarray, bins: int = 10) -> str:
    counts, _ = np.histogram(u, bins=bins, range=(0.0, 1.0))
    expect = u.size / bins
    peak = max(counts.max(), 1)
    out = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(20 * c / peak))
        out.append(f"  [{i/bins:.1f},{(i+1)/bins:.1f})  {c:4d}  "
                   f"{c/expect:5.2f}x  {bar}")
    return "\n".join(out)


def main() -> None:
    gated = pl.read_parquet(PANELS / "surprise_panel.parquet")
    raw = pl.read_parquet(PANELS / "surprise_panel_ungated.parquet")
    for p in (gated, raw):
        assert_no_oos(p, time_col="close_time")

    print(f"ungated rows {raw.height}   gated rows {gated.height}\n")

    for panel, label in ((raw, "ungated"), (gated, "gated")):
        tab = summarise(panel, label)
        print(f"=== per-series PIT calibration ({label}) ===")
        with pl.Config(tbl_rows=40, float_precision=3):
            print(tab.sort("n", descending=True))
        u = panel["pit"].to_numpy()
        u = u[np.isfinite(u)]
        d, p = ks_uniform(u)
        print(f"\npooled  n={u.size}  mean_u={u.mean():.3f}  "
              f"tail20={((u < .1) | (u > .9)).mean():.3f}  "
              f"KS D={d:.3f}  p={p:.3g}")
        print("(pooled KS is descriptive only: rows share release instants, so "
              "the draws are not independent)")
        print(f"\npooled PIT histogram ({label}), 10 bins, x = observed/expected:")
        print(histogram(u))
        print()

    # Is the surprise measure's *shape* what drives it, or the mean's placement?
    print("=== mean-placement check ===")
    for panel, label in ((raw, "ungated"), (gated, "gated")):
        s = panel.select(
            (pl.col("resolved_value") > pl.col("implied_mean")).mean().alias("above_mean"),
            (pl.col("resolved_value") > pl.col("implied_median")).mean().alias("above_median"),
        ).row(0)
        print(f"{label:8s}  P(resolved > implied_mean) = {s[0]:.3f}   "
              f"P(resolved > implied_median) = {s[1]:.3f}   (0.5 under no bias)")

    print("\n=== per-series bias: the §5B t-test vs the PIT sign test (gated) ===")
    with pl.Config(tbl_rows=40, float_precision=3):
        print(bias_tests(gated))
    print("\n=== same, ungated ===")
    with pl.Config(tbl_rows=40, float_precision=3):
        print(bias_tests(raw))


def bias_tests(panel: pl.DataFrame) -> pl.DataFrame:
    """Two tests of the same null — "the market is not systematically wrong".

    ``t_z``   the §5B test: a one-sample t on ``surprise / implied_std``.  It
              asks whether the *mean* error is zero, and it is what
              ``edge_economics.md`` §5B ran when it concluded no series shows an
              ex-ante bias (max |t| = 1.66 over 17).
    ``sign``  an exact binomial on ``u > 0.5``.  It asks whether outcomes land
              above the middle of the implied distribution more often than not.

    They can disagree, and the direction of disagreement is informative: the t
    is diluted by the heavy dispersion of ``implied_std`` across events, while
    the sign test spends no power on magnitude.  Reported side by side with a
    BH correction over the series actually tested.
    """
    from stg.structure.stats import benjamini_hochberg

    rows = []
    for series in sorted(panel["series"].unique().to_list()):
        d = panel.filter(pl.col("series") == series)
        u = d["pit"].to_numpy()
        z = (d["surprise"] / d["implied_std"]).to_numpy()
        z = z[np.isfinite(z)]
        if u.size < MIN_N or z.size < MIN_N:
            continue
        t, p_t = stats.ttest_1samp(z, 0.0)
        k = int((u > 0.5).sum())
        p_s = float(stats.binomtest(k, u.size, 0.5).pvalue)
        rows.append(dict(series=series, n=int(u.size), mean_z=float(z.mean()),
                         t_z=float(t), p_t=float(p_t),
                         frac_above=k / u.size, p_sign=p_s))
    out = pl.DataFrame(rows)
    return out.with_columns(
        pl.Series("sign_bh", benjamini_hochberg(out["p_sign"].to_numpy(), 0.10)),
        pl.Series("t_bh", benjamini_hochberg(out["p_t"].to_numpy(), 0.10)),
    ).sort("p_sign")


if __name__ == "__main__":
    main()
