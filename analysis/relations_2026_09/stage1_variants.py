#!/usr/bin/env python
"""Item 1b — does Stage 1 survive a change of surprise measure?
(relations_study_plan §1.4, §1.6)

    venv/bin/python analysis/relations_2026_09/stage1_variants.py

Re-runs the *entire* Stage-1 sweep — same grid, same horizon, same permutation
machinery, same BH q — four times, changing only the number fed in as the
trigger's surprise:

    surprise         the published measure, ``resolved - implied_mean``
    s_pit            2u - 1, the unit-free PIT surprise (§1.4)
    z_surprise       ``surprise / implied_std``, what Stage 2 already consumes
    surprise_median  ``resolved - implied_median``, immune to the open-tail
                     problem §14.2 left in the mean (§1.6)

``estimate_adjacency`` reads the column named ``surprise``, so each variant is
the same panel with that column swapped.  Spearman is invariant to *monotone*
transforms, so a variant moves ρ only where the transform reorders events —
which ``s_pit`` and ``z_surprise`` both do, because each event has its own
implied distribution.  ``surprise_median`` reorders through a different
estimate of the centre.  This is therefore a real robustness test and not an
arithmetic identity, and the *baseline* row also serves as a reproduction check
against the published ``adjacency_report.md``.

In-sample only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.splits import assert_no_oos
from stg.structure import estimate_adjacency

PANELS = Path("artifacts/panels")
OUT = Path("analysis/relations_2026_09/out")

VARIANTS = {
    "surprise": pl.col("surprise"),
    "s_pit": pl.col("s_pit"),
    "z_surprise": pl.col("surprise") / pl.col("implied_std"),
    "surprise_median": pl.col("surprise_median"),
}


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)

    frames = []
    for name, expr in VARIANTS.items():
        t0 = time.time()
        panel = sp.with_columns(expr.alias("surprise")).drop_nulls("surprise")
        panel = panel.filter(pl.col("surprise").is_finite())
        edges = estimate_adjacency(panel, horizon="dormant", min_n=10, q=0.10,
                                   n_perm=2000, markets=mk, trades=tr)
        edges = edges.with_columns(pl.lit(name).alias("measure"))
        frames.append(edges)
        surv = edges.filter(pl.col("survives"))
        print(f"\n=== {name}  ({panel.height} rows, {time.time() - t0:.0f}s) ===")
        print(f"pairs {edges.height}   nominal p<0.05 "
              f"{int((edges['p_permutation'] < 0.05).sum())}   "
              f"BH survivors {surv.height}")
        with pl.Config(tbl_rows=30, float_precision=4):
            print(surv.select("trigger", "target", "side", "n", "rho",
                              "p_permutation", "same_release"))
        sys.stdout.flush()

    all_edges = pl.concat(frames)
    OUT.mkdir(parents=True, exist_ok=True)
    all_edges.write_parquet(OUT / "stage1_variants.parquet")

    print("\n=== survivor overlap across measures ===")
    surv = (all_edges.filter(pl.col("survives"))
            .with_columns((pl.col("trigger") + "->" + pl.col("target")
                           + "/" + pl.col("side")).alias("pair")))
    print(surv.group_by("pair").agg(
        pl.col("measure").sort().alias("measures"),
        pl.len().alias("n_measures")).sort("n_measures", descending=True))

    print("\n=== rho on the four published edges, by measure ===")
    published = [("CPI", "FED", "any"), ("PAYROLLS", "FEDDECISION", "hike"),
                 ("PAYROLLS", "FED", "any"), ("CPICOREYOY", "FEDDECISION", "cut")]
    keep = pl.any_horizontal([
        (pl.col("trigger") == t) & (pl.col("target") == g) & (pl.col("side") == s)
        for t, g, s in published])
    with pl.Config(tbl_rows=40, float_precision=4):
        print(all_edges.filter(keep)
              .select("trigger", "target", "side", "measure", "n", "rho",
                      "p_permutation", "survives")
              .sort("trigger", "target", "side", "measure"))


if __name__ == "__main__":
    main()
