#!/usr/bin/env python
"""Stage 2: does the Stage-1 structure predict *direction* out of fold?

    venv/bin/python scripts/run_direction_study.py [--rebuild] [--n-perm 5000]

The AGCRN study (reports/agcrn_study.md) asked a magnitude-regression question
at a horizon where the record says there is no signal, and answered "no". Its
post-mortem prescribes the replacement: same graph, but the **dormant horizon**
(trigger resolution -> the target's 3rd subsequent trade) and a **direction**
label, learned with a transparent complexity ladder instead of a sequence
network. This script runs that ladder.

Reads  artifacts/panels/surprise_panel.parquet  (run build_panels.py first)
Caches artifacts/panels/pair_panel_dormant.parquet
Writes artifacts/direction_ladder.parquet   pooled out-of-fold metrics per rung
       artifacts/direction_folds.parquet    per-fold accuracy
       artifacts/direction_edges.parquet    per-fold edge stability
       artifacts/direction_report.md        human-readable summary

Every edge weight is re-fitted inside each fold on training rows only, so no
rung consumes the published in-sample adjacency. In-sample only; 2026 is never
read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.direction import (
    build_pair_panel, coverage_composition, edge_stability, pair_counts, run_ladder,
)
from stg.panel._io import load_markets, scan_trades
from stg.splits import BURN_IN_END, OOS_START, PURGE_DAYS, assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("artifacts")
PAIR_PANEL = PANELS / "pair_panel_dormant.parquet"


def load_panel(rebuild: bool) -> pl.DataFrame:
    if PAIR_PANEL.exists() and not rebuild:
        panel = pl.read_parquet(PAIR_PANEL)
    else:
        sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
        panel = build_pair_panel(sp, markets=load_markets(is_only=True),
                                 trades=scan_trades(is_only=True))
        PAIR_PANEL.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(PAIR_PANEL)
    assert_no_oos(panel, "t0")
    # burn-in (ZIRP, ~86K trades in the whole year) is excluded from estimation
    return panel.filter(pl.col("t0") >= BURN_IN_END)


def fmt(x: float, nd: int = 3) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def ladder_table(summary: pl.DataFrame, subsets=("all", "p05", "bh")) -> list[str]:
    lines = ["| rung | subset | n | acc | perm null | majority | bal. acc | AUC | perm p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for sub in subsets:
        for r in summary.filter(pl.col("subset") == sub).iter_rows(named=True):
            lines.append(
                f"| {r['rung']} | {sub} | {r['n']} | {fmt(r['acc'])} | "
                f"{fmt(r['perm_null'])} | {fmt(r['base_rate'])} | {fmt(r['bal_acc'])} | "
                f"{fmt(r['auc'])} | {fmt(r['perm_p'], 3)} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the pair panel from the surprise panel (~3 min)")
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()

    panel = load_panel(args.rebuild)

    def run(df: pl.DataFrame):
        return run_ladder(df, n_folds=args.n_folds, start_frac=args.start_frac,
                          q=args.q, n_perm=args.n_perm)

    summary, folds, structures = run(panel)
    variants = {
        "no same-release pairs": panel.filter(~pl.col("same_release")),
        "no WTI trigger": panel.filter(pl.col("trigger") != "WTI"),
    }
    var_summaries = {}
    for name, df in variants.items():
        try:
            var_summaries[name] = run(df)[0]
        except ValueError:
            pass

    OUT.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(OUT / "direction_ladder.parquet")
    folds.write_parquet(OUT / "direction_folds.parquet")
    edges = edge_stability(structures)
    edges.write_parquet(OUT / "direction_edges.parquet")

    counts = pair_counts(panel)
    n_folds_used = folds["fold"].n_unique()
    lines = [
        "# Stage-2: dormant-horizon direction prediction",
        "",
        f"built: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}   "
        f"in-sample only (wall {OOS_START.date()}, burn-in < {BURN_IN_END.date()})",
        f"folds: {n_folds_used} expanding, purge {PURGE_DAYS}d   BH q: {args.q}   "
        f"permutations: {args.n_perm}",
        "",
        "## Panel",
        f"- labelled (trigger event -> next target event) rows: **{panel.height}** "
        f"over {panel['pair'].n_unique()} ordered pairs, "
        f"{panel['trigger'].n_unique()} triggers, {panel['target'].n_unique()} targets",
        f"- span {panel['t0'].min():%Y-%m-%d} to {panel['t0'].max():%Y-%m-%d}; "
        f"up-moves {100 * (panel['y'] > 0).mean():.1f}%",
        f"- scored out of fold: {int(folds.filter(pl.col('rung') == 'base_rate')['n_test'].sum())}",
        "",
        "Label = sign of the target's dormant-window move (p at its 3rd trade after "
        "the trigger resolves, minus its last trade before). Edge weights are "
        "re-estimated on each fold's training rows only — the published adjacency "
        "is never fed in.",
        "",
        "## Coverage gates",
        "",
        "**Reading the table.** `perm null` is the accuracy the same predictions "
        "score against labels shuffled within trigger series — that, not "
        "`majority`, is what `acc` is tested against. The two differ because "
        "always-guess-the-majority-class is a different strategy: it can score "
        "0.59 on a 59%-up subset while carrying no directional information at "
        "all, which is why `bal. acc` (mean of the two class recalls, 0.500 for "
        "any constant rule) is reported beside it.",
        "",
        "`all` every scored row; `p05` rows whose pair carries a nominally "
        "significant train-fold edge; `bh` rows whose pair survives BH-FDR *within "
        "that fold*. `bh` is the out-of-fold analogue of the 75.9% sign agreement "
        "in `adjacency_report.md`, which conditions on in-sample survival and so "
        "cannot be read as predictive.",
        "",
        "## Ladder",
        "",
    ] + ladder_table(summary)

    lines += ["", "## Per-fold accuracy (sign rule)", "",
              "| fold | cut | n train | n test | edges | BH | acc | majority |",
              "|---|---|---|---|---|---|---|---|"]
    for r in folds.filter(pl.col("rung") == "sign_rule").iter_rows(named=True):
        lines.append(f"| {r['fold']} | {r['cut']} | {r['n_train']} | {r['n_test']} | "
                     f"{r['n_edges']} | {r['n_bh']} | {fmt(r['acc'])} | {fmt(r['base'])} |")

    lines += ["", "## Edge stability across folds", "",
              "Pairs estimable in every fold, ordered by how often they survive BH.",
              "",
              "| pair | folds | BH folds | rho mean | rho min | rho max | sign stable |",
              "|---|---|---|---|---|---|---|"]
    for r in edges.filter(pl.col("bh_folds") > 0).iter_rows(named=True):
        lines.append(f"| {r['pair']} | {r['folds']} | {r['bh_folds']} | "
                     f"{fmt(r['rho_mean'])} | {fmt(r['rho_min'])} | {fmt(r['rho_max'])} | "
                     f"{'yes' if r['sign_stable'] else 'no'} |")

    for name, s in var_summaries.items():
        lines += ["", f"## Robustness — {name}", ""]
        lines += ladder_table(s.filter(pl.col("subset") != "all"), ("p05", "bh"))

    for gate in ("bh", "p05"):
        comp = coverage_composition(panel, structures, gate)
        if comp.is_empty():
            continue
        lines += ["", f"## What the `{gate}`-covered rows are made of", "",
                  "| pair | rows | folds | sign-rule acc | same-release |",
                  "|---|---|---|---|---|"]
        for r in comp.iter_rows(named=True):
            lines.append(f"| {r['pair']} | {r['n']} | {r['folds']} | "
                         f"{fmt(r['sign_acc'])} | {'yes' if r['same_release'] else 'no'} |")

    lines += ["", "## Largest pairs by sample size", "",
              "| pair | n | up rate |", "|---|---|---|"]
    for r in counts.head(10).iter_rows(named=True):
        lines.append(f"| {r['pair']} | {r['n']} | {fmt(r['up_rate'])} |")

    (OUT / "direction_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/direction_report.md and 3 parquet artifacts "
          f"({panel.height} rows, {n_folds_used} folds)")


if __name__ == "__main__":
    main()
