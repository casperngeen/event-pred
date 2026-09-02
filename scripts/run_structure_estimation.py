#!/usr/bin/env python
"""Stage 1: estimate the validated cross-market influence adjacency.

    venv/bin/python scripts/run_structure_estimation.py [--horizon dormant]

Reads ``artifacts/panels/surprise_panel.parquet`` (run build_panels.py first)
and writes:

    artifacts/adjacency_is.parquet   edge table (rho, p, BH, permutation, ...)
    artifacts/adjacency_report.md    human-readable summary

In-sample only. The 2026 block is never read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.panel import universe
from stg.panel.surprise import usable_triggers
from stg.structure import estimate_adjacency, estimate_by_horizon, mediation_grid
from stg.structure.stats import block_permutation_sign_p
from stg.splits import OOS_START

PANELS = Path("artifacts/panels")
OUT = Path("artifacts")

MEDIATION_TRIPLES = [
    ("WTI", "CPI", "FED", "any"),
    ("CPI", "CPIYOY", "FED", "any"),
    ("PAYROLLS", "U3", "FED", "any"),
    ("CPI", "PAYROLLS", "FED", "any"),
    ("CPICORE", "CPI", "FED", "any"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="dormant", choices=["dormant", "liquid"])
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)

    edges = estimate_adjacency(sp, horizon=args.horizon, min_n=args.min_n,
                               q=args.q, n_perm=args.n_perm, markets=mk, trades=tr)
    OUT.mkdir(parents=True, exist_ok=True)
    edges.write_parquet(OUT / "adjacency_is.parquet")

    surv = edges.filter(pl.col("survives"))
    surv_edges = [(r["trigger"], r["target"], r["side"]) for r in surv.iter_rows(named=True)]
    hz = estimate_by_horizon(sp, surv_edges, markets=mk, trades=tr) if surv_edges else pl.DataFrame()
    med = mediation_grid(sp, MEDIATION_TRIPLES, horizon=args.horizon, markets=mk, trades=tr)

    n = edges.height
    exp = args.q  # placeholder; real expected-by-chance below
    lines = [
        "# Stage-1 structure estimation",
        "",
        f"built: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}   "
        f"in-sample only (wall {OOS_START.date()})",
        f"horizon: {args.horizon}   min_n: {args.min_n}   BH q: {args.q}",
        "",
        "## Grid",
        f"- ordered (trigger -> target) pairs searched: **{n}**",
        f"- usable triggers: {', '.join(usable_triggers(sp, args.min_n))}",
        f"- uncorrected p<0.05: {(edges['p_asymptotic'] < 0.05).sum()}  "
        f"(expected by chance: {0.05 * n:.1f})",
        f"- **BH-FDR survivors at q={args.q}: {int(edges['survives'].sum())}**",
        "",
        "## Surviving edges",
        "",
        "| trigger | target | side | n | rho | p (asym) | p (perm) | same-release |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in surv.iter_rows(named=True):
        lines.append(
            f"| {r['trigger']} | {r['target']} | {r['side']} | {r['n']} | "
            f"{r['rho']:.3f} | {r['p_asymptotic']:.2g} | "
            f"{r['p_permutation']:.4f} | {'yes' if r['same_release'] else 'no'} |"
        )

    # pooled sign test with the block-permutation null
    cells = []
    for r in surv.iter_rows(named=True):
        from stg.panel.targets import response_panel
        rp = response_panel(sp.filter(pl.col("series") == r["trigger"]),
                            r["target"], r["side"], horizon=args.horizon,
                            markets=mk, trades=tr)
        if rp.height:
            s = rp["surprise"].to_numpy().astype(float)
            resp = rp["response"].to_numpy().astype(float)
            sign = np.sign(r["rho"]) or 1.0
            cells.append((r["trigger"], s * sign, resp))
    if cells:
        bp = block_permutation_sign_p(cells, n_perm=3000)
        lines += [
            "",
            "## Sign agreement *among the surviving edges* (descriptive)",
            f"- n = {bp['n']}, aligned sign agreement = {bp['sign_agreement']*100:.1f}%",
            f"- block-permutation null {bp['null_mean']*100:.1f}% "
            f"± {bp['null_sd']*100:.1f}%   one-sided p = {bp['p']:.4f}",
            "",
            "_This conditions on BH survival, so it is a coherence check on the "
            "recovered edges, not an independent test. The independent pooled "
            "sign result (66.4%, p=0.001 over theory-specified cells) is in "
            "research_log.md §3._",
        ]

    if hz.height:
        lines += ["", "## Term structure of diffusion (rho by target days-to-close)",
                  "", "| edge | 0-7d | 7-21d | 21-60d |", "|---|---|---|---|"]
        for (t, tg, sd), g in hz.group_by("trigger", "target", "side"):
            by = {(r["dtc_lo"], r["dtc_hi"]): r for r in g.iter_rows(named=True)}
            def cell(k):
                r = by.get(k)
                return "–" if not r or not np.isfinite(r["rho"]) else f"{r['rho']:.2f} (n{r['n']})"
            lines.append(f"| {t}->{tg}/{sd} | {cell((0,7))} | {cell((7,21))} | {cell((21,60))} |")

    if med.height:
        lines += ["", "## Mediation — is A->C routed through B?",
                  "", "| A -> C \\| B | n | rho(A,C) | partial | change |",
                  "|---|---|---|---|---|"]
        for r in med.iter_rows(named=True):
            lines.append(
                f"| {r['a']}->{r['c']} \\| {r['b']} | {r['n']} | {r['rho']:.3f} | "
                f"{r['partial']:.3f} | {r['change']:+.3f} |")
        lines += ["",
                  "_A large negative change = the apparent A->C edge is routed "
                  "through B (genuine multi-hop). Unchanged = independent bilateral "
                  "edges._"]

    (OUT / "adjacency_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/adjacency_is.parquet ({edges.height} edges, "
          f"{int(edges['survives'].sum())} survive) and adjacency_report.md")


if __name__ == "__main__":
    main()
