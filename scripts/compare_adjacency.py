#!/usr/bin/env python
"""Does AGCRN's learned adjacency Ã recover the directly-estimated structure?

    venv/bin/python scripts/compare_adjacency.py

Reads artifacts/agcrn_learned_adj.parquet (from train_agcrn.py) and
artifacts/adjacency_is.parquet (from run_structure_estimation.py). Writes
artifacts/adjacency_comparison.md.

Ã ≥ 0 by construction (softmax(ReLU(EEᵀ))), so:
  - recovery: is the rank of |Ã_ij| aligned with the rank of |ρ̂_ij|? do the
    top-m Ã entries contain the BH-FDR survivors?
  - grid-rejected mass: how much of Ã sits on pairs the FDR grid rejected?
  - same-release loading: does Ã prefer same-release pairs (co-movement) over
    cross-domain pairs (candidate propagation)?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel.registry import is_same_release

OUT = Path("artifacts")


def _rank_corr(a, b):
    from stg.structure.stats import spearman
    return spearman(np.asarray(a), np.asarray(b))


def main() -> None:
    la = OUT / "agcrn_learned_adj.parquet"
    sa = OUT / "adjacency_is.parquet"
    if not la.exists() or not sa.exists():
        sys.exit("run train_agcrn.py (full) and run_structure_estimation.py first")

    learned = pl.read_parquet(la)
    stage1 = pl.read_parquet(sa)
    s1 = {(r["trigger"], r["target"]): r for r in stage1.iter_rows(named=True)}
    survivors = {(r["trigger"], r["target"]) for r in
                 stage1.filter(pl.col("survives")).iter_rows(named=True)}

    L = ["# Learned Ã vs the Stage-1 adjacency", "",
         f"Stage-1: {stage1.height} searched pairs, {len(survivors)} BH-FDR survivors.",
         ""]

    for model in learned["model"].unique().to_list():
        m = learned.filter(pl.col("model") == model)
        pairs = [(r["trigger"], r["target"], r["a_ij"]) for r in m.iter_rows(named=True)]
        pairs.sort(key=lambda x: -x[2])
        total_mass = sum(a for _, _, a in pairs) or 1.0

        # align to the Stage-1 grid where both exist
        common = [(t, g, a) for t, g, a in pairs if (t, g) in s1]
        a_vals = [a for _, _, a in common]
        rho_abs = [abs(s1[(t, g)]["rho"]) for t, g, _ in common]
        rc = _rank_corr(a_vals, rho_abs) if len(common) >= 5 else float("nan")

        top_m = set((t, g) for t, g, _ in pairs[:len(survivors)])
        recovered = survivors & top_m
        rejected_mass = sum(a for t, g, a in pairs
                            if (t, g) in s1 and not s1[(t, g)]["survives"])
        sr_mass = sum(a for t, g, a in pairs if is_same_release(t, g))

        L += [f"## {model}", "",
              f"- rank corr( |Ã_ij| , |ρ̂_ij| ) over {len(common)} shared pairs: "
              f"**{rc:+.2f}**",
              f"- top-{len(survivors)} Ã entries recover **{len(recovered)}/{len(survivors)}** "
              f"BH survivors" + (f" ({', '.join(f'{t}->{g}' for t, g in sorted(recovered))})"
                                 if recovered else ""),
              f"- Ã mass on FDR-*rejected* Stage-1 pairs: "
              f"**{100*rejected_mass/total_mass:.0f}%**",
              f"- Ã mass on same-release pairs: **{100*sr_mass/total_mass:.0f}%** "
              f"(these are co-movement, not propagation)",
              "",
              "| rank | Ã edge | Ã_ij | Stage-1 ρ̂ | BH? |",
              "|---|---|---|---|---|"]
        for t, g, a in pairs[:10]:
            r = s1.get((t, g))
            rho = f"{r['rho']:+.2f}" if r else "–"
            bh = "**yes**" if (t, g) in survivors else ("no" if r else "not searched")
            L.append(f"| {pairs.index((t, g, a)) + 1} | {t}→{g} | {a:.3f} | {rho} | {bh} |")
        L.append("")

    L += ["## Reading",
          "",
          "- rank corr near 0 or negative ⇒ Ã is not tracking the validated structure.",
          "- low survivor recovery + high rejected-pair mass ⇒ Ã places weight where "
          "the FDR grid found nothing — a concrete finding about adaptive graph "
          "learning applied without validation (update_2026_08.md §2, Stage 2).",
          "- high same-release mass ⇒ Ã is picking up two-contracts-one-print "
          "co-movement, which the direct estimator flags and conditions on."]
    (OUT / "adjacency_comparison.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}/adjacency_comparison.md")


if __name__ == "__main__":
    main()
