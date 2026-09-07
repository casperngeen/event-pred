# AGCRN as a validation study — result and post-mortem

*Companion to `update_2026_08.md` §2 (the pivot) and `graph_definition.md`.
Reproduce with `scripts/train_agcrn.py` + `scripts/compare_adjacency.py`; the
diagnostic analysis is `analysis/agcrn_diagnostics_2026_09/`.*

## The question

The CA report proposed an inductive AGCRN adaptation as the primary model and
methodological contribution. `update_2026_08.md` §2 argued — on capacity,
architecture, and a linear complexity ladder — that AGCRN is better positioned as
an object of study than as the primary model, with direct structure estimation
(`stg_infra/stg/structure/`) as the standalone deliverable. This study runs AGCRN honestly
to test that argument rather than assert it.

## Setup

Node = a macro series' market-implied belief (19 nodes); target = k=3-snapshot-
ahead change in `implied_mean`, per-series z-scored (primary) and ATM `yes_price`
Δ in cents (robustness). 8 expanding walk-forward folds inside the in-sample
block, 3 seeds, `PURGE_DAYS` train/val gap, **2026 never loaded**. Ladder: linear
baselines (zero / AR(1) / +momentum / +neighbour {all, same-release,
Stage-1-weighted}); AGCRN at minimal (~5k params) and Bai-default (~290k at
cheb_k=2) with learned-E and shared-MLP-E; plus an AGCRN variant with the Stage-1
signed adjacency frozen in as Ã.

## Result

**No model beats predict-zero out of sample.** On the primary target, R² vs zero:

| rung | R² vs zero | dir. acc |
|---|---|---|
| linear zero / own / +neighbour (all) | 0 to −0.004 | 46–47% |
| AGCRN minimal, Stage-1-prior *(least bad)* | −0.18 | 47% |
| AGCRN minimal, shared-MLP E | −0.26 | 48% |
| AGCRN default (290k), shared-MLP E | −0.61 | 48% |
| AGCRN minimal / default, learned E | −1.04 / −1.39 | 46–47% |

More capacity is worse. On the ATM-price target every AGCRN config collapses to
predicting ≈0. The learned Ã does not track the validated structure: rank corr
with |ρ̂| ≈ +0.14…+0.22, **0/8 BH survivors** among its top-8 edges, ~48% of its
mass on FDR-rejected pairs; the shared-MLP variants saturate Ã onto `ISMPMI`, the
thinnest node.

## Why it broke down (`analysis/agcrn_diagnostics_2026_09/`)

This is a **task/horizon mismatch, not an architecture failure** — no model,
including leaky oracles, scores positive here.

1. **The signal is at a horizon the model cannot see.** The same
   trigger→target pairs give pooled aligned sign agreement of **76%** at the
   dormant horizon (next 3 trades, ~0.5 h), **57%** at the liquid horizon, and
   **~52% — coin flip** at the snapshot horizon AGCRN predicts (~weeks).
   `update_2026_08.md` §5: ~48% of the signed move is in the first print, 12 min
   after resolution. The AGCRN label is measured after the effect has decayed and
   been absorbed.

2. **The target is a magnitude; the signal is a sign.** `research_log.md` §1–2:
   every magnitude-weighted test is null, only rank/sign survives. An MSE/MAE
   regression dilutes a sign-only signal toward zero, so a model can be right on
   direction and still score R² < 0. On trigger-adjacent cells at the snapshot
   horizon, corr(signal, Δz) = +0.08 and corr(|signal|, |Δz|) = −0.03.

3. **No predictable central tendency.** Between releases a belief is ~a
   random-walk increment: a per-node-mean oracle that *leaks the future* scores
   R² ≈ 0.006. The apparent AR(1) predictability at k ≥ 2 is an
   overlapping-window artifact — lag-1 autocorrelation +0.48 (k=3) on overlapping
   windows, **−0.07 on non-overlapping** ones.

4. **The panel is ~entirely "post-trigger".** Macro releases are dense (median
   inter-snapshot gap = 1 day), so 87% of cells sit within 10 days of a Stage-1
   in-neighbour's resolution. There is no "quiet regime" for the model to
   contrast against, and moves are *smaller* on trigger-adjacent cells (the
   surprise is already priced).

5. **Low information density per node.** For 15 of 19 nodes only 8–25% of
   snapshots fall near their own release; the rest are forward-filled flat
   beliefs plus cross-sectional noise. Hence learned-E ≫ shared-MLP-E in badness,
   and default ≫ minimal: more capacity, same absent signal, more room to fit
   noise and the overlap artifact.

## Conclusion for the thesis

The negative result is **evidence about the prediction task**, coherent with the
entire in-sample record ("Pearson fails, sign survives"). It supports
repositioning AGCRN to a Stage-2 validation study with direct estimation as the
primary deliverable — the signal that exists is fast, sign-only and
rank-detectable, which is what `stg_infra/stg/structure/` is built for and what a neural Δ
regressor over macro-resolution snapshots is not. If AGCRN is trained further, it
should target the **dormant horizon** as a **direction classifier**; even then
the diagnostics put the ceiling low.
