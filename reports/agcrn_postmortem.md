# Why AGCRN (and every rung) failed to beat predict-zero

Regenerate with `analysis/agcrn_diagnostics_2026_09/postmortem.py`, run from the repo root. In-sample only.

The headline from `artifacts/agcrn_report.md` — 0 models beat zero, R² vs zero −0.18 to
−1.4, directional accuracy 41–48% — is **not an AGCRN architecture failure**. It
is a target/horizon mismatch: the regression task has essentially no signal for
*any* model, and that is fully consistent with what Stage-1 found.

---

## 1. The signal is at a horizon the model cannot see

Same BH-surviving trigger→target pairs, pooled aligned sign agreement:

| horizon | what it is | sign agreement | corr(\|surprise\|, \|response\|) |
|---|---|---|---|
| **dormant** | trigger resolution → next 3 trades (~0.5 h) | **75.9%** | +0.26 |
| liquid | → mean over target's final 7 days | 56.5% | +0.24 |
| **snapshot** (AGCRN target) | Δ implied_mean over next 3 snapshots (~weeks) | **~52%** (coin flip) | ~0.00 |

`update_2026_08.md` §5 already measured this: ~48% of the signed move lands in the
*first* print after resolution (median 12 min later), the 3rd trade adds nothing.
The AGCRN label — belief revision over the following weeks — is measured after the
effect has fully decayed and been absorbed. There is nothing left in the window
to predict.

## 2. The target is a magnitude; the signal is a sign

`research_log.md` §1–2: every magnitude-weighted test of cross-market
underreaction came back null; only rank/sign statistics survive. On
trigger-adjacent cells at the snapshot horizon:

- sign agreement (incoming Stage-1 signal vs Δ): **0.52**
- corr(signal, Δz): **+0.08**; corr(|signal|, |Δz|): **−0.03**

An MSE/MAE regression is magnitude-weighted by construction, so it dilutes a
sign-only signal toward zero — the model can be right on direction and still
score R² < 0. Stage-1's rank/sign estimator sidesteps this; a neural regressor
cannot.

## 3. There is no predictable central tendency to learn

R² vs zero of oracles that **no causal model can beat**:

| horizon k | per-node mean (leaks future) | causal expanding mean | causal AR(1) |
|---|---|---|---|
| 1 | +0.002 | −0.023 | −0.039 |
| 3 | **+0.006** | −0.021 | +0.283 † |
| 5 | +0.011 | −0.016 | +0.487 † |

† The AR(1) "predictability" at k ≥ 2 is an **overlapping-window artifact**:
k-step-ahead Δ at consecutive snapshots shares k−1 steps. Lag-1 autocorrelation
is +0.48 (k=3) / +0.70 (k=5) on overlapping windows but **−0.07 / −0.10 on
non-overlapping (stride-k) windows**. A sequence model chases this in-sample; it
is not out-of-sample signal (same trap as the CA report's consecutive-day rule,
`research_summary.md` §3). Between releases, a belief is ~a random-walk
increment: the leaky per-node-mean oracle scores R² ≈ 0.006.

## 4. The panel is ~entirely "post-trigger" — the regime can't be isolated

Macro releases are dense — median inter-snapshot gap is **1 day** (something
resolves most business days). So the fraction of labelled cells within *W* days
of a Stage-1 in-neighbour's resolution is:

| W | trigger-adjacent share | \|Δz\| adjacent | \|Δz\| quiet |
|---|---|---|---|
| 5 d | 78% | 0.21 | 0.27 |
| 10 d | 87% | 0.21 | 0.35 |
| 21 d | 95% | 0.21 | 0.48 |

Nearly every snapshot is "just after some trigger", so a model cannot learn
"trigger regime vs quiet regime" — there is no quiet regime. And moves are
*smaller* on trigger-adjacent cells, not larger: the surprise is already priced.
A surprise-weighted linear predictor scored only on trigger-adjacent cells:
**R² vs zero = −0.006, dir. acc = 0.48**.

## 5. Low information density per node → overfitting

For 15 of 19 nodes, only **8–25%** of their snapshots fall near their own
release; the rest carry forward-filled (flat) beliefs while cross-sectional noise
is injected through the graph. So the GRU's input is mostly constant. This is why
the *learned-E* AGCRN (R² −1.04 / −1.39) is far worse than *shared-MLP-E*
(−0.26 / −0.61) and why the ~290k-param default is worse than the ~5k minimal:
more capacity, same absent signal, more room to fit noise and the overlap
artifact.

---

## What this means for the thesis

- The negative result is **evidence about the task, not the model**. A
  magnitude regression of a weeks-ahead belief revision has no signal, for AGCRN
  or a linear model or an oracle. This is coherent with the whole IS record
  (`research_log.md`: "Pearson fails, sign survives").
- It **strengthens** the pivot framing: the signal that exists is a fast,
  sign-only, rank-detectable effect — exactly what the Stage-1 estimator
  (`stg_infra/stg/structure/`) is built for, and exactly what a neural sequence regressor
  is not.
- If AGCRN is to be trained at all as the Stage-2 validation study, it should
  predict at the **dormant horizon** (a short, trade-indexed window, or a
  next-print target) and as a **sign/direction classifier**, not a Δ regression
  over macro-resolution snapshots. Even then §3–4 suggest the ceiling is low.
- The learned Ã collapsing onto `ISMPMI` / same-release pairs
  (`artifacts/adjacency_comparison.md`) is the same story from the graph side: with no
  predictive signal, the adaptive adjacency has nothing to organise around and
  saturates on the thinnest node.
