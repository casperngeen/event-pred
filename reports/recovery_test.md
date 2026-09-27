# Can a spatial-temporal graph learn the economic graph? A recovery test

*2026-09-26. Code: `analysis/recovery_2026_09/recovery.py`. Tables:
`analysis/recovery_2026_09/out/recovery.txt`. In-sample calendar only; no OOS data
touched.*

## The question

Every result since the AGCRN checklist has the same shape. The economic
structure, imposed (HAWKISH signs on three channels), predicts, while the same
structure learned from prices does not. That raises one objection: *is the
structure being imposed because it cannot be learned, or because the learner
was never given a fair chance?*

The two cases can't be told apart on real data, because the true graph is
unknown. So this test plants a known graph in data that is otherwise real, and
asks each learner to find it.

## Design

**Semi-synthetic.** Everything except the cross-series signal comes from the
event-time panel (`analysis/event_time_2026_09/`: 154 release instants from
2021-11 to 2025-12, 17 series).

| component | source |
|---|---|
| Release calendar | Bootstrap of real instants: which series release, which are active, which are labelled, and all 14 non-surprise input features over the 6-step window |
| Surprises | Fresh draws from N(0, C) on the releasing series. C is the real cross-series surprise correlation, so the CPI family stays one factor. |
| Noise | The real immediate-repricing label, resampled per node. Heavy tails and 15% zeros are kept; any real signal is broken. |
| Signal | y[t, b] = σ_b · (e + κ Σ_a W*[a, b] z[t, a]), with κ = ρ/√(1−ρ²) |

**The planted graph W\*** is the structure the event-time study imposes: HAWKISH
signs on labour→labour, inflation→inflation and labour→policy, cross-release
only. That is 25 directed, signed edges (6 negative, rank 4) among 142 candidate
trigger→target pairs.

**The knobs.**
- ρ is the per-edge correlation with one parent firing, one of {0, 0.2, 0.4, 0.6}. ρ = 0 is the null.
- The calendar length is 1×, 3×, 10× or 30× the real one. 1× = 154 instants ≈ 4 years; 10× ≈ 40 years; 30× ≈ 125 years.
- Repetitions: 10 / 10 / 4 / 2 respectively.

**How strong is the real effect?** On the real imm label:
- The per-edge theory-signed Spearman averages **+0.12** over the 11 true edges with n ≥ 10 (median n = 13).
- The pooled sign agreement of Σ W\*·z with y is **0.634**, which is equivalent to **ρ ≈ 0.41** for a Gaussian. It is optimistic, since the channels were selected in-sample.

So the realistic range is **ρ ≈ 0.1–0.4**.

**Learners.** All fit on the first 85% of steps. The torch models early-stop on
the last 15% of that (Adam, lr 3e-3, ≤ 600 epochs, patience 30).

| learner | what it is |
|---|---|
| pairwise | The Stage-1 estimator: Spearman(z_a, y_b) per candidate, BH-FDR at q = 0.10 |
| lasso | y_b = Σ_a W_ab z_a per target, cross-validated. A one-layer linear graph at full rank. |
| low-rank graph | The same model with W = (E_src E_dstᵀ) ⊙ candidates at rank 4, fitted by gradient descent. The smallest STG that can express W\*. |
| AGCRN | The event-time configuration: per-step masking, zero-init head, hidden 16, d_emb 2, learned embedding. |
| AGCRN signed | Identical, but with adjacency E_dst E_srcᵀ (d = 4): directed and signed, no softmax. |
| … z-only input | Ablation: the two AGCRNs fed only [released, z] instead of 15 features × 6 lags. |

**Scoring.** Each learner is scored on how it ranks the 142 candidates.
- For every learner: its own edge matrix (ρ̂, coefficients, Ã or E_dst E_srcᵀ).
- For the torch models, additionally the effective edge ∂ŷ_b/∂z_a, averaged over held-out windows.
- Headline metric: **AUROC of |score|, true edges vs false**. 0.5 is chance.
- Also reported: sign accuracy on the true edges, precision at 25, and discovery power/FDP where a learner produces a discovery set.

## Result

**AUROC, true vs false edges** (mean over repetitions; full tables including
standard deviations in `out/recovery.txt`):

| learner | ρ | 1× (154) | 3× (462) | 10× (1,540) | 30× (4,620) |
|---|---|---|---|---|---|
| pairwise (Stage-1) | 0.2 | 0.52 | 0.61 | 0.74 | 0.83 |
| | 0.4 | 0.53 | 0.65 | 0.81 | 0.90 |
| lasso | 0.2 | 0.54 | 0.56 | 0.69 | 0.76 |
| | 0.4 | 0.63 | 0.68 | 0.82 | 0.90 |
| low-rank graph | 0.2 | 0.53 | 0.52 | **0.81** | **0.84** |
| | 0.4 | 0.51 | 0.68 | **0.85** | **0.90** |
| AGCRN, ∂ŷ/∂z | 0.4 | 0.46 | 0.47 | 0.54 | 0.60 |
| | 0.6 | 0.52 | 0.48 | 0.51 | 0.59 |
| AGCRN, Ã | any | 0.35–0.37 | 0.36–0.38 | 0.37–0.39 | 0.37–0.39 |
| AGCRN signed, ∂ŷ/∂z | 0.4 | 0.48 | 0.47 | 0.45 | 0.47 |
| | 0.6 | 0.53 | 0.50 | 0.53 | 0.67 ± 0.23 |
| AGCRN signed, z-only, ∂ŷ/∂z | 0.4 | 0.47 | 0.48 | 0.47 | – |
| | 0.6 | 0.52 | 0.53 | **0.86** | – |

Under the null (ρ = 0), pairwise, lasso and low-rank all sit at 0.43–0.55.
**AGCRN does not.** Its Ã sits at 0.36–0.37, and its ∂ŷ/∂z drifts down to
0.37–0.40 as data grows. That is a structural ranking bias that exists with no
signal at all.

### 1. At the real sample size, nothing recovers the graph

At 154 instants the median true edge has **5 training instants** where its
source released and its target was labelled.
- In the realistic range (ρ ≤ 0.4), every learner's AUROC is 0.35–0.63.
- The Stage-1 estimator's power is 2–6% at ρ = 0.2–0.4, and 14% even at ρ = 0.6.

This is not an STG failure. **No method recovers the graph edge by edge from
this calendar,** including the method built for it.

This is the result that justifies imposing structure. At this n, the data
cannot choose the graph, but it can *test* a graph chosen in advance. That is
the channel-pooling design (theory fixes the grouping and the sign; the data
estimates one pooled effect). The imposition is now a measured necessity, not
a preference.

### 2. Simple graph learners recover it with about 10× the data

At 10× (about 60 training instants per edge), the three simple learners reach
AUROC **0.69–0.85** at ρ = 0.2–0.4. At 30× they reach **0.76–0.90**, with sign
accuracy ≥ 0.95 and held-out R² matching the oracle.

Among them, the **rank-4 signed, directed graph is the best STG-shaped
learner**. At ρ = 0.2 it gets 0.81 at 10×, against 0.69 for lasso and 0.74 for
pairwise. That is what a graph's low-rank structure should buy.

For discovery claims, use pairwise + BH, not lasso:

| | FDP at 10–30× |
|---|---|
| pairwise + BH | 0.02–0.20 (roughly at q) |
| cross-validated lasso | 0.58–0.66 (0.88–0.94 under the null) |

### 3. AGCRN does not recover it even at 30×

With strong signal (ρ = 0.6) and 30× the data, canonical AGCRN's effective
edges reach AUROC 0.59. The simple learners reach 0.92 on the same data. Its
held-out R² is +0.059 against an oracle of +0.142.

**Its adjacency Ã never carries the truth.** Ã's AUROC is 0.35–0.39 in every
cell, *including the null*. Ã's ranking is set by which series are active, not
by the signal. **Reading Ã as "the learned economic network" would report
structure that is not there,** and that structure would look the same with or
without a true graph.

### 4. Why AGCRN fails: diluted inputs first, graph expressivity second

Three ablations separate the explanations.

| change | effect at 10×, ρ = 0.6 | reading |
|---|---|---|
| Remove the negative edges (`--truth pos`) | AGCRN ∂ŷ/∂z 0.44, signed 0.49: no better | The softmax graph's inability to express a negative edge is **not** the binding constraint |
| Make the graph directed and signed (AGCRN signed) | 0.53: no better with full inputs | Expressivity alone does not fix it |
| Feed only [released, z] (AGCRN signed, z-only) | **0.86** (own adjacency 0.92), R² +0.113 vs oracle +0.127 | **Signal dilution is the binding constraint.** The surprise is 1 of 15 features × 6 lags × 17 nodes, and gradient descent through a GRU with early stopping does not find it at this signal-to-noise. |

Two further points:
- Canonical AGCRN with z-only input improves only to 0.62, and its Ã stays at 0.35–0.42. So once the inputs are focused, the softmax adjacency does become the limit.
- Even the z-only signed model fails at ρ = 0.4 at 10× (0.47), where the low-rank graph gets 0.85.

**The ordering is consistent throughout:** the fewer parameters stand between
z_a and ŷ_b, the less data recovery needs.

## Is there a non-linear signal to learn? (real panel)

The recovery test plants a linear signal. A separate question is whether the
*real* data holds a non-linear one: a shape a flexible learner could exploit
and a linear rung would miss. `analysis/event_time_2026_09/nonlinear.py`
(output in `out/nonlinear.txt`) tests the shapes earlier findings point to:
- **sign**, **tanh**: saturation. Only sign/rank has survived before (research_log §1).
- **asym**: separate slopes for hawkish and dovish news.
- **p-scaled**: the same news moves a 50¢ contract more than a 95¢ one, the bounded-probability form.
- **σ-scaled**: bigger updates when the prior is wider.
- **parent-mean**: averages rather than sums the surprises from co-released parents.

Each shape costs one extra parameter at most and is fitted walk-forward, with
and without an intercept. There is also a small gradient-boosted model on
[s, price, width, liquidity, days to close], scored against the same model
without s and a permutation null.

- **Nothing beats linear.** Every CI on ΔR² vs linear includes zero or favours
  linear. On all 142 edges (immediate label), sign and tanh are significantly
  *worse* (−0.018 [−0.035, −0.002], −0.012 [−0.026, −0.001]).
- **Saturation is the one hint, in the BH channels only.** There, sign and tanh
  have the best point estimates on both labels (ΔR² +0.02 to +0.08). The shape
  table steps at zero and flattens at the top (quintile means −0.16, −0.10,
  +0.12, +0.29, +0.22). With 129–160 test cells the CIs are about ±0.1, so this
  is consistent with the earlier sign-only finding, not new evidence for it.
- **The bounded-probability form is rejected, not supported.** p-scaling is
  worse in all four cells, significantly so on the BH immediate label
  (−0.055 [−0.141, −0.001], no intercept). In the in-sample shape table the
  theory-signed response is *largest* below 25¢ (+0.36 vs +0.15, n = 35).
- **The boosted model gains nothing from the surprise.** Its gain from s has
  permutation p = 0.10, 0.11, 0.31 and 0.49 across the four cells. What it does
  find, on the settlement label, is own-state predictability (R² +0.024 on all
  edges, +0.052 on BH channels, CIs through zero). That is the price-level drift
  already documented as always-NO and favourite-longshot, not a network effect.
- **The intercept is a trap at this n.** With 32 training cells in the first
  fold, the fitted intercept (+0.46) meets a test-fold mean of −0.87. That is
  why the linear rung scores −0.18 on the BH immediate label while its slope is
  positive in every fold (+0.06 to +0.29). The theory-consistent model has no
  intercept (no news, no expected move).

So at this n there is no non-linear network signal for an STG to find that a
one-parameter sign rule doesn't already capture. The recovery test's ordering
applies here too: a flexible learner needs more data than a linear one to find
the same signal, not less.

## What this means for the thesis

1. **The negative AGCRN result becomes a positive, quantified statement.** At
   the macro calendar's size (~5 co-firing instants per edge), learning the
   graph from prices is infeasible *for any learner*. A graph learner of the
   right shape needs about 10× the data, and a full AGCRN needs more than 30×
   (or focused inputs). That is the answer to "did you just tune it badly?".
2. **Imposing the structure is the identification strategy, not a shortcut.**
   It is the only way to use the data the calendar provides. The theory fixes
   what the data cannot choose (the grouping and sign); the data tests what it
   can (a pooled effect per channel).
3. **An STG that could learn this graph looks like the low-rank signed graph.**
   It is directed, signed and low-rank, and its message comes from the surprise
   rather than from the full node state. That specification carries over to a
   setting with about 100× the events, such as the sports arm
   (`progress_2026_09.md` §9, decision 4).
4. **Never interpret AGCRN's Ã as a finding.** Under a known truth it ranks
   true edges *below* chance in every cell, including the null.

## Caveats

- **The planted signal is linear and contemporaneous,** which favours the
  linear learners by construction. This is the structure the theory posits. A
  learner that can't approach linear on the linear case gives no grounds for
  expecting it to do better on a harder one, but the test says nothing about
  non-linear truths.
- **The calendar is bootstrapped.** 10× means 40 years of the *same* release
  pattern. More data from a different calendar, such as more series or denser
  releases, would change the per-edge counts.
- **30× has 2 repetitions per cell,** and AGCRN signed at 30×, ρ = 0.6 is
  bimodal (0.67 ± 0.23: one run recovers, one doesn't). The z-only ablation has 3.
- **The early-stopping budget is fixed** across learners. The z-only signed
  model that succeeded ran to epoch ~200; runs that stopped at 20–60 epochs
  failed. Some of AGCRN's failure is optimisation under noisy early stopping,
  which is part of what "cannot learn it at this n" means in practice.
