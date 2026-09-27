# Is AGCRN truly worse than the linear baselines? An implementation checklist

*Companion to `agcrn_study.md` and `agcrn_postmortem.md`. Reproduce with
`analysis/agcrn_checklist_2026_09/step{1,2,3}_*.py`. In-sample only; same
target (k=3 per-series z-scored Δ implied_mean), same 8-fold walk-forward
harness, 3 seeds.*

## Bottom line

1. **The implementation on `main` does handicap AGCRN, and most of its gap to
   the linear ladder is an artefact.** The biggest cost is not the graph at all.
   A randomly initialised output head, combined with early stopping that often
   keeps an epoch-0 state, makes AGCRN emit noise with sd ≈ 0.2–0.3 against a
   label sd of 0.86. Zeroing the head at init, and changing nothing else, moves
   AGCRN from R² vs zero **−0.33 → −0.05** (learned E) and **−0.19 → −0.02**
   (shared-MLP E).
2. **There are also real bugs.** Padded nodes leak into the graph and the GRU
   (37–58% of Ã's mass sits on padded nodes). The Stage-1 prior was transposed.
   The learned-Ã export had trigger and target swapped. The shared-MLP
   embedding was pooled over the batch, so a window's prediction depended on
   which other windows were in the batch. All four are fixed behind flags.
3. **With every fix, AGCRN matches the linear ladder, and neither has any
   signal.** The best AGCRN rung scores R² −0.008 against linear own_momentum's
   +0.0002, and it beats that linear model in 5 of 8 folds. Even optimally
   rescaled with the evaluation labels, no model in the study exceeds
   **R² +0.006** (corr ≤ 0.07). The linear models' 55% directional accuracy is
   the base rate: 54% of labels are up and the ridge predicts "up" 94% of the
   time.

4. **The economic sign prior (`HAWKISH`, strategy_spec §3.3) doesn't change
   this.** Frozen into AGCRN as a signed graph it scores −0.12, and added to the
   ridge it scores −0.0001. Its signal has 51% sign agreement with the AGCRN
   label, because the effect is priced by the end-of-day snapshot the label
   starts from (§3b).

So "simpler models work better" is really "predicting ≈0 works best", which is
the checklist's *zero-baseline* case. The postmortem's conclusion (a task/horizon
mismatch, not an architecture failure) **survives**. Two of its supporting
observations were implementation artefacts, though; see §4.

## 1. Implementation checks (step 1)

| check | verdict | evidence |
|---|---|---|
| **Padding & masking** | **Bug.** | 69% of the dense tensor is padding (6.9 of 22 nodes active per snapshot). Padded cells are zero-filled *before* standardisation, so they become −μ/σ: mean \|z\| **9.7** on `implied_mean` (max 55), against 0.8 on real cells. The legacy mask is "active anywhere in the batch × window", which is **22/22** nodes under full-batch training, so it removes nothing. Scrambling *only* padded cells moves predictions on real labelled nodes by up to 0.15 at init. With `masking="per_step"` the change is exactly 0 (and unit-tested). |
| Batch-pooled shared-MLP E | **Bug.** | `MLP(x_t).mean(0)` pools over the batch. Sample 0 predicted alone vs inside a 128-window batch differs by 0.09. |
| Stage-1 orientation | **Bug.** | The prior is `[trigger, target]`, but the model aggregates `A @ x` (row = receiver). On `main`, FED received nothing and CPI/PAYROLLS received FED. Fixed, with a test. |
| **Target scaling** | OK | In raw units PAYROLLS would be 90% of the loss. The harness z-scores per series, so no node exceeds 13%. |
| **Can it overfit?** | OK | Every variant, minimal and 290k default, legacy and fixed, drives Huber loss on 16 windows from 0.27 to ≤0.0002 within 300–3000 steps. Optimisation works. |
| **Train vs val** | Neither pure under- nor over-fitting: **there is nothing to fit**. | Under the harness, the kept state's fit loss is 0.99–1.04× predict-zero, while early-stop loss is ≥0.99× and test loss 1.1–2.2×. Train 600 epochs and fit loss falls to 0.13–0.72× while early-stop loss rises to 1.6–10.6×. At no epoch does any variant beat zero on validation. |
| **Zero baseline** | **The whole story.** | See the table in §3. Linear R² ≈ 0 because ridge shrinks to ≈0 (pred sd 0.09). AGCRN on `main` has pred sd 0.23–0.31 and corr ≈ 0, so its negative R² is its own variance. |

The **zero-init head** isn't on the checklist, but it's the fix that matters most.
Early stopping picks the best *model state* and never compares it with
predict-zero. On a signal-free target the best state is often epoch 0 (12–42% of
fold-fits), which is just the random head. Ridge never faces this problem.

## 2. Structural checks (step 2, last fold, minimal config)

| diagnostic | legacy / learned E | legacy / shared-MLP | fixed / learned | fixed / shared-MLP | fixed / hybrid + prior + top-3 |
|---|---|---|---|---|---|
| row entropy H / log n_active | 1.21 (spread over padded nodes too) | 0.34 | 1.00 (uniform) | 0.33 | 0.13 |
| effective neighbours exp(H) | 16.0 | 5.0 | 10.1 (= all active) | 3.8 | 1.5 |
| Ã mass on **padded** senders | **37%** | **58%** | 0% | 0% | 0% |
| Ã moved from its initialisation | **0.0%** | 52% | 0.1% | 0.3% | 15% |
| CV of Ã_ij across snapshots | 0 | 0 | 0.15 | 1.31 | 2.65 |
| rank corr with \|ρ̂\| (oriented) | +0.32 | +0.03 | +0.18 | +0.18 | +0.46 |

- **Near-uniform adjacency: confirmed for learned E.** E is initialised at
  0.05·randn, so ReLU(EEᵀ) ≈ 0 and the softmax is flat. Early stopping lands
  before E moves at all. Every node simply averages the graph, and on `main` that
  average includes the padded junk.
- **"Feature similarity, not influence": partly.** The shared-MLP Ã is not a
  cosine-similarity kernel (corr ≈ 0). It is a dot-product kernel driven by
  feature *magnitude*, redrawn every snapshot (CV 1.3), and it barely trains
  (0.3% movement). So there is no stable "WTI → CPI" relation in it. Adding a
  series-id embedding (the checklist's event-type embedding; here each node *is*
  an event type) does not change that on this data.
- **The prior works as a prior.** λ·|ρ̂| raises rank agreement with Stage-1 to
  +0.41 (and +0.46 with top-3), but it buys no predictive accuracy (§3).
- **Capacity**: 2,478 effective labels (ICC 0.20), cut a further ~3× by the
  overlapping k=3 windows. Minimal AGCRN has 1.6–2.7 parameters per effective
  label, and the Bai default has **117**.
- **Surprise**: the in-neighbour surprise, weighted through the Stage-1 graph,
  has corr **−0.001** with the k=3 target (sign agreement 0.50). So even when
  given the surprise, the graph has no directional signal to propagate at this
  horizon. A node's *own* surprise correlates +0.24 with its own next Δ. That
  is most likely the node rolling to its next event after resolution, not
  spillover.

## 3. Fixes applied cumulatively (step 3, 8 folds × 3 seeds)

| model | params | R² vs zero | dir. acc (y≠0) | pred sd | corr | R² at best rescale † | folds > linear |
|---|---|---|---|---|---|---|---|
| linear zero | – | 0 | – | 0 | – | 0 | – |
| linear own_momentum | – | **+0.0002** | 0.550 | 0.092 | +0.048 | +0.003 | – |
| linear neighbour_stage1 | – | −0.0009 | 0.549 | 0.094 | +0.045 | +0.003 | 2/8 |
| linear + surprise | – | +0.0026 | 0.528 | 0.114 | +0.071 | +0.006 | 5/8 |
| **A0 main, learned E** | 5,341 | **−0.325** ±0.22 | 0.498 | 0.306 | +0.034 | +0.001 | 0/8 |
| **A0 main, shared-MLP E** | 6,195 | **−0.189** ±0.02 | 0.506 | 0.230 | −0.019 | +0.000 | 0/8 |
| A1 + zero-init head, learned | 5,341 | −0.053 ±0.03 | 0.514 | 0.127 | +0.022 | +0.001 | 3/8 |
| A1 + zero-init head, shared-MLP | 6,195 | −0.022 ±0.01 | 0.496 | 0.089 | −0.029 | +0.001 | 4/8 |
| A2 + per-step masking, learned | 5,533 | −0.168 ±0.18 | 0.501 | 0.226 | −0.022 | +0.000 | 1/8 |
| A2 + per-step masking, shared-MLP | 6,451 | −0.013 ±0.00 | 0.485 | 0.071 | −0.012 | +0.000 | 4/8 |
| A3 + series-id embedding (hybrid) | 6,795 | −0.024 ±0.01 | 0.495 | 0.075 | −0.020 | +0.001 | 3/8 |
| **A4 + event-study prior (λ=2)** | 6,795 | **−0.008** ±0.01 | 0.514 | 0.065 | +0.028 | +0.001 | **5/8** |
| A5 + top-3 | 6,795 | −0.025 ±0.01 | 0.490 | 0.078 | −0.011 | +0.000 | 4/8 |
| A6 + shared W, dropout 0.2, wd 1e-3 | 4,059 | −0.012 ±0.00 | 0.509 | 0.073 | −0.004 | +0.000 | 5/8 |
| A7 + surprise channel | 4,219 | −0.013 ±0.01 | 0.518 | 0.061 | +0.031 | +0.002 | 4/8 |
| frozen Stage-1 graph (fixed orientation) | 5,533 | −0.068 ±0.05 | 0.528 | 0.183 | +0.005 | +0.000 | 4/8 |
| A2, default size (learned) | 294,045 | −0.190 ±0.04 | 0.507 | 0.274 | −0.006 | +0.000 | 3/8 |

† Least-squares rescaling fitted on the evaluation labels. It is an upper bound
on what recalibration could buy, not an out-of-sample number, and it equals
corr². A model with no directional information stays at ≈0 however it is scaled.

What the ladder shows:

- **Every AGCRN rung, the linear rungs and the oracle rescaling all sit inside
  ±0.01 of zero.** Once the head starts at zero, the ranking among rungs mostly
  tracks prediction spread (pred sd). Noisier models lose more, because there is
  no signal to trade the variance against. That is also why *more* capacity is
  worse (default −0.19 vs minimal −0.01 to −0.17).
- **Per-step masking is a correctness fix, not a score fix.** It removes the
  junk (§1–2), but there is no signal underneath for it to uncover. The
  learned-E rung even gets noisier (−0.05 → −0.17, seed sd 0.18), because the
  un-junked inputs let it fit harder to noise.
- The checklist's structural fixes (event-type embedding, prior, top-k, reduced
  capacity, surprise) each move R² by ≤0.015. They are not separable from
  seed noise, which is as large as the effects.

Caveats: the ladder is cumulative, so marginal effects are confounded with
order. λ and k were not tuned (tuning on these folds would be selection on the
evaluation). The harness's early-stop tail is not purged from the fit set, so
k=3 labels overlap across that inner boundary. That leak can only *help* AGCRN,
so it cannot explain the underperformance.

## 3b. The economic sign prior (step 4)

The step-3 priors were estimated from Stage-1 ρ̂. Step 4 uses the a-priori
prior from `strategy_spec.md` §3.3 instead, with zero fitted parameters. The
edge A→B is `HAWKISH[A] × HAWKISH[B]` for every cross-release pair, and the
input is A's winsorised z-surprise, placed on the trigger node at its release
snapshot. A softmax graph can't carry the −1 edges (U3, JOBLESSCLAIMS), so the
prior goes in as a **frozen signed adjacency**. The full graph has 142 edges, 48
of them negative. The version restricted to the three BH-surviving channels of
`leadlag_findings.md` (labour→labour, inflation→inflation, labour→policy) has 25.

**Is the signal visible at snapshot resolution at all?** Sign agreement between
`Σ_i G[i,j]·z_i` and the target's Δ implied_mean (binomial p; cells treated as
independent, which is optimistic):

| graph | AGCRN label (t → t+3, after the release day) | diagnostic label (t−1 → t+2, includes the release day) |
|---|---|---|
| all cross-release | 0.512 (n 469, p 0.64) | 0.523 (n 430, p 0.36) |
| BH channels | 0.519 (n 108, p 0.77) | 0.578 (n 102, p 0.14) |
| labour→policy only | 0.596 (n 47) | 0.675 (n 40) |

The signal is flat on the label AGCRN is trained on. It leans the right way only
when the release day is included, and it's strongest in labour→policy, which
matches where the lead-lag study found it. Even there it has too few cells to
separate from chance. This is the postmortem's §1 horizon point measured
directly with the economic prior: the effect is priced by the end-of-day
snapshot, so the Δ that AGCRN predicts starts after the signal has gone.

**Walk-forward (8 folds × 3 seeds, per-step masking, zero-init head):**

| model | R² vs zero | dir. acc (y≠0) | pred sd | corr |
|---|---|---|---|---|
| linear own_momentum | +0.0002 | 0.550 | 0.092 | +0.048 |
| linear + economic signal (all) | −0.0001 | 0.547 | 0.093 | +0.047 |
| linear + economic signal (BH channels) | −0.0009 | 0.548 | 0.095 | +0.045 |
| AGCRN, frozen economic graph (all) + z | −0.117 ±0.07 | 0.527 | 0.235 | −0.002 |
| AGCRN, frozen economic graph (BH channels) + z | −0.126 ±0.06 | 0.524 | 0.186 | −0.009 |
| AGCRN, frozen economic graph (all) + z, shared weights | −0.047 ±0.00 | 0.489 | 0.186 | +0.002 |
| AGCRN, adaptive (hybrid) + z, no prior (control) | −0.016 ±0.01 | 0.473 | 0.077 | −0.008 |

The economic prior does not help AGCRN, and it does not help the linear model
either: the ridge gives the signal a ≈0 slope. The frozen signed graph is
actually *worse* than the adaptive control. A fixed row-normalised aggregation
pushes every in-neighbour's state into the target, and the model can't learn to
switch that off, so prediction spread roughly triples while correlation stays at
zero. This is a statement about the **weeks-ahead Δ implied_mean target**, not
about the prior. The same prior earns its result in `leadlag_findings.md` on a
different label (first post-release print → settlement, per leg). Testing AGCRN
with this prior fairly means moving AGCRN to that label.

## 3c. On the release clock (`analysis/event_time_2026_09/`)

The daily panel was rebuilt on an event-time grid: 154 release instants × 17
series. Node state comes from each contract's last print *before* the release,
and there are two labels on each node's lead contract: **imm** (≤3 prints within
24 h after the release, minus the pre-release print) and **settle** (settlement
minus the first post-release print). Walk-forward purging now uses each window's
label end, since settlement resolves up to 60 days later.

**The economic signal becomes visible.** Sign agreement of the zero-parameter
`Σ HAWKISH[a]·HAWKISH[b]·z_a`, with 95% CIs from a bootstrap over release
instants:

| | imm | settle |
|---|---|---|
| all 142 cross-release edges (the pre-specified graph) | 0.512 [0.47, 0.56] | 0.505 [0.47, 0.54] |
| 3 BH channels | **0.634 [0.55, 0.71]** | 0.544 [0.46, 0.63] |
| labour→policy | **0.704 [0.57, 0.82]** | **0.661 [0.54, 0.78]** |
| inflation→policy | **0.648 [0.52, 0.78]** | 0.576 [0.47, 0.70] |
| labour→inflation (wrong way) | 0.452 [0.37, 0.53] | 0.432 [0.36, 0.51] |

On the daily panel the same BH channels scored 0.519. The effect exists; the
daily clock was hiding it. inflation→policy fits the attention reading in
`leadlag_findings.md`: CPI moves FED **immediately** (0.65), so nothing is left
by settlement (0.58, CI spans 0.5). labour→policy survives to settlement.

**Caveats that bound this.**
- The three channels were selected by BH in the lead-lag study on the same
  in-sample events (settlement label), so the BH rows are not independent
  confirmation. The pre-specified all-edge graph is flat on both labels.
  Positive and negative channels cancel.
- The signal fires on only 134 (imm) / 159 (settle) evaluated cells, over ~85
  release instants.

**Learning it still loses to imposing it.** On the cells where the BH signal
fires, the zero-parameter rule is right 63%. Every fitted model does worse:

| model (imm label) | R² vs zero, all cells | R² / dir. acc on firing cells |
|---|---|---|
| linear, econ signal only (all edges) | **+0.005** | +0.017 / 0.596 |
| linear, econ signal only (BH) | −0.003 | −0.009 / 0.578 |
| AGCRN, adaptive (hybrid), no prior | −0.020 | −0.003 / 0.560 |
| AGCRN, frozen econ graph (BH) | −0.013 | −0.030 / 0.459 |
| AGCRN, frozen econ graph (all) | −0.020 | −0.015 / 0.514 |

On the settle label, the frozen-BH-graph AGCRN is the best AGCRN in the whole
study: R² −0.000, corr +0.06, 0.57 direction on firing cells. It still doesn't
beat the econ-only ridge (+0.008). The settle label's "own state" linear rung
reaches 74% directional accuracy, but that's the lead contract's price level
(a 90¢ contract settles YES about 90% of the time), not a signal. It is the same
trap `direction_study.md` retracted.

So the clock was the problem for *seeing* the effect, and with the right clock
the effect is real in the policy channels. With ~150 release instants, a
7k-parameter graph model can't learn it better than the zero-parameter sign
restriction. That is the same "impose, don't learn" result as
`leadlag_findings.md`, now shown for AGCRN.

## 3d. What trading it earns (`analysis/event_time_2026_09/backtest.py`)

The trade: when a strategy has a view on a target, take the lead contract at its
**first post-release print** (YES if positive, NO if negative) and hold to
settlement. The median entry is 3.0 h after the release. Costs follow
`leadlag_2026_09/economics.py`: the taker fee `ceil(0.07·C·P(1−P))` on
100-contract tickets plus half the measured spread; settlement is free. Net is
in cents per contract, with 95% CIs from a bootstrap over release instants, and
the permutation null shuffles surprise vectors across releases.

**Walk-forward test folds (694 candidate trades, 89 releases):**

| strategy | trades | NO share | hit | net ¢/contract [95% CI] | perm p |
|---|---|---|---|---|---|
| a-priori sign rule, all edges (nothing fitted) | 694 | 40% | 49% | **−5.77** [−8.82, −2.62] | 0.86 |
| linear, econ signal only (the model) | 694 | 73% | 53% | +2.51 [−0.27, +5.27] | – |
| *control: intercept only (sign of past mean residual)* | 694 | 69% | 53% | **+3.00** [+0.26, +5.80] | – |
| *control: always NO* | 694 | 100% | 49% | +4.80 [+2.26, +7.38] | – |
| linear, econ only, BH-channel cells | 159 | 82% | 59% | +7.57 [+1.33, +13.23] | – |
| *control: intercept only, same cells* | 159 | 87% | 60% | **+9.64** [+4.40, +14.47] | – |
| linear, own state (price level) | 694 | 50% | 75% | +3.07 [+0.47, +5.70] | – |
| AGCRN, frozen econ graph (all) | 694 | 46% | 55% | −1.80 [−5.11, +1.50] | – |
| rule, **labour→policy only** | 35 | 49% | 71% | **+6.03** [+1.42, +11.20] | **0.015** |

- **The linear model's profit is not the economic signal.** Its side is 73% NO,
  and in this sample YES is overpriced: always-NO earns +4.8¢, mostly in 2025.
  An intercept-only model earns *more* than the econ model on the same cells
  (+3.00 vs +2.51 overall, +9.64 vs +7.57 on BH cells), so the signal adds
  nothing over "lean NO". That drift is not stable either: the same intercept
  control over all releases nets +0.07¢.
- **The "own state" profit is the favourite-longshot bias:** 98–100% NO below
  25¢ and ~100% YES above 90¢ (the tail miscalibration in `price_structure.py`).
- **The a-priori sign rule loses money** (−3.55¢ over all 955 releases, −5.77¢
  on test folds). Shuffled surprises do as well 70–86% of the time.
  inflation→inflation is the worst channel (−10¢).
- **The one channel that pays is labour→policy:** +5.89¢ net [+1.01, +11.07]
  over all 59 releases, 47% NO (so not the drift), permutation p = 0.010,
  +10% return on capital. That is **+$348 in total at 100 contracts per trade**,
  across ~4 years. It was selected by BH in the lead-lag study on these same
  events, so it is an in-sample estimate that still needs the out-of-sample test
  in `strategy_spec.md` §9.

**Buy, then sell once the surprise is captured (`exits.py`).** This uses the
same trades and sides, but exits after the repricing instead of holding: at the
target's 3rd post-release print (≤24 h, the window the `imm` label measures),
or at 24 h. It compares taker and maker on each leg.

- A maker entry rests at the pre-release price from the release and fills only
  if a later print trades at or through it within 24 h, so adverse selection is
  built in (`maker_fill.py`).
- A maker exit rests at the exit price and falls back to crossing after 24 h.
- The maker fee is 0.0175·C·P(1−P), with 0 and the full taker fee as bounds.
- Queue position is ignored, which flatters the maker. The "strict" variant
  counts only prints *through* the limit.

labour→policy rule, all 59 releases (net ¢/contract, 95% clustered CI):

| plan | fills | hold | gross | costs | net |
|---|---|---|---|---|---|
| taker → hold to settlement | 100% | 26 d | +7.35 | 1.45 | **+5.89** [+1.01, +11.07] |
| maker → hold to settlement | 69% | 20 d | +8.19 | 0.20 | **+8.00** [+1.50, +14.70] |
| maker (strict fill) → hold | 46% | 20 d | +9.37 | 0.23 | +9.13 [−0.24, +18.98] |
| taker → taker @ capture | 100% | 0.4 h | +0.36 | 2.89 | −2.54 [−3.81, −1.41] |
| taker → maker @ capture | 100% | 1.2 h | +0.41 | 1.68 | −1.27 [−2.52, −0.15] |
| maker → taker @ capture | 69% | 0.3 h | −0.26 | 1.66 | −1.92 [−2.91, −1.03] |
| maker → maker @ capture | 69% | 1.3 h | −0.31 | 0.41 | −0.73 [−1.66, +0.15] |
| maker → maker @ 24 h | 69% | 26 h | +0.80 | 0.54 | +0.25 [−1.85, +2.41] |

- **There is nothing left to capture on a round trip.** A taker entering at the
  first post-release print sees a further move of only +0.36¢ by the capture
  point: the repricing *is* that first print. A maker resting at the
  pre-release price is filled only when the market comes back against it, so its
  gross is negative (−0.3¢), which is adverse selection. The second crossing
  then costs more than the move in every strategy. No capture exit has a
  positive net anywhere, and the same holds for the all-edge rule, the linear
  model and both controls.
- **Maker entry helps only when holding to settlement.** Costs fall from 1.45¢
  to 0.20¢, and labour→policy nets +8.0¢ per contract. But only 69% of signals
  fill (46% under the strict rule), so total dollars are *lower*: $328 (maker)
  and $247 (strict) against $348 (taker), at 100 contracts per trade.
- For the linear model, maker hold (+4.75¢) matches the intercept-only (+4.85¢)
  and always-NO (+4.73¢) controls, so it is still the NO drift, not the signal.

**Every channel, not just labour→policy.** Each of the 13 channels of the
a-priori graph with ≥10 signals, traded on its own signal alone, over all
releases (net ¢/contract, maker fee 0.0175):

| channel | n | maker fill | taker hold | maker hold | maker strict hold | TT cap | TM cap | MT cap | MM cap | TT 24h | MM 24h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| labour→policy | 59 | 69% | **+5.89** | **+8.00** | +9.13 | −2.54 | −1.27 | −1.92 | −0.73 | −1.03 | +0.25 |
| growth→inflation | 60 | 50% | +5.90 | +10.29 | +7.41 | −2.29 | −0.32 | −3.11 | +0.97 | −2.11 | +2.29 |
| labour→growth | 44 | 52% | +2.56 | +9.65 | +10.11 | −3.22 | −1.54 | −4.20 | −1.19 | −3.00 | −0.76 |
| growth→labour | 21 | 71% | +5.31 | +4.46 | +18.35 | −2.92 | −1.14 | −3.92 | −0.34 | −2.34 | +4.60 |
| inflation→labour | 141 | 60% | −1.39 | −0.42 | +1.99 | −4.69 | −0.32 | −3.86 | +2.06 | −4.93 | +1.62 |
| policy→inflation | 59 | 47% | −4.20 | −1.97 | −4.94 | −1.90 | −5.74 | −5.15 | −4.65 | −1.67 | −2.64 |
| inflation→policy | 66 | 64% | −1.72 | −4.29 | −8.25 | −2.07 | −2.07 | −3.03 | −3.51 | −1.73 | −3.86 |
| inflation→growth | 46 | 61% | −5.62 | +1.91 | +1.84 | −2.88 | −3.61 | −1.59 | −0.37 | −2.42 | −2.12 |
| policy→labour | 25 | 88% | −7.87 | −4.72 | −0.82 | −4.37 | −7.05 | −2.18 | −3.84 | −5.61 | −0.09 |
| labour→labour | 50 | 68% | −6.49 | −14.37 | −21.25 | −4.91 | +1.74 | −3.68 | −0.59 | −4.87 | −2.74 |
| growth→policy | 13 | 77% | −7.05 | −8.86 | −11.30 | −2.48 | −1.78 | −1.74 | −1.15 | −2.69 | −1.21 |
| inflation→inflation | 84 | 51% | −10.12 | −12.73 | −15.15 | −4.44 | −4.45 | −3.90 | −5.85 | −4.66 | −8.17 |
| labour→inflation | 287 | 56% | −6.32 | −5.68 | −6.52 | −4.93 | −3.93 | −4.21 | −2.92 | −4.62 | −3.21 |

TT/TM/MT/MM = taker or maker on entry→exit; "cap" exits at the 3rd post-release
print (≤24 h).

- **Capture exits:** 71 of the 78 channel × capture/24h cells are negative. The
  7 positive ones (best: growth→labour MM 24h +4.60, n = 15; inflation→labour
  MM cap +2.06, n = 85) all have clustered CIs spanning zero, and they are the
  best of six plans per channel, so they are selected.
- **Hold to settlement:** only labour→policy beats shuffled surprises
  (permutation p = 0.007). Across 13 channels its BH q is **0.088**, so it does
  not clear 5% FDR once the other channels are counted. growth→inflation is next
  (+5.90 taker, +10.29 maker, p = 0.11, CIs span zero).
- **The losing channels are not a wrong sign.** inflation→inflation (−10.1¢) and
  labour→inflation (−6.3¢) have CIs below zero, but they are no worse than
  shuffled surprises (lower-tail p = 0.15, 0.11). Costs and the YES drift make
  the shuffled baseline negative too, so reversing them is not a strategy.
- **The maker leg** raises the per-contract hold net in 8 of 13 channels, but
  fills only 47–88% of signals, and the strict-through variant swings widely at
  these sample sizes (labour→labour −21¢, growth→labour +18¢ on ~10–15 fills).

So the edge is a *settlement* edge, as `exit_rules.py` found for the lead-lag
spec. The market reprices the surprise at the first print, too fast to trade,
and what is left is a mispricing of the eventual outcome in the labour→policy
channel.

## 4. What changes in the existing writeups

- `agcrn_study.md` / `artifacts/agcrn_report.md`: "no model beats predict-zero"
  **stands**. "AGCRN is far worse than linear" should now read "AGCRN on `main` is
  worse because of an init/early-stopping artefact plus a padding leak. Fixed,
  it is indistinguishable from the linear ladder, which is itself
  indistinguishable from zero."
- `agcrn_postmortem.md` §5: the learned-E ≫ shared-MLP-E gap and "default ≫
  minimal" are mostly **prediction variance from the random head**, not
  "information density per node". The ISMPMI saturation of the shared-MLP Ã is
  **the padding leak**: 58% of Ã's mass sits on padded nodes, and ISMPMI is
  padded 97% of the time.
- `artifacts/adjacency_comparison.md` used the swapped export. The "top edges"
  it lists are reversed, and it should be regenerated after re-running
  `scripts/train_agcrn.py`.
- Recommended default for any further AGCRN runs:
  `masking="per_step", zero_head=True`. `scripts/train_agcrn.py` still uses the
  legacy defaults so its published numbers reproduce.

## 5. The ablation's missing rungs: graph-free surprise and a structure null (steps 5–6, 2026-09-25)

Added to complete the ablation table (pooled surprise → identified graph →
shuffled graph → learned adjacency). Same harness, target and folds as §3.
Surprise (`s_pit`) sits on the triggering node. In the node set the BH graph
is 2 edges (CPI→FED, PAYROLLS→FED), because FEDDECISION is not a node, and
the searched-ρ̂ graph is 118 edges on 21 nodes. A shuffle relabels the
graph's own endpoint nodes, so weights, signs and degree profile are kept and
only *which* markets connect changes.

**Graph-free rungs (step 5a).**

| rung | R² vs zero | corr |
|---|---|---|
| linear own_momentum | +0.0002 | +0.048 |
| + own surprise, **no graph** | **+0.0028** | +0.071 |
| + pooled surprise (complete unweighted graph) | −0.0001 | +0.049 |
| + surprise through Stage-1 ρ̂ (all / BH) | +0.0026 / +0.0026 | +0.071 |

The node's *own* surprise accounts for all of §3's "linear + surprise" gain,
and the graph term adds nothing. §2 had already read that own-surprise
correlation as the node rolling to its next event, not as spillover.

**Shuffled graph, linear (step 5b, 200 draws).** Real ρ̂ graph R² +0.0026 against a
null median of +0.0026 (p = 0.48); BH graph p = 1.00. The specific structure
does not matter at this horizon.

**Shuffled graph, frozen-graph AGCRN + surprise (step 5c, 5 draws × 3 seeds × 8 folds).**

| graph | real R² | permuted mean (range) | real corr | permuted corr |
|---|---|---|---|---|
| all searched ρ̂ | −0.096 | −0.193 (−0.253 … −0.116) | −0.010 | −0.021 … +0.032 |
| BH survivors | −0.074 | −0.099 (−0.119 … −0.084) | −0.015 | −0.021 … −0.010 |

The real graph ranks first of six in both, which is the smallest p that five
draws allow (1/6 ≈ 0.17). It is less bad **because it predicts less**:
pred sd 0.165 against 0.18–0.33 for the shuffles. Corr and the
best-rescaling R² are ≈ 0 for every graph, real or shuffled. So this does
not show that the identified structure carries signal. At most, the true
graph lets the model sit closer to predict-zero. Only 3 distinct BH shuffles
exist, and two repeat.

**Does Ã recover Stage-1 beyond chance? (step 6, last fold, 3 seeds, 5,000 draws).**
Ã is averaged only over windows where both endpoints are active, so a pair's
weight no longer depends on how often it trades. §2's +0.18 used the
unconditional mean, which is diluted by inactive windows.

| variant | rank corr with \|ρ̂\| | p (node-label null) | p (value shuffle) | BH-survivor percentile (p) |
|---|---|---|---|---|
| per-step / learned E | −0.04 | 0.59 | 0.64 | 0.82 (0.17) |
| per-step / shared-MLP E | −0.14 | 0.80 | 0.91 | 0.65 (0.30) |
| per-step / hybrid | −0.17 | 0.85 | 0.95 | 0.46 (0.58) |
| hybrid + λ\|ρ̂\| prior *(positive control)* | **+0.36** | **0.013** | **<0.001** | 0.75 (0.19) |

**No learned Ã recovers the Stage-1 structure beyond chance.** The control,
which is given ρ̂ in its logits, passes both nulls, so the test has the power
to detect recovery when it exists. This replaces §2's un-nulled
"+0.18 rank corr" with a proper negative.

**Reading for the ablation table.** At the k=3 snapshot horizon: surprise
alone (no graph) ≈ identified graph ≈ shuffled graph ≈ learned adjacency ≈
predict-zero. Learned adjacency does not beat the economically identified
prior, but the prior does not beat a shuffled graph either. The finding is
that there is nothing to propagate at this horizon, not that one graph beats
another (see `agcrn_postmortem.md` on the horizon mismatch).
