# Graph models against linear: every metric, the ablations, the signs, gross returns

*2026-09-26. Event-time panel (154 release instants × 17 series, in-sample,
walk-forward with 8 folds purged on label end, 3 seeds). Scripts in
`analysis/event_time_2026_09/`: `metrics.py`, `ablation.py`, `returns.py`
(outputs in `out/metrics.txt`, `out/ablation.txt`, `out/returns.txt`), and
`scoped.py` (`out/scoped.txt`, `out/scoped_t2.txt`, added 2026-09-27).
Synthetic sign check: `analysis/recovery_2026_09/recovery.py`, tag `signs`.*

Four questions:
1. Does any graph model beat the linear models on any metric?
2. Does the STG beat temporal-only and spatial-only versions of itself?
3. Can the graph models learn the *signs* of the edges, with and without the economic prior?
4. How do the models compare on gross returns, before costs?
5. If the graph is scoped to the relations the data supports, and trained only
   on the cells they can explain, does it learn them then?
6. Which of the usual explanations for an STG failing on this kind of signal
   apply here (§6: a response to external review comments)?

## Summary

1. **On prediction metrics, no graph model beats linear on anything.**
   - Across R², directional accuracy, balanced accuracy, F1 (up class and
     macro) and AUC, on both labels, every CI on ΔAUC and Δbalanced-accuracy
     against the best linear rung includes zero or favours linear.
   - The only point estimates in a graph model's favour are +0.010 AUC
     (AGCRN with the frozen all-edge graph, immediate label, CI ±0.08) and a
     slightly higher F1-up. Both are within noise, and the F1-up gain comes
     from predicting "up" more often.
2. **The STG is not better than its parts.** Removing the graph, the history,
   or both never hurts AGCRN by more than noise, and "neither" (a per-node
   network on the current state) has the best R² of the AGCRN rungs on the
   immediate label. The linear pattern is the same: the economic signal on its
   own is the best linear rung on that label, and adding own-state history
   makes it worse.
3. **Without the signs given, the graph models don't learn them. Given the
   signs, AGCRN mostly keeps them.**
   - With no prior, the adaptive and signed AGCRNs score balanced sign
     accuracy of 0.44–0.64, p ≥ 0.25.
   - An unsigned structural prior (which edges exist, not their signs) doesn't
     help either: 0.37–0.57.
   - The frozen signed all-edge graph keeps the theory sign, at 0.88–1.00
     (p ≤ 0.001) except on the settlement label's BH edges (0.72, p = 0.10).
     The frozen BH graph keeps it on the settlement label (0.88) but *reverses*
     it on the immediate label (0.50).
   - The only no-prior learner with a sign signal is the **linear** free graph,
     on the immediate label's BH edges (0.78, p = 0.023).
   - With a known planted graph, the no-prior low-rank linear graph reaches
     0.94–1.00 from 3× the data. No-prior AGCRN stays at 0.62–0.78.
4. **On gross returns, the best signals are linear and come from own state, not
   the network.**
   - Settlement models on all cells: linear own-state +5.20¢/trade gross,
     +5.22¢ over a random side with the same long/short mix, CI [+2.6, +7.9].
   - The best graph model is AGCRN with the frozen all-edge graph and shared
     W: +3.15¢ over random side [+0.6, +5.9], but −0.55¢ gross, because it goes
     long 77% of the time into a market where YES was overpriced.
   - The STG ablation shows no gain from combining graph and history.
5. **Scoping the graph to the supported relations doesn't make it learn them.**
   - On this panel only one channel has t ≥ 2 on both labels: labour→policy
     (PAYROLLS +, U3 −, JOBLESSCLAIMS − → FED), plus GDP→FED on the
     immediate label.
   - Scoped to it and trained only on its cells, every AGCRN ranks *worse* than
     chance on the immediate label (AUC 0.23–0.41, full-sample and walk-forward
     selection), including the one given the correct signs.
   - The zero-parameter sign rule is the best model in that scope: AUC
     0.76–0.78 on the immediate label, and +7.3¢/trade over a random side on
     the settlement label, CI [+2.5, +12.7].
   - Selected honestly (inside each fold), the settlement channel never reaches
     t ≥ 2 in any training fold. It can be imposed from theory but not
     discovered from the data available at the time.
   - The one scoped graph result with a CI clear of zero (signed AGCRN, BH
     scope, settlement) does not carry over to the t ≥ 2 scope.
6. **External review comments.** Of six proposed causes, five had already been
   tested and ruled out as the binding constraint. The sixth (surprise
   encoding) was checked and is not a bug. The binding constraint remains the
   one `recovery_test.md` measured: too few events per edge, and a surprise
   diluted among the other inputs.

## 1. Every metric

Scores are in per-series z units, on seed-averaged out-of-fold predictions.
"acc" is directional accuracy on cells with y ≠ 0; "bal acc" averages the
up and down hit rates. ΔAUC and Δbal acc are against the reference linear
rung, with a bootstrap over release instants.

**Immediate repricing** (reference: linear econ signal, all edges; 53.3% up):

| model | R² | acc | bal acc | F1 up | F1 macro | AUC | ΔAUC [95% CI] |
|---|---|---|---|---|---|---|---|
| linear econ signal (all) — ref. | **+0.005** | 0.540 | 0.521 | 0.651 | 0.488 | 0.513 | – |
| linear own + econ (BH) | −0.025 | 0.529 | 0.510 | 0.643 | 0.476 | **0.562** | +0.049 [−0.008, +0.106] |
| linear both: own lags + econ (BH) | −0.106 | **0.546** | **0.537** | 0.613 | **0.532** | 0.558 | +0.045 [−0.019, +0.103] |
| AGCRN frozen econ graph (all) | −0.009 | 0.544 | 0.520 | 0.671 | 0.465 | 0.523 | +0.010 [−0.068, +0.088] |
| AGCRN spatial only (adaptive) | −0.010 | **0.546** | 0.516 | **0.695** | 0.405 | 0.492 | −0.021 [−0.096, +0.055] |
| AGCRN STG (adaptive) | −0.016 | 0.515 | 0.491 | 0.652 | 0.426 | 0.471 | −0.042 [−0.123, +0.042] |

- Every model's balanced accuracy is 0.47–0.54, so nothing clearly beats a
  coin flip once the base rate is removed.
- AGCRN's higher F1-up (up to 0.695) comes from calling "up" more often
  (in the up-heavy sample), not from skill: its F1 macro is the lowest in the
  table.

**Settlement residual** (reference: linear own state; 51.3% up):
- Linear own state scores accuracy 0.746 and AUC 0.809. Every graph model is
  0.17–0.30 AUC lower, with CIs well clear of zero.
- **That accuracy is not skill.** The residual is y = 100·[YES] − p_entry, so
  its sign is the event's outcome, and a contract's price already predicts its
  outcome about 75% of the time. On this label accuracy and AUC measure the
  market's own probability, and a model scores well by reading the price level.
  The skill question is the expected P&L, which is §4.

## 2. STG vs temporal-only vs spatial-only

The same AGCRN (7,799 parameters) with parts removed. R² vs zero, seed mean
(± sd):

| rung | imm R² | imm dir | settle R² | settle dir |
|---|---|---|---|---|
| STG: adaptive graph + history | −0.016 ±0.009 | 0.515 | −0.033 ±0.018 | 0.513 |
| temporal only (no graph) | −0.020 ±0.007 | 0.521 | −0.035 ±0.009 | 0.552 |
| spatial only (adaptive graph, no history) | −0.010 ±0.003 | 0.546 | −0.003 ±0.010 | 0.487 |
| neither (per-node network, current state) | **−0.004 ±0.007** | 0.531 | −0.020 ±0.002 | 0.559 |
| STG: economic BH graph + history | −0.020 ±0.003 | 0.496 | −0.072 ±0.015 | 0.530 |
| spatial only: economic BH graph | −0.007 ±0.009 | 0.496 | −0.047 ±0.021 | 0.537 |

- **No rung benefits from combining graph and history.** Every rung is below
  zero. Removing the history *improves* R² in every pair (STG → spatial only,
  temporal only → neither).
- Linear analogues on the immediate label:
  - own state at the last instant: −0.024;
  - own-state lags over 6 instants (standardised, clipped at ±3 sd): −0.102;
  - economic signal: +0.005;
  - own lags + signal: −0.106.

  So linear history hurts too, and the only above-zero linear rung is the
  spatial economic signal on its own.

## 3. Do the graph models learn the signs?

**Read-out.** Each fitted model's effective edge ∂ŷ_b/∂z_a, averaged over
held-out windows where a released and b is labelled, over folds and seeds.

**Why raw agreement misleads.** Most theory edges are positive: 94 of 142, and
19 of the 25 BH edges. So a model whose effective edges are all positive
("move with your neighbours' surprise") already agrees with theory on 66% / 76%
of edges. The adaptive AGCRN's raw 0.66 / 0.70 is exactly that. The test used
here is **balanced sign accuracy**: the mean of the hit rates on
theory-positive and theory-negative edges, where 0.5 means no sign learned,
with a one-sided Fisher exact test.

**Real data** (bal / p, all candidate edges | BH edges):

| model | immediate | settlement |
|---|---|---|
| AGCRN adaptive, no prior | 0.50 / 0.62 \| 0.53 / 0.60 | 0.44 / 0.90 \| 0.64 / 0.40 |
| AGCRN signed directed, no prior | 0.51 / 0.48 \| 0.56 / 0.55 | 0.54 / 0.25 \| 0.56 / 0.49 |
| AGCRN + unsigned BH prior | 0.53 / 0.26 \| 0.57 / 0.45 | 0.50 / 0.64 \| 0.37 / 0.94 |
| AGCRN frozen signed BH graph | 0.50 / 0.72 (same 20 edges) | **0.88 / 0.004** |
| AGCRN frozen signed all-edge graph | **0.88 / <0.001 \| 1.00 / <0.001** | **0.90 / <0.001** \| 0.72 / 0.10 |
| linear free graph, no prior | 0.57 / 0.10 \| **0.78 / 0.023** | 0.55 / 0.22 \| 0.57 / 0.44 |

For reference, the data's own full-sample sign (Spearman of z_a with y_b,
n ≥ 5) agrees with theory on 81% of the 16 readable BH edges on the immediate
label, and 50% on the settlement label.

- **No prior → no sign.** Neither AGCRN's balanced accuracy clears chance.
  Where one gets the negative edges right, it gets the positive ones wrong: the
  signed AGCRN on the immediate label's BH edges hits 0.80 of the negative edges
  but 0.31 of the positive ones. The adaptive AGCRN mostly reads everything as
  positive (hit− 0.10–0.20 on the immediate label).
- **An unsigned prior doesn't supply signs,** by construction. It tells the
  softmax which edges exist, and the softmax cannot represent a negative edge.
- **A signed prior mostly survives training, but not always.** The frozen BH
  graph on the immediate label ends at 0.50: the node weights reverse the given
  sign on most of the negative edges (hit− 0.25).
- **The linear free graph is the only no-prior learner with a sign signal**,
  on the one label and subset where the data itself agrees with theory (BH,
  immediate).

**With a known planted graph** (recovery test, 3 reps per cell, balanced sign
accuracy of ∂ŷ/∂z on the 25 true edges):

| learner | 1×, ρ = 0.4 | 1×, ρ = 0.6 | 3×, ρ = 0.6 | 10×, ρ = 0.4 | 10×, ρ = 0.6 |
|---|---|---|---|---|---|
| low-rank linear graph, no prior | 0.72 | 0.84 | 0.99 | **1.00** | 0.99 |
| AGCRN, no prior | 0.62 | 0.66 | 0.65 | 0.78 | 0.70 |
| AGCRN signed, no prior | 0.62 | 0.59 | 0.59 | 0.64 | 0.74 |
| AGCRN + true graph, unsigned prior | 0.65 | 0.66 | 0.71 | 0.80 | 0.85 |
| AGCRN + true signed graph, frozen | **0.87** | **0.88** | **0.98** | 0.82 | **0.95** |

- The same ordering holds. A small linear graph learns signs from about 3× the
  data with no prior.
- AGCRN without the signed prior stays well short even at 10×.
- Given the correct signed graph, AGCRN keeps the signs at the real size
  (0.87–0.88), but not reliably: it gets 0.60 in the 3×, ρ = 0.4 cell.
- Each cell has only 6 negative edges and 3 repetitions, so individual cells
  are noisy (±0.1). The ordering is the result, not the decimals.

## 4. Gross returns, before costs

**The trade.** Take the lead contract at its first post-release print, on the
predicted side, and hold to settlement. Gross P&L is side × (100·[YES] − p_entry)
cents per contract.

**The skill measure.** In this sample always-NO earns +6.93¢ gross, because YES
was overpriced, mostly in 2025. So a short-leaning model earns for that reason
alone. **"vs random side"** subtracts that: it is the model's P&L minus that of a
random side with the model's own long/short mix.

**Settlement-label models, all 694 cells** (¢/trade, 95% CI over instants):

| signal | % short | gross | vs random side |
|---|---|---|---|
| always NO | 100% | +6.93 [+4.45, +9.50] | 0 |
| zero-parameter econ sign rule (BH, firing cells) | 34% | −3.62 [−9.81, +2.55] | −0.12 [−5.42, +5.71] |
| **linear own state** | 50% | **+5.20** [+2.55, +7.87] | **+5.22** [+2.57, +7.89] |
| linear own + econ (BH) | 49% | +4.15 [+1.38, +6.93] | +4.25 [+1.47, +7.05] |
| linear econ signal (BH) | 63% | +4.52 [+1.58, +7.34] | +2.66 [−0.15, +5.42] |
| AGCRN frozen econ (all), shared W | 23% | −0.55 [−3.29, +2.31] | +3.15 [+0.56, +5.90] |
| AGCRN STG, adaptive | 30% | −5.77 [−8.28, −3.34] | −2.95 [−5.14, −0.85] |
| AGCRN temporal only | 24% | −3.86 | −0.23 [−2.61, +2.20] |
| AGCRN spatial only | 54% | +0.13 | −0.49 [−3.03, +1.93] |
| AGCRN neither | 25% | −2.86 | +0.59 [−1.69, +2.89] |
| AGCRN signed directed, no prior | 43% | +0.10 | +1.08 [−1.91, +4.05] |

- **The best signal is linear own state,** and it is not a network signal. It
  reads the contract's price level and recent ladder drift, i.e. the
  favourite-longshot and YES-overpricing effects.
- **The linear economic-signal rungs earn their gross mostly by leaning
  short** (63–73% short): their excess over random side has a CI through zero.
- **No AGCRN beats the matched linear rung on gross.** The adaptive STG has a
  significantly *negative* skill (−2.95¢): it leans long into overpriced YES.
- **Using the immediate-label models' direction for the settlement trade**
  (543 cells), linear own state again leads (+3.80¢ over random side, CI
  [+1.5, +6.0]). The graph-free AGCRN rungs are +1.7 to +2.5¢, slightly above
  the STG (−1.55¢).
- **One large cell to treat with caution.** On the 134 cells where the BH
  signal fires, the immediate-label *linear own-state-lags* model earns
  +11.4¢ gross (+11.7 over random side, CI [+6.5, +16.3]). That is consistent
  with the short-term momentum that helps the lead-lag spec
  (`strategy_spec.md`), but it is one of about 100 CIs in `returns.txt`, on a
  subset, and it does not appear on all cells (+3.3, CI through zero). It is a
  hypothesis to pre-register for the out-of-sample test, not a finding.

## 5. Scoping the graph to the relations the data supports

`scoped.py`. Each model sees only a scope: a set of edges, the series they
touch (the others are masked out of the inputs), and only the *firing* cells.
A firing cell is a target's label at an instant when one of its in-scope
sources released; the models are trained and scored on those cells only. The
walk-forward, purge and training are the same as `models.py`.

**Scopes:**

| scope | edges | chosen how |
|---|---|---|
| hub | labour + inflation → FED (10 edges, 2 negative) | Stage-1 structure, fixed in advance |
| BH | the three BH channels (25 edges, 6 negative) | lead-lag study, fixed in advance |
| t2 | every type→type channel with theory-signed slope t ≥ 2 on the full sample | this panel, full sample |
| wf | edges with Spearman p < 0.10, n ≥ 8, on each fold's training windows, data-signed | inside each fold |
| t2wf | the t ≥ 2 channel rule applied inside each fold | inside each fold |

hub, BH and t2 were chosen on 2021–2025 data that includes the test blocks,
so they are optimistic. wf and t2wf are honest.

**Which channels have t ≥ 2.** The regression is y_b (per-series z units) on
Σ_a HAWKISH[a]·HAWKISH[b]·z_a over the channel's cells, with an intercept and
SEs clustered by release instant, on the full sample:

| channel | immediate: n, t | settlement: n, t |
|---|---|---|
| labour→policy | 57, **+3.03** | 58, **+2.02** |
| growth→policy (GDP→FED) | 12, **+4.56** | 12, −1.12 |
| inflation→policy | 64, +1.63 | 65, +1.14 |
| labour→labour | 45, +1.49 | 50, +0.66 |
| inflation→inflation | 58, +0.68 | 84, −0.51 |
| growth→labour | 19, −0.09 | 20, +1.98 |
| growth→inflation | 40, +0.22 | 59, +1.88 |

The BH channels come from the lead-lag study's full-ladder panel. On this
panel only labour→policy clears t ≥ 2 on both labels, so the t2 scope is
essentially **labour → FED**.

**Immediate repricing** (balanced accuracy / AUC on the scope's test cells;
sign = balanced sign accuracy of the effective edges vs theory):

| model | hub (73 cells) | BH (134) | t2 (41) | t2wf (35, honest) | wf (71, honest) |
|---|---|---|---|---|---|
| zero-parameter sign rule | **0.70 / 0.69** | **0.61 / 0.60** | **0.69 / 0.78** | **0.68 / 0.76** | 0.62 / 0.69 |
| linear, sign imposed | 0.62 / 0.69 | 0.52 / 0.53 | 0.56 / 0.71 | 0.66 / 0.74 | 0.63 / 0.65 |
| linear free (sign) | 0.64 / 0.72 (0.75) | 0.54 / 0.53 (0.67) | 0.56 / 0.59 (1.00, 4 edges) | 0.66 / 0.70 (0.89) | **0.66 / 0.70** (0.42) |
| AGCRN adaptive (sign) | 0.49 / 0.50 (0.50) | 0.51 / 0.46 (0.44) | 0.48 / 0.29 (0.50) | 0.50 / 0.32 (0.50) | 0.55 / 0.45 (0.50) |
| AGCRN signed (sign) | 0.49 / 0.49 (0.50) | 0.56 / 0.62 (0.49) | 0.50 / 0.39 (0.50) | 0.55 / 0.41 (0.50) | 0.56 / 0.47 (0.69) |
| AGCRN frozen (sign) | 0.49 / 0.50 (**1.00**) | 0.49 / 0.46 (0.52) | 0.41 / 0.32 (1.00) | 0.39 / 0.23 (**1.00**) | 0.46 / 0.45 (0.56) |
| AGCRN adaptive, trained on *all* cells | 0.52 / 0.52 | 0.52 / 0.56 | 0.55 / 0.60 | 0.55 / 0.61 | 0.56 / 0.62 |
| linear own + econ, trained on *all* cells | 0.64 / 0.78 | 0.58 / 0.63 | 0.68 / 0.84 | 0.71 / 0.84 | 0.60 / 0.73 |

- **Scoping doesn't let AGCRN learn the channel.** The scoped AGCRNs reach at
  most AUC 0.62 (signed, BH scope) and are mostly near chance. In the
  labour→FED scopes they are clearly below it (AUC 0.23–0.41), including the frozen one whose edge signs are
  correct (1.00). Its predictions must therefore come from FED's own-state
  inputs, not from the edges.
- The same AGCRN trained on all cells does *better* on the scoped cells
  (0.52–0.62). Restricting its training to the firing cells cost it
  information rather than removing noise.
- **The best models in every scope are the simplest:** the zero-parameter sign
  rule, and in the honest scopes the imposed-sign and free linear models.
  The highest AUC, linear own + econ trained on all cells, adds the target's
  own state.

**Settlement residual** (balanced accuracy / AUC; excess ¢/trade over a
random side with the same long/short mix, 95% CI over instants):

| model | hub (74) | BH (159) | t2 (35) |
|---|---|---|---|
| zero-parameter sign rule | 0.62 / 0.60; +3.9 [−0.4, +8.9] | 0.55 / 0.55; −0.1 [−5.4, +5.6] | **0.73 / 0.77; +7.3 [+2.5, +12.7]** |
| linear, sign imposed | 0.59 / 0.51; +0.7 | 0.54 / 0.58; +1.6 | 0.67 / 0.75; +3.2 [−2.0, +8.2] |
| linear free | 0.49 / 0.47; −0.5 | 0.56 / 0.53; +2.7 | 0.47 / 0.42; −2.4 |
| AGCRN adaptive | 0.49 / 0.51; −2.2 | 0.59 / 0.61; +4.6 [−1.4, +9.6] | 0.48 / 0.36; −0.4 |
| AGCRN signed (sign) | 0.63 / 0.74; +3.4 (0.25) | 0.60 / 0.62; **+5.6 [+1.3, +9.7]** (**0.75**, p = 0.03) | 0.60 / 0.62; +2.8 [−2.8, +8.7] |
| AGCRN frozen | 0.59 / 0.63; +3.8 | 0.47 / 0.52; −5.5 | 0.61 / 0.56; +2.0 |
| linear own + econ, trained on *all* cells | 0.90 / 0.91; **+5.9 [+1.4, +10.4]** | 0.72 / 0.79; +2.4 | 0.86 / 0.91; +4.2 |

- **The zero-parameter labour→FED rule** (t2 scope, +7.3¢ over random side) is
  the same effect as the earlier costed backtest (labour→policy +5.89¢ net
  taker, `agcrn_checklist.md` §3d), now seen in the scoped frame.
- **The signed AGCRN in the BH scope** gets the edge signs right (0.75,
  p = 0.03) and earns +5.6¢ over random side. But:
  - it is 75% short on cells where always-NO earns +10.9¢;
  - the BH scope was chosen on overlapping data;
  - it is one of about 60 scoped comparisons;
  - it shrinks to +2.8¢ (CI through zero) in the t ≥ 2 scope.

  Treat it as a candidate for the out-of-sample test, not a finding.
- **The honest scopes fail on the settlement label.**
  - wf: 32 test cells; no model has a CI clear of zero.
  - t2wf: labour→policy (full-sample t = 2.02) never reaches t ≥ 2 inside any
    training fold. The folds pick inflation→inflation (3 of 8) or
    inflation→labour (2 of 8) instead, leaving 25 test cells on 10 instants,
    and every model fails.
  - So the settlement-label labour→FED effect could not have been discovered
    ahead of time from the data. It can only be imposed from theory.

## 6. Response to external review comments (2026-09-27)

A reviewer proposed six reasons an AGCRN would fail on a sparse, directional,
release-driven lead-lag signal, two diagnostics, and four fixes. Each was
checked against what has been run:

| # | proposed cause | applies? | where it was tested | result |
|---|---|---|---|---|
| 1 | Symmetric EEᵀ graph can't represent one-way diffusion | Yes, for the standard adjacency | Signed directed adjacency E_dst E_srcᵀ (strictly more expressive than MTGNN's M₁M₂ᵀ − M₂M₁ᵀ, which is one-way but non-negative): recovery test, §3, §5 | Not the binding constraint. It doesn't recover the planted graph at 10× with full inputs (AUROC 0.53), and on real data it doesn't learn signs (§3) or scoped relations (§5) |
| 2 | Static edges pass noise on quiet days; the signal is event-conditional | Largely designed out | The event-time panel steps are release instants, not days. The surprise channel is exactly 0 unless the series released, so a message carrying z is gated by construction | The remaining leak is the non-surprise features flowing along edges. That is the dilution the recovery test measured: feeding only [released, z] recovers the graph at 10× (AUROC 0.86) |
| 3 | Quiet-day gradients swamp the release windows | Yes | §5: loss only on firing cells, i.e. the extreme version of up-weighting events | Doesn't help. Scoped AGCRNs are at or below chance, and worse than the same model trained on all cells |
| 4 | Softmax row normalisation dilutes one strong edge among N−1 noise edges | Yes, for the adaptive adjacency | Top-k (checklist step 3), signed adjacency without softmax, frozen graphs (§3, §5) | No gain from any of them |
| 5 | Surprise encoding washes out release values; NaN→0 is ambiguous | Checked | `event_nodes.parquet`, `_panel.py` | Not a bug. z = 0 on every non-release step, with a separate release flag. Per-series standardisation shifts those zeros to a small constant (about −0.07 sd for FED) but doesn't squash release values. The recovery test used the same scaling and still recovered the graph once inputs were focused |
| 6 | Daily bars collapse an intraday lag | Yes, for the daily panel | The event-time panel (`agcrn_checklist.md` §3c): state as of just before the release, response from post-release prints | Fixed by design; all results in this report use it |

**The two diagnostics:**
- *Inspect the learned adjacency.* Done (checklist steps 2 and 6, and
  `recovery_test.md`). Ã is near-uniform on real data, and under a known graph
  it ranks the true edges *below* chance even with no signal.
- *A simple linear test on the identified pairs.* Done (§1, §5). The one-slope
  imposed-sign linear model, and the zero-parameter rule, match or beat every
  AGCRN on the identified channels.

**The four fixes:**

| fix | status |
|---|---|
| explicit surprise encoding | Already in place (see row 5 above) |
| supply the graph (fixed, directed, sparse, or as a prior) | Frozen signed graphs (§3, §5) and an unsigned soft prior. With the correct signs frozen in, AGCRN keeps the signs but still doesn't predict better than the linear rule |
| event-conditional attention | Not built as GAT, but its core (a directed message carrying the surprise only when the source releases) is what the scoped free linear model computes. That model is among the best scoped models on the immediate label, as the reviewer predicted, but it is a linear rule, not an STG |
| up-weight event windows | §5 |

**The reviewer's test and its result.** The reviewer's criterion was: "if the
fixed directed graph plus proper surprise encoding helps, the issue was graph
learnability, not absence of signal." Here the fixed directed graph plus
surprise encoding does *not* help AGCRN, while the zero-parameter rule on that
same graph works. The signal is real on the labour → FED channel, but it
amounts to about one parameter's worth. The recovery test puts the data needed
to learn it with an STG at 10× the current calendar or more.

## What this means

On every axis tested (prediction metrics, the component ablation, sign
learning and gross returns), the graph models either match or trail simple
linear models. The one thing the graph structure carries is **the economic
sign, when it is imposed**. It carries it only when frozen in, and even then
training can reverse it.

This is the same conclusion as `recovery_test.md`, reached on the real panel:
at this sample size the network has to be supplied, not learned, and the
signal that is there is small enough that a one-parameter linear rung captures
all of it.

## Caveats

- In-sample (2021-11 to 2025-12). The out-of-sample 2026 test is unspent.
- About 25 models × 4 return blocks × 2 metrics. Expect several nominal CI
  exclusions by chance. The conclusions above rest on patterns that hold
  across rungs, not on single cells.
- The settlement label's accuracy and AUC are dominated by the price level
  (§1). Only the P&L columns measure skill on that label.
- Linear own-state lags needed standardising and clipping at ±3 sd
  (`d_q50_7d` reaches 87 sd out of sample). Unclipped, the rung scores
  R² −0.55. The AGCRN rungs standardise their inputs by construction.
