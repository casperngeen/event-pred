# Intraday post-release paths: how fast news reaches each market

*2026-09-27. Plan: `intraday_path_plan.md`. Scripts: `analysis/intraday_2026_09/`
(outputs in its `out/`, git-ignored). In-sample (2021-10 → 2025-12). Nothing
from 2026 was read, and no path is forward-filled past 2026-01-01.*

## Summary

The across-release STG found no temporal dependence to learn (`graph_ablation.md`
§9). This study moves the clock inside a release: every target's price is
followed on 5-min → hourly → daily bars for 7 days after each of 139 Kalshi
releases and 1,030 non-Kalshi calendar releases.

| question | answer |
|---|---|
| Q1 speed | **Kalshi cross-series responses are fast:** half by 15 min, all by ~2 h (+0.44¢ per z at 24 h). **labour → policy is the fastest channel** (half-life 5 min, complete by 1 h), not a drift. The one slow channel is **labour → growth** (half-life ~3 h, 58% by 8 h). The releasing series' own next contract is slower (half-life ~1.75 h). |
| Q2 clock vs trade time | **The first post-release print is not the whole move.** It carries 0.67 [0.45, 0.87] of the own series' 24 h response and ~0.5 of the cross-series one; the third print carries 0.85, the fifth to tenth ~1.0. Most clock-time slowness is waiting for a trade, except labour → growth, which is slow in trade time too. |
| Q3 order | **No propagation through the network.** The source market's own move adds nothing to a target's remaining move beyond the headline surprise (t < 1 at 15 min / 1 h / 4 h; walk-forward ΔR² < 0). Markets react to the release in parallel. |
| Q4 temporal model | **No.** On last-trade prices, a GRU trained on traded bars edges the best linear rung at 1 h (ΔR² +0.004 [+0.001, +0.007]), but that is bid-ask bounce: on 2-print prices it vanishes, and no sequence or graph model beats a linear rung anywhere. The AGCRN stays at predict-zero. The one learnable piece is the own next contract's unfinished response to its z, which **one pooled absorption curve** predicts best (24 h R² +0.13, AUC 0.68). LightGBM, a random forest, an MLP and a Bayesian hierarchical graph (§5) don't change this: none beats a linear rung on R², and all trail the curve on the own contract. |

Two of the plan's stated expectations fail:
- labour → policy was expected to drift, and it doesn't;
- "most of the response by the first print" holds only from the third print on.

The third, "sequence models tie the per-channel curve", holds: the GRU ties or
loses, and the AGCRN loses.

## 0. The panel (`build_paths.py`)

**Release times.** A Kalshi market closes before its release: usually 5 min
before, but some closed at midnight on the release day. The event-time panel's
`instant` is that close. Here τ = 0 is the **economic-calendar release minute**
of the instant's series, matched within ±24 h:
- 146 of 154 instants match, merged into 139 release minutes;
- 8 have no calendar release within a day and are dropped. Seven are in the
  late-2025 government shutdown, when markets closed but nothing was released;
  the eighth is a GDP release missing from the calendar;
- the 1,030 calendar-arm minutes are `spillover_2026_09/releases.py`'s trigger
  set, grouped by minute.

**Cells.** A cell is a (release, target) pair: the target's lead contract at the
release, with p₀ = the last print before it, at most 7 days old. There are
1,131 Kalshi cells (148 own, 923 cross) and 8,753 calendar cells. A path stops
at the next Kalshi release or at the contract's close; 4% (Kalshi) and 10%
(calendar) stop within 24 h.

**Sanity checks:**
- **No pre-release drift:** β at −60 / −30 / −15 min is −0.03 / −0.03 / −0.01¢
  per z, all CIs through 0.
- **The placebo arms are flat.** Kalshi cross-series +24 h: −0.11 [−0.26, +0.06].
  Own next contract: +0.29 [−0.50, +1.14].
- **Coverage is the binding constraint.** Only 10% of Kalshi cross cells trade
  within 5 min, 35% within 1 h, 56% by 4 h and 75% by 24 h. FED trades within
  5 min at 53% of releases (median first print 4 min). The thin CPI components
  (apparel, food, gas, used cars, shelter) and ISM take ~4 h to their first
  print, and their p₀ is 1–2 days old.

## 1. Speed per channel: clock time (`curves.py`)

At each bar τ and for each channel, r(τ) = p(τ) − p₀ is regressed on the
theory-signed surprise, with an intercept:
- **cross targets:** jointly on one signal per source family, so a CPI + GDP
  release is split between its channels;
- **the releasing series' next contract:** on its own z.

Absorption is β(τ)/β(24 h), using pooled slopes rather than per-cell ratios.
CIs come from a bootstrap over releases, and p from sign flips per release.

**Kalshi arm:**

| channel | β 24 h, ¢/z [95% CI] | p | 5 min | 15 min | 1 h | 4 h | 8 h | half-life [CI] |
|---|---|---|---|---|---|---|---|---|
| own next contract | **+5.22** [+2.83, +7.46] | 0.001 | 0.27 | 0.42 | 0.32 | 0.60 | 0.89 | 105 min [5 min, 4.5 h] |
| cross, pooled | **+0.44** [+0.20, +0.69] | 0.003 | 0.43 | 0.52 | 0.87 | 1.15 | 1.18 | 15 min [5, 50 min] |
| cross, liquid (≥ 15 prints / 7 d) | +0.63 [+0.36, +1.01] | 0.001 | 0.57 | 0.58 | 1.07 | 1.13 | 1.26 | 5 min [5, 25 min] |
| cross, thin | +0.24 [−0.10, +0.63] | 0.105 | 0.07 | 0.35 | 0.37 | 1.20 | 0.99 | 75 min [5 min, 4 h] |
| **labour → policy** | **+3.28** [+1.44, +5.20] | 0.002 | 0.60 | 0.72 | 1.13 | 1.18 | 1.20 | **5 min** [5, 20 min] |
| inflation → policy | +0.91 [+0.25, +1.79] | 0.005 | 0.83 | 0.72 | 1.29 | 1.33 | 1.28 | 5 min [5, 40 min] |
| labour → labour | +1.23 [+0.17, +2.52] | 0.021 | 0.34 | 0.57 | 0.66 | 0.59 | 1.09 | 15 min [5 min, 7 h] |
| labour → inflation | +0.43 [−0.01, +0.85] | 0.039 | 0.00 | 0.11 | 0.33 | 0.66 | 0.99 | 4 h [10 min, 6 h] |
| **labour → growth** | **+1.04** [+0.38, +1.71] | 0.001 | 0.09 | 0.10 | 0.06 | 0.51 | 0.58 | **195 min** [10 min, 11 h] |

The five channels shown pass BH (q = 0.10) of 11 with enough co-firing cells.
The other six have CIs through 0.

- **labour → policy is not a slow channel in the lead contract.** 60% of the
  24 h slope is there by +5 min and all of it by +1 h. The propagation study's
  multi-day repricing (`propagation_findings.md`) was on far-meeting contracts.
  The lead contract here is the most-traded one, usually the next meeting.
- **labour → growth is slow:** 6% by 1 h, half by ~3 h, 58% by 8 h, still
  rising at 3–7 days. The GDP legs are thin (median first post-release print
  1 h, p₀ 13 h old).
- **The own next contract is slower than the cross targets.** Its path wobbles
  (0.44 at 30 min, 0.32 at 1 h), which is noise on 143 cells, but it rises
  steadily to 0.89 by 8 h. It also keeps rising past 24 h (1.15 at 3 d,
  1.68 at 7 d), but the CIs are wide. §2 shows most of this is waiting for
  trades.
- **Robustness.** A 2-print mean price, against bid-ask bounce, gives cross
  +0.40 with half-life 10 min and own +6.12 with half-life 2 h. Cutting paths
  at any release, calendar ones included, gives cross +0.50 with half-life
  20 min. The conclusions are unchanged.

**Calendar arm** (1,030 releases with a consensus but no Kalshi market):

| channel | β 24 h, ¢/z [95% CI] | p | 15 min | 1 h | 4 h | 8 h | half-life [CI] |
|---|---|---|---|---|---|---|---|
| cross, pooled | +0.15 [+0.06, +0.26] | 0.003 | 0.11 | 0.26 | 0.65 | 0.96 | 165 min [25 min, 7 h] |
| cross, liquid | +0.17 [+0.04, +0.31] | 0.006 | 0.26 | 0.46 | 0.99 | 1.09 | 65 min [5, 165 min] |
| cross, thin | +0.13 [−0.03, +0.28] | 0.053 | −0.11 | −0.05 | 0.12 | 0.75 | 7 h [80 min, 10 h] |
| activity → growth | **+0.89** [+0.52, +1.34] | 0.001 | 0.11 | 0.19 | 0.73 | 0.84 | 165 min [105, 270 min] |
| activity → inflation | +0.28 [+0.07, +0.47] | 0.005 | −0.12 | −0.02 | 0.31 | 0.69 | 7 h [165 min, 14 h] |

Two of 16 channels pass BH.

- **Calendar news reaches Kalshi markets several times more slowly than Kalshi
  releases do.** The effects are about a third the size.
- **Placebo caveat.** On liquid targets the calendar placebo also has a 24 h
  slope: +0.13 [+0.01, +0.25], p = 0.01, against +0.17 real. Its path is flat
  through 8 h (≤ 0.34 of its 24 h value), while the real one is at 0.99 by 4 h.
  The *shape* is therefore release-driven, but the calendar arm's 24 h *level*
  on liquid targets is not cleanly separated from background drift. The pooled
  placebo is +0.05 [−0.04, +0.13].

## 2. Clock time vs trade time

The slope at the target's k-th post-release print (within 24 h) is divided by
its 24 h slope, on the same cells. Only pooled groups and BH channels are
shown: a ratio to a 24 h slope near 0 is noise.

| channel | k = 1 | k = 2 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|---|
| own next contract | **0.67** [0.45, 0.87] | 0.80 [0.63, 0.98] | **0.85** [0.71, 1.01] | 0.80 | 0.68 (n = 42) |
| cross, pooled | 0.53 [0.17, 0.91] | 0.57 | 0.56 [0.23, 0.88] | 0.96 | 1.10 [0.76, 1.70] |
| labour → policy | 0.35 [0.06, 0.96] | 0.15 | 0.26 | 0.59 [0.32, 1.14] | 1.05 [0.65, 2.05] |
| inflation → policy | 0.57 [0.06, 1.45] | 0.83 | 0.85 | 1.56 | 1.07 |
| labour → growth | 0.33 [−0.27, 0.99] | 0.09 | 0.12 | −0.04 | 0.16 (n = 37) |
| calendar: activity → growth | 0.47 [0.17, 0.76] | 0.71 | 0.69 | 0.70 | 1.08 [0.60, 3.51] |

- **"Reprices once, completely, at its first print"** (`research_log.md` §5e)
  **is about ⅔ right for the own next contract.** The first print carries 0.67 of
  the 24 h response and the third carries 0.85. In cents, sign(z)·move is
  4.3¢ at the first print, 6.7¢ at the third and 6.3¢ at +24 h. The own
  contract is slow in clock time because its first print arrives at a median
  29 min.
- **Cross targets:** about half at the first print, all of it by the 5th–10th.
- **labour → policy is gradual in trade time, fast in clock time.** The FED
  ladder takes its first ~10 trades to reprice fully (0.35 → 1.05), and FED
  trades so often that this happens within minutes.
- **labour → growth is slow in both.** After 10 prints the GDP contract still
  has only ~0.2 of its 24 h response (n = 37–103, wide CIs). This is the only
  channel where the lag isn't just the wait for a trade.

## 3. Order: propagation or common reaction? (`order.py`)

The regression, at τ = 15 min / 1 h / 4 h, on cells untruncated to +24 h:

  r_B(τ → 24 h) ~ 1 + sig_B + r_A(0 → τ) + r_B(0 → τ) + flow_B(0 → τ)

A is the releasing series with the largest |z|, via its next contract, and its
move is oriented by HAWKISH[A]·HAWKISH[B]. SEs are clustered by release.

| cells | τ | sig_B | r_A (source move) | r_B (own move) | flow_B |
|---|---|---|---|---|---|
| Kalshi cross (495 cells, 76 releases) | 15 min | +0.15 (0.9t) | +0.025 (**0.96t**) | −0.17 (−1.1t) | +0.20 (1.6t) |
| | 1 h | +0.15 (1.0t) | +0.009 (0.46t) | −0.14 (−1.3t) | −0.03 |
| | 4 h | +0.09 (1.0t) | +0.014 (0.95t) | −0.14 (−1.7t) | −0.05 |
| Kalshi own (143, 86) | 15 min | **+3.93 (3.3t)** | – | −0.32 (−4.5t; −1.2t on 2-print) | −0.24 |
| | 1 h | **+3.97 (3.5t)** | – | −0.26 (−2.7t; −2.3t) | +0.28 |
| | 4 h | +2.55 (2.6t) | – | −0.14 (−1.9t) | +0.03 |
| calendar cross (7,917, 955) | 15 min | +0.14 (2.8t) | – | −0.14 (−2.0t; −0.3t) | +0.04 |
| | 1 h | +0.12 (2.5t) | – | −0.16 (−2.7t; −0.6t) | +0.02 |
| | 4 h | +0.08 (2.0t) | – | −0.15 (−3.9t; −0.2t) | +0.04 (2.2t) |

- **The source market's move carries nothing the headline doesn't.** r_A is
  t < 1 at every τ. Adding it lowers walk-forward R²: Δ −0.042 / −0.043 /
  −0.009, with CIs at or below 0 at every τ. News does not travel *through* the
  network, from the released series' contract to the others. Each market reads
  the release itself.
- **The own next contract underreacts in clock time.** After 15 min and 1 h,
  the remaining move still loads on z (t = 3.3, 3.5). This is §1–2's slow own
  path: most own contracts have not traded yet. The walk-forward gain from the
  path state is not significant (ΔR² +0.06 [−0.09, +0.26] at 15 min).
- **The own-move reversal (r_B < 0) is mostly bid-ask bounce.** On the 2-print
  mean price it shrinks to t ≈ −0.3 to −1.2 everywhere except the own contract
  at 1 h (−2.3t).
- **Cross targets have nothing left after 15 min.** sig_B on the remaining
  move is t ≈ 1. Calendar releases still have some left at 4 h (t = 2.0),
  consistent with their slower curves.

## 4. Can a temporal model learn it? (`models.py`, `torch_check.py`)

**Task.** At every bar τ ≤ 24 h, predict the target's move over the next
window from what is known at τ:
- **1 h:** to the first bar ≥ τ + 60 min;
- **24 h:** to +24 h.

Both arms are trained together, walk-forward over 8 expanding folds by release
(purged), with labels in per-series sd units. Rungs:

| rung | what |
|---|---|
| 1 pooled curve | s·(β_g(τ′) − β_g(τ)): the surprise times the remaining slope of one in-fold absorption curve per group (Kalshi own / Kalshi cross / calendar cross) |
| 2 channel curve | ridge on channel × τ-bucket signal columns |
| 3 linear path state | rung 2 + the path so far × τ-bucket: own move, traded flag, prints, time since last print, signed flow, the source contract's move |
| 3b | rung 3 fitted only on bars where the target trades in the window |
| 4 GRU | one GRU over the bars per target, weights shared, learned node id, a head at every bar |
| 4b | the GRU with the loss on traded bars only |
| 5a / 5b AGCRN | an AGCRN cell stepped over the bars: nodes = the 17 targets, messages = the other markets' state so far; adaptive graph / frozen signed economic graph |

Torch rungs train on minibatches of 32 releases, with early stopping on the
last 15% of training releases, over 2 seeds. A first full-batch version took
one optimiser step per epoch and left every sequence model at predict-zero
(`torch_check.py`).

**Test R² vs 0 / AUC, on bars where the target trades inside the window**
(the untestable no-trade bars are excluded; all-bar results are in
`out/models.txt`):

| rung | 1 h all | 1 h Kalshi own | 1 h Kalshi cross | 1 h calendar | 24 h all | 24 h Kalshi own | 24 h Kalshi cross | 24 h calendar |
|---|---|---|---|---|---|---|---|---|
| 1 pooled curve | +.000 / .52 | −.001 / .56 | +.002 / .50 | +.000 / .52 | +.004 / .51 | +.074 / .63 | −.004 / .50 | +.000 / .51 |
| 2 channel curve | +.000 / .51 | −.000 / .58 | −.000 / .51 | +.000 / .51 | +.002 / .50 | +.057 / .62 | −.011 / .49 | +.000 / .50 |
| 3 linear path state | +.007 / .56 | +.033 / **.66** | −.006 / .54 | +.006 / .56 | **+.012** / .54 | +.084 / .62 | +.009 / .52 | +.008 / .54 |
| 3b (traded loss) | +.004 / .56 | **+.047** / .65 | −.085 / .54 | **+.012** / .56 | +.010 / .54 | **+.085** / .61 | −.017 / .51 | +.008 / .54 |
| 4 GRU | +.002 / .55 | +.003 / .61 | +.001 / .53 | +.002 / .55 | +.003 / .53 | +.015 / .58 | +.007 / .50 | +.002 / .53 |
| 4b GRU (traded loss) | **+.011** / **.57** | +.024 / .63 | **+.007** / **.55** | +.010 / **.57** | +.006 / .53 | +.037 / .59 | **+.020** / .52 | +.003 / .53 |
| 5a AGCRN adaptive | +.000 / .53 | +.000 / .52 | +.000 / .51 | +.000 / .53 | −.000 / .50 | −.000 / .43 | +.002 / .54 | −.000 / .50 |
| 5b AGCRN econ graph | +.000 / .50 | +.000 / .54 | −.000 / .48 | +.000 / .50 | −.000 / .51 | +.000 / .44 | +.001 / .54 | −.000 / .51 |

Releases scored (1 h / 24 h): 700 / 648 overall, 50 / 47 Kalshi own, 91 / 87
Kalshi cross.

**Δ against the best linear rung** (1–3b, picked per column by R²), 95% CI from
a bootstrap over releases:
- **GRU 4b, 1 h, all traded bars:** ΔR² **+0.0040 [+0.0013, +0.0069]**, ΔAUC
  +0.009 [+0.000, +0.020]. This is the only ΔR² CI above zero in the table,
  and it doesn't hold within either arm:
  - Kalshi: +0.004 [−0.008, +0.019];
  - calendar: −0.002 [−0.007, +0.003] against 3b.
  - It wins the pooled column because it is second-best in both arms, while
    each linear rung is best in only one.
- **Every other ΔR²** has a CI through 0 or below it. This includes 24 h Kalshi
  cross (4b +0.011 [−0.021, +0.045]) and 24 h Kalshi own (4b −0.048). On 1 h
  Kalshi cross, every path-state rung, linear or GRU, beats the pooled curve
  on AUC (+0.03 to +0.05, CIs above 0), not on R².
- **On all bars,** the GRU and AGCRN beat the pooled curve on 24 h Kalshi cross
  (+0.010 [+0.002, +0.017] and +0.006 [+0.001, +0.011]). That is only because
  the curve's R² there is negative (−0.004): both models sit near predict-zero.
- **The AGCRN never leaves predict-zero** (|R²| ≤ 0.002 everywhere). Its AUC on
  the own next contract at 24 h is 0.43, below chance (ΔAUC vs linear −0.19
  [−0.30, −0.09]). `torch_check.py` shows why:
  - its training loss falls, so it can fit its training bars;
  - its early-stop loss never beats predict-zero by more than 0.05%, under any
    batch size, learning rate (3e-4 to 1e-2) or loss mask;
  - its best epoch is usually 0.

  The GRU beats predict-zero on the early-stop set by 0.05–0.85%.

**What the path-state rungs learn** (full-sample rung 3, in label sd per 1 sd of
the feature):

| feature | ≤ 15 min | 15 min–1 h | 1–2 h | 2–6 h | 6–24 h |
|---|---|---|---|---|---|
| own move so far, 24 h label | **−0.20** | **−0.18** | −0.14 | −0.09 | −0.05 |
| own move so far, 1 h label | −0.15 | −0.11 | −0.13 | −0.05 | −0.02 |
| signed flow, 24 h label | +0.08 | +0.03 | +0.03 | +0.01 | +0.01 |
| source contract's move, 24 h label | +0.04 | +0.03 | +0.03 | +0.02 | +0.01 |

The signal is a **reversal of the move so far**, strongest in the first hour
and fading by 6 h. The source contract's move contributes little, as in §3.

**Bid-ask bounce check** (`models.py --price mean2`). The plan requires
checks before believing a sequence model that beats the linear rungs. The
check that fits here is bounce: consecutive prints alternate between bid and
ask, so a last-trade price above the mid predicts a fall mechanically. The
same models were rerun with every move (features and labels) measured on the
mean of the last two prints. Results on traded bars:

| column | best linear | GRU 4b | Δ 4b vs best linear [95% CI] | best rung |
|---|---|---|---|---|
| 1 h, all | 3: +0.0018 | +0.0011 | −0.0007 [−0.0031, +0.0018] | 3 |
| 1 h, Kalshi own | 3b: +0.0178 | +0.0086 | −0.0092 [−0.065, +0.055] | 3b |
| 1 h, calendar | 3: +0.0009 | +0.0007 | −0.0001 [−0.0021, +0.0016] | 3 |
| 24 h, Kalshi own | **1 pooled curve: +0.131** (AUC 0.68) | +0.036 | −0.096 [−0.178, +0.024] | **1** |
| 24 h, Kalshi cross | 1: −0.009 | −0.003 | +0.006 [−0.008, +0.019] | 5a: +0.001 |
| 24 h, calendar | 3: +0.0013 | −0.0011 | −0.0024 [−0.0067, +0.0033] | 5a: +0.0014 |

- **The GRU's 1 h gain was bounce.** On the 2-print price, 1 h R² falls from
  ≈ 0.01 to ≤ 0.002 on the pooled and calendar bars; the own contract keeps
  +0.018. No sequence model beats the best linear rung in any column.
- **The reversal coefficient halves** (−0.20 → −0.08 sd per sd at ≤ 15 min,
  24 h label). What remains is small, and signed flow takes over as a
  continuation signal (+0.14 at ≤ 15 min, 1 h label).
- **The one within-release signal that survives is the own next contract's
  unfinished response to its own surprise** (§1–3). The pooled absorption
  curve, rung 1 (one curve, the theory sign), predicts it best: R² +0.13 and
  AUC 0.68 on 24 h traded bars. This rests on 40 releases. The GRU reaches
  +0.03, and the AGCRN reaches −0.001 with AUC 0.49.

**Answer to Q4: no.** Once bounce is removed, no sequence model, and no
graph-message model, predicts the within-release path better than a linear
rung. The structure that exists is carried by simple pieces:
- the absorption curve for the own contract, whose theory sign is fixed;
- a small linear flow effect.

The AGCRN on the intraday grid ends where the across-release one did: at
predict-zero. This time the within-release lag is real (§1–2); it is just too
small and too simple for a graph recurrence to add anything over one curve.

## 5. Standard ML and a Bayesian graph on the same task (`models_extra.py`)

§4's only nonlinear baseline was the GRU, and it had no Bayesian graph model.
This section adds both, on the same bars, labels, folds and scoring:

| rung | what |
|---|---|
| 6a / 6b LightGBM | gradient-boosted trees on the 22 per-bar features, plus node id (categorical) and target type; rounds early-stopped on the last 15% of training releases. 6b: loss on traded bars only |
| 6c random forest | sklearn, 200 trees, ≥ 200 bars per leaf, 100k bars per tree |
| 6d / 6e MLP per bar | 64 → 32 ReLU on the GRU's inputs, with no recurrence and no graph. 6e: traded-bar loss |
| 7a Bayes graph, hard sign | `graph_ablation.md` §8's hard-sign model on the intraday grid. Every edge (source family or own z → target node) × τ-bucket has a slope ≥ 0 on the theory-signed surprise, partially pooled within channel × bucket. Rung 2 is its one-slope-per-channel limit |
| 7b Bayes STG | §8's soft-sign prior, plus rung 3's path-state features × bucket, with one slope per node, pooled across nodes. This is §9's Bayesian STG: its node-adaptive block is AGCRN's node-specific part in linear form |

The Bayes rungs use Gibbs sampling on sufficient statistics. Each
(release, target) cell has weight 1 spread over its bars, so ~50 overlapping
bars don't swamp the pooling. There are 2 chains × (400 + 800) draws.
- 7a: max R̂ 1.01–1.12.
- 7b: max coefficient R̂ 1.24–1.34, from ~6% of weakly identified edge slopes.
  Its predictions have converged: on the last fold, R̂ over test predictions
  has median 1.001 and 99th percentile 1.04, and the two chains' posterior-mean
  predictions correlate at 0.985.

**Test R² / AUC on traded bars, 2-print price (bounce removed):**

| rung | 1 h all | 1 h Kalshi own | 1 h Kalshi cross | 24 h all | 24 h Kalshi own | 24 h Kalshi cross |
|---|---|---|---|---|---|---|
| best linear (§4) | 3: +.002 / .52 | 3b: +.018 / .63 | 2: +.003 / .50 | **1: +.008** / .52 | **1: +.131 / .68** | 1: −.009 / .50 |
| 4b GRU | +.001 / .52 | +.009 / .56 | −.001 / .48 | +.001 / .53 | +.036 / .61 | −.003 / .52 |
| 5b AGCRN econ graph | +.000 / .52 | +.000 / .55 | +.000 / .50 | +.001 / .52 | +.000 / .45 | +.001 / .52 |
| 6a LightGBM | +.002 / .51 | +.021 / **.67** | +.002 / .50 | +.001 / .51 | +.030 / .59 | −.006 / .49 |
| 6b LightGBM, traded loss | +.002 / .52 | +.020 / .59 | +.002 / .51 | +.001 / .51 | +.034 / .63 | −.013 / .47 |
| 6c random forest | +.003 / **.53** | +.017 / .64 | +.003 / .51 | +.001 / .53 | +.057 / .63 | −.022 / .47 |
| 6d MLP | +.000 / .52 | +.002 / .57 | +.001 / .52 | +.003 / .52 | +.030 / .63 | +.003 / .51 |
| 6e MLP, traded loss | +.002 / **.53** | +.014 / .62 | +.002 / .52 | +.003 / .52 | +.061 / .62 | −.005 / .50 |
| 7a Bayes graph, hard sign | +.002 / .52 | +.004 / .58 | **+.008** / .52 | +.003 / .52 | +.025 / .64 | **+.005** / .51 |
| 7b Bayes STG | **+.003** / **.53** | +.014 / .63 | +.006 / .51 | +.007 / .52 | +.077 / .65 | −.012 / .49 |

Last-trade results and every subset are in `out/models_extra[_mean2].txt`.

- **No new rung beats the best linear rung on R² in any column.** Every ΔR² CI
  includes 0 or lies below it, on both prices.
- **Two AUC edges survive on the 2-print price, both at 1 h on all traded bars,
  and both are tiny:**
  - random forest: ΔAUC +0.012 [+0.000, +0.025];
  - 7b: ΔAUC +0.009 [+0.000, +0.019], ΔR² +0.0014 [−0.0002, +0.0028].

  They are 2 of ~100 uncorrected Δ cells and don't carry to R².
- **The trees pick up bounce as the GRU did.** On last-trade prices, the random
  forest's 1 h AUC edge is +0.028 [+0.017, +0.039] and LightGBM 6b's ΔR² is
  +0.0046 [+0.000, +0.010]. On the 2-print price these shrink to the numbers
  above.
- **On the one real signal, the own next contract at 24 h, the pooled curve
  beats everything.** Its AUC is 0.68, against 0.59–0.63 for trees and MLPs
  and 0.64–0.65 for the Bayes graphs. Their R² sits between +0.03 and +0.08,
  against the curve's +0.13, and several Δbalanced-accuracy CIs are below 0.
  A flexible model spends its data rediscovering one monotone curve with a
  known sign, and it gets that curve worse.
- **The Bayes graph ranks where it should.** It is the best of the graph
  models, ahead of both AGCRNs everywhere and level with the trees. Pooling
  and the sign prior keep it from overfitting, but it still doesn't beat the
  one-parameter curve. The same holds across releases (`graph_ablation.md` §8).
  7a's cross-target cells (+0.008 at 1 h, +0.005 at 24 h) are the only
  positive Kalshi-cross R² on both horizons, but the CIs include 0.

**Answer:** adding gradient boosting, a random forest, an MLP and a Bayesian
graph changes nothing in §4. The ranking on this task is roughly:

  pooled curve / linear ≥ Bayes graph ≈ trees ≈ MLP ≥ GRU > AGCRN.

The ranking is set by how much structure each model is *given*, not by how
much it can learn.

## Expectations, stated before running, against results

| expectation (`intraday_path_plan.md`) | result |
|---|---|
| Own next contract and most cross targets: most of the response by the first print, β(τ)/β(24 h) ≥ 0.8 in trade time | **Partly.** 0.67 (own) and 0.53 (cross) at the first print; ≥ 0.8 from the 2nd–3rd (own) and 5th (cross) print |
| labour → policy: drift after the first print in clock time, partly surviving in trade time | **No clock-time drift** (half-life 5 min). Gradual in trade time (0.35 at the first print, 1.05 at the 10th), but those prints come within minutes |
| Phase 3 on macro: sequence models tie the per-channel curve | **Holds, or worse for the sequence models.** The GRU ties the linear rungs once bounce is removed. The AGCRN sits at predict-zero and is below chance on the own contract. The pooled curve is the best rung where there is signal. |
| Phase 4 (sports) | Not run (decided at the start: macro Phases 0–3 only) |

## Caveats

- **Effective sample:** 139 Kalshi and 1,030 calendar releases. The Kalshi
  channels rest on 46–279 co-firing cells, so half-life CIs are wide.
- **One contract per target:** the lead contract, i.e. the most prints in the
  7 days before the release. Far-meeting FED contracts, which drift over days
  in `propagation_findings.md`, are not followed.
- **Forward-filled prices.** A bar with no new print carries the last price.
  Curves over all cells mix speed of trading and speed of updating; §2
  separates the two.
- **The calendar arm's liquid-target 24 h level is not cleanly above its
  placebo** (§1).
- **Multiple testing:** BH within each arm's channel table. Trade-time and
  order tables are shown only for pre-specified or BH-passing groups.
