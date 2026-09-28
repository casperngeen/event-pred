# Propagation speed: which contract, and how long until it is priced

*Written 2026-09-26. Scripts and run order in `analysis/propagation_2026_09/`
(`build_panel.py`, `profile.py`, `controls.py`, `economics.py`). In-sample only
(pre-2026). Extends `leadlag_findings.md`, which only ever looked at the next
target event.*

> **Rerun on the backfilled archive (2026-09-28).** The numbers below are the
> 2026-09-26 run. With the in-sample backfill (mostly PAYROLLS 2022–23 trades,
> true settlement values for 172 more events) the panel is 32,230 rows over 225
> trigger events, and labour → Fed gets **stronger**:
> - d = 0 by horizon: **+0.5 / +4.2 / +3.8 / +5.2pp** (next meeting, then
>   2–3 / 3–4 / 4–6 months out);
> - 2–3m on 10–75c legs: **+16.9 / +15.0 / +5.7 / +2.2pp** at d = 0 / 1 / 7 / 30;
> - with all three controls: **+14.0pp**, CI [+3.3, +23.7].
>
> The costed far-meeting rule selects the same positions, so its P&L is
> unchanged (+6.24c; always-YES +6.30c). Its permutation p moves from 0.008 to
> **0.022**, because the pool of surprises being shuffled changed. Conclusions
> unchanged. Detail: `backfill_rerun_2026_09.md`.

## Summary

The question: does a trigger's surprise reach some targets slowly, so that it
is still unpriced in the target's *later* contracts, or still unpriced *weeks*
after the release, and could you hold the information until a suitable
contract is available?

| claim | evidence |
|---|---|
| **One channel has a horizon profile: labour → Fed.** | Aligned residual at entry, d = 0: next meeting **+0.3pp** (p = 0.21), meetings 2–3 / 3–4 / 4–6 months out **+3.0 / +2.6 / +3.5pp** (p = 0.01 / 0.03 / 0.03). PAYROLLS→FED alone rises monotonically: +1.2 → +3.6 → +4.5 → +6.8pp. |
| **It is slow across contracts, fast in time.** | 2–3m horizon, 10–75c legs: **+14.3pp** at d = 0 (BH), +11.8 at d = 1 (BH), +4.0 at d = 7, −0.4 at d = 30. Gone within a week. |
| **Withholding for a month earns nothing.** | Every labour→policy cell at d = 30 is ≈ 0 (+0.2 to +1.9pp on all legs, none significant). For CPI/PCE targets, whose legs rarely trade until about two weeks before release, the deferred entry shows nothing at all. |
| **It is the original surprise, not later news.** | The target's own prior surprise, its intervening surprises and the trigger's next surprise correlate with the signal at +0.001 / +0.010 / +0.022. With all three controlled, the 2–3m coefficient is **+11.7pp**, CI [+1.5, +20.7], BH. |
| **Net of costs, the far-meeting rule clears friction on mid-priced legs.** | 2–6m, d = 0, 10–75c: **+6.24c** net, CI [+0.81, +13.27], block-permutation **p = 0.008**; survives 2× spread (+5.24c, P(≤0) = 0.029). All legs: +1.04c, CI [−0.98, +3.58]. |
| **But it does not beat buying YES on every far leg.** | Always-YES on the same legs: **+6.30c**. That is a **2022** effect (+33.9c in 2022, −19.4c in 2025, the hiking and cutting cycles). The rule beats always-YES in 2023, 2024 and 2025 (+25, +2, +15c), and loses to it by 28c in 2022. |

The one-sentence version: **payrolls surprises reach Fed meetings two to six
months out more slowly than the next meeting, and the gap closes in days, not
months. What that is worth in-sample cannot be separated from which way the
Fed cycle was going.**

Panel: 31,514 rows (trigger event × target event × entry delay × leg), 222
trigger events, 115 target events, 20 pairs in 5 groups.

---

## 1. Design

### The grid

`leadlag_2026_09/build_panel.py` pairs each trigger resolution with the first
target event closing after it (≤ 60 days) and enters at the first print after
the release. This study adds two dimensions:

* **horizon**: every target event settling within 180 days of the trigger,
  banded by days from resolution to settlement (0–1m, 1–2m, 2–3m, 3–4m, 4–6m);
* **delay d**: entry at the first print after `t_res + d`, for d = 0, 1, 7 and 30
  days. The print must come within 7 days of the cut, unless the leg has never
  traded (**deferred**). In that case the entry is its first print whenever that
  is: the "wait until a suitable contract appears" case.

### Horizon, not "the k-th event"

The obvious index is k, the k-th event of C after A. It is wrong in this
archive, because the event series have holes: PCECORE jumps 272 and 420 days
between consecutive events, CPI jumps 216 and 337. With k, a contract nine months out
would be filed as "the next event". Indexing on days to settlement makes a hole
cost rows rather than mislabel them. `k` is kept in the panel for reference.

### Which pairs

Fixed in advance, not searched: leadlag §4b's BH channels (labour→labour,
PCE↔CPI, labour→policy), `edge_economics.md`'s BH survivors (CPI/CPICORE→FED,
CPIYOY→PCECORE), and a few with a strong prior (PCECORE, CPIYOY, CPICOREYOY
and GDP → FED). Each carries a `source` tag in `PAIRS`. Leadlag §4b already
showed that per-pair search is noise at this sample size, and the grid
multiplies the cell count by roughly 20.

The sign restriction and threshold-only scope are leadlag's, copied, so the
two studies cannot disagree about the economics.

### How often the "withhold" case actually arises

| target | listed ahead (median) | first print relative to release |
|---|---|---|
| FED | 268 d | trades continuously; median entry 0.3 d after a labour release even at 2–3m |
| CPI family | 62 d | legs rarely print before ~2 weeks out: 60% of 0–1m rows and ~100% beyond are deferred |
| PCECORE, ISMPMI | ~27 d | as CPI |
| JOBLESSCLAIMS | 6 d | — |

So for FED the question is "is the far contract already priced?", and for CPI
and PCE it is "is it priced when trading starts?". Only the second is the
withholding strategy.

---

## 2. The profile, no controls

`profile.py`. Statistic: leadlag §4b's **aligned residual**,
`mean(sign(signal)·(win − p_entry))` in pp. Anything absorbed before entry is
already in `p_entry`, so a positive cell means A's information was still
unpriced at that delay. Inference is unchanged from leadlag: clustered
bootstrap on `target_event`, `z_surprise` block-permuted among trigger events
within each trigger series, BH at q = 0.10 within each table.

### Pooled: nothing past the next contract

All 20 pairs pooled, only **0–1m at d = 0** survives BH (+1.28pp, perm p =
0.00, clustered CI [−1.64, +4.78]). Every cell from 1–2m out is between
−2.5 and +0.6pp with p ≥ 0.13. Pooling hides the one channel that has a profile.

### By group, all legs (aligned pp; + perm p < 0.05)

| group | horizon | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|---|
| labour→policy | 0–1m | +0.3 | −0.1 | −0.4 | – |
| | 1–2m | +1.1 | +0.1 | −0.0 | +0.2 |
| | **2–3m** | **+3.0+** | **+2.5+** | +0.8 | +0.2 |
| | **3–4m** | **+2.6+** | +1.4 | +0.5 | +1.9 |
| | **4–6m** | **+3.5+** | +1.9 | +0.8 | +0.7 |
| inflation→policy | 0–1m | −0.1 | −0.7 | −1.0 | – |
| | 1–2m, 2–3m | −0.0, −0.0 | −0.6, −1.2 | −0.4, −1.1 | +0.1, +1.2 |
| | 3–4m, 4–6m | +3.1, +4.3 | +1.7, +3.5 | +2.3, +2.8 | +1.8, +3.4 (none p < 0.20) |
| labour→labour | 0–1m | **+6.6+** | **+6.1+** | **+6.3+** | – |
| | 1–2m … 4–6m | −9.7 to +2.3, none significant | | | |
| PCE↔CPI | 1–2m | **+2.7+** | **+2.4+** | **+2.6+** | +1.5 |
| | 2–3m, 3–4m | −6.8, −6.1 (every delay) | | | |
| growth→policy | all | −4.4 to +0.5, none significant | | | |

Three things are visible:

1. **labour→policy is flat at the next meeting and positive further out.** This
   is the pattern the hypothesis predicts.
2. **labour→labour and PCE↔CPI survive delay**, but only because their targets
   are deferred. The leg has not traded, so d = 0, 1 and 7 are the same entry.
   They are not evidence that information persists for a week.
3. **inflation→policy stays flat.** CPI→FED was flat at the next meeting in
   leadlag §4b, and it is still flat 4–6 months out (+4.3pp, p = 0.20, CI
   [−0.04, +9.30]). The more-watched channel shows no slow component either.

### labour→policy, by pair and by year

Aligned residual, d = 0, all legs:

| pair | 0–1m | 1–2m | 2–3m | 3–4m | 4–6m | target events |
|---|---|---|---|---|---|---|
| PAYROLLS→FED | +1.2 | +3.6 | +4.5 | +6.8 | +5.4 | 19–21 |
| U3→FED | −0.5 | +0.4 | +3.7 | +2.8 | +3.5 | 22–26 |
| JOBLESSCLAIMS→FED | +0.5 | −0.9 | −1.0 | −4.2 | −2.8 | 2–3 |

The effect is PAYROLLS and U3, which print in the same release. JOBLESSCLAIMS
reaches only three FED events and contributes nothing. By year, d = 0 over
all horizons: 2022 +1.8, 2023 +6.0, 2024 +3.0, 2025 +1.0pp. That is the same
decay leadlag found.

### Delay kills it

On 10–75c legs (not pre-specified; leadlag §2 found all the pooled signal
there), labour→policy:

| horizon | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|
| 1–2m | **+14.4** BH, CI [+2.2, +26.1] | +6.7+ | +9.1+ | −0.8 |
| 2–3m | **+14.3** BH, CI [+3.2, +24.9] | **+11.8** BH, CI [+0.6, +22.8] | +4.0 | −0.4 |
| 3–4m | +7.6+ | +4.3 | +1.3 | +4.8 |
| 4–6m | +8.8+ | +5.8+ | +1.6 | +0.3 |

PAYROLLS→FED alone, far meetings pooled: +21.6 → +16.1 → +12.4 → +5.9pp across
d = 0, 1, 7, 30. The market does get there. It takes days for the far meetings,
where the next meeting is priced by the first print.

---

## 3. With controls

`controls.py`. A payrolls surprise and a Fed meeting three months out have two
more Fed meetings and two more payrolls prints between them. Three controls,
added one at a time to `win − p_entry = a + b·sign(signal) + c·controls`:

* **own_pre**: the target's own latest surprise before entry. It is public at
  entry, so it asks whether A is getting credit for C's own momentum.
* **own_mid**: the target's own surprises between entry and settlement, i.e.
  the intervening Fed decisions. This is mediation: does A predict C_k only
  through C's next prints?
* **trig_mid**: the trigger series' next surprises in the same window. This
  asks whether the effect is persistence of payrolls surprises.

Each control is the sum of clipped z (|z| ≤ 3) in its window. Perm p holds the
controls fixed (Frisch–Waugh). `b` with no controls differs from the aligned
residual only by the intercept.

**The controls are nearly orthogonal to the signal.** Their correlations with
`sign(signal)` are +0.001 (`own_pre`), +0.010 (`own_mid`) and +0.022
(`trig_mid`). They cannot absorb much, and they don't:

| labour→policy, d = 0 | b (none) | + own_pre | + own_mid | + trig_mid | CI (all controls) |
|---|---|---|---|---|---|
| all legs, 2–3m | +2.3+ | +2.4+ | +2.2+ | +2.3+ | [−1.4, +5.1] |
| all legs, 4–6m | +2.5+ | +2.6+ | +2.6+ | +2.5+ | [−2.0, +6.5] |
| 10–75c, 1–2m | +13.3+ | +14.1+ | +14.5+ | **+16.6** BH | [+4.6, +29.7] |
| 10–75c, 2–3m | **+13.8** BH | **+13.0** BH | +11.6+ | **+11.7** BH | [+1.5, +20.7] |
| 10–75c, 3–4m | +7.3+ | +8.1+ | +7.0+ | +7.1+ | [−8.2, +20.2] |

Control coverage on FED rows: `own_mid` is non-empty for 31% of rows,
`trig_mid` for 75%.

**The one place a control bites is labour→labour at 0–1m.** On 10–75c legs,
adding `trig_mid` takes it from +12.7pp (+12.2, +11.4 with the first two
controls) to **+7.9pp**. About a third of "claims predicts payrolls" is the
following weeks' claims prints, which is the obvious mechanism for a weekly
series leading a monthly one. It is still positive, on 15 target events.

**One BH cell to disregard.** inflation→policy 2–3m d = 30 on 10–75c legs
comes out at +20.7pp, but sits between −17.4pp and −14.7pp neighbours on 20
target events. It is noise.

---

## 4. Costs

`economics.py`. The rule has zero parameters. At each labour release, net the
aligned signals of every trigger that printed at that instant, and take
`sign(net)` on each FED leg: YES if positive, NO if negative. Hold to
settlement.

Netting matters: PAYROLLS and U3 are one BLS release, and counting them as two
trades would double every position. Netting takes 6,795 trigger × leg rows to
5,065 positions, 1,730 of them joint PAYROLLS+U3.

**The far-meeting rule (2–6m, d = 0) was chosen after reading §2.** Every
number below is an in-sample upper bound on a rule selected by looking.

### Spreads on far FED legs

Leadlag's cost model is the Kalshi taker fee (7% × p(1−p), 100 contracts) plus
half a flat 2c spread (1c in the wings), from `research_log` §11.2 on near-dated
legs. Far-dated FED legs trade less, so the spread was measured with
`stg.direction.tradability.effective_spread` by days to settlement and
moneyness:

| FED legs | 60 s pairing, 24 h after a labour release | 60 s, all trades | 1 h, all trades |
|---|---|---|---|
| 0–1m centre | 1.0c (64 pairs) | 1.0c (1,377) | 1.0c (4,995) |
| 1–2m centre | 1.0c (243) | 1.0c (2,017) | 1.0c (5,812) |
| 2–3m centre | — | — | 1.0c (666, 14% negative) |
| 3–4m, 4–6m wing | — | — | 0.0c (55–57 pairs) |

**No far cell has enough opposite-side prints within 60 seconds to estimate a
spread at all.** Only the 1 h window reaches them. There the drift
contamination shows up as 14% negative pairs, and the far wings come out at an
implausible 0.0c. So the charge is the larger of the measured median and the
flat schedule, which in practice means the flat schedule: 1.00c half-spread on
10–75c legs.
The 2× and 3× stresses below are the real guard.

### Net P&L by horizon and delay (cents per contract)

All legs:

| horizon | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|
| 0–1m | −0.39 | −0.64 | −1.16 | – |
| 1–2m | −0.53 | −1.21 | −1.21 | −0.92 |
| 2–3m | **+2.37** [−1.28, +6.35] | +1.97 | −0.14 | −0.93 |
| 3–4m | +1.18 | −0.11 | −1.66 | +0.12 |
| 4–6m | −0.06 | −1.80 | −1.77 | −2.66 |

10–75c legs:

| horizon | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|
| 1–2m | +7.66 [−5.17, +20.71] | +1.95 | +6.17 | −4.01 |
| 2–3m | **+14.80** [+0.99, +28.26] | **+13.82** [−0.56, +27.67] | +2.63 | −4.01 |
| 3–4m | +5.71 | +3.36 | −1.99 | +1.94 |
| 4–6m | +2.76 | −0.93 | −3.39 | −5.92 |

The same shape as §2 after costs: the next meeting loses the friction, the
2–3m meeting clears it, and a week's delay removes it.

### The headline rule: 2–6m, d = 0

| | all legs | 10–75c legs |
|---|---|---|
| positions / target events | 857 / 30 | 270 / 30 |
| gross | +2.44c | +8.62c |
| fee / half-spread | 0.68c / 0.72c | 1.38c / 1.00c |
| **net** | **+1.04c**, CI [−0.98, +3.58], P(≤0) 0.17 | **+6.24c**, CI [+0.81, +13.27], P(≤0) 0.011 |
| spread × 2 | +0.32c, P(≤0) 0.39 | +5.24c, P(≤0) 0.029 |
| spread × 3 | −0.40c, P(≤0) 0.63 | +4.24c, P(≤0) 0.067 |
| block-permutation p (net) | 0.029 | 0.008 |
| return on capital (capital-weighted) | +2.1% | +13.1% |
| annualised (capital-days) | +6.7% | +39.5% |
| median hold | 111 d | 118 d |

### The control that matters: always YES

The permutation null averages **−3.3c** (all legs) and **−7.3c** (10–75c),
well below zero cost. Random signals lose more than friction, which means one
side paid systematically in-sample. Taking every far leg on one side,
regardless of the signal:

| 2–6m, d = 0 | rule | always YES | always NO |
|---|---|---|---|
| all legs | +1.04c | **+3.76c** [−2.12, +11.02] | −6.57c [−13.85, −0.68] |
| 10–75c legs | +6.24c | **+6.30c** [−10.73, +25.32] | −11.05c [−30.05, +5.95] |

**The rule does not beat always-YES on average.** Far-dated FED YES legs were
underpriced in-sample, and a rule that is YES about 57% of the time inherits
that drift. By year, 10–75c:

| year | target events | rule | always YES | rule − always YES |
|---|---|---|---|---|
| 2022 | 9 | +5.71 | **+33.90** | −28.19 |
| 2023 | 9 | **+22.96** [+17.65, +29.33] | −2.18 | +25.14 |
| 2024 | 10 | +12.51 | +10.36 | +2.15 |
| 2025 | 7 | −4.01 | **−19.35** | +15.35 |

Always-YES is the policy cycle: a market that kept underestimating how far and
how long rates would rise in 2022, and how far they would fall in 2025. The
rule's excess over that baseline is positive in three of four years. It loses
only in 2022, the year when simply being long rates was worth 34c a contract.

Within each side, the signal's trades beat the side's baseline:

| 10–75c | rule's trades | always that side | signal adds |
|---|---|---|---|
| YES (150) | +13.20c | +6.30c | +6.9c |
| NO (120) | −2.46c | −11.05c | +8.6c |

(All legs: +1.5c on the YES side, +2.0c on the NO side.)

So there are two things in the P&L, and they cannot be separated in-sample:
the signal's timing, which is real (permutation p = 0.008, positive within
each side), and the regime drift, which is larger, one-directional, and would
be a bet on the Fed cycle rather than on propagation.

---

## 5. Reading

**On the hypothesis.** There is a speed difference, but it is across
contracts, not across calendar time. A payrolls surprise is priced into the
next Fed meeting by the first print. It reaches the meetings two to six months
out more slowly, and is absorbed there within about a week. The version asked
about, holding the information for one or two months until a contract for C
is available, shows nothing. That is true for FED (d = 30 ≈ 0 everywhere) and
for the CPI/PCE targets, where waiting is forced by thin trading.

**Why this is plausible.** It is leadlag §4b's attention reading, moved one
step further out. CPI→FED is the most-watched channel and is flat at every
horizon. The next Fed meeting is where attention concentrates after a jobs
report. The meetings further out get repriced by the same news, only later,
and they are thinner: no far cell even has a measurable 60-second spread.
`liquidity_findings.md` §1 found the same thing in the cross-section: thin
legs take 30 hours to reprice, liquid ones 1 hour.

**Why it is not a trading result yet.** The far-meeting rule was selected
after looking. Its net depends on a 10–75c band that was also chosen from
earlier in-sample results. It sits on 30 Fed meetings, and in-sample it cannot
be told apart from a bet on the direction of the Fed cycle. The cleanest
version of the test is one that the cycle cannot contaminate: a spread
position that is long the signal's side in the far meeting and short it in the
next meeting, which moves with the same cycle but was shown here to be
efficiently priced. That is not tested.

## 6. Caveats

* **Small.** 26–30 FED target events per cell, and PAYROLLS and U3 are one
  release, so they are not independent evidence. All-legs CIs include zero
  everywhere except 2–3m d = 0 before controls (§2).
* **Post hoc.** The far-meeting band, the 10–75c band and the d = 0 entry were
  all read off in-sample tables. BH is applied within tables whose cells
  overlap (a deferred entry is the same across delays), so it is a rough guard
  and does not control the family-wise error rate.
* **Decaying.** 2023 is the best year on every cut, and 2025 is negative in
  absolute terms. The rule's excess over the drift baseline in 2025 is
  +15c, but on 7 target events.
* **Spread unmeasured where it matters.** Far FED legs are too thin for the
  standard estimator, so the charge is the near-dated schedule. The 3×
  stress brings 10–75c to +4.24c, P(≤0) = 0.067.
* **Capacity untested.** Thin far legs, and the entry is the first print
  after the release, which is someone else's trade at a price you might not
  get.

## 7. What next

1. **Freeze a spec and spend the holdout on it:** far-meeting (2–6m) FED legs,
   d = 0, 10–75c, netted labour signals, taker, hold to settlement, reported
   next to always-YES. It is a specification with provenance for every
   parameter, like `strategy_spec.md`.
2. **Build the cycle-neutral version**, far minus next meeting on the same
   release (§5), in-sample first. If the excess over always-YES is
   propagation, it should survive, and it no longer depends on the cycle.
3. Measure far-leg spreads from order-book snapshots, if the archive can get
   them. The trade tape cannot.
