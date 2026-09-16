# Trading strategy specification — cross-market lead-lag, held to settlement

*Written 2026-09-15. The complete signal→execution chain as actually implemented
in `analysis/leadlag_2026_09/`, with every parameter and its provenance.*

**Status: an in-sample specification, not a demonstrated edge.** It is written
down in full so that it can be **frozen** and tested once on the untouched 2026
block. §7 states exactly what that test is. Nothing here has touched OOS data
(`stg.splits.OOS_START = 2026-01-01`; every script routes through
`assert_no_oos`).

---

## 1. Scope

| | |
|---|---|
| Venue | Kalshi |
| Contract type | **Threshold ladders only** (`"Above X"`, settles YES iff `X > K`) |
| Excluded | Bucket ladders (WTI, WTIW) and categorical (FEDDECISION) |
| Sample | 2021–2025, in-sample block only |
| Panel | 10,714 (trigger event × target leg) rows, 262 trigger events, 401 target events, 135 ordered pairs |

Bucket ladders are excluded because a bucket settles on `lo ≤ X ≤ hi`, which is
not monotone in `X`, so a directional signal cannot be applied to the ladder
coherently. WTI is additionally the series `TODO.md:204` proposes dropping from
the trigger set on principle (no scheduled information release).

---

## 2. Stage A — pairing (`build_panel.py`)

For each **trigger event** A and each candidate **target series** B:

1. **Trigger universe.** `usable_triggers(surprise_panel, 10)` — series with ≥10
   usable events — intersected with the `HAWKISH` table (§3).
2. **Target universe.** `trigger_universe(5, markets)` restricted to
   `kind == "threshold"` and present in `HAWKISH`.
3. **Exclude same-release pairs** via `registry.is_same_release(A, B)`. CPI and
   CPICORE print from one BLS release; a relation between them is simultaneity,
   not lead-lag.
4. **Exclude self-pairs** (`A == B`).
5. **Target event selection.** The **first** event of series B whose
   `close_time` is *strictly after* A's resolution instant `t_res`, subject to
   `gap_days ≤ MAX_GAP_DAYS = 60`. If the nearest such event is beyond 60 days,
   the observation is dropped.
6. **Take every leg** of that target event — not one representative. This is the
   deliberate departure from Stage 1 / `direction_study` / `channel_pooling`,
   all of which collapse the event to its most-traded leg.
7. **Leg admissibility.** The leg must have at least one print *strictly after*
   `t_res`, and that first post-resolution price `p_entry` must satisfy
   `1 ≤ p_entry ≤ 99`.

Recorded per row: `p0` (last print at or before `t_res`), `p_entry` (first print
strictly after), `t_entry`, `strike`, `win` (the market's own `result == "yes"`),
`gap_days`, `yr` (= `t_res.year`).

---

## 3. Stage B — the signal

### 3.1 Trigger-side measure

```
surprise   = resolved_value − implied_mean          (stg/panel/surprise.py)
z_surprise = surprise / implied_std
```

`implied_mean` / `implied_std` are the first two moments of the pdf recovered
from A's own strike ladder on the last pre-resolution day carrying
`MIN_FRESH_LEGS = 3` freshly traded legs. `resolved_value` prefers the true
printed `expiration_value` (§14.1).

Rows with non-finite `z_surprise` or `s_pit` are dropped.

### 3.2 Winsorisation

`z` is clipped to ±`quantile(|z|, 0.99)` (= 2.89 on this sample). Applied once,
before any split. Monotone, so it changes no sign.

### 3.3 Sign restriction — **zero fitted parameters**

```
direction = HAWKISH[A] × HAWKISH[B]
signal    = direction × z_winsorised
```

`HAWKISH[s]` answers: does a *higher* print from series `s` push the policy path
up (+1) or down (−1)?

| +1 | −1 |
|---|---|
| CPI, CPICORE, CPIYOY, CPICOREYOY, PCECORE, CPIGAS, CPIUSEDCAR, CPISHELTER, CPIFOOD, CPIAPPAREL, PAYROLLS, ADP, GDP, ISMPMI, FED | **U3, JOBLESSCLAIMS** |

The two `−1` entries are what make this falsifiable rather than a relabelling: a
hawkish CPI surprise predicts U3 prints *lower*, so U3's YES legs must become
*less* likely. Confirmed firing correctly (`signal_model.py` §5: the raw
association reverses in the `direction = −1` block, −4.48pp, while the aligned
one does not, +4.48pp).

Table copied verbatim from `analysis/relations_2026_09/channel_pooling.py` so
the two studies cannot silently disagree about the economics.

A positive `signal` predicts B's YES legs are **underpriced**.

---

## 4. Stage C — position rule

### 4.1 Price bucketing

Bucket on **`p0`** (the pre-resolution price), so both the maker and taker arms
decide on identical information:

```
BUCKETS = [(1,5), (5,10), (10,25), (25,50), (50,75), (75,90), (90,95), (95,99)]
```

Half-open `[lo, hi)`. A bucket is used only if it holds ≥60 rows overall.

### 4.2 Walk-forward terciles — **fitted, but never on the scored year**

Within each price bucket, for each year `Y` after the first:

```
train = rows with yr < Y        (require ≥60)
test  = rows with yr == Y
t_lo, t_hi = percentile(signal[train], [33.3, 66.7])

side = +1   if signal > t_hi      (buy YES)
     = −1   if signal ≤ t_lo      (sell YES / buy NO)
     =  0   otherwise             (no position)
```

### 4.3 Confirmation filter

```
move = (p_entry − p0) × sign(side)
require  move ≥ 2c
```

The market must already have moved **in the signal's direction** by at least 2
cents at the first post-resolution print. Rationale and controls in §6.

### 4.4 Price floor

```
entry_price = p2        if side = +1
            = 100 − p2  if side = −1
require  entry_price ≥ 20c
```

Removes positions where the ~1.8c friction is a large fraction of notional.

---

## 5. Stage D — execution and payoff

### 5.1 Entry

Cross at **`p2`, the SECOND print strictly after `t_res`.**

This is not a tuning choice — it removes a look-ahead. `p_entry` (the *first*
post-resolution print) is used as the **confirmation signal** in §4.3, so it
cannot also be the fill. Measured: the median gap between the first and second
print is **58.7 minutes** and the median price difference is **0.0c**, so the
confirmation is observable with time to act at an unchanged price.

### 5.2 Costs — taker, one crossing

```
fee    = ceil(0.07 × 100 × p × (1−p) × 100) / 100 × 100/100   [cents], p = entry/100
spread = 2.0c if |entry − 50| ≤ 40 else 1.0c      (research_log §11.2)
cost   = fee + spread / 2
```

Half the spread because only one crossing is required. **No exit fee** —
settlement is not a trade, and this is the structural reason the strategy holds
rather than exits.

### 5.3 Payoff

```
payoff = 100 × win        if side = +1
       = 100 × (1 − win)  if side = −1
net    = payoff − entry_price − cost
```

`win` is the leg's own settlement (`result == "yes"` from the markets table).
Ground truth, no reconstruction. Position is held to settlement; there is no
exit rule, no stop, and no intermediate rebalancing.

### 5.4 Maker execution is **not** used

Tested and rejected (`maker_fill.py`). Resting a limit at `p0` saves ~0.5–1c of
spread and gives back 1.7–3.1c to adverse selection: positions that never filled
would have been worth +14.83c gross against +0.05c for those that did. No maker
arm beat the taker even at a zero maker fee and with queue position ignored.

---

## 6. Parameter provenance — read this before quoting any number

| parameter | value | how chosen |
|---|---|---|
| `HAWKISH` signs | ±1 per series | **a priori**, from economics; copied from `channel_pooling.py`; falsification cell fires |
| same-release exclusion | on | **a priori** (simultaneity, not lead-lag) |
| `MAX_GAP_DAYS` | 60 | inherited from `targets.py` |
| `MIN_TRIGGER_EVENTS` | 10 | inherited repo convention |
| winsorisation | \|z\| ≤ p99 | author's choice, **not tuned** on outcomes |
| price buckets | 8 bands | author's choice, **not tuned**; finer in the wings by design |
| tercile cutpoints | 33.3 / 66.7 | **walk-forward** — fitted on prior years only |
| entry at 2nd print | — | **correctness fix** (removes look-ahead), not tuning |
| **confirmation `move ≥ 2c`** | 2c | **chosen in-sample**; {0,1,2,3,5} were compared. Walk-forward version picks k=1 (2024), k=2 (2025) |
| **price floor `entry ≥ 20c`** | 20c | **chosen in-sample**; three variants compared |
| spread model | 1–2c | `research_log` §11.2, measured on liquid legs |

**The last two rows are the problem.** Together with the variants compared,
roughly five outcome-dependent choices sit between the raw panel and the
headline. No p-value below is corrected for that.

### Controls that constrain the interpretation

| control | result |
|---|---|
| momentum only (trade the move, ignore the signal) | **−0.27c** |
| market moved ≥3c but signal disagrees | **−0.19c** |
| filter on \|signal\| magnitude instead of move | **−0.27c** |
| block-permuted signal (whole strategy re-run) | **gross ≈ 0** |

So it is the **conjunction** of signal direction and market confirmation — not
momentum, not large surprises, and not the signal alone.

---

## 7. Results as measured (in-sample)

Final specification (walk-forward terciles + `move ≥ 2c` + `entry ≥ 20c`,
entry at the 2nd print, taker costs):

| | |
|---|---|
| positions | 889 over **214 target events** |
| gross | +7.1c |
| friction | ~1.8c |
| **net** | **+5.32c** |
| 95% CI (clustered on `target_event`) | **[+0.49, +10.10]** |
| P(≤0) | **0.015** |
| return on capital, equal-dollar | +10.6% |
| return on capital, equal-contract | +7.7% |

**By year:** 2023 **+15.34c** (n=72) · 2024 **+4.64c** (n=314) · 2025 **+4.31c**
(n=503).

For comparison, without the two in-sample filters: net +0.40c, P(≤0) = 0.38, and
the year path decays 2023 → 2025 (+11.35 / +4.12 / +2.04). The price floor is
what flattens the decay.

---

## 7b. Ablation and benchmarks

`ablation.py`. Every variant on the same panel, same costs, same entry
convention, same event-clustered bootstrap.

| variant | n | events | net | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| **FULL SPEC** (signal + confirm + floor) | 889 | 214 | **+5.41c** | [+0.59, +10.21] | **0.01** |
| − drop FLOOR | 1096 | 234 | +3.60c | [−0.47, +7.78] | 0.04 |
| − drop CONFIRM | 2869 | 287 | +2.06c | [−0.88, +5.04] | 0.09 |
| − drop BOTH (signal only) | 4330 | 296 | +0.53c | [−1.88, +2.99] | 0.34 |
| signal removed: **momentum** side | 2931 | 352 | +0.67c | [−1.79, +3.12] | 0.30 |
| signal removed: **random** side | 1479 | 321 | +0.39c | [−2.30, +3.08] | 0.38 |
| passive: always buy YES | 5670 | 373 | **−2.38c** | [−5.57, +0.73] | 0.93 |
| passive: always sell YES | 5099 | 375 | +0.83c | [−2.90, +4.33] | 0.33 |
| confirm on **\|move\|** not direction | 1432 | 254 | +2.83c | [−1.21, +6.99] | 0.08 |
| **CEILING**: perfect foresight | 7921 | 380 | **+16.50c** | [+15.01, +18.04] | 0.00 |

Every component earns its place, and the ladder is monotone: signal alone
+0.53 → +floor +2.06 → +confirm +3.60 → both +5.41. The strategy captures
**33% of the perfect-foresight ceiling**.

Removing the signal but keeping the whole apparatus collapses it to +0.39c
(random) or +0.67c (momentum). Requiring the move to be in the *signal's
direction* rather than merely large is worth +2.6c (5.41 vs 2.83), so the
confirmation is not an activity or liquidity proxy.

### The honest headline is +3.76c, not +5.41c

Permuting the signal **through the entire specification** — z shuffled among
trigger events within trigger series, signal rebuilt, terciles refit, both
filters reapplied:

| | |
|---|---|
| observed | **+5.41c** |
| permuted null | **+1.65c ± 1.39** (200 draws) |
| draws ≥ observed | **0 of 200** → p = 0.0050 by (r+1)/(m+1) |
| position of observed | **2.7 sd** above the null mean |
| null max | +5.4116c, against an observed +5.4122c |

**The null mean is the number that matters.** The two filters applied to a
signal containing no information still earn **+1.65c** — the apparatus is not
neutral. So the signal's marginal contribution is **+3.76c**, not +5.41c, and
any claim about "the strategy" should quote the ablation, not the raw total.

Two cautions. The p-value floor is 0.005 at 200 draws, so `p = 0.0050` means
"nothing exceeded it", not "overwhelming". And the null's maximum came within
0.0006c of the observed — the result sits exactly at the edge of its own null,
which is what 2.7 sd looks like.

---

## 8. Not modelled

- **Market impact and depth.** Fills are assumed at the print price for
  arbitrary size. No book was available (`data/kalshi_orderbooks.jsonl` is
  sports-only, late 2025).
- **Opportunity cost of collateral.** Kalshi collateralises at full notional;
  capital is locked from entry to settlement (median hold tens of days) and no
  financing charge is applied.
- **Partial fills / queue.** Not applicable to the taker arm, but it means the
  rejected maker arm was evaluated optimistically and still lost.
- **Per-leg spread.** A 1–2c moneyness model is used, not a measured book.
- **Portfolio construction.** Positions are independent single contracts. One
  CPI print settles many legs at once, so a real book carries correlated
  settlement risk that is not sized for here.

---

## 8b. Look-ahead audit

`leakage_audit.py`. Every quantity checked against the instant it is used.

**Hard wall: clean.** Zero rows with `close_time` or `t_res` at or after
`OOS_START = 2026-01-01`. Latest `close_time` in the panel is 2025-12-19.

| quantity | known at | verdict |
|---|---|---|
| `resolved_value(A)` | `t_res` | OK |
| `implied_mean/std(A)` | pre-`t_res` ladder | OK |
| `HAWKISH` signs | a priori | OK |
| `p0` | ≤ `t_res` | OK |
| price bucket of `p0` | `t_res`, boundaries fixed a priori | OK |
| tercile cutpoints | years < Y, refit annually | OK |
| `p_entry` (confirmation) | `t_entry` | OK |
| `p2` (the fill) | `t_entry2`, median 58.7 min later | OK |
| `win` | outcome only, never an input | OK |
| winsorisation limit | **full sample** p99 of \|z\| | **leak** |
| bucket eligibility (≥60 rows) | **full sample** counts | **leak** |
| leg must print after `t_res` | **future** | **survivorship** |
| leg must print twice (for `p2`) | **future** | **survivorship** |
| confirm 2c / floor 20c | full sample | search, not look-ahead |
| spread model 1–2c | full sample | minor |

### The two removable leaks are immaterial

Re-run with expanding-window winsorisation (p99 of \|z\| over prior years only)
and expanding bucket eligibility:

| variant | n | events | net | 95% CI | P(≤0) |
|---|---|---|---|---|---|
| as specified | 889 | 214 | +5.41c | [+0.59, +10.21] | 0.01 |
| **both leaks removed** | **889** | **214** | **+5.41c** | **[+0.59, +10.21]** | **0.01** |

Identical, and for a reason: winsorisation clips only the top 1% of \|z\|, and
the position rule is **rank-based** (terciles), so clipping cannot move a value
across a tercile boundary. The bucket gate was never binding — every bucket had
≥60 prior-year rows anyway.

### The survivorship leak is the real one

Entry requires the target leg to have printed after `t_res`. At `t_res` a trader
does not know which legs will trade.

| | |
|---|---|
| legs **listed** on matched target events | 3,685 |
| legs that printed and entered the panel | **2,462 (66.8%)** |
| per-event coverage | p10 0.33, median 0.82, p90 1.00 |
| of panel legs, share with a 2nd print (for `p2`) | 93.4% |
| events where a YES-settling leg is present | **97.6%** |

So a third of listed legs are absent, and they are the illiquid ones — this is
`research_log.md` §12's convenience-sample problem in another form. It cannot be
removed without quote data, and `data/kalshi_orderbooks.jsonl` is sports-only
from late 2025.

**What bounds the damage:** the winning leg is visible in 97.6% of events, and
the band most exposed to illiquid legs (entry 90–100c, where you are shorting a
cheap YES or buying an expensive one) is **not** carrying the result. That band
has a hit rate of 1.000 over 284 positions — which looked alarming until checked:
it spans **115 distinct target events** with a median of 2 legs each, and
**removing it entirely improves the strategy** to +6.26c, CI [−0.88, +13.07],
P(≤0) = 0.042. It dilutes rather than drives.

---

## 9. The frozen OOS test

To be run **once**, on the 2026 block, with no further tuning:

1. Rebuild the panel on 2026 using §2, unchanged.
2. Compute signals using §3, with `HAWKISH` unchanged.
3. Fit tercile cutpoints on **all in-sample years** (2021–2025) per price
   bucket; apply to 2026. Nothing refitted within 2026.
4. Apply `move ≥ 2c` and `entry ≥ 20c` exactly as above.
5. Enter at the second post-resolution print; costs per §5.2; hold to settlement.
6. Report: n, target events, gross, net, bootstrap CI clustered on
   `target_event`, and P(≤0). **One number, reported whatever it says.**

Pre-registered expectation, stated before the test: given the decay measured in
§7 and in every other effect in this project, the point estimate should be
**lower than +5.32c**. A result in the +2 to +4c range with a CI including zero
would be consistent with the in-sample finding; a negative result falsifies it.

---

## 10. Known weaknesses, ranked

1. **Multiplicity.** ~5 outcome-dependent choices; `P(≤0) = 0.015` is
   uncorrected and would not survive a serious adjustment. This is the binding
   limitation and only the OOS test resolves it.
2. **Three year-clusters.** Year-clustered inference is degenerate here — a
   reported P(≤0) = 0.000 on three clusters is an artifact of the bootstrap and
   must not be cited. Event-clustered figures are the honest ones.
3. **Decay.** Every effect in this project shrinks across 2023–2025. The price
   floor flattens it (4.64 → 4.31) but three points is not a trend.
4. **`data/trades/` is a convenience sample** (`research_log` §12). Entry
   requires a post-resolution print, so illiquid legs are absent and the
   selection acts on the population under test.
5. **2023 rests on 72 positions / ~40 events**, and it is the strongest year.
6. **In-sample only**, and the specification was built by someone who had read
   every other result in this repo. The walk-forward controls the tercile
   cutpoints and the threshold; it does not control the analyst.
