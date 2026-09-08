# Do the edges make economic sense, and can they be traded?

*Companion to `direction_study.md`, which established that the Stage-1 edges
carry out-of-fold directional information. Two questions that raises and does
not answer. Reproduce with `venv/bin/python scripts/run_edge_economics.py`;
generated tables in `artifacts/edge_economics.md`. In-sample only.*

Note on sample sizes: ρ̂ here is computed on the direction panel, which excludes
the ZIRP burn-in (pre-2022), so counts run slightly below
`artifacts/adjacency_report.md` (e.g. CPI→FED n = 41 here vs 46 there). Signs
and magnitudes are unaffected.

---

## 1. The recovered graph is a policy hub, not a web

Five of the eight BH-surviving edges point at `FED` or `FEDDECISION`. The rest
are inside the CPI family. So the "structure" the thesis recovers is not a
richly connected network of cross-market influence — it is one dominant
channel, **macro data → the policy path**, plus a same-release clique.

That is a more modest claim than "prediction markets form a spatio-temporal
graph", and it is the honest one. It is also the *right* channel to have found:
it is the best-documented transmission mechanism in macro-finance (Kuttner 2001;
Gürkaynak, Sack & Swanson 2005, both already cited in `research_summary.md`
§6.2/§7.3), which is what makes the sign predictions in §2 available *a priori*
rather than fitted.

## 2. Sign restrictions: does each channel fire in the direction theory demands?

`research_summary.md` §8.3 item 2 argues that sign restrictions are
falsification a validation loss cannot provide — a model can score well with
economically nonsensical coefficients and never reveal it. So the signs below
were fixed from theory before consulting the estimates. `FED` contracts pay on
*higher* rates, so "+" means the surprise pushes the expected policy path up.

| channel | predicted | ρ̂ | n | p | verdict |
|---|---|---|---|---|---|
| CPICORE → CPI | + | **+0.744** | 24 | <0.0001 | matches (mechanical) |
| CPI → FED | + | **+0.505** | 41 | 0.0003 | matches |
| CPICORE → FED | + | **+0.482** | 34 | 0.0019 | matches |
| PAYROLLS → FED | + | **+0.532** | 25 | 0.0026 | matches |
| PAYROLLS → FEDDECISION/hike | + | +0.867 | 9 | <0.0001 | matches (n=9) |
| CPIYOY → FEDDECISION/hike | + | +0.600 | 10 | 0.034 | matches (n=10) |
| CPIYOY → PCECORE | + | **+0.720** | 12 | 0.0010 | matches |
| **U3 → FED** | **−** | **−0.138** | 33 | 0.44 | **sign flips as predicted** (ns) |
| **U3 → FEDDECISION/hike** | **−** | **−0.400** | 9 | 0.25 | **sign flips as predicted** (ns) |
| WTI → CPIGAS | + | +0.039 | 80 | 0.73 | **fails** |
| WTI → CPI | + | +0.010 | 296 | 0.86 | **fails** |
| WTI → JOBLESSCLAIMS | *(none)* | −0.755 | 11 | 0.0006 | no prior to test |

Three readings.

**(a) The hawkish/dovish sign structure is respected.** Every channel where a
hotter reading implies tighter policy comes back positive; both channels where
theory demands the *opposite* sign — unemployment, the registry's designated
falsification cell — come back negative. Neither U3 cell is significant, so
this is a consistency check rather than a result, but it is the check that
would have exposed the estimator fitting generic co-movement. It doesn't.

**(b) The one mechanically certain link is absent, and that is informative.**
Oil passes into the gasoline CPI subcomponent by construction — if any edge in
this grid should exist, WTI → CPIGAS is it. Estimated: ρ̂ = +0.04 on n = 80,
p = 0.73. Nothing.

The explanation is not a power failure, it is the definition of the trigger.
**WTI has no information event.** Its settle is public and continuously
observable, so by the time a WTI contract resolves there is no news in it —
the "surprise" is a measurement artefact of when the ladder was snapped, not a
release. CPI, payrolls and the FOMC decision are scheduled disclosures of
genuinely unknown information; WTI is not.

This makes the direction study's finding that WTI is *mass without structure*
(56% of panel rows, one BH edge, and dropping it as a trigger takes the sign
rule from p = 0.041 to p = 0.007) not a data quirk but an economic prediction
that came true. It also sharpens the inclusion criterion for the node universe:
**a trigger needs a scheduled information release, not merely a resolution
timestamp.**

**(c) WTI → JOBLESSCLAIMS is very likely a false positive.** It survives BH
(ρ̂ = −0.76, n = 11) but has no directional prior — a story is tellable for
either sign — and it sits in the one trigger family whose mechanically certain
edge is flat. An estimator that cannot find oil in gasoline prices should not
be believed when it finds oil in jobless claims. Recommend reporting it as a
flagged survivor, not a channel.

## 3. What this means for the identification problem

Ranked by how much the economics licenses a causal reading:

| channel | status |
|---|---|
| CPICORE → CPI, CPI → CPICORE, CPIYOY → CPI | **mechanical** — one print resolves both. Arithmetic, not propagation. |
| CPI → FED, CPICORE → FED, PAYROLLS → FED, and the FEDDECISION versions | **theory-backed transmission** — a documented reaction function, with the sign fixed in advance and confirmed. The strongest available candidates for a genuine cross-market claim. |
| CPIYOY → PCECORE | **theory-backed lead** — CPI precedes PCE and shares source collections. The most interesting non-policy edge, but n = 12. |
| WTI → JOBLESSCLAIMS | **unmotivated** — flag, do not interpret. |

This is the sharpest form of `research_summary.md` §5 available without new
data: the mechanical edges are separable *by construction*, and the remainder
is one economically coherent channel rather than a scattering of pairs. The
direction study's same-release robustness cut (63.0% → 60.7%, p = 0.041 →
0.13) is exactly the price of removing row 1 of this table.

---

## 4. Can they be traded? No — and the reason is precise

Every out-of-fold `sign_rule` signal on `bh`-covered rows, priced as a round
trip. 81 signals, ~34 per year.

| | cents per trade |
|---|---|
| gross, measured from `p0` (what the estimator sees) | **+0.86** |
| gross, from the first post-resolution print (what a trader can get) | **+0.01** |
| — lost to entering after the first print | −0.85 |
| effective spread (taker, one round trip) | −1.38 *(see note)* |
| Kalshi fees, both legs | −2.90 |
| **net, taker (central case)** | **−4.25** |
| net, maker on both legs (upper bound, fill risk ignored) | **−1.50** |

t = −4.94 on n = 81. Median net −3.00c; the sample loses 340c in total.

> **Note — the spread charged here is optimistic.** 1.38c is each contract's
> *unconditional* effective spread, averaged over its whole life. Measured
> conditionally, the book widens at exactly the moment a signal fires: median
> 2.0c within 0.5 h of a release (mean 2.84c) against 1.0c quiet (mean 1.98c).
> Substituting the release-conditional median puts net at ≈ **−4.9c**, and the
> conditional mean at ≈ −5.7c. The verdict does not change; the number in the
> table is the one most favourable to the strategy.
> (`research_log.md` §11.2)

### Why it vanishes: it was never a drift

Split the signed move at the first executable price:

| leg | cents | hit rate | median abs move |
|---|---|---|---|
| jump: `p0` → first post-resolution print (**untradable**) | **+0.85** | 0.543 | 2.0c |
| drift: first → third print (**tradable**) | **+0.01** | 0.395 | 1.0c |

**The tradable leg is below chance.** The estimator is not detecting a
dislocation that decays over the following minutes — it is detecting a single
repricing that is *complete at the first print*. There is no decay curve to
arrive early on, because there is no decay.

Two facts explain why the two legs differ so much.

**(a) `p0` is not a live price.** It is the last trade before resolution, and in
these books it is old: median **8.8 hours**, 75th percentile 19.6 h, 58% of
signals more than 6 hours stale. So `p1 − p0` spans a median of ~9 hours of
accumulated repricing, of which only the final ~30 minutes (17.5 min to the
first print, then 13 min to the third) is executable. The estimator's reference
frame and a trader's reference frame are simply not the same window.

**(b) The first print has already absorbed the information.** Once a fresh price
exists, the residual is a coin flip (0.395). That is what an efficient — if
infrequent — market looks like: it does not reprice continuously, but when it
finally reprices, it does so in one step.

### Is this just the staleness artifact §4.1 warned about?

No — and this is worth stating loudly, because `research_summary.md` §4.1 named
stale prices as the threat that could invalidate the whole thesis and said to
check it first. If stale reference prices manufactured the edge, accuracy would
**rise** with staleness. It falls, monotonically:

| p0 age | n | accuracy | signed jump | signed drift |
|---|---|---|---|---|
| < 1h | 29 | **0.690** | +1.24c | +0.76c |
| 1–24h | 39 | **0.615** | +1.92c | +0.36c |
| > 24h | 13 | **0.538** | −3.23c | −2.69c |

corr(staleness, |jump|) = +0.09, near zero. **Stale prices dilute the finding
rather than create it**, and the freshest third of the sample carries the
strongest result (69%) — including the only positive drift leg in the table
(+0.76c on n = 29, far too small to lean on, but pointing the right way).

One further fact ties this back to the whole in-sample record: the jump's
*size* is unrelated to the size of the news — corr(|surprise|, |jump|) = **+0.02**
(and −0.04 for the drift). The first print moves in the right direction slightly
more often than chance, by an amount that has nothing to do with how large the
surprise was. That is `research_log.md` §1–2's "sign survives, magnitude does
not", now visible in the execution frame.

### The signal is gone before any cost is charged

This is the finding, and it is not about costs at all. Measured from `p0` — the
last trade *before* the trigger resolved — the sign rule earns +0.86c. Measured
from the first print *after* resolution, it earns **+0.01c**. The hit rate falls
from 63.0% to **39.5%** on the same 81 signals.

The entire edge lives in the jump between the last pre-resolution trade and the
first post-resolution trade. By the time a price exists that anyone could
transact at, there is nothing left. `update_2026_08.md` §5 measured ~48% of the
signed move landing in the first print; on the executable margin it is
effectively 100%.

The median lag from resolution to that first print is **17.5 minutes**, and the
median holding time to the exit print is **13 minutes**. So this is not a case
of a slow trader missing a fast move — it is that no tradable price exists
between the two, because these markets simply do not print in between.

### Costs then bury what is left

Even granting a gross edge, the cost stack is decisive, and **fees dominate the
spread**: 2.90c of fees against 1.38c of spread. Kalshi's per-contract fee runs
on p(1−p), which is worst exactly where these markets sit — near 50c — and it is
charged on both legs. A signal would need to clear ~4.3c per round trip to break
even. The largest per-pair gross edge in the table is 4.08c, on the *mechanical*
CPICORE→CPI pair, and it still nets −1.69c.

### The maker escape hatch does not apply at this horizon

`research_summary.md` §6.4 argues for resting orders rather than crossing, on
the grounds that a multi-day drift needs no immediacy — which would flip the
spread from a ~2c cost to a ~1c credit. **That argument does not transfer to the
dormant horizon.** The window closes in ~13 minutes; a resting order that goes
unfilled misses the move entirely, and fill probability is precisely lowest when
the price is running away from you. Even granting the most favourable case —
both legs filled passively, fill risk ignored — the maker bound is still −1.50c.

`JOBLESSCLAIMS` could not be costed at all: not one pair of adjacent
opposite-direction trades within 60s exists in the whole in-sample block. That
is not a missing measurement, it is the answer.

### Reading this as a result, not a failure

`research_summary.md` §6.4 anticipated exactly this and pre-committed to the
right framing: **economic significance, not profitability**. The finding is a
clean, quantified statement of the limits to arbitrage that let the pattern
persist —

> The signal is real and statistically detectable in the estimator's reference
> frame, and simultaneously unavailable to any trader, because the price at
> which it would be captured never exists. The friction that creates the
> anomaly — books so thin that 17 minutes pass with no print — is the same
> friction that makes it unexploitable.

That closes the §6.1 loop more convincingly than a positive backtest would, and
it is a stronger claim than "the margin was too thin": the margin is not thin,
it is *absent* at the executable price, and that is measurable rather than
asserted.

### What it does *not* say

- It does not weaken `direction_study.md`. Direction prediction from `p0` is a
  statement about information structure; it stands on its own terms.
- It does not rule out a tradable version at a **different horizon**. The
  liquid-window horizon (days) is where §6.4's maker argument does apply — but
  the sign agreement there is 56.5%, near coin-flip.
- It is in-sample only, and it prices a *specific* execution: cross to enter at
  the first print, cross to exit at the third. A limit-order study with modelled
  fill probability is the honest extension, and `analysis/exploratory_2026_08/
  fill_prob.py` already started one.

## 5. Would holding *before* resolution work?

If the whole edge is a jump at the first print, the only way to capture it is to
be positioned already. Three paths; two are closed.

**A — hold on the surprise signal. Impossible by construction.** The surprise is
`resolved_value − implied_mean`; it does not exist until resolution.

**B — hold on an ex-ante bias in the implied mean.** If the market's central
forecast were systematically off, `E[surprise] ≠ 0` would be known in advance and
the edge would supply the target's expected direction. Tested across every series
with ≥ 8 in-sample events:

| series | n | mean surprise | t | mean z |
|---|---|---|---|---|
| WTI | 369 | −0.074 | −1.64 | −0.03 |
| CPI | 48 | −0.005 | −0.34 | −0.09 |
| U3 | 46 | −0.017 | −0.89 | −0.13 |
| CPIYOY | 37 | −0.027 | −1.66 | −0.22 |
| CPICORE | 41 | −0.004 | −0.25 | −0.09 |
| PAYROLLS | 32 | +16,716 | +1.20 | +0.29 |

No series is biased — max |t| = 1.66 across 17 series, uncorrected, which is what
noise alone produces. **The ladder-recovered implied mean is a well-calibrated
forecast.** That is a result worth reporting on its own (it validates
`implied.py`'s PDF recovery against an economic criterion rather than an internal
one) and it closes this path.

**C — hold on a coherence violation. The live option.** Before resolution, the
trigger's implied distribution implies something about the target *through the
estimated edge*. Where the target's own price does not reflect it, take the
target position and hold through resolution. This is the one worth building:

1. it is the only version that is positioned before the print carrying the move;
2. it **restores §6.4's maker argument** — hours or days to establish the
   position rather than 17 minutes, so resting orders are viable and the spread
   turns from a ~1.4c cost toward a credit;
3. it is already scoped as a contribution (the coherence-violation counter,
   §5 / Phase 3 item 10), and needs the Phase C quote/ladder panel — which moves
   that data-prep item onto the critical path.

Two caveats. Holding through resolution means bearing the **full event
variance** (median |jump| = 2c, mean |jump| ≈ 4c), so the edge must be large
relative to that, not merely positive. And it changes the claim: that is trading
a *static cross-market mispricing*, not observing propagation — arguably more
interesting, but a different sentence in the abstract.

## 6. Verdict

**Economically sensible:** yes, and testably so. The graph is one well-documented
channel (macro data → policy path) plus a mechanical CPI clique; every
theory-signed channel fires in the predicted direction, including the two
dovish-flip falsification cells. The one edge with no economic prior sits in the
trigger family whose mechanically certain edge is flat, and should be treated as
a false positive.

**Tradable:** no, and for a sharper reason than cost. It is not a decaying
dislocation but a single repricing, complete at the first executable print; the
tradable leg is below chance. It is *not* a staleness artifact — accuracy rises
with freshness, answering §4.1. Report it as a limits-to-arbitrage result, which
is what `research_summary.md` §6.4 pre-committed to.

**The remaining tradable hypothesis** is pre-resolution coherence (§5C), and it
needs Phase C panels.
