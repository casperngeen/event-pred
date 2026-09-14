# Modelling the relations: better surprise measures, richer edge metrics, and structure beyond the pair

*Written 2026-09-14. A design document for Stage 1, not a findings document —
nothing here has been run except the two measurement tables in §1.2 and §3.1,
which were computed ad hoc this session against
`artifacts/panels/surprise_panel.parquet` (reproduction snippets in the
appendix). Companion to `research_log.md` (findings), `graph_definition.md`
(the node/edge definition this would amend) and `TODO.md` (sequencing).*

> **Why this document exists.** Stage 1 currently estimates one scalar
> association — Spearman(`surprise`, `response`) — per ordered pair, over 141
> pairs, and reports 4 BH survivors (`artifacts/adjacency_report.md`). Both the
> input and the output of that estimator are thinner than the data supports:
> the trigger side discards a recovered distribution and keeps its mean, the
> target side discards a ladder and keeps one leg's cent change, and the
> relation itself is one number per pair with no state dependence. Every
> proposal below is an attempt to extract more structure **without spending
> more degrees of freedom**, because `n` is the binding constraint on this
> project and has been since August.

**Scope reminder.** Everything below is in-sample only. The 2026 block stays
untouched, including for feasibility checks (`splits.py::assert_no_oos`).

---

## 1. Surprise

### 1.1 What it is now

`stg_infra/stg/panel/surprise.py:240,352`:

```
surprise = resolved_value − implied_mean
```

`implied_mean` is the first moment of the PDF recovered from the strike ladder
(`stg/events/implied.py:233 recover_pdf` → `:289 pdf_implied_stats`) on the last
pre-resolution day carrying `MIN_FRESH_LEGS = 3` legs that actually traded.
`resolved_value` prefers the true printed `expiration_value` since §14.1.

Stage 1 feeds the **raw** `surprise` into a rank correlation
(`structure/estimator.py`). Stage 2 uses `z_surprise = surprise / implied_std`
(`direction/dataset.py:116`). The two stages therefore do not use the same
quantity, which is itself worth fixing (§1.6).

### 1.2 One threat that is already closed — do not spend time on it

A natural worry: `snap_date` is "the last day with a real cross-section", which
could be well before the print, so `surprise` would be part news and part belief
drift between the snapshot and resolution. Measured, `close_time.date() −
snap_date`:

| series | n | median | p90 |
|---|---|---|---|
| WTI | 198 | 0 | 0 |
| CPI | 41 | 0 | 1 |
| U3 | 38 | 0 | 0 |
| PAYROLLS | 32 | 0 | 0 |
| CPICORE | 31 | 0 | 1 |
| CPISHELTER | 9 | 1 | 8 |
| CPIAPPAREL | 7 | 2 | 10 |
| CPIFOOD | 6 | 3 | 12 |

Overall median 0, p75 0, p90 1. **The implied mean is a same-day quantity for
every series that acts as a trigger.** Only the thin subcomponents are stale,
and they are `target_only` in the registry anyway. Combined with §14.1's finding
that CPI/U3/PAYROLLS close at 12:25 UTC, five minutes before the 08:30 ET BLS
release, the trigger-side timing is sound. Record this and move on.

### 1.3 The three weaknesses that are real

1. **It is a first moment of a distribution that was recovered and then
   discarded.** `implied_std`, `implied_entropy`, `implied_skew` and
   `implied_kurtosis` are computed per event and none of them reach Stage 1. The
   mean is also the moment most damaged by §14.2's open tails — two tails placed
   `spacing / 2` past the extreme strikes drag the mean toward the ladder centre.
2. **It is not unit-free**, so it cannot be pooled across series. This is
   precisely what stops the channel-pooling plan (`research_log.md` §11) from
   being anything more sophisticated than a sign.
3. **Magnitude does not cross-validate** — §2's CPI vs CPIYOY check on 37 matched
   release dates: signed `surprise` r = 0.686, `|surprise|` r = 0.242. The
   project's response was to abandon magnitude entirely (§6 item 3). §1.5 argues
   that conclusion was drawn on the wrong magnitude estimator.

### 1.4 Candidate — PIT / quantile surprise *(do this first)*

Replace the difference-from-mean with the realised value's position in the
implied distribution:

```
u      = F_implied(resolved_value)          # CDF; free from recover_pdf's (mids, probs)
s_pit  = 2u − 1                             # bounded in [−1, +1], signed
                                            # or Φ⁻¹(u) if a Gaussian scale is wanted
```

The recovered object is a **pmf over bins**, not a continuous density, so use the
mid-PIT correction or the result is discretisation-biased:

```
u = P(X < bin(resolved)) + 0.5 · P(X = bin(resolved))
```

**Why it is better.** Unit-free by construction, bounded, comparable across
pp-of-CPI and thousands-of-claims, and it reads the *whole* distribution rather
than one moment — a surprise landing in a fat left tail scores differently from
one landing the same absolute distance out in a thin one. It is the natural
input to channel pooling: averaging `s_pit` across the inflation channel means
something; averaging raw surprises does not.

**It also yields a standalone finding, independent of any edge.** Under a
calibrated market `u ~ Uniform(0, 1)`. A PIT histogram plus a
Kolmogorov–Smirnov test per series is a **calibration study of the Kalshi macro
ladder** — the distributional generalisation of the §5B unbiasedness result,
which tests only the first moment. Overdispersion (`u` piling at the extremes)
would say the market is systematically overconfident. That is citable whether or
not any edge is real, and it pairs naturally with the coherence-violation
contribution proposed in `update_2026_08.md` §1. Build it from
`build_surprise_panel(gated=False)` for the same reason §14.2 gives — the gated
panel is bounded by construction.

**Cost.** Low. The CDF is a `cumsum` over the `probs` array both contract paths
already build. Add `pit` and `s_pit` columns to `_SCHEMA`.

**What the outcome would mean.** If Stage 1 re-run on `s_pit` recovers the same
4 edges, the surprise measure is robust and the thesis gains a robustness line.
If it recovers *more*, the unit problem was costing power. If it recovers fewer,
the existing edges were partly an artefact of scale differences across triggers,
which is important to know before the OOS test is spent.

### 1.5 Candidate — surprisal as the magnitude channel *(the highest-information test here)*

```
m = −log p_implied(bin containing resolved_value)
```

This is "surprise" in the information-theoretic sense: unsigned, and far less
sensitive to the estimated mean than `|resolved − mean|`, which inherits every
bit of the mean's estimation error *and* the unit problem.

**The test.** Re-run the §2 consistency check — CPI vs CPIYOY, matched on
release date, n = 37 — with `m` and with `|s_pit|` in place of `|surprise|`.

**Why this matters more than it looks.** "The signal is in the direction, not
the size" (`research_log.md` §2, §6 item 3) is one of the load-bearing design
decisions of the whole project: it is why Stage 1 is rank-based, why Stage 2 is a
direction classifier, and why the AGCRN regression target was abandoned. That
decision rests on a single number, `|surprise|` r = 0.242. If surprisal
cross-validates at r ≈ 0.6 where `|surprise|` gave 0.24, the claim becomes a
statement about a bad estimator rather than about the world, and a large amount
of design space reopens. If it also gives ~0.24, the original conclusion is
confirmed on a much stronger measure and the thesis can say so.

**Cost.** One afternoon. Binary outcome either way. Note the coarseness caveat:
with 5–10 ladder bins, `m` takes few distinct values, so report it alongside
`|s_pit|`, which is continuous.

### 1.6 Three cheap robustness items

- **Median instead of mean.** `resolved_value − implied_median`. The median is
  placed by interpolation *inside* the observed support and is immune to §14.2's
  open-tail problem. `implied_median` already exists in the node panel
  (`panel/nodes.py`) but not in the surprise panel. If the Stage-1 table moves
  materially, that is a measurement-sensitivity finding; if not, it is a free
  robustness line.
- **Standardise in Stage 1 too.** Stage 2 already uses `surprise / implied_std`.
  Within a pair, dividing by a varying `implied_std` reorders ranks, so this is
  **not** a no-op for Spearman. One line, and it makes the two stages consume the
  same quantity.
- **Report the tie rate per pair.** §14.4 established 86.2% ties on `surprise`
  and 98.1% on `response` pooled. Per pair it is the effective sample size, and
  it is currently invisible in `adjacency_report.md`.

### 1.7 Out of scope but worth stating — consensus-relative surprise

The textbook macro definition is `actual − consensus forecast` (Bloomberg,
Econoday). The interesting version here is not replacement but **decomposition**:
`implied_mean − consensus` is the market's disagreement with professional
forecasters, and it is a genuine identification lever for §5 — it isolates
market-specific information from the common public signal that both contracts in
a same-release pair are reading. Requires external data the repo does not have,
so this is a limitations-section item or future work, not an October item.

---

## 2. Metrics beyond surprise

Stage 1 is scalar-in, scalar-out: one number from the trigger, one leg's cent
change from the target. Several richer objects already sit unused in the panels.

### 2.1 Better responses (same edge, better measurement)

| candidate | why | cost | caveat |
|---|---|---|---|
| **Signed order-flow response** | `response` is a difference of integer cents — **98.1% tied** (§14.4). Signed taker imbalance over the same window is continuous and essentially untied, and measures the same thing (direction of belief revision) at much higher resolution. `net_flow` already exists in the node panel. | low | needs the trade-level window, not the daily aggregate |
| **Distributional response** — ΔKL or Wasserstein between the target's implied PDF before and after the trigger | Removes the "which leg represents this event" problem entirely — the problem that produced the §14.5 look-ahead and cost the `CPICORE→CPI` edge. Uses the whole ladder instead of one strike. | medium | §11.2(b): within 30 min the median event has **one** strike trading, so this only works at daily granularity — a horizon where the effect is weaker. A second axis, not a replacement. |
| **Uncertainty response** — Δ`implied_std` / Δ`implied_entropy` | A variance-spillover channel, entirely separate from the mean channel, and it survives the objection that kills the mean channel: uncertainty resolution is a *large* effect that does not require getting a sign right. | medium | conflates resolution-of-own-uncertainty with transmitted uncertainty; needs the trigger's own resolution controlled for |
| **Attention response** — trade arrival intensity | A volume jump is an order of magnitude larger than a 1c price move, and it is a direct test of the mechanism: if attention does not propagate, information cannot. Gives edges where the price test is underpowered. | low | attention ≠ information; report as a mechanism check, not an edge |

`net_flow` as the response is probably the single cheapest power gain available
anywhere in this project, because it attacks the tie problem at its source rather
than correcting for it.

### 2.2 Different relational objects (not trigger→target at all)

**Daily belief co-movement graph.** Partial correlation or graphical lasso on
`d_implied_mean` across nodes, using the **6,419-row node panel instead of the
519-row event panel** — an order of magnitude more observations. It conflates
common exposure with propagation, so report it as *belief co-movement structure*,
not influence. Its value is the comparison: where the co-movement graph and the
Stage-1 influence graph agree you have common exposure; where influence appears
without co-movement you have something closer to propagation. It also gives the
AGCRN successor a structured prior, which it never had
(`agcrn_study.md`), and it is a second adjacency to put beside
`artifacts/adjacency_comparison.md`.

**Connectedness index on implied_std.** Diebold–Yilmaz on the panel of node
uncertainties gives an independently-estimated adjacency with a standard,
citable method behind it. Useful precisely because it does not depend on the
surprise measure at all.

**Per-edge adjustment speed.** §13 reports a pooled median 6.2 min repricing.
Per edge it is a relational metric with a clean interpretation — informational
efficiency by pair — and the ledger code already computes the components.

### 2.3 Different statistics on the same edge — where the *patterns* are

These reuse the existing rows and answer "under what conditions does this edge
fire", which is a richer object than one ρ per pair.

- **Magnitude-conditional ρ.** Does the edge fire only on large surprises?
  `structure/horizon.py` already performs exactly this cut along the target's
  days-to-close; add `|z|` (or `|s_pit|`) as a second axis. An attention-threshold
  story — only surprises past ~1sd get noticed — is economically standard and
  testable on the rows already in hand.
- **Sign asymmetry.** Separate ρ for hawkish and dovish surprises; Fed reaction
  asymmetry is well documented. Costs parameters, so run it pooled at channel
  level (§3.4), never per pair.
- **Regime stability.** The 2022–23 hiking cycle and the 2024–25 easing cycle are
  different worlds. `artifacts/direction_edges.parquet` **already holds the
  per-fold refits**, and `direction/structure.py`'s docstring flags the
  refit-vs-published gap as reportable. Nobody has reported it. This is nearly
  free — a plot of existing output — and "the recovered structure is / is not
  stable across the policy cycle" is a substantive claim in either direction.

---

## 3. Beyond the pair

Four different things get conflated under "higher-order structure". They have
very different prospects, and §3.1 measures the fact that decides which.

### 3.1 Multiple simultaneous triggers → one target

This is not exotic in this data — it is the **normal case**. Counting trigger
series sharing a `close_time` in the surprise panel:

| series resolving at the same instant | releases |
|---|---|
| 1 | 322 |
| 2 | 41 |
| 3 | 10 |
| 4 | 3 |
| 5 | 3 |
| 6 | 6 |
| 7 | 2 |
| 8 | 1 |

So **66 release instants carry ≥2 trigger series and 25 carry ≥3**; one CPI print
(2024-04-10) carries 8. §14.6 independently found 29.2% of pair-panel rows share
a `t0` with a sibling. The pairwise design treats these as independent
observations of one event and then patches the dependence with block permutation.
The better specification is release-level: the unit is the **release**, carrying a
*vector* of surprises, and the target's response is regressed on the vector.

Whether that is worth doing turns on whether the simultaneous surprises carry
independent information. Standardised surprises (`surprise / implied_std`),
matched on `close_time`:

| pair | n | r(z) | sign agreement |
|---|---|---|---|
| CPI ↔ CPICORE | 22 | **+0.880** | 95% |
| CPI ↔ CPICOREYOY | 12 | +0.867 | 92% |
| CPICORE ↔ CPICOREYOY | 15 | +0.893 | 93% |
| CPI ↔ CPIYOY | 15 | +0.697 | 73% |
| CPICORE ↔ CPIYOY | 14 | +0.694 | 64% |
| **PAYROLLS ↔ U3** | **24** | **−0.180** | **50%** |

Two clear and opposite readings.

**(a) The CPI family is one factor, not five.** At r = 0.88–0.89 and 93–95% sign
agreement there is no room for a "does core or headline drive the policy path"
contrast: the regressors are collinear and n = 22. The right move is the opposite
of adding structure — **collapse the CPI family to a single latent inflation
surprise factor**, e.g. the first principal component, or (more practical given
that family members are not all present on every date) the mean of available
`s_pit` across the family. This:

- raises events per node, the binding constraint;
- removes the near-duplicate dependence that block permutation currently has to
  compensate for (`research_log.md` §3: the naive null overstated significance by
  an order of magnitude precisely because of this);
- shrinks the pair grid. 6 of the 14 usable triggers are CPI-family; collapsing
  the four headline/core × MoM/YoY variants to one takes the grid from ~141
  ordered pairs to ~111, a ~21% cut in the BH multiplicity burden, which directly
  buys survivors.

*Measure before collapsing:* `CPIGAS` and `CPIUSEDCAR` are subcomponents and were
**not** included in the table above. They may carry genuinely separate
information and should be tested on the same matched-date correlation before
being folded in. This change also amends the node definition, so
`graph_definition.md` needs updating with it rather than after it.

**(b) The Employment Situation is the real double trigger.** PAYROLLS and U3
resolve at the same instant from the same BLS print with r = −0.18 and sign
agreement at exactly chance. They carry near-orthogonal information. So

```
response_FED ~ β₁·z_PAYROLLS + β₂·z_U3        theory signs: β₁ > 0, β₂ < 0
```

is properly identified on n = 24. Better still, the zero-parameter version — a
**labour-market strength index**

```
z_labour = z_PAYROLLS − z_U3        (hawkish-positive by construction)
```

which fits nothing, is theory-signed, and should **dominate either component
alone** if both carry signal. This is a sharp test, it is cheap, and it
rehabilitates U3, which has been the project's weakest falsification cell since
§5 largely because it was being tested in isolation rather than as the second
component of a two-dimensional release. If `z_labour` beats both `z_PAYROLLS`
and `z_U3` on the same rows, that is positive evidence for the release-vector
specification generally, not just for this cell.

### 3.2 Sequential double triggers — conditional transmission

ADP precedes NFP; CPI precedes FOMC. The testable question: **does a CPI surprise
transmit more strongly to the policy path when the preceding payrolls surprise
agreed in sign (confirmation) than when it conflicted?**

Implement rank-safely by comparing ρ(A→C) on the agree and disagree subsamples
rather than fitting an `s_A · s_B` interaction — the interaction term is
magnitude-weighted and §1.3(3) says magnitude is the unreliable half.
`direction/learners.py::neighbour_signal` is the nearest existing machinery, and
§14.6 already fixed its simultaneity handling, so the "what did this target
already know" primitive exists. This is the most faithful reading of "double
trigger → target" and it is viable on current `n` because it is one contrast, not
one parameter per pair.

### 3.3 Multi-hop and path structure — expect a negative, and report it as a finding

`structure/mediation.py` exists and found nothing: every partial in
`adjacency_report.md` is unchanged to within ±0.036. The standing interpretation
is "no evidence of multi-hop structure, low power". I think the stronger and more
honest reading is **structural**: with 4 surviving edges all pointing macro →
policy, the recovered graph is a **bipartite star, not a network**. There is no
interior node for message passing to pass through.

The only genuine 2-path in the current universe is `CPI → FED → FEDDECISION`,
since `FED` is both a usable trigger and a target. Test it explicitly and report
it, but do not expect depth.

Real multi-hop needs either time-layered nodes or the sports arm, where the graph
is **mechanically known** (game → season win total → division → championship,
`research_log.md` §11). That is the strongest argument in `TODO.md` for promoting
sports and it is an argument about *graph topology*, not event count — worth
putting to the supervisor in that form: the macro universe cannot support the
thesis's graph claim because it has no depth, and that is a finding about the
domain rather than a failure of method.

### 3.4 Block / hypergraph structure — the highest-value item

Channel pooling is already `TODO.md`'s top analysis item, but it is framed there
as a pooling trick. It is better framed as a **stochastic block model on the node
set**: nodes carry types (inflation, labour, growth, energy, policy), edges are
estimated at block level, and 141 free pair parameters collapse to ~4
theory-signed channel coefficients.

That is higher-order relational structure in exactly the sense this document is
about, it is the only proposal here that *adds* power rather than spending it
(§11: 456 rows vs 81, p = 0.021, zero fitted parameters), and same-release groups
become **hyperedges** rather than a nuisance to be permuted around.

Composed with §1.4's unit-free surprise and §3.1(a)'s CPI-family collapse, these
three are one coherent redesign rather than three patches:

- `s_pit` makes surprises commensurable **so that** pooling is meaningful;
- the family collapse removes the within-block collinearity **so that** block
  estimates are not dominated by five copies of one number;
- the block model spends 4 parameters where 141 were spent.

### 3.5 A note on the graph the model should learn

If several of the above land, the object is no longer one adjacency but
**several typed adjacencies** — same-release, channel, belief co-movement,
uncertainty spillover, attention. That is a multi-relational (R-GCN) setting, and
it is a much better fit for an AGCRN successor than a single learned `Ã`.
`graph_definition.md` already records that `softmax(ReLU(EEᵀ))` **cannot
represent edge signs** — a typed, signed, externally-estimated set of adjacencies
sidesteps that mismatch rather than asking the architecture to overcome it.

---

## 4. Suggested order

`n` is the binding constraint everywhere, so the ranking is by power added per
parameter spent, not by interest.

| # | item | § | buys | cost |
|---|---|---|---|---|
| 1 | PIT surprise + calibration study | 1.4 | unit-free input (unlocks 4); a standalone finding independent of any edge | low |
| 2 | Re-run §2 cross-validation with surprisal | 1.5 | decides whether magnitude is recoverable — reopens or confirms a load-bearing design decision | low |
| 3 | CPI-family collapse + PAYROLLS−U3 labour index | 3.1 | more events per node, ~21% less BH burden, a properly identified double trigger | medium |
| 4 | Channel pooling as a block model, on `s_pit` | 3.4 | ~5.6× rows at zero fitted parameters | medium |
| 5 | Order-flow response beside price response | 2.1 | attacks the 98% tie rate at source | medium |
| 6 | Per-fold edge stability report | 2.3 | a substantive claim from artifacts that already exist | ~free |
| 7 | Magnitude-conditional and sign-asymmetric ρ | 2.3 | state dependence — the "pattern" layer | low |
| 8 | Belief co-movement graph on the node panel | 2.2 | 10× the observations; a second adjacency to compare against | medium |
| 9 | Sequential conditional transmission | 3.2 | the genuine double-trigger test | medium |
| 10 | Distributional / uncertainty / attention responses | 2.1–2.2 | new edge types where the price test is underpowered | high |

**Sequencing note against the current `TODO.md`.** Items 1–3 are all cheaper than
"fix the Stage-2 ladder's design", which `TODO.md` currently lists as the
highest-value change. Item 3 in particular changes the node set that Stage 2
would be run *on*, so it belongs before the `p0c` fix, not after it — otherwise
the ladder gets re-specified twice.

**What must not change.** None of this touches the OOS wall, and none of it is a
reason to re-open the pre-registered OOS cell (`sign_rule` × `bh`, same-release
excluded, dormant horizon). If the redesign here produces a better specification,
the OOS cell should be **re-pre-registered before it is spent**, not silently
swapped.

---

## Appendix — reproducing the two measured tables

Run from the repo root. Both are ad hoc against the built surprise panel; neither
is in a script yet, and both should be folded into `scripts/build_panels.py`
diagnostics if they are going to be cited.

**§1.2 snapshot lag:**

```python
import polars as pl
sp = pl.read_parquet('artifacts/panels/surprise_panel.parquet').with_columns(
    (pl.col('close_time').dt.date() - pl.col('snap_date')).dt.total_days().alias('snap_lag'))
print(sp.group_by('series').agg(
    pl.len().alias('n'),
    pl.col('snap_lag').median().alias('med'),
    pl.col('snap_lag').quantile(0.9).alias('p90')).sort('n', descending=True))
```

**§3.1 simultaneity and cross-surprise correlation:**

```python
import itertools, numpy as np, polars as pl
sp = pl.read_parquet('artifacts/panels/surprise_panel.parquet')

# how many trigger series share a close_time
print(sp.group_by('close_time').agg(pl.len().alias('k'))['k'].value_counts().sort('k'))

fam = ['CPI', 'CPICORE', 'CPIYOY', 'CPICOREYOY', 'PAYROLLS', 'U3']
w = (sp.filter(pl.col('series').is_in(fam))
       .with_columns(z=pl.col('surprise') / pl.col('implied_std'))
       .pivot(values='z', index='close_time', on='series'))
for a, b in itertools.combinations([c for c in w.columns if c != 'close_time'], 2):
    d = w.select(a, b).drop_nulls()
    if d.height >= 10:
        x, y = d[a].to_numpy(), d[b].to_numpy()
        print(f'{a:12s} {b:12s} n={d.height:3d} r={np.corrcoef(x, y)[0,1]:+.3f} '
              f'sign-agree={(np.sign(x) == np.sign(y)).mean():.0%}')
```
