# Intraday post-release paths: how fast news reaches each market (plan)

*2026-09-27. Plan, not findings. In-sample only (2021-10 → 2025-12): never read
`data/_OOS_DO_NOT_USE_*` or anything dated 2026.*

## Why this study

The temporal side of the STG has so far been tested *across* releases. The
event-time panel steps from one release instant to the next (median gap one
week), and the GRU looks back over the last 6 of them (~54 days). That found
nothing (`graph_ablation.md` §2, §9):
- lag windows and decays of earlier surprises carry no signal;
- AGCRN does best with its GRU removed.

Each release is absorbed before the next one arrives, so there is no
cross-release dependence to model.

Any lag sits **inside the hours after a single release**:
- **labour → policy:** FED contracts keep moving after payrolls
  (`propagation_findings.md`: far-meeting contracts reprice over days;
  `graph_ablation.md` §5: the one channel with t ≥ 2);
- **sports:** about a third of a game result reaches the division market only
  after its first post-game print (drift t = 6.0, `reports/sports_study.md` on
  branch `casper/sports-graph`);
- **against that:** the released series' own market reprices "once,
  completely, at its first print" (`research_log.md` §5e). Thin legs take ~30 h
  to their first print against ~1 h for liquid ones (`liquidity_findings.md`).

This study moves the time axis inside the release. For each release, follow
every market's price on an intraday grid for the next 24 h (and on a coarser
grid to +7 d), and ask **how fast and in what order the news reaches each
market**. For the thesis, this adds a *speed* and an *order* to each edge of the
graph, which so far only records *which* markets respond and with what sign.

## Questions

1. **Speed per channel.** What share of the eventual response has arrived by
   +5 min, 15 min, 1 h, 4 h, 24 h? What is each channel's half-life?
2. **Clock time vs trade time.** Is a slow channel slow because traders update
   slowly, or only because nobody trades? Measure the path against minutes
   since the release and against the target's k-th post-release print.
3. **Order: propagation through the network, or a common reaction?** Does the
   source market's own move (e.g. next month's CPI contract) by τ predict the
   target's *remaining* move after τ, beyond the surprise itself? Does the
   target's partial move predict its own remainder (underreaction)?
4. **Is there a temporal model to learn?** Does a sequence model over the
   intraday bars beat a per-channel absorption curve at predicting the
   remaining move? This is the test the across-release GRU failed.

## Data

- **Tape:** reuse `analysis/spillover_2026_09/_tape.py`. It loads threshold
  legs for the 17 `HAWKISH` series, with a VWAP per print and signed taker
  flow. It also has `lead_contract` (deterministic tie-break) and `asof`.
- **Triggers:**
  - the 154 Kalshi release instants (`event_time_2026_09/build_panel.py`,
    `out/event_nodes.parquet`), with their standardised surprises;
  - optional second arm: the 1,030 calendar releases with a consensus
    (`spillover_2026_09/releases.py`, `out/releases.parquet`, surprise =
    (actual − consensus)/sd, family and hawkish sign in
    `calendar_triggers.py`).
- **Targets:** each of the 17 series' lead contract at the release, chosen from
  prints *before* the release.
  - For the releasing series, this is its next unresolved contract (e.g. next
    month's CPI), since this month's resolves at the release.
  - Record the pre-release price p₀ and its age.
- **Sign convention:** as everywhere else, the response is theory-signed by
  sign(z)·HAWKISH[source]·HAWKISH[target]. Positive means the theory direction.

## Design

**Grid** (τ = time since release):

| stretch | bars |
|---|---|
| −60 min → 0 | a 60-min pre-release reference |
| 0 → +2 h | 5 min |
| +2 h → +6 h | 15 min |
| +6 h → +24 h | 1 h |
| +1 d → +7 d | 1 day (for labour → policy's far meetings) |

**Per (release, target, bar):**
- the as-of price (last print, forward-filled);
- whether the target has traded since the release;
- prints since the release;
- time since the last print;
- cumulative signed flow since the release.

**Truncation:** cut a path at the next release that touches that target (13% of
instants have another release within 24 h). Group releases in the same minute
into one instant, as `build_panel.py` does.

**Absorption curve:**
- Per channel c and τ, regress the theory-signed response r(τ) = p(τ) − p₀ on
  |z| across (release, target) cells. Report β_c(τ) and the ratio
  β_c(τ)/β_c(24 h).
- Use pooled slopes rather than per-cell fractions r(τ)/r(24 h), which blow up
  when r(24 h) ≈ 0.
- CIs: bootstrap over release instants. Null: flip each release's surprise
  sign at random.

**Trade time:** the same curve with τ replaced by the k-th post-release print of
the target (k = 1, 2, 3, 5, 10), on cells that reach k prints within 24 h.

## Phases

**Phase 0: build the path panel.** Script `build_paths.py` writes
`out/paths.parquet`, one row per (instant, target, bar).

Sanity checks:
- coverage: the share of cells with at least one print by each τ, by series;
- median pre-release staleness;
- the releasing series' next contract has a jump at its first print, matching
  `research_log.md` §5e;
- the curves at placebo times are flat (`_tape.placebo_time`).

**Phase 1: absorption curves** (`curves.py`), for Q1 and Q2.
- β_c(τ) and half-life per channel, for:
  - the 15 channels;
  - own series vs cross series;
  - liquid vs thin targets (by pre-release print count).
- The same curves in trade time.
- **Expected:** most macro channels are flat after the first print. Labour →
  policy may drift. Report the calendar-release arm separately.

**Phase 2: order and lead-lag** (`order.py`), for Q3. At each τ ∈ {15 min,
1 h, 4 h}:

  r_B(τ → 24 h) ~ z_signed + r_A(0 → τ) + r_B(0 → τ) + flow_B(0 → τ)

- A is the source's next contract and B the target.
- Clustered by instant, walk-forward as below.
- A significant r_A term, over z, is propagation *through* the network: the
  source market's move carries information about the target's move that the
  headline doesn't.
- A positive r_B term is target underreaction; a negative one is overshoot.

**Phase 3: can a temporal model learn it?** (`models.py`), for Q4.
- **Task:** at each bar τ, predict the target's move over the next window
  (τ → τ + 1 h, and τ → 24 h), using information up to τ only.
- **Rungs, in order:**
  1. zero-parameter: theory sign × |z| × (1 − fitted absorption so far);
  2. per-channel curve (ridge on channel × τ-bucket dummies × z);
  3. linear with the path state so far (r_A, r_B, flow, traded flags);
  4. GRU over the bars, per target;
  5. AGCRN on the intraday grid: step = bar, nodes = the 17 targets, messages
     = other markets' moves so far. This is the STG where the temporal axis
     finally has dependence in it.
- **Walk-forward:** the 8 expanding folds by release instant used elsewhere
  (`stg.models.train._fold_cuts`), purged on label end. Every bar of a release
  goes to the same fold.
- **Metrics:** R² vs 0, balanced accuracy, AUC. Score on all bars and on bars
  where the target trades in the prediction window, since a forecast of a
  price that never trades is untestable. Bootstrap over instants for Δ vs the
  best linear rung.

**Phase 4 (optional, separate branch):** the same pipeline on MLB 2025 games
on `casper/sports-graph`, with no shared code with the macro study. There the
drift is known to exist, so it is the positive control for Phase 3: if a
sequence model can't beat the absorption curve on sports, it won't on macro.

## Pitfalls

- **The effective sample is the number of releases** (154 Kalshi; ~1,180 with
  the calendar arm), not bars × targets. Bars within a release are highly
  dependent: bootstrap and fold by instant.
- **Stale prices.** Forward-filled bars are not prices anyone traded at. Keep
  the traded-since flags, report coverage, and score Phase 3 on bars that
  trade.
- **Bid-ask bounce.** Consecutive prints alternate between sides. Report the
  curves with and without a taker-side adjustment (or on 2-print VWAP) to
  check that early "drift" isn't bounce.
- **Coincident releases.** 08:30 ET often carries several releases. Keep them
  as one instant with a multi-source surprise vector, as in the event-time
  panel.
- **Overnight.** A 24 h window after an 08:30 release spans a thin night.
  Trade time (Q2) is the check.
- **Multiple testing** across channels × τ: BH within each figure, and state
  expectations before looking (below).

## Expectations, stated before running

- The releasing series' next contract and most cross-series targets: most of
  the response by the first print, β(τ)/β(24 h) ≥ 0.8 in trade time.
- Labour → policy: a detectable drift after the first print in clock time,
  partly surviving in trade time.
- Phase 3 on macro: sequence models tie the per-channel curve.
- Phase 4 on sports: the path-state rung beats the curve, because the drift is
  there.

A result that breaks these, especially a macro sequence model beating the
curve, gets the leakage and alignment checks of `graph_ablation.md` §7 before
it is believed.

## Deliverables

`analysis/intraday_2026_09/`:
- `build_paths.py`, `curves.py`, `order.py`, `models.py`;
- a README with run lines;
- outputs in `out/`, git-ignored.

Writeup: `reports/intraday_path_findings.md`, linked from `INDEX.md` and from
`graph_ablation.md` §9.

## Decisions to confirm at the start

1. The grid above (5 / 15 / 60 min, to 24 h, plus daily to 7 d).
2. Whether to include the calendar-release arm from Phase 1, or add it after.
3. Whether to run Phase 4 (sports) before Phase 3 as the positive control.
