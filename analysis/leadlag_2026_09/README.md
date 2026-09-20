# Lead-lag to settlement, resolved by entry price (2026-09-15)

A from-scratch rebuild of the lead-lag question, with two changes from every
previous study in this repo:

1. **The target is settlement, not a price change.** Entry is the first print
   strictly after the trigger resolves; the position is held to the target's
   close. Payoff is 0 or 100c, so there is no exit fee and no exit spread.
2. **The whole target ladder is used, not one representative leg.** Stage 1,
   `direction_study` and `channel_pooling` all collapse a target event to its
   most-traded leg. That throws away the dimension this study is about: a leg
   at 10c and a leg at 90c are different bets on the same statistic, with
   different calibration, different payoff geometry and different fees.

The writeup is `reports/leadlag_findings.md`.

    venv/bin/python analysis/leadlag_2026_09/build_panel.py      # ~4 min, writes the panel
    venv/bin/python analysis/leadlag_2026_09/price_structure.py  # market baseline by price
    venv/bin/python analysis/leadlag_2026_09/signal_model.py     # does the signal add anything?
    venv/bin/python analysis/leadlag_2026_09/relations.py        # per-channel / per-pair, learned vs imposed
    venv/bin/python analysis/leadlag_2026_09/economics.py        # ~6 min (400 permutations)
    venv/bin/python analysis/leadlag_2026_09/maker_fill.py       # maker vs taker execution

`build_panel.py` must run first; the rest read its parquet and are independent
of each other.

| script | question | answer |
|---|---|---|
| `build_panel.py` | One row per (trigger event, target leg). | 10,714 legs, 262 trigger events, 401 target events, 135 pairs. |
| `price_structure.py` | How does the market itself behave by entry price? | Brier 0.0912 vs 0.2499 base. Calibrated except 5-10c (−3.24pp) and 90-95c (+4.34pp). |
| `signal_model.py` | Does the aligned lead-lag signal predict settlement? | Yes: +3.80pp top-vs-bottom tercile, block-permutation p = 0.0005. Concentrated in 10-75c, absent above 75c. But no Brier/log-loss improvement. |
| `relations.py` | Which specific relations carry it, and does learning beat imposing? | Learning loses monotonically: imposed (0 params) +0.86pp, per-channel (15) +0.47pp, per-pair (135) −0.07pp. 0/88 pairs survive BH; 3/14 channels do. |
| `maker_fill.py` | Does resting a limit order beat crossing? | **No.** Adverse selection −1.7 to −14.8c against a 0.5-1c spread saving; no maker arm beats the taker even at zero fee. |
| `economics.py` | What does holding to close earn? | Gross +2.36c vs 0.00c for a permuted signal; friction 1.55c; net +0.81c with a clustered CI straddling zero. Decaying by year. |
| `move_filter.py` | Does requiring the market to have already moved your way help? | **Yes — this is where the confirmation filter comes from.** Net rises 0.40c (no filter) to 2.30c at `move >= 2c`. Controls: momentum alone −0.27c, move-but-signal-disagrees −0.19c. |
| `confirmation_stability.py` | Does the confirmation filter decay? | No — positive in all three years. |
| `ablation.py` | What is the signal worth over the apparatus? | +3.76c marginal, not the +5.41c headline. Dropping signal, confirm and floor leaves +0.53c. |
| `leakage_audit.py` | Any look-ahead in the chain? | Two candidates found, both immaterial; spec unchanged. |
| `exit_rules.py` | Does an early exit beat holding to settlement? | No; spec unchanged. |
| `sizing.py` | Does price-dependent sizing help? | No; spec unchanged. |
| `blotter.py` | The full trade list. | `out/trade_blotter.csv`, with `yes_price_at_fill` so SELL YES rows are unambiguous. |
| `tie_handling.py` | Is "last trade price" a rule when several prints share an instant? | **No.** The committed tie-break sat at the 98th percentile of 60 random ones; headline falls to ~+4.7c. VWAP and max/min are order-invariant, `.last()` is not. |
| `spec_v2.py` | The two audit fixes applied together. | VWAP-collapsed tape + median-of-3 confirmation: **+4.10c, CI [−1.34, +9.56]**, block-permutation p = 0.005. |
| `liquidity.py` | Is the edge larger where fewer people are watching? | Thin legs reprice 30x slower and show a monotone profit gradient — but see `liquidity_spread_diag.py`. Within-event contrast +5.60c, P = 0.070. |
| `liquidity_gates.py` | How much liquidity selection is already baked in? | No volume filter in the pipeline; four implicit gates. **G0, the archive, misses 44% of thin traded legs against 7% of liquid.** Relaxing our own gate mildly improves the result. |
| `liquidity_spread_diag.py` | Why did the thin−liquid spread swing 0.94–7.43c? | It did not. No CI had been put on the difference (~±9c), and changing `k` churns a third of the portfolio (Jaccard 0.61). Population held fixed: **~+3c, P(≤0) ≈ 0.3**. |

Captured output is in `out/`. Writeups: `reports/leadlag_findings.md`, `reports/strategy_spec.md`, and `reports/liquidity_findings.md` for the three `liquidity*.py` scripts.

## Design notes

* **Threshold targets only.** A threshold leg settles YES iff `X > K`, so a
  shift in the target's central value moves every leg monotonically. Bucket
  legs (WTI) settle on `lo <= X <= hi`, which is not monotone in `X`, and WTI
  is the series `TODO.md:204` proposes dropping from the trigger set on
  principle. It is out of both sides here.
* **Same-release pairs excluded** — CPI and CPICORE print together, so a
  relation between them is simultaneity, not lead-lag.
* **The sign restriction is imposed, not fitted**:
  `direction = HAWKISH[trigger] * HAWKISH[target]`, zero free parameters.
  `U3`/`JOBLESSCLAIMS` carry `-1`, which makes it falsifiable — and pass 5 of
  `signal_model.py` confirms the raw association reverses in exactly those
  cells while the aligned one does not.
* **Clustering on `target_event` everywhere.** One print settles every leg of
  an event, and the same target event is matched by several different
  triggers, so the leg count is never the sample size.
* **The null is a block permutation** — `z_surprise` shuffled among trigger
  events *within each trigger series*, preserving the CPI/CPIYOY dependence
  that `research_log.md` §3 showed a naive shuffle destroys.

Everything is in-sample only.
