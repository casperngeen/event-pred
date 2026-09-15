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
    venv/bin/python analysis/leadlag_2026_09/economics.py        # ~6 min (400 permutations)

`build_panel.py` must run first; the rest read its parquet and are independent
of each other.

| script | question | answer |
|---|---|---|
| `build_panel.py` | One row per (trigger event, target leg). | 10,714 legs, 262 trigger events, 401 target events, 135 pairs. |
| `price_structure.py` | How does the market itself behave by entry price? | Brier 0.0912 vs 0.2499 base. Calibrated except 5-10c (−3.24pp) and 90-95c (+4.34pp). |
| `signal_model.py` | Does the aligned lead-lag signal predict settlement? | Yes: +3.80pp top-vs-bottom tercile, block-permutation p = 0.0005. Concentrated in 10-75c, absent above 75c. But no Brier/log-loss improvement. |
| `economics.py` | What does holding to close earn? | Gross +2.36c vs 0.00c for a permuted signal; friction 1.55c; net +0.81c with a clustered CI straddling zero. Decaying by year. |

Captured output is in `out/`.

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
