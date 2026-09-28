# Propagation speed, 2026-09

`leadlag_2026_09` pairs each trigger with the **next** target event only, and
enters at the first print after the trigger resolves. A relation that reaches
its target slowly -- A moves C's contract two or three releases out, or the
market takes weeks to notice -- is invisible to it. This study widens the panel
along two axes and asks where a trigger's surprise is still unpriced:

* **horizon** -- days from the trigger's resolution to the target event's
  settlement, banded monthly out to six months;
* **entry delay d** -- enter at the first print after `t_res + d`, for
  d = 0, 1, 7, 30 days.

Pairs are fixed in advance, not searched: the relations earlier studies
identified (leadlag §4b BH channels, `edge_economics.md` BH survivors,
event_time's labour→policy) plus a few with a strong prior (PCECORE, CPIYOY,
CPICOREYOY and GDP → FED). `PAIRS` in `build_panel.py` tags each one. Same
`HAWKISH` sign restriction, same threshold-only scope as leadlag. In-sample only.
Writeup: `reports/propagation_findings.md`.

    venv/bin/python analysis/propagation_2026_09/build_panel.py   # ~12 s, writes out/depth_legs.parquet
    venv/bin/python analysis/propagation_2026_09/profile.py       # ~6 s, reads it
    venv/bin/python analysis/propagation_2026_09/controls.py      # ~20 s, reads it
    venv/bin/python analysis/propagation_2026_09/economics.py     # ~1 min, reads it

| script | what it does |
|---|---|
| `build_panel.py` | One row per (trigger event, target event settling within 180 d, delay, threshold leg). Entry is the first print after the cut and must follow it within 7 d, unless the leg had never traded (`deferred`), in which case it is the leg's first print whenever that is. |
| `profile.py` | No controls. `relations.py`'s aligned residual `mean(sign(signal)·(win − p_entry))` per (group, horizon, delay), with a clustered bootstrap on `target_event`, the within-series block permutation, and BH at q = 0.10. Repeated on 10–75c legs (not pre-specified). |
| `controls.py` | OLS of `win − p_entry` on the trigger's sign, adding in turn the target's own latest pre-entry surprise (`own_pre`), its own surprises between entry and settlement (`own_mid`, mediation), and the trigger series' next surprises (`trig_mid`, persistence). The perm p holds controls fixed (Frisch–Waugh). |
| `economics.py` | labour→policy only. Nets same-release signals into one position per (release, FED leg), takes `sign(net)`, holds to settlement. Taker fee + half the spread, measured on FED legs by days to settlement and moneyness (never charged below leadlag's flat schedule). Net P&L by horizon × delay, the post hoc far-meeting rule with 2×/3× spread stresses, always-YES / always-NO controls, by year, and the block-permutation null. |

## Results

Horizon is indexed by days, not by "the k-th event": the archive has holes
(PCECORE jumps 272 and 420 days, CPI 216 and 337), so "the k-th event in the
data" would file a nine-month-out contract as the next one.

**labour→policy is the only group with a horizon profile, and it is the
pattern the hypothesis predicts in horizon but not in delay.** PAYROLLS/U3 →
FED, all legs, d = 0:

| horizon | 0–1m | 1–2m | 2–3m | 3–4m | 4–6m |
|---|---|---|---|---|---|
| aligned residual (pp) | +0.3 | +1.1 | **+3.0** | **+2.6** | **+3.5** |
| perm p | 0.21 | 0.06 | 0.01 | 0.03 | 0.03 |
| PAYROLLS→FED alone | +1.2 | +3.6 | +4.5 | +6.8 | +5.4 |

The next Fed meeting is priced efficiently. Meetings two to six months out are
not, **at the first print after the release** -- and the effect is gone within
a week. 2–3m horizon, 10–75c legs:

| delay | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|
| no controls (pp) | **+13.8** (BH) | **+11.4** (BH) | +4.9 | +0.6 |
| all three controls (pp) | **+11.7** (BH), CI [+1.5, +20.7] | **+10.1** (BH) | +3.4 | +2.8 |

So the propagation is **slow across contracts, fast in time**: the market
reprices the far meetings more slowly than the near one, but in days, not
months. Withholding for a month and entering then earns nothing (d = 30 ≈ 0 in
every labour→policy cell).

**The controls explain almost none of it.** Their correlation with the
trigger's sign is +0.001 (`own_pre`), +0.010 (`own_mid`), +0.022 (`trig_mid`);
the labour→policy coefficient moves by < 2pp. The far-meeting effect is not the
intervening Fed decisions and not the next payrolls print -- it is the original
surprise. The one place a control bites is **labour→labour at 0–1m** (claims →
payrolls/ADP/U3): 12.7 → 7.9pp on 10–75c legs once `trig_mid` is in, i.e.
about a third of it is the *next* weeks' claims prints, which is the obvious
mechanism for a weekly series leading a monthly one.

**Nothing else has a profile.** inflation→policy, growth→policy and PCE↔CPI
are flat or negative across horizons once CIs are read. The one BH cell outside
labour→policy -- inflation→policy 2–3m d = 30 on 10–75c legs (+20.7pp) -- sits
between −17pp and −15pp neighbours on 20 target events and is noise.

## Caveats

* **Small.** labour→policy is 26–29 FED target events per cell, and PAYROLLS
  and U3 print together, so the two triggers are not independent draws. The
  clustered CIs on the all-legs cells all include zero; only the 10–75c cells
  exclude it, and that band was chosen after leadlag §2, not before.
* **Decaying.** labour→policy at d = 0, all horizons: 2022 +1.8pp, 2023 +6.0,
  2024 +3.0, 2025 +1.0 -- the same decay leadlag found.
* **Gross.** The tables above are before costs; `economics.py` prices them
  (below).
* **Inflation targets rarely trade early.** CPI/PCE legs mostly print for the
  first time within ~2 weeks of their release, so beyond 0–1m nearly every
  PCE↔CPI row is `deferred`, and d = 0/1/7 land on the same entry. For those
  targets "withhold until a contract trades" is the only option, and it shows
  nothing.
* **Cells overlap.** A deferred leg is the same entry at several delays, and
  every horizon shares trigger events, so BH is a rough guard, not a family
  error rate.

## Costs (`economics.py`)

Far meetings (2–6m), entry at d = 0, chosen after reading the tables above, so
this is an in-sample upper bound:

| | all legs | 10–75c legs |
|---|---|---|
| net | +1.04c, CI [−0.98, +3.58] | **+6.24c**, CI [+0.81, +13.27] |
| permutation p | 0.029 | 0.008 |
| spread × 3 | −0.40c | +4.24c, P(≤0) 0.067 |
| **always YES, same legs** | **+3.76c** | **+6.30c** |

The rule does not beat buying YES on every far leg. That baseline is the Fed
cycle (10–75c: 2022 +33.9c, 2025 −19.4c), and the rule beats it in 2023, 2024
and 2025 but loses to it by 28c in 2022. Far FED legs are too thin for a 60 s
spread estimate at all, so the charge is the near-dated schedule. Details and
reading in `reports/propagation_findings.md` §4–5.

## Rerun 2026-09-28 on the backfilled archive

All four scripts were rerun. The tables above are the 2026-09-26 run. The panel
is now 32,230 rows over 225 trigger events. labour → policy, d = 0:

| horizon | 0–1m | 1–2m | 2–3m | 3–4m | 4–6m |
|---|---|---|---|---|---|
| aligned residual (pp) | +0.5 | +1.2 | **+4.2** | **+3.8** | **+5.2** |
| perm p | 0.15 | 0.04 | 0.00 | 0.02 | 0.01 |

2–3m on 10–75c legs, by entry delay:

| controls | d = 0 | d = 1 | d = 7 | d = 30 |
|---|---|---|---|---|
| none (pp) | **+15.8** (BH) | **+14.0** (BH) | +6.5 | +3.2 |
| all three (pp) | **+14.0** (BH), CI [+3.3, +23.7] | **+12.8** (BH) | +4.9 | +5.6 |

`economics.py` P&L is unchanged: the rule selects the same positions. Its
permutation p is now 0.052 on all legs and **0.022** on 10–75c legs. Detail:
`reports/backfill_rerun_2026_09.md`.

`out/` is untracked; see `analysis/README.md`.
