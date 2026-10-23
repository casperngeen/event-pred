# In-game model ablations: run sheet

What each part of the in-game model contributes, for the report's ablation table. Each step below is one model
version: train it on the cluster, average its seeds there, copy the average back, simulate its trades on the laptop,
and rebuild the table. Do the steps in order; each one stands alone, so the table is usable after any of them.

## Ground rules

- **Test folds only.** Every version is trained and scored on the same 5 walk-forward test folds as the frozen
  model. Nothing here touches the holdout (`_HOLDOUT_DO_NOT_USE`), and none of it changes the frozen rule. These
  are descriptive results for the report, not candidates for deployment.
- **5 seeds per version, averaged.** The old ablations (`analysis/graph_2026_10/out/b11_ablation_pnl.txt`) used
  one seed, and changing only the seed moved a contract's P&L about as much as removing the graph did. They were
  also run on the model with futures contracts, dropped on 7 Oct, so they are superseded by this sheet.
- **Scored two ways.** The table (`analysis/graph_2026_10/b33_ablation_table.py`) shows, per contract type:
  - forecast skill: R² of the 120-s price change;
  - trading result under the frozen rule: decide every 5 s, one order per contract at a time (MLB: skip
    15–60 s after the last known pitch), fixed exit at 120 s, frozen sizing, net cents per contract after fees;
  - the per-game P&L difference against the full model, with a paired t.
- **Everything else stays as in the frozen model:** feed delay 10 s, horizons 30 s / 1 min / 2 min / 30 min,
  traded-price panel, GRU memory, 2 rounds of message passing.

## Already in the table (no runs needed)

| Row | Source |
|---|---|
| Full model (frozen), all six contract types | the existing 5-seed test-fold runs |
| MLB: every pitch vs plate appearances (both without the skip window) | existing `_tp_pitch` and `_tp` runs |
| MLB: double plays coded correctly | existing `_dpfix` runs |

## Setup (once)

1. Copy the code changes to the cluster (only `stg_infra` and `cluster`, as always):

       rsync -av --exclude '__pycache__/' --exclude '*.egg-info/' --exclude '.pytest_cache/' stg_infra cluster xlogin:~/event-pred/

   New since the last sync: `--drop` in `cluster/train_ingame.py` and `stg/models/ingame.py` (feature-group
   ablations) and `--skip-existing`, `SEEDS` in `cluster/train_ingame.sbatch` (all 5 seeds in one submission),
   `cluster/average_seeds.py`, and `cluster/submit_ablations.sh` / `cluster/average_ablations.sh` (the quick path).
2. On the cluster, check that the panels are there: `ls ~/event-pred/panels/` should show `MLB_pitch_v3`, `NBA_v3`
   and `NFL_v3` (the same panels the frozen models were trained on).
3. **Disk.** Per-seed forecasts take ~0.6 GB per seed for MLB and NBA (5 folds), so one version is ~3 GB on the
   cluster until it is averaged. `average_seeds.py --delete-seeds` removes them once the average is written. Check
   `quota -s` before a big step (the delay sweep is 11 versions). On the laptop, each averaged version is
   0.1–0.4 GB until `b33_ablation_trades.sh --delete` removes it after the trades are simulated.

## Quick path: queue everything at once (recommended)

The cluster allows 32 queued or running array tasks per user. Each task can run many trainings back to back, so
all 875 runs fit in **23 tasks**: one per version for NBA + NFL together (50 runs, ~1 h), one per version for MLB
(25 runs, ~1 h), and the delay sweeps split into 4 + 3 tasks.

    # cluster, from ~/event-pred
    cluster/submit_ablations.sh                       # all steps; or STEPS="A1 A2 A3" for some; DRY_RUN=1 to preview
    squeue -u $USER                                   # only ~2 tasks run at a time; the rest wait in the queue
    cluster/average_ablations.sh                      # rerun whenever tasks finish: averages every finished version

- A task stops at the partition's 3-h limit (rare: a few slow NBA / NFL runs). Rerun the same
  `submit_ablations.sh` line (e.g. `STEPS=B2`): every run whose `.json` already exists is skipped
  (`--skip-existing`), so only the missing runs are trained again.
- `average_ablations.sh` prints each version as running (n/25), averaged now, or already averaged, with its
  run name. For each averaged version, on the laptop: `analysis/graph_2026_10/b33_pull.sh <run name>` copies it
  back, simulates its trades for each league it covers, deletes the forecasts and rebuilds the table (steps c
  and d below in one command; `DRY_RUN=1` prints the steps).
- With ~2 tasks running at once, everything takes roughly 15–20 hours of queue time, depending on GPU
  availability. Submit A and C first if you want those results earliest (`STEPS="A1 A2 A3 C"`, 11 tasks), then B.

## The four commands of every step (manual route)

Use these for one version at a time, or for steps c and d after the quick path.

Each step below gives its exact values. The pattern:

**a. Train (cluster, from `~/event-pred`).** MLB and NBA/NFL use different panels, so they are separate
submissions. `PACK=5` runs 5 trainings per task (~10–20 min per task). The array size is
(5 folds × delays × leagues × 5 seeds) / 5 − 1.

    LEAGUES="NBA NFL" PANEL_SUFFIX=_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=5 HORIZONS=30s,60s,120s,1800s \
        EXTRA_ARGS="..." [MEMORY=none] cluster/submit.sh cluster/train_ingame.sbatch --array=0-9
    LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=5 HORIZONS=30s,60s,120s,1800s \
        EXTRA_ARGS="..." [MEMORY=none] cluster/submit.sh cluster/train_ingame.sbatch --array=0-4

Check `squeue -u $USER`, then `grep -L wrote cluster/logs/stg-ingame-<jobid>_*.out` (lists any task that did not
finish).

**b. Average the seeds (cluster).** numpy only; run it on a compute node:

    srun -p normal --mem=16G --time=00:30:00 venv-python cluster/average_seeds.py --league NBA --name <NAME> --delete-seeds

(`venv-python` = `${VENV:-.venv-cluster}/bin/python`.) `<NAME>` is the run name between `fold<k>_` and `_seed<s>`;
each step lists it per league.

**c. Copy back (laptop, from the repo root).** The averaged forecasts and every seed's small `.json` file:

    rsync -av 'xlogin:~/event-pred/results/ingame/ens5/*_<NAME>_ens5_seed0.*' results/ingame/ens5/
    rsync -av 'xlogin:~/event-pred/results/ingame/*_<NAME>_seed[0-4].json' results/ingame/

**d. Simulate the trades and rebuild the table (laptop).** `<TAG>` = `<NAME>` without the leading
`<memory>_d<delay>`:

    analysis/graph_2026_10/b33_ablation_trades.sh <league> <memory> <delay> <TAG> --delete
    venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py

The table is `analysis/graph_2026_10/out/b33_ablation_table.txt` (and `.csv`).

## Steps, in priority order

Runs: 25 per league per version (5 folds × 5 seeds), about 2–3 min each on an A100.

### A. Model components (the core table)

| Step | Version | Train (a) | `<NAME>` NBA / NFL | `<NAME>` MLB | Runs |
|---|---|---|---|---|---|
| A1 | no graph: contracts don't exchange messages | `EXTRA_ARGS="--rounds 0"` | `gru_d10_h30-60-120-1800_tp_r0` | `gru_d10_h30-60-120-1800_tp_r0_pitch` | 75 |
| A2 | no play memory: the GRU over past plays is replaced by zeros | `MEMORY=none` | `none_d10_h30-60-120-1800_tp` | `none_d10_h30-60-120-1800_tp_pitch` | 75 |
| A3 | neither: only the current game state and each contract's own features | `MEMORY=none EXTRA_ARGS="--rounds 0"` | `none_d10_h30-60-120-1800_tp_r0` | `none_d10_h30-60-120-1800_tp_r0_pitch` | 75 |

Example, A1 in full:

    # cluster
    LEAGUES="NBA NFL" PANEL_SUFFIX=_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=5 HORIZONS=30s,60s,120s,1800s \
        EXTRA_ARGS="--rounds 0" cluster/submit.sh cluster/train_ingame.sbatch --array=0-9
    LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=5 HORIZONS=30s,60s,120s,1800s \
        EXTRA_ARGS="--rounds 0" cluster/submit.sh cluster/train_ingame.sbatch --array=0-4
    # cluster, once all 75 have finished
    for L in NBA NFL; do srun -p normal --mem=16G --time=00:30:00 .venv-cluster/bin/python cluster/average_seeds.py \
        --league $L --name gru_d10_h30-60-120-1800_tp_r0 --delete-seeds; done
    srun -p normal --mem=16G --time=00:30:00 .venv-cluster/bin/python cluster/average_seeds.py \
        --league MLB --name gru_d10_h30-60-120-1800_tp_r0_pitch --delete-seeds
    # laptop
    rsync -av 'xlogin:~/event-pred/results/ingame/ens5/*_tp_r0*_ens5_seed0.*' results/ingame/ens5/
    rsync -av 'xlogin:~/event-pred/results/ingame/*_gru_d10_h30-60-120-1800_tp_r0*_seed[0-4].json' results/ingame/
    analysis/graph_2026_10/b33_ablation_trades.sh NBA gru 10 _h30-60-120-1800_tp_r0 --delete
    analysis/graph_2026_10/b33_ablation_trades.sh NFL gru 10 _h30-60-120-1800_tp_r0 --delete
    analysis/graph_2026_10/b33_ablation_trades.sh MLB gru 10 _h30-60-120-1800_tp_r0_pitch --delete
    venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py

For A2 and A3, pass `none` as the memory to `b33_ablation_trades.sh` (e.g. `... NBA none 10 _h30-60-120-1800_tp_r0`).

### C. Feed delay (can the model trade at the delays the live feeds really have?)

Live, MLB plays arrive ~7–15 s after the pitch, and ESPN's NBA and NFL plays ~45 s after. Each delay is a
separately trained model, scored at the same delay. The trading rule stays the frozen one (including MLB's
15–60-s skip window), so the comparison isolates what the model loses.

| Step | Leagues | Train (a) | `<NAME>` | Runs |
|---|---|---|---|---|
| C1 | NBA, NFL | `DELAYS="45"`, `--array=0-9` | `gru_d45_h30-60-120-1800_tp` | 50 |
| C2 | NBA, NFL | `DELAYS="0 30 60"`, `--array=0-29` | `gru_d0_…`, `gru_d30_…`, `gru_d60_…` (`…` = `h30-60-120-1800_tp`) | 150 |
| C3 | MLB | `DELAYS="0 5 15"`, `--array=0-14` | `gru_d0_h30-60-120-1800_tp_pitch`, `gru_d5_…`, `gru_d15_…` | 75 |

C1 first: it answers whether NBA and NFL could be traded live at all with ESPN's feed. In step d, the delay is the
third argument, e.g. `b33_ablation_trades.sh NBA gru 45 _h30-60-120-1800_tp --delete`.

### B. Inputs (which information the model uses)

Each version sets one group of inputs to zero in training and testing (`--drop`). The groups are defined in
`stg/models/ingame.py` (`DROP_GROUPS`):

| Step | Version | `EXTRA_ARGS` | Inputs set to zero | `<NAME>` suffix | Runs |
|---|---|---|---|---|---|
| B1 | no play types | `--drop play_type` | what happened on each play (home run, strikeout, turnover, …); score and time stay | `_no-play_type` | 75 |
| B2 | no win probability | `--drop wp` | the published win probability, its change per play, and the structural baseline built from it | `_no-wp` | 75 |
| B3 | no structural baseline | `--drop baseline` | each contract's distance to its threshold and its baseline price | `_no-baseline` | 75 |
| B4 | no recent trades / flow | `--drop prints` | each contract's last trade price and age, order flow and volume | `_no-prints` | 75 |
| B5 | no quotes (lowest priority) | `--drop quotes` | each contract's bid, ask and quote age | `_no-quotes` | 75 |

`<NAME>` = the full model's name plus the suffix, e.g. `gru_d10_h30-60-120-1800_tp_no-wp` (NBA / NFL) and
`gru_d10_h30-60-120-1800_tp_pitch_no-wp` (MLB). B4 matters most for the report: the stale-price findings suggest
much of the signal comes from a contract's last trade lagging the market.

### Total

| Group | Versions | Runs | A100 hours (approx.) |
|---|---|---|---|
| A. components | 3 | 225 | 9 |
| C. feed delay | 7 | 275 | 11 |
| B. inputs | 5 | 375 | 15 |

## Results that already exist (cite, don't rerun)

These are strategy-side or older comparisons, already in `analysis/graph_2026_10/out/`:

| Question | File |
|---|---|
| One seed vs the average of 5 (forecast skill) | `b13_seed_ensemble_gru_d10_h30-60-120-1800_tp.txt`, `..._tp_pitch.txt` |
| With vs without futures contracts in the graph (older decision rule) | `b14_futures_ablation.txt`, `b14_nofutures_pnl.txt` |
| Deciding every 5 s vs once a minute, stacking vs one at a time | `b24_scale_in.txt` |
| Fixed exit vs holding to settlement | `b16_exit_vs_hold.txt` |
| Minimum signal and sizing rule | `b23_min_signal_one.txt`, `b21_signal_sizing.txt`, `b13_sizing_check.txt` |
| MLB skip window after a pitch | `b20_mlb_timing_tp_pitch_ens5.txt` |
| Simple baselines (no change, structural baseline, simple rule) | `b10_baselines.txt`, `b10_simple_rule.txt` |
