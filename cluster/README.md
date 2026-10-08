# Running on the SoC GPU cluster (Slurm)

Partitions (from `sinfo`): `gpu` / `gpu-long` (3 h / 3 days) for GPU jobs,
`normal` / `long` (3 h / 3 days) for CPU jobs. Ask for a GPU by name:
`--gres=gpu:a100-40:1` (default), `gpu:h100-47:1` (an H100 slice),
`gpu:a100-80:1`. Avoid the generic `gpu:nv`, which gives an older card.

The models are small, so what matters is queue time, not GPU size. Between-game
models run fine on a laptop CPU; the cluster is for the in-game model and large
grids.

## Setup (once)

The login node is reached through the stujump jump host. Add to `~/.ssh/config`
on your laptop, so `ssh`, `rsync` and `scp` all go through it:

    Host xlogin
        HostName xlogin.comp.nus.edu.sg
        User casperng
        ProxyJump casperng@stujump.comp.nus.edu.sg

From your laptop, copy the code, never `data/` (holdout data must not leave the
laptop; jobs get prepared, working-block tensors only):

    rsync -av --exclude '__pycache__/' --exclude '*.egg-info/' --exclude '.pytest_cache/' \
          stg_infra cluster xlogin:~/event-pred/

Copy only `stg_infra/` and `cluster/`: never the repo root, which holds `.git`, notebooks,
`analysis/*/out` (derived data) and `kalshi_api_key.txt`.

On the cluster, from `~/event-pred`:

    bash cluster/setup_env.sh          # .venv-cluster with CUDA torch + stg
    sbatch cluster/smoke_gpu.sbatch    # ~1 min once it starts

CUDA torch is ~3 GB. If home is too small (`quota -s`), put the venv on a
bigger disk and pass the same path to jobs:

    VENV=/bigger/disk/stg-venv bash cluster/setup_env.sh
    VENV=/bigger/disk/stg-venv sbatch cluster/smoke_gpu.sbatch
    cat cluster/logs/stg-smoke-*.out   # GPU name and s/step at the plan's sizes

## Later

Training jobs will be Slurm job arrays, one task per (fold, seed, variant),
each writing its own result file, plus a collect step:
- in-game model: `gpu` partition, one GPU (or an H100 slice) per task;
- between-game grid: `normal` partition, one CPU per task.

These come with the trainer (sports plan, Phase 3b).

## In-game model (stg.models.ingame)

Build the panel on the laptop (it refuses to write anything reaching the 2026-07-01 holdout), then
copy only that directory; raw `data/` never leaves the laptop:

    venv/bin/python -W ignore analysis/graph_2026_10/b3_table.py NFL     # training table
    venv/bin/python -W ignore analysis/graph_2026_10/b4_export.py NFL    # -> data/sports/derived/ingame/NFL/panel_v2
    rsync -av data/sports/derived/ingame/NFL/panel_v2/ xlogin:~/event-pred/panels/NFL/
    rsync -av --exclude '__pycache__/' --exclude '*.egg-info/' --exclude '.pytest_cache/' stg_infra cluster xlogin:~/event-pred/

On the cluster:

    sbatch cluster/train_ingame.sbatch                 # GRU, folds 1-5 x feed delays 0/2/5/10/15/20 s
    MEMORY=none DELAYS="0 15" sbatch --array=0-9 cluster/train_ingame.sbatch     # the no-memory ablation
    ls results/ingame/                                 # one JSON per task

Copy the results back with `rsync -av xlogin:~/event-pred/results/ingame/ results/ingame/`. The NFL also
runs on the laptop in minutes: `bash cluster/run_ingame_local.sh NFL`.

## Final models for the holdout test

The frozen spec is `reports/holdout_spec.md`. Train each league's final model (5 seeds, every training
game, weights saved) on the same `panels/<league>_v3` copies as the test-fold runs (MLB: `panels/MLB_pitch_v3`, with
`LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 sbatch --array=0-4 cluster/train_final.sbatch`):

    sbatch cluster/train_final.sbatch                  # NBA MLB NFL x seeds 0-4 = 15 tasks
    ls results/ingame/final/                           # <league>_final_gru_d10_h30-60-120-1800_tp_seed<s>.json + .pt

Copy the weights back with `rsync -av xlogin:~/event-pred/results/ingame/final/ results/ingame/final/`.
Holdout data never go to the cluster: the holdout panel, forecasts and backtest run on the laptop.

## Holdout forecasts (the one exception to "holdout data stay on the laptop")

The laptop runs out of memory forecasting the MLB holdout, so the forecasts alone run here, under these rules:
the holdout panels go to `holdout_panels/` (never `panels/`, which training jobs read), nothing else from the
holdout comes here (no trades, no tables), and both the panels and the forecasts are deleted once copied back.
The backtest and the report stay on the laptop.

    # laptop
    rsync -avz --exclude '__pycache__/' --exclude '*.egg-info/' --exclude '.pytest_cache/' stg_infra cluster xlogin:~/event-pred/
    rsync -avz data/sports/_HOLDOUT_DO_NOT_USE/derived/ingame/MLB_pitch/panel_v3/ xlogin:~/event-pred/holdout_panels/MLB/   # MLB: every-pitch panel
    rsync -avz data/sports/_HOLDOUT_DO_NOT_USE/derived/ingame/NFL/panel_v3/ xlogin:~/event-pred/holdout_panels/NFL/
    # cluster
    sbatch cluster/predict_holdout.sbatch              # MLB, NFL (one task each)
    # laptop, when done: forecasts back into the holdout folder, then delete the cluster copies
    rsync -az xlogin:~/event-pred/results/holdout/ data/sports/_HOLDOUT_DO_NOT_USE/derived/ingame/results/
    ssh xlogin 'rm -rf ~/event-pred/holdout_panels ~/event-pred/results/holdout'
