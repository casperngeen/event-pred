#!/bin/bash
# In-game graph model (stg.models.ingame): one Slurm array task per (league, test fold, feed delay).
# One array job can cover several leagues (useful when only one job may be active at a time).
# From the repo root on the cluster, after copying the panel (see cluster/README.md, "In-game model"):
#     sbatch cluster/train_ingame.sbatch                                  # NFL, GRU, 5 folds x 6 delays
#     LEAGUE=NBA sbatch cluster/train_ingame.sbatch
#     LEAGUES="NBA MLB" DELAYS="0 5 10 15" sbatch --array=0-39 cluster/train_ingame.sbatch   # both, one job
#     MEMORY=none DELAYS="0 15" sbatch --array=0-9 cluster/train_ingame.sbatch
#     cat cluster/logs/stg-ingame-<jobid>_<task>.out
# Run r -> fold = 1 + r % 5, delay = DELAYS[(r / 5) % n_delays], league = LEAGUES[r / (5 n_delays)];
# there are 5 x n_delays x n_leagues runs. Each array task runs PACK consecutive runs (default 1), one
# after another, so the array needs (number of runs) / PACK tasks: size --array to that minus 1. Use
# PACK > 1 when the cluster caps how many jobs (array tasks) you may submit. LEAGUES defaults to LEAGUE.
#     LEAGUES="NBA MLB" DELAYS="0 5 10 15" PACK=2 sbatch --array=0-19 cluster/train_ingame.sbatch
#     HORIZONS=30s,60s,120s LEAGUES="NBA MLB" DELAYS="0 5 10 15" PACK=2 sbatch --array=0-19 cluster/train_ingame.sbatch
# PANEL_SUFFIX picks another panel copy, e.g. PANEL_SUFFIX=_mid -> panels/NBA_mid (panel_v4m, fresh-mid targets);
# EXTRA_ARGS passes trainer options, e.g. EXTRA_ARGS="--huber 1000 --lambda-y 0". Folds 3-5 only: --array=2-4 (one delay).
#     LEAGUE=NBA PANEL_SUFFIX=_mid DELAYS="10" HORIZONS=30s,60s,120s,1800s EXTRA_ARGS="--huber 1000" sbatch --array=2-4 cluster/train_ingame.sbatch
# Ablations (run name gets _r0 for no message passing, "none" for no play memory) and a second seed:
#     LEAGUES="NBA MLB" DELAYS="10" HORIZONS=30s,60s,120s,1800s EXTRA_ARGS="--rounds 0" sbatch --array=0-9 cluster/train_ingame.sbatch
#     LEAGUES="NBA MLB" DELAYS="10" HORIZONS=30s,60s,120s,1800s MEMORY=none sbatch --array=0-9 cluster/train_ingame.sbatch
#     LEAGUES="NBA MLB" DELAYS="10" HORIZONS=30s,60s,120s,1800s SEED=1 sbatch --array=0-9 cluster/train_ingame.sbatch
# Results: results/ingame/<league>_fold<k>_<memory>_d<delay>_seed<s>.json (+ .npz predictions).
#SBATCH --job-name=stg-ingame
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"
MEMORY=${MEMORY:-gru}
SEED=${SEED:-0}
read -r -a D <<< "${DELAYS:-0 2 5 10 15 20}"
read -r -a LG <<< "${LEAGUES:-${LEAGUE:-NFL}}"
T=${SLURM_ARRAY_TASK_ID:-0}
PACK=${PACK:-1}
NRUNS=$(( 5 * ${#D[@]} * ${#LG[@]} ))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for (( i = 0; i < PACK; i++ )); do
  R=$(( T * PACK + i ))
  (( R < NRUNS )) || break
  FOLD=$(( 1 + R % 5 ))
  DELAY=${D[$(( (R / 5) % ${#D[@]} ))]}
  LEAGUE=${LG[$(( R / (5 * ${#D[@]}) ))]}
  echo "task $T run $R: league $LEAGUE, fold $FOLD, delay $DELAY s, memory $MEMORY"
  python cluster/train_ingame.py --panel "panels/${LEAGUE}${PANEL_SUFFIX:-}" --test-fold "$FOLD" --memory "$MEMORY" --delay "$DELAY" \
      --seed "$SEED" --games-per-batch 16 --steps-per-game 256 --save-preds \
      ${HORIZONS:+--horizons "$HORIZONS"} ${EXTRA_ARGS:-}
done
