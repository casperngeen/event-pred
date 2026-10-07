#!/bin/bash
# The frozen model for the holdout test (reports/holdout_spec.md): one Slurm array task per (league, seed),
# each trained on every game in the training panel (cluster/train_ingame.py --final), weights saved.
# From the repo root on the cluster, with panels/<league>_v3 = the local panel_v3 (no futures):
#     sbatch cluster/train_final.sbatch                       # NBA MLB NFL x seeds 0-4 = 15 tasks
#     LEAGUES="MLB" sbatch --array=0-4 cluster/train_final.sbatch
# Run r -> league = LEAGUES[r / n_seeds], seed = SEEDS[r % n_seeds]; PACK runs per task as in train_ingame.sbatch.
# Results: results/ingame/final/<league>_final_gru_d10_h30-60-120-1800_tp_seed<s>.json + .pt (weights)
#SBATCH --job-name=stg-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"
read -r -a LG <<< "${LEAGUES:-NBA MLB NFL}"
read -r -a SD <<< "${SEEDS:-0 1 2 3 4}"
T=${SLURM_ARRAY_TASK_ID:-0}
PACK=${PACK:-1}
NRUNS=$(( ${#LG[@]} * ${#SD[@]} ))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for (( i = 0; i < PACK; i++ )); do
  R=$(( T * PACK + i ))
  (( R < NRUNS )) || break
  LEAGUE=${LG[$(( R / ${#SD[@]} ))]}
  SEED=${SD[$(( R % ${#SD[@]} ))]}
  echo "task $T run $R: league $LEAGUE, seed $SEED (final model)"
  python cluster/train_ingame.py --panel "panels/${LEAGUE}_v3" --final --save-model --memory gru --delay 10 \
      --horizons 30s,60s,120s,1800s --seed "$SEED" --games-per-batch 16 --steps-per-game 256 --out results/ingame/final
done
