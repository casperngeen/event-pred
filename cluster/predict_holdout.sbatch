#!/bin/bash
# The frozen model's holdout forecasts (cluster/predict_holdout.py), one array task per league.
# The holdout panels sit in holdout_panels/<league>/ only for this run (never panels/); see cluster/README.md.
#     sbatch cluster/predict_holdout.sbatch                  # MLB NFL
#     LEAGUES="MLB" sbatch --array=0 cluster/predict_holdout.sbatch
# Results: results/holdout/<league>_fold99_gru_d10_h30-60-120-1800_tp[_pitch]_{seed<s>,ens5_seed0}.npz (+ .json) (MLB: _pitch)
# GPU: a100-40 by default; another type for one submission (options on the command line override this line):
#     cluster/submit.sh <this script> [options]: the first modern GPU type with free capacity (never the old nv cards)
#     sbatch --gres=gpu:a100-80:1 ...     or --gres=gpu:h100-47:1 (half an H100) / --gres=gpu:h100-96:1 / --gres=gpu:h200-141:1
#SBATCH --job-name=stg-holdout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"
read -r -a LG <<< "${LEAGUES:-MLB NFL}"
LEAGUE=${LG[${SLURM_ARRAY_TASK_ID:-0}]}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python cluster/predict_holdout.py --league "$LEAGUE" --panel "holdout_panels/$LEAGUE" --weights results/ingame/final --out results/holdout
