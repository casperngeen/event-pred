#!/bin/bash
# GPU smoke test. From the repo root on the cluster:
#     sbatch cluster/smoke_gpu.sbatch
#     cat cluster/logs/stg-smoke-<jobid>.out
# Another GPU: --gres=gpu:h100-47:1 (an H100 slice) or gpu:a100-80:1.
#SBATCH --job-name=stg-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV:-.venv-cluster}/bin/activate"     # same VENV as setup_env.sh; sbatch passes it through
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python cluster/bench_agcrn.py
