#!/bin/bash
# One-off setup on the cluster login node, from the repo root:
#     bash cluster/setup_env.sh
#     VENV=/path/on/bigger/disk/stg-venv bash cluster/setup_env.sh   # if home is too small
# Creates a venv (default .venv-cluster/) with a CUDA build of PyTorch and the
# stg package. CUDA torch is ~3 GB, so pip's download cache is disabled (it
# would double that under ~/.cache). The PyPI torch wheel for Linux bundles
# CUDA; bench_agcrn.py confirms the GPU is visible. If python3 is too old (stg
# needs >= 3.9), load a newer one first (`module avail python`) and rerun.
# Jobs find the venv through the same VENV variable (see smoke_gpu.sbatch).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-cluster}

python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir numpy polars scikit-learn torch
pip install --no-cache-dir -e stg_infra
mkdir -p cluster/logs
# login nodes cap process memory, so importing torch can fail here with "failed
# to map segment"; that is not an install error. The smoke job checks torch on
# a GPU node.
python -c "import stg; print('stg ok')"
echo "venv: $VENV"
