#!/bin/bash
# The same grid as train_ingame.sbatch, run in sequence on this machine (Apple MPS or CPU): for each
# feed delay and memory variant, the five walk-forward folds. NFL takes minutes per task here; NBA /
# MLB are about 5x larger, so use the cluster for those.
#     bash cluster/run_ingame_local.sh NFL
#     DELAYS="0 15" MEMORIES="gru none" bash cluster/run_ingame_local.sh NFL
#     HORIZONS=30s,60s,120s DELAYS="0 5 10 15" bash cluster/run_ingame_local.sh NFL
#     PANEL=panel_v3 HORIZONS=30s,60s,120s DELAYS="0 10" bash cluster/run_ingame_local.sh NFL   (traded-price targets)
set -euo pipefail
cd "$(dirname "$0")/.."
LEAGUE=${1:-NFL}
for DELAY in ${DELAYS:-0 2 5 10 15 20}; do
  for MEMORY in ${MEMORIES:-gru}; do
    for FOLD in 1 2 3 4 5; do
      venv/bin/python -W ignore cluster/train_ingame.py --panel "data/sports/derived/ingame/${LEAGUE}/${PANEL:-panel_v2}" \
          --test-fold "$FOLD" --memory "$MEMORY" --delay "$DELAY" --save-preds \
          ${HORIZONS:+--horizons "$HORIZONS"} ${EXTRA_ARGS:-}
    done
  done
done
