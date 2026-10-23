#!/bin/bash
# Queue the in-game ablations of reports/ablation_runsheet.md as a few big array jobs: one per version for NBA + NFL
# together (same panel suffix), one per version for MLB, and the feed-delay sweeps split into a few tasks. Every
# version is 5 test folds x 5 seeds per league. 23 array tasks in all, under the cluster's 32-job cap (MaxSubmit).
# From ~/event-pred on the cluster:
#     cluster/submit_ablations.sh                       # every step
#     STEPS="A1 A2 A3" cluster/submit_ablations.sh      # some steps (A1-A3, B1-B5, C)
#     DRY_RUN=1 cluster/submit_ablations.sh             # print the sbatch commands only
# Each task runs PACK trainings one after another (~1 h typical, 3 h limit). Runs whose .json exists are skipped
# (--skip-existing), so a task that hits the time limit or fails is redone by rerunning the same command.
# When a version's runs have all finished: cluster/average_ablations.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
HZ=30s,60s,120s,1800s
ALL="A1 A2 A3 B1 B2 B3 B4 B5 C"
for step in ${STEPS:-$ALL}; do
  MEM=gru; EXTRA=""
  case $step in
    A1) EXTRA="--rounds 0";;
    A2) MEM=none;;
    A3) MEM=none; EXTRA="--rounds 0";;
    B1) EXTRA="--drop play_type";;
    B2) EXTRA="--drop wp";;
    B3) EXTRA="--drop baseline";;
    B4) EXTRA="--drop prints";;
    B5) EXTRA="--drop quotes";;
    C) ;;
    *) echo "unknown step $step"; exit 1;;
  esac
  if [ "$step" = C ]; then
    # NBA + NFL at 0 / 30 / 45 / 60 s: 200 runs, 50 per task; MLB at 0 / 5 / 15 s: 75 runs, 25 per task
    LEAGUES="NBA NFL" PANEL_SUFFIX=_v3 DELAYS="0 30 45 60" SEEDS="0 1 2 3 4" PACK=50 HORIZONS=$HZ \
      cluster/submit.sh cluster/train_ingame.sbatch --array=0-3 --job-name=abl-C-nba-nfl
    LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 DELAYS="0 5 15" SEEDS="0 1 2 3 4" PACK=25 HORIZONS=$HZ \
      cluster/submit.sh cluster/train_ingame.sbatch --array=0-2 --job-name=abl-C-mlb
  else
    # NBA + NFL: 50 runs in one task; MLB: 25 runs in one task
    MEMORY=$MEM EXTRA_ARGS="$EXTRA" LEAGUES="NBA NFL" PANEL_SUFFIX=_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=50 HORIZONS=$HZ \
      cluster/submit.sh cluster/train_ingame.sbatch --array=0 --job-name=abl-$step-nba-nfl
    MEMORY=$MEM EXTRA_ARGS="$EXTRA" LEAGUES="MLB" PANEL_SUFFIX=_pitch_v3 DELAYS="10" SEEDS="0 1 2 3 4" PACK=25 HORIZONS=$HZ \
      cluster/submit.sh cluster/train_ingame.sbatch --array=0 --job-name=abl-$step-mlb
  fi
done
