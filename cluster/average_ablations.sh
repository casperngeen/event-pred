#!/bin/bash
# Average the seeds of every ablation version whose 25 runs (5 folds x 5 seeds) have all finished, on a CPU compute
# node, deleting the per-seed forecasts afterwards (cluster/average_seeds.py --delete-seeds). Versions already
# averaged or not finished are skipped, so this can be rerun as jobs finish. From ~/event-pred on the cluster:
#     cluster/average_ablations.sh
# Prints, per version and league, the run name to copy back to the laptop (reports/ablation_runsheet.md, step c).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${VENV:-.venv-cluster}/bin/python
R=results/ingame
mkdir -p $R/ens5
exec 9> $R/ens5/.average_ablations.lock      # one copy at a time: two would delete each other's input files
flock -n 9 || { echo "another copy of average_ablations.sh is running; wait for it (pgrep -af average_ablations)"; exit 1; }
name() {   # league memory delay rounds drop -> run name between fold<k>_ and _seed<s> (as cluster/train_ingame.py writes it)
  local t="_h30-60-120-1800_tp"
  [ "$4" = 0 ] && t+="_r0"
  [ "$1" = MLB ] && t+="_pitch"
  [ -n "$5" ] && t+="_no-$5"
  echo "$2_d$3$t"
}
versions() {   # league -> "memory delay rounds drop" per version
  echo "gru 10 0 -"; echo "none 10 2 -"; echo "none 10 0 -"
  for d in play_type wp baseline prints quotes; do echo "gru 10 2 $d"; done
  if [ "$1" = MLB ]; then for d in 0 5 15; do echo "gru $d 2 -"; done
  else for d in 0 30 45 60; do echo "gru $d 2 -"; done; fi
}
for L in NBA NFL MLB; do
  versions $L | while read -r mem delay rounds drop; do
    [ "$drop" = - ] && drop=""
    n=$(name $L $mem $delay $rounds "$drop")
    done_runs=$(find $R -maxdepth 1 -name "${L}_fold[1-5]_${n}_seed[0-4].json" | wc -l | tr -d ' ')
    averaged=$(find $R/ens5 -maxdepth 1 -name "${L}_fold[1-5]_${n}_ens5_seed0.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$averaged" -eq 5 ]; then
      echo "averaged   $L $n"
    elif [ "$done_runs" -lt 25 ]; then
      echo "running    $L $n ($done_runs/25 runs)"
    else
      srun -p normal --mem=16G --time=00:30:00 $PY cluster/average_seeds.py --league $L --name $n --delete-seeds < /dev/null
      echo "averaged   $L $n (now)"
    fi
  done
done
