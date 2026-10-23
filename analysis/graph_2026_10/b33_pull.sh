#!/bin/bash
# Laptop side of one ablation version (reports/ablation_runsheet.md): copy its seed-averaged forecasts and the seeds'
# .json files back from the cluster, simulate its trades for each league it covers (b33_ablation_trades.sh, which
# then deletes the forecasts), and rebuild the table. <name> is what cluster/average_ablations.sh printed:
#     analysis/graph_2026_10/b33_pull.sh gru_d10_h30-60-120-1800_tp_r0          # NBA and NFL
#     analysis/graph_2026_10/b33_pull.sh gru_d10_h30-60-120-1800_tp_r0_pitch    # MLB
#     DRY_RUN=1 analysis/graph_2026_10/b33_pull.sh <name>                        # print the steps only
set -euo pipefail
cd "$(dirname "$0")/../.."
N=$1
MEM=${N%%_d*}                                   # gru / none
REST=${N#*_d}                                   # <delay>_h...
D=${REST%%_h*}
TAG=_h${REST#*_h}
case $N in *_pitch*) LGS="MLB";; *) LGS="NBA NFL";; esac
run() { echo "+ $*"; [ -n "${DRY_RUN:-}" ] || "$@"; }
run rsync -av "xlogin:~/event-pred/results/ingame/ens5/*_${N}_ens5_seed0.*" results/ingame/ens5/
run rsync -av "xlogin:~/event-pred/results/ingame/*_${N}_seed[0-4].json" results/ingame/
for L in $LGS; do
  run analysis/graph_2026_10/b33_ablation_trades.sh "$L" "$MEM" "$D" "$TAG" --delete
done
run venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py
