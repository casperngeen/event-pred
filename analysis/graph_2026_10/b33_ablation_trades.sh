#!/bin/bash
# Trade logs of one ablation version under each league's frozen decision rule (test folds), from the seed-averaged
# forecasts that cluster/average_seeds.py wrote and that were copied to results/ingame/ens5/. One fold at a time
# (memory). The rule: every 5 s, one order or position per contract at a time; MLB also skips decisions 15-60 s
# after the last known pitch. Logs: out/b7_maker_trades_<L>_d<delay><tag>_ens5[_none]_dec5_one[_skip15-60]_fold<k>.parquet
# (a few MB). --delete removes the version's forecast files afterwards (~0.1-0.4 GB per fold): the table only
# needs the logs and the .json files.
#     analysis/graph_2026_10/b33_ablation_trades.sh <league> <memory> <delay> <tag> [--delete]
#     analysis/graph_2026_10/b33_ablation_trades.sh NBA gru 10 _h30-60-120-1800_tp_r0
#     analysis/graph_2026_10/b33_ablation_trades.sh MLB none 10 _h30-60-120-1800_tp_pitch --delete
# <tag> is the run name after d<delay> and before _seed<s>, as cluster/train_ingame.py wrote it.
# Then: venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py
set -euo pipefail
cd "$(dirname "$0")/../.."
L=$1; MEM=$2; D=$3; TAG=$4; DEL=${5:-}
case $L in
  MLB) PANEL=data/sports/derived/ingame/MLB_pitch/panel_v3; RULE=(SKIP_AGE=15,60);;
  NBA|NFL) PANEL=data/sports/derived/ingame/$L/panel_v3; RULE=();;
  *) echo "league: MLB, NBA or NFL"; exit 1;;
esac
ENS=results/ingame/ens5
for k in 1 2 3 4 5; do
  f=$ENS/${L}_fold${k}_${MEM}_d${D}${TAG}_ens5_seed0.npz
  [ -f "$f" ] || { echo "missing $f (copy it from the cluster first)"; exit 1; }
done
for k in 1 2 3 4 5; do
  env MEMORY="$MEM" RUN_TAG="${TAG}_ens5" DECIDE_S=5 ONE_AT_A_TIME=1 ROW_KINDS=spread,total FOLD=$k ${RULE[@]+"${RULE[@]}"} \
    venv/bin/python -W ignore analysis/graph_2026_10/b7_maker_trades.py $ENS "$L" "$D" "$PANEL"
done
if [ "$DEL" = --delete ] && [ "$MEM" = gru ] && [ "$D" = 10 ] && { [ "$TAG" = _h30-60-120-1800_tp ] || [ "$TAG" = _h30-60-120-1800_tp_pitch ]; }; then
  echo "not deleting: these are the frozen model's forecasts"; DEL=
fi
if [ "$DEL" = --delete ]; then
  rm -f $ENS/${L}_fold[1-5]_${MEM}_d${D}${TAG}_ens5_seed0.npz
  echo "deleted the forecast files of ${L} ${MEM} d${D}${TAG} (the .json files stay)"
fi
