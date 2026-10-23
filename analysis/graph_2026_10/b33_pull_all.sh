#!/bin/bash
# Laptop side of every ablation version at once (reports/ablation_runsheet.md), once cluster/average_ablations.sh
# prints "averaged" for all of them: one rsync for every version's seed-averaged forecasts and seeds' .json files,
# then b33_ablation_trades.sh for each version and league (which deletes the forecasts), then the table.
#     analysis/graph_2026_10/b33_pull_all.sh copy      # the rsync only (asks for the cluster password)
#     analysis/graph_2026_10/b33_pull_all.sh trades    # trades + table only, on what has been copied
#     analysis/graph_2026_10/b33_pull_all.sh           # both
# Needs ~11 GB free while copying; space comes back as each version is traded. One version: b33_pull.sh <name>.
set -euo pipefail
cd "$(dirname "$0")/../.."
T=h30-60-120-1800_tp
NAMES=(gru_d10_${T}_r0 none_d10_${T} none_d10_${T}_r0
       gru_d10_${T}_no-play_type gru_d10_${T}_no-wp gru_d10_${T}_no-baseline gru_d10_${T}_no-prints gru_d10_${T}_no-quotes
       gru_d0_${T} gru_d30_${T} gru_d45_${T} gru_d60_${T}
       gru_d10_${T}_r0_pitch none_d10_${T}_pitch none_d10_${T}_r0_pitch
       gru_d10_${T}_pitch_no-play_type gru_d10_${T}_pitch_no-wp gru_d10_${T}_pitch_no-baseline
       gru_d10_${T}_pitch_no-prints gru_d10_${T}_pitch_no-quotes
       gru_d0_${T}_pitch gru_d5_${T}_pitch gru_d15_${T}_pitch)
STEP=${1:-all}
if [ "$STEP" = copy ] || [ "$STEP" = all ]; then
  INC=(--include='ens5/')
  for n in "${NAMES[@]}"; do INC+=(--include="*_${n}_ens5_seed0.*" --include="*_${n}_seed[0-4].json"); done
  # keepalives: the jump host drops quiet connections; --partial keeps a cut-off file so a rerun resumes it
  rsync -av --partial -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10" "${INC[@]}" --exclude='*' "xlogin:~/event-pred/results/ingame/" results/ingame/
fi
if [ "$STEP" = trades ] || [ "$STEP" = all ]; then
  failed=()
  for n in "${NAMES[@]}"; do
    MEM=${n%%_d*}; R=${n#*_d}; D=${R%%_h*}; TAG=_h${R#*_h}
    case $n in *_pitch*) LGS="MLB";; *) LGS="NBA NFL";; esac
    for L in $LGS; do
      if ! ls results/ingame/ens5/${L}_fold[1-5]_${n}_ens5_seed0.npz >/dev/null 2>&1; then
        echo "== $L $n: no forecasts copied, skipped"; failed+=("$L $n (not copied)"); continue
      fi
      echo "== $L $n"
      analysis/graph_2026_10/b33_ablation_trades.sh "$L" "$MEM" "$D" "$TAG" --delete || failed+=("$L $n")
    done
  done
  venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py
  if [ ${#failed[@]} -gt 0 ]; then printf 'FAILED: %s\n' "${failed[@]}"; else echo "all versions traded"; fi
fi
