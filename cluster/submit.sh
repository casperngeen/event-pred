#!/bin/bash
# Submit a GPU job on whichever modern GPU type has free capacity now (a100-40, a100-80, h100-47, h100-96,
# h200-141, in that order of preference), instead of always the a100-40 in the script's #SBATCH line. The
# cluster rejects an untyped "--gres=gpu:1", so a type is always named; with none free, it queues for the
# first one. The model is small and mostly CPU-bound, so every type gives the same results at about the same
# speed; the old "nv" cards are never picked.
#     cluster/submit.sh cluster/train_final.sbatch
#     LEAGUES="MLB" SEED=1 PACK=5 PANEL_SUFFIX=_pitch_v3 DELAYS="10" HORIZONS=30s,60s,120s,1800s \
#         cluster/submit.sh cluster/train_ingame.sbatch --array=0
#     GPU_TYPES="h100-96 a100-80" cluster/submit.sh ...    # your own order of preference
#     DRY_RUN=1 cluster/submit.sh ...                       # print the sbatch command only
# Environment variables (LEAGUES, SEED, ...) pass through to the job as with plain sbatch.
set -euo pipefail
[ $# -ge 1 ] || { echo "usage: cluster/submit.sh <script.sbatch> [sbatch options]"; exit 1; }
script=$1; shift
read -r -a prefer <<< "${GPU_TYPES:-a100-40 a100-80 h100-47 h100-96 h200-141}"
# one line per node of the gpu partition: "<gres> <state>", e.g. "gpu:a100-40:4(S:0-1) mix"
avail=$(sinfo -h -N -p "${PARTITION:-gpu}" -o "%G %t" 2>/dev/null || true)
pick=""
for t in "${prefer[@]}"; do
  if echo "$avail" | grep -E "gpu:${t}:[0-9]" | grep -qE " (idle|mix)$"; then pick=$t; break; fi
done
if [ -z "$pick" ]; then pick=${prefer[0]}; echo "no preferred GPU type has free capacity now; queueing for $pick"; fi
cmd=(sbatch --gres="gpu:${pick}:1" "$@" "$script")
echo "${cmd[*]}"
[ -n "${DRY_RUN:-}" ] || "${cmd[@]}"
