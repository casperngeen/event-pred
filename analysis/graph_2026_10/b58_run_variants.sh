#!/bin/bash
# Sequential guard-off then late-block-off replays on the Mac: 2 workers (~3 GB), disk checked before each run.
cd "$(dirname "$0")" || exit 1
for v in noguard nolate; do
  free=$(df -g / | awk 'NR==2{print $4}')
  if [ "$free" -lt ${MIN_FREE:-12} ]; then echo "stop: only ${free} GiB free before $v"; exit 1; fi
  echo "$(date -u +%H:%M:%S) start $v (free ${free} GiB)"
  VARIANT=$v OUT_NAME=b57_$v ARMS=host SIZING=20x OMP_NUM_THREADS=1 WORKERS=${WORKERS:-2} GAMES=$(cat out/b57_lists/${v}_games.txt) \
    /usr/bin/time -l /usr/local/bin/python3 b57_variant_replay.py --local-only 2>&1 | grep -E "^\{|maximum resident|real|Error" 
  echo "$(date -u +%H:%M:%S) done $v"
done
