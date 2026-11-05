#!/bin/bash
# One-command refresh of the Section 6.5 results after new games are recorded (Mac only; cached games are skipped).
#   ./refresh_results.sh            full refresh: pull new games, main replay, variants, baselines, tables
#   ./refresh_results.sh tables     only rebuild thesis/generated/results_tables.tex from existing replays
# Games whose live runner had no contract list (NBA, 23 Oct style) are listed at the end: backfill them with
#   python3 analysis/graph_2026_10/b59_backfill_inputs.py apply GAME_ID...   then rerun this script.
# Memory: peaks about 2 GiB per worker; WORKERS (default 2) and the 12 GiB free-disk check bound the load. Do not run on a live host.
set -e
cd "$(dirname "$0")/analysis/graph_2026_10"
PY=/usr/local/bin/python3
if [ "$1" != "tables" ]; then
  free=$(df -g / | awk 'NR==2{print $4}'); [ "$free" -ge ${MIN_FREE:-12} ] || { echo "only ${free} GiB free; clean up first (see reports/results_rebuild_log.md)"; exit 1; }
  export SIZING=20x OMP_NUM_THREADS=1 WORKERS=${WORKERS:-2}
  # 1. main run: downloads new games from the sandbox and VM, replays the games not yet in out/b55_main_20x
  OUT_NAME=b55_main_20x ARMS=host $PY b49_event_replay_v2.py --monitor | grep -E '^\{"state": "(completed|failed)' || true
  # 2. games where a safeguard acted, then the guard-off and late-block-off arms on them
  $PY b63_variant_lists.py
  bash b58_run_variants.sh
  # 3. baselines on every scored game
  bash b62_run_baselines.sh
fi
# 4. tables
$PY b61_make_tables.py
echo "excluded games still waiting for inputs:"
$PY - <<'P'
import json
for e in json.load(open('out/b55_main_20x/inventory.json'))['excluded']:
    if e.get('reason') != 'preseason':
        print(' ', e['date'], e['league'], e['game'], '-', e['reason'])
P
