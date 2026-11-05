#!/bin/bash
# Run the six baseline arms (b60) on every game scored in the main run that has no baseline result yet.
cd "$(dirname "$0")"
free_gib=$(df -g / | awk 'NR==2{print $4}')
[ "$free_gib" -ge ${MIN_FREE:-12} ] || { echo "only $free_gib GiB free"; exit 1; }
GAMES=$(/usr/local/bin/python3 - <<'P'
import glob, json, os
done = {os.path.basename(os.path.dirname(f)) for f in glob.glob('out/b60_baselines/games/*/*/result.json')}
# The six arms replay the whole book stream each (about 25-60 min per basketball or football game), so basketball and football use a systematic
# subsample: every STEP-th scored game in id order (all games now: STEP 1 everywhere; a typical football game takes about 3 minutes).
STEP = {'MLB': 1, 'NBA': 1, 'NFL': 1}
scored = {}
for f in sorted(glob.glob('out/b55_main_20x/games/*/*/result.json')):
    r = json.load(open(f))
    if r.get('results'):
        scored.setdefault(r['league'], []).append(r['game'])
ids = []
for lg, gs in scored.items():
    if lg in STEP:
        ids += [g for g in sorted(gs)[::STEP[lg]] if lg + '_' + g not in done]
print(','.join(ids))
P
)
[ -n "$GAMES" ] || { echo "nothing to run"; exit 0; }
OUT_NAME=b60_baselines SIZING=20x OMP_NUM_THREADS=1 WORKERS=${WORKERS:-2} GAMES=$GAMES \
  /usr/bin/time -l /usr/local/bin/python3 b60_baseline_replay.py --local-only
