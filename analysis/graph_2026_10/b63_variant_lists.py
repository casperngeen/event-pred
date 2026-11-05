"""Games on which a safeguard acted in the main run (a skip decision of its kind), written to out/b57_lists/{noguard,nolate}_games.txt.
A variant arm only needs re-simulating on these games; elsewhere switching the safeguard off changes nothing."""
import glob
import gzip
import json
import os

GUARD = ('skip: feed', 'skip: ball in play', 's since the last known pitch', 'signal differs on real-time stamps', 'skip: review', 'skip: placeholder')
OUT = os.path.join(os.path.dirname(__file__), 'out')
lists = {'noguard': [], 'nolate': []}
for f in sorted(glob.glob(f'{OUT}/b55_main_20x/games/*/*/result.json')):
    r = json.load(open(f))
    if not r.get('results'):
        continue
    acts = set()
    with gzip.open(os.path.join(os.path.dirname(f), 'host/decisions.jsonl.gz'), 'rt') as h:
        for line in h:
            x = json.loads(line)
            for d in (x if isinstance(x, list) else [x]):
                a = d.get('action') if isinstance(d, dict) else None
                if isinstance(a, str) and a.startswith('skip'):
                    acts.add('late' if 'late game' in a else 'guard' if any(g in a for g in GUARD) else 'other')
    if 'guard' in acts:
        lists['noguard'].append(r['game'])
    if 'late' in acts and r['league'] == 'NBA':
        lists['nolate'].append(r['game'])
os.makedirs(f'{OUT}/b57_lists', exist_ok=True)
for k, v in lists.items():
    open(f'{OUT}/b57_lists/{k}_games.txt', 'w').write(','.join(v))
    print(k, len(v))
