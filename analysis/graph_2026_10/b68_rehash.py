"""Re-key cached replay results after a source change that cannot affect them.

b49 caches each game's result.json under rule_hash, a digest over b49, event_execution, live_trade, live_model and the weights.
Removing the baseball 15-60 s skip window (31 Oct) edited live_trade.py, so every cached result looks stale, although basketball
and football never used the window (SKIP_AGE was None for them) and the baselines never call the engine's skip logic.
After verifying on one basketball and one football game that a fresh replay reproduces the cached numbers (b67_verify), this
rewrites rule_hash of those cached results to the current value. Baseball results are never re-keyed: they are recomputed.

  b68_rehash.py DIR[,DIR...] [--arms host|baselines]
"""
import hashlib
import json
import sys
from pathlib import Path

import b49_event_replay_v2 as BB

B = BB.B
arms = ('host',) if '--arms' not in sys.argv or sys.argv[sys.argv.index('--arms') + 1] == 'host' else tuple(f'{b}_{t}' for b in ('sym', 'micro', 'flow') for t in ('clock', 'model'))
rule = dict(arms=arms, sizing_mode=BB.SIZING, sizing='round_half_up(20 * live_trade.size_of(signal,width),0.01)', models=B.MODELS, latency_seconds=BB.LATENCY,
            queue='exchange-time book; no advancement from cancellations; through prints fill in full; prints at limit consumed once; whole-cent snap; Kalshi fee ledger',
            sources={str(p.relative_to(B.ROOT)): B.digest(p) for p in [Path(BB.__file__), B.ROOT / 'scripts/event_execution.py', B.ROOT / 'scripts/live_trade.py', B.ROOT / 'scripts/live_model.py']})
rule['model_weights'] = {str(p.relative_to(B.ROOT)): B.digest(p) for p in (B.ROOT / 'results/ingame/deploy').glob('*.pt') if any(name in p.name for name in B.MODELS.values())}
new = hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()
print('current rule_hash', new[:12], 'arms', arms)
n = 0
for d in sys.argv[1].split(','):
    for f in sorted((Path(__file__).with_name('out') / d).glob('games/*/*/result.json')):
        r = json.loads(f.read_text())
        if r.get('league') == 'MLB' or r.get('rule_hash') == new:
            continue
        r['rule_hash'] = new
        f.write_text(json.dumps(r, indent=2, default=float) + "\n")
        n += 1
print('re-keyed', n)
