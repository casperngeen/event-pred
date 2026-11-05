"""Simple market-making baselines in the b49 event replay (same simulator, fees, 60 s entry / 120 s exit, 20x sizing rule).

No feed information, no training, no tuning: each rule reads only the order book or prints the engine already holds.

  sym     symmetric quote: bid at ref-1c and offer at ref+1c on alternating clock ticks (Avellaneda-Stoikov with zero skew)
  micro   micro-price skew (Stoikov 2018): quote on the side the size-weighted micro-price leans towards
  flow    trade-flow skew (Cont-Kukanov-Stoikov 2014 style): quote with the sign of net taker volume of the last 60 s
  clock every 60 s on every modelled contract, no stalled-feed guard, no late-game block
  model only at the (contract, time) pairs where the main model placed an order (isolates the signal from the timing)

Entry price: ref -/+ 1c exactly as the model. Size: 1 contract per order (the forecast sets no size here), so per-contract
metrics are the comparable ones. The forecaster is replaced by a stub (the baselines do not read it) which makes the run fast.
Run like b49: OUT_NAME, GAMES, WORKERS env (ARMS selects a subset of the six), --local-only; no --monitor.
"""
import gzip
import hashlib
import json
import math
import os
import sys
from functools import lru_cache

import numpy as np

import b49_event_replay_v2 as BB
import b64_fast_stream  # noqa: F401  (orjson loader, identical stream)

TAGS = [f'{b}_{t}' for b in ('sym', 'micro', 'flow') for t in ('clock', 'model')]
for _t in TAGS:
    BB.ARM_SPECS[_t] = ('host', {})
BB.ARMS = tuple(os.environ.get('ARMS', ','.join(TAGS)).split(','))   # one replay, all six rules as arms (the book stream is shared)
T = BB.B.T
SIZE = 1.0
MAIN = BB.Path(__file__).with_name('out') / os.environ.get('MAIN_NAME', 'b55_main_20x')


class Stub:
    horizons = ['120s']

    def __init__(self, league):
        pass

    def forecast(self, steps, px, pt, pstate, L, delay):
        return [dict(r, t=s['t'], k=s['k'], plays_known=0, delta=np.zeros(1), level={'120s': r['anchor']})
                for s in steps for r in s['rows']]


BB.B.SharedForecaster = Stub


class _Json:
    """json with a one-entry cache on loads: b49 parses every feed document once per arm, and the six arms read the same bytes.
    The baselines only read the parsed document (they build the play list and the first-play time from it)."""
    def __init__(self):
        self.key = self.val = None

    def loads(self, s, *a, **kw):
        if a or kw or not isinstance(s, (bytes, str)):
            return json.loads(s, *a, **kw)
        if self.key is None or self.key != s:
            self.key, self.val = s, json.loads(s)
        return self.val

    def __getattr__(self, name):
        return getattr(json, name)


BB.json = _Json()


@lru_cache(maxsize=None)
def recorded_model_times(day):
    out = set()
    with gzip.open(day / 'host/decisions.jsonl.gz', 'rt') as f:
        for line in f:
            if '"order"' not in line:
                continue
            for d in (json.loads(line) if line.lstrip().startswith('[') else [json.loads(line)]):
                if isinstance(d, dict) and d.get('action') == 'order':
                    out.add((d['ticker'], round(d['t'], 1)))
    return out


def model_times(eng):
    """(ticker, t) pairs of the selected main run's orders for this game."""
    g = eng.g
    day = next(MAIN.glob(f"games/*/{g['league']}_{g['game_id']}"), None)
    return recorded_model_times(day) if day else set()


def side_of(eng, base, tk, p, n, t, k):
    if base == 'sym':
        return 1 if k % 2 == 0 else -1
    if base == 'micro':
        lv = eng.book.levels.get(tk)
        if not lv or not lv['yes'] or not lv['no']:
            return 0
        by, bn = max(lv['yes']), max(lv['no'])
        qy, qn = lv['yes'][by], lv['no'][bn]
        bid, ask = by / 100, 100 - bn / 100
        micro = (bid * qn + ask * qy) / (qy + qn)          # a thin ask side pulls the micro-price towards the ask
        s = micro - (bid + ask) / 2
        return int(np.sign(round(s, 6)))
    w = (p['t'][:n] > t - 60)
    return int(np.sign(float((p['n'][:n][w] * np.where(p['yes'][:n][w], 1, -1)).sum())))


def decide(self, t, now, rows):
    base, timing = self.arm.split('_')
    if not hasattr(self, '_b0'):
        self._b0, self._mt = t, (model_times(self) if timing == 'model' else None)
    k = int(round((t - self._b0) / T.LM.STEP))
    if timing == 'clock' and k % 12:
        return
    out = []
    for r in rows:
        tk = r['ticker']
        if r['kind'] not in self.kinds:
            continue
        if timing == 'model' and (tk, round(t, 1)) not in self._mt:
            continue
        p = self.mk.p[tk]
        n = int(np.searchsorted(p['t'], t, side='right'))
        d = dict(t=t, ticker=tk, kind=r['kind'], level=r['level'][T.HORIZON])
        if not n or t - p['t'][n - 1] > T.REF_MAX_S:
            continue
        ref = float(p['px'][n - 1])
        side = side_of(self, base, tk, p, n, t, k // 12)
        lim = ref - T.OFFSET if side == 1 else ref + T.OFFSET
        if side == 0 or not 0 < lim < 100 or self.book.is_busy(tk):
            continue
        self.book.place(d | dict(action='order', side=side, limit=lim, size=SIZE, signal=0., ref=ref))
        out.append(d | dict(action='order', side=side, limit=lim, size=SIZE))
    if out:
        self.log('decisions.jsonl', out)


_init = T.Engine.__init__


def init(self, g, fc, out, *a, **kw):
    _init(self, g, fc, out, *a, **kw)
    self.arm = BB.Path(out).name                       # b49 names each arm's directory by its tag


T.Engine.__init__ = init
T.Engine._decide = decide

_run = BB.run


def run_with_model_times(entry, rule_hash):
    day = next(MAIN.glob(f"games/*/{entry['league']}_{entry['game']}"))
    signature = hashlib.sha256(json.dumps(sorted(recorded_model_times(day))).encode()).hexdigest()
    code = hashlib.sha256(BB.Path(__file__).read_bytes()).hexdigest()
    effective = hashlib.sha256((rule_hash+signature+code).encode()).hexdigest()
    result = _run(entry, effective)
    result.update(model_times_sha256=signature, baseline_code_sha256=code)
    BB.write(BB.OUT/'games'/entry['date']/f"{entry['league']}_{entry['game']}"/'result.json', result)
    return result


BB.run = run_with_model_times

if __name__ == '__main__':
    sys.exit(BB.main())
