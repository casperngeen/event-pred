"""Does adding WTI (702 events, now that bucket contracts parse) change anything?

WTI was the highest-n trigger in the CA report's sweep (WTI->CPI, n=811) and was
invalidated by the threshold-parse bug. bucket_surprise.py restores it. This
asks whether it actually widens the EFFECTIVE universe or just the pair count.

  1  Pair count vs independent target count.
  2  Sign test with WTI cells added, block permutation.
  3  The E|R| perfect-foresight bound, widened.

Pre-2026 only. Run from event-pred/.
"""
from __future__ import annotations
import sys, math, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, '/Users/caspe2/NUS/FYP/analysis/exploratory_2026_08')
sys.path.insert(0, 'stg_infra')
from sign_diagnostics import surprises, reps_for, SPECS, spearman
from bucket_surprise import build_bucket_surprise

RNG = np.random.default_rng(7)
N_PERM = 5000
def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

# WTI surprise: oil dearer than expected -> inflationary -> hawkish
WTI_SPECS = [('WTI', 'FEDDECISION', 'hike', +1), ('WTI', 'FEDDECISION', 'cut', -1),
             ('WTI', 'FED', 'any', +1), ('WTI', 'CPI', 'any', +1)]
_wti = None
def surp_of(series):
    global _wti
    if series in ('WTI', 'WTIW'):
        if _wti is None:
            _wti = build_bucket_surprise(series, verbose=False)
        return _wti
    return surprises(series)

def cell(trig, tgt, side, sign, horizon='dormant'):
    s = surp_of(trig)
    if s.height == 0: return None
    rep, tr = reps_for(tgt, side)
    reps = list(zip(rep['close'].to_list(), rep['rep'].to_list()))
    ev, S, R, TGT = [], [], [], []
    for r in s.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r['close_time']]
        if not nxt: continue
        c, t = min(nxt, key=lambda x: x[0])
        gap = (c - r['close_time']).total_seconds() / 86400
        if not (0 <= gap <= 60): continue
        sub = tr.filter(pl.col('ticker') == t)
        pre = sub.filter(pl.col('created_time') <= r['close_time'])
        post = sub.filter(pl.col('created_time') > r['close_time'])
        if pre.height == 0 or post.height < 3: continue
        p0 = float(pre['yes_price'][-1]); p1 = float(post['yes_price'][2])
        ev.append(r['event']); S.append(float(r['surprise']) * sign)
        R.append(p1 - p0); TGT.append(t)
    if not ev: return None
    return ev, np.array(S), np.array(R), TGT

rule('1  PAIRS vs INDEPENDENT TARGETS')
print('WTI resolves weekly; FOMC meets 8x a year. Many WTI triggers therefore\n'
      'point at the SAME next target contract, which inflates pair counts\n'
      'without adding information — the n=811 problem in the CA report.\n')
print(f'{"cell":<26}{"pairs":>7}{"distinct targets":>18}{"pairs/target":>14}')
built = {}
for trig, tgt, side, sign in SPECS + WTI_SPECS:
    out = cell(trig, tgt, side, sign)
    if out is None or len(out[0]) < 8: continue
    built[(trig, tgt, side)] = out
    ev, S, R, TGT = out
    nt = len(set(TGT))
    tag = ' <- NEW' if trig == 'WTI' else ''
    print(f'{trig+"->"+tgt+"/"+side:<26}{len(ev):>7}{nt:>18}{len(ev)/nt:>14.1f}{tag}')

old = {k: v for k, v in built.items() if k[0] != 'WTI'}
new = {k: v for k, v in built.items() if k[0] == 'WTI'}
def tally(dd):
    pairs = sum(len(v[0]) for v in dd.values())
    tgts = len({t for v in dd.values() for t in v[3]})
    return pairs, tgts
po, to_ = tally(old); pn, tn = tally(new); pa, ta = tally(built)
print(f'\n  existing cells : {po:>5} pairs, {to_:>3} distinct target contracts')
print(f'  WTI cells      : {pn:>5} pairs, {tn:>3} distinct target contracts')
print(f'  combined       : {pa:>5} pairs, {ta:>3} distinct target contracts')
print(f'\n  pair count {"+%.0f%%" % (100*pn/max(po,1))}, but distinct targets '
      f'{"+%.0f%%" % (100*(ta-to_)/max(to_,1))}.')

rule('2  SIGN TEST WITH WTI ADDED (block permutation by release period)')
def pooled(datasets):
    hits = tot = 0; rhos = []
    for S, R in datasets:
        m = (S != 0) & (R != 0)
        hits += int((S[m] * R[m] > 0).sum()); tot += int(m.sum())
        rho = spearman(S, R)
        if not np.isnan(rho): rhos.append(rho)
    return (hits / tot if tot else np.nan), (float(np.mean(rhos)) if rhos else np.nan), tot

def period(e):
    p = e.replace('KX', '').split('-')
    return p[1] if len(p) > 1 else e

def run(dd, label):
    cells = [(k[0], v) for k, v in dd.items()]
    ds = [(v[1], v[2]) for _, v in cells]
    obs_s, obs_r, n = pooled(ds)
    per_trig = {}
    for trig, v in cells:
        per_trig.setdefault(trig, []).append(v)
    null_s, null_r = [], []
    for _ in range(N_PERM):
        pm = {}
        for trig, vs in per_trig.items():
            seen, vals = [], []
            for ev, S, R, _T in vs:
                for e, s_ in zip(ev, S):
                    if e not in seen: seen.append(e); vals.append(s_)
            pm[trig] = dict(zip(seen, RNG.permutation(vals)))
        pds = [(np.array([pm[trig][e] for e in v[0]]), v[2]) for trig, v in cells]
        s_, r_, _ = pooled(pds)
        null_s.append(s_); null_r.append(r_)
    null_s = np.array(null_s); null_r = np.array(null_r)
    print(f'{label:<22}n={n:>4}  sign {100*obs_s:.1f}%  rho {obs_r:+.3f}   '
          f'p(sign) {float((null_s>=obs_s).mean()):.4f}  p(rho) {float((null_r>=obs_r).mean()):.4f}')

print(f'{"universe":<22}{"":>6}')
run(old, 'existing cells'); run(built, 'with WTI added')
run(new, 'WTI cells only')

rule('3  PERFECT-FORESIGHT BOUND, WIDENED')
def bound(dd, label):
    R = np.concatenate([v[2] for v in dd.values()])
    Sg = np.concatenate([np.sign(v[1]) * v[2] for v in dd.values()])
    m, se = np.abs(R).mean(), np.abs(R).std(ddof=1) / math.sqrt(len(R))
    sm, sse = Sg.mean(), Sg.std(ddof=1) / math.sqrt(len(Sg))
    print(f'{label:<22}n={len(R):>4}  E|R| {m:5.2f}c  95% CI [{m-1.96*se:4.2f}, {m+1.96*se:4.2f}]'
          f'   signed edge {sm:+5.2f}c +-{1.96*sse:4.2f}')
    return m + 1.96 * se
print(f'{"universe":<22}')
b1 = bound(old, 'existing cells'); b2 = bound(built, 'with WTI added')
bound(new, 'WTI cells only')
print(f'\nfees alone, round trip: 3.50c')
for nm, b in (('existing', b1), ('widened', b2)):
    print(f'  {nm:<10} upper bound on perfect foresight {b:.2f}c -> '
          f'{"profitability EXCLUDED" if b < 3.50 else "not excluded"}')
