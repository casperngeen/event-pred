"""Stage 1: relation discovery over the full trigger->target grid.

This is the CA report's actual contribution (S3.2: "no existing work has
characterised whether these cross-market belief updates are STRUCTURED"),
estimated directly rather than learned from prediction loss.

Why direct estimation rather than AGCRN's adaptive adjacency: an edge here
gets a point estimate, a standard error and an FDR-corrected p-value, so it
can be reported as a finding and tested out-of-sample. A loss-optimised
adjacency has none of those.

  1  Full ordered grid, sign/rank statistics, BH-FDR. Reports how many pairs
     were searched — the CA report's 22 relationships never state this.
  2  Permutation verification of the survivors.
  3  Mediation: is A->C direct, or explained by A->B->C? This is the part a
     pairwise sweep cannot do and a graph can — the actual case for graph
     structure over a list of edges.

Pre-2026 only. Run from event-pred/.
"""
from __future__ import annotations
import sys, math, itertools, json
import numpy as np, polars as pl

sys.path.insert(0, '/Users/caspe2/NUS/FYP/analysis/exploratory_2026_08')
sys.path.insert(0, 'stg_infra')
from liquid_window import mk_all, pdf_surprise
from sign_diagnostics import reps_for, spearman
from bucket_surprise import build_bucket_surprise

SCR = '/private/tmp/claude-503/-Users-caspe2-NUS-FYP/a09a383d-5b57-43d1-9091-34a28c3fb458/scratchpad'
MIN_N = 10
MAX_GAP = 60
Q = 0.10
RNG = np.random.default_rng(11)
def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

mm = pl.read_parquet(f'{SCR}/macro_markets.parquet')
SERIES = (mm.group_by('series').agg(pl.col('event_ticker').n_unique().alias('ev'))
            .filter(pl.col('ev') >= 5).sort('ev', descending=True)['series'].to_list())
BUCKET_SERIES = {'WTI', 'WTIW'}

_sc = {}
def surp(s):
    if s not in _sc:
        _sc[s] = (build_bucket_surprise(s, verbose=False) if s in BUCKET_SERIES
                  else pdf_surprise(s))
    return _sc[s]

TRIGGERS = [s for s in SERIES if surp(s) is not None and surp(s).height >= MIN_N]
print(f'trigger series with a usable surprise measure : {len(TRIGGERS)}')
print(f'target series                                 : {len(SERIES)}')

def responses(trig, tgt, side='any'):
    s = surp(trig)
    rep, tr = reps_for(tgt, side)
    reps = list(zip(rep['close'].to_list(), rep['rep'].to_list()))
    S, R, TG = [], [], []
    for r in s.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r['close_time']]
        if not nxt: continue
        c, t = min(nxt, key=lambda x: x[0])
        if not (0 <= (c - r['close_time']).total_seconds() / 86400 <= MAX_GAP): continue
        sub = tr.filter(pl.col('ticker') == t)
        pre = sub.filter(pl.col('created_time') <= r['close_time'])
        post = sub.filter(pl.col('created_time') > r['close_time'])
        if pre.height == 0 or post.height < 3: continue
        S.append(float(r['surprise'])); R.append(float(post['yes_price'][2]) - float(pre['yes_price'][-1]))
        TG.append(t)
    return np.array(S), np.array(R), TG

def rho_p(S, R):
    """Two-sided Spearman with asymptotic p. Direction is READ OFF, not imposed."""
    n = len(S)
    if n < MIN_N: return np.nan, np.nan, n
    r = spearman(S, R)
    if np.isnan(r) or abs(r) >= 1: return r, np.nan, n
    t = r * math.sqrt((n - 2) / (1 - r * r))
    from math import erf
    z = abs(t) / math.sqrt(2)
    p = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(1))))   # normal approx, n>=10
    return r, min(max(p, 0.0), 1.0), n

rule('1  FULL GRID — every ordered trigger->target pair')
grid = []
for trig in TRIGGERS:
    for tgt in SERIES:
        if tgt == trig: continue
        sides = ['hike', 'cut'] if tgt == 'FEDDECISION' else ['any']
        for side in sides:
            try:
                S, R, TG = responses(trig, tgt, side)
            except Exception:
                continue
            r, p, n = rho_p(S, R)
            if np.isnan(r) or np.isnan(p): continue
            grid.append({'trigger': trig, 'target': tgt, 'side': side, 'n': n,
                         'rho': r, 'p': p, 'n_targets': len(set(TG))})
g = pl.DataFrame(grid).sort('p')
print(f'\npairs SEARCHED (n>={MIN_N}) : {g.height}')
print(f'pairs with p<0.05 uncorrected : {(g["p"]<0.05).sum()}')
print(f'expected by chance at 5%      : {0.05*g.height:.1f}')

# Benjamini-Hochberg
p = g['p'].to_numpy(); m = len(p)
thresh = Q * (np.arange(1, m + 1)) / m
passing = np.where(p <= thresh)[0]
kmax = passing.max() + 1 if passing.size else 0
g = g.with_columns(pl.Series('bh_rank', np.arange(1, m + 1)),
                   pl.Series('bh_crit', thresh),
                   pl.Series('survives', np.arange(m) < kmax))
print(f'\nBenjamini-Hochberg at q={Q}: {kmax} edges survive')
print(f'\n{"trigger":<14}{"target":<14}{"side":<6}{"n":>5}{"rho":>8}{"p":>10}'
      f'{"BH crit":>10}{"targets":>9}')
for r in g.head(20).iter_rows(named=True):
    mark = ' *' if r['survives'] else ''
    print(f'{r["trigger"]:<14}{r["target"]:<14}{r["side"]:<6}{r["n"]:>5}{r["rho"]:>8.3f}'
          f'{r["p"]:>10.4f}{r["bh_crit"]:>10.4f}{r["n_targets"]:>9}{mark}')
g.write_parquet(f'{SCR}/structure_grid.parquet')

rule('2  PERMUTATION CHECK on the survivors')
surv = g.filter('survives')
if surv.height == 0:
    print('nothing survived BH; permutation check skipped.')
    print(f'\nFor reference the CA report claims 22 relationships. On this grid of')
    print(f'{g.height} searched pairs, BH at q={Q} supports {kmax}.')
else:
    print(f'{"edge":<32}{"rho":>8}{"perm p":>9}{"n":>5}')
    for r in surv.iter_rows(named=True):
        S, R, _ = responses(r['trigger'], r['target'], r['side'])
        obs = spearman(S, R); null = []
        for _ in range(2000):
            null.append(spearman(RNG.permutation(S), R))
        null = np.array(null)
        pp = float((np.abs(null) >= abs(obs)).mean())
        print(f'{r["trigger"]+"->"+r["target"]+"/"+r["side"]:<32}{obs:>8.3f}{pp:>9.4f}{len(S):>5}')

rule('3  MEDIATION — direct edge, or routed through a third market?')
print('A pairwise sweep returns a LIST of edges. The graph question is whether\n'
      'A->C survives once B is controlled for. Partial Spearman on events where\n'
      'both A and B surprises exist, matched to the same C contract.\n')

def joint(a, b, c, side='any'):
    """Surprises of A and B matched to the same next C contract."""
    sa, sb = surp(a), surp(b)
    rep, tr = reps_for(c, side)
    reps = list(zip(rep['close'].to_list(), rep['rep'].to_list()))
    def tag(s):
        out = {}
        for r in s.iter_rows(named=True):
            nxt = [(cc, t) for cc, t in reps if cc > r['close_time']]
            if not nxt: continue
            cc, t = min(nxt, key=lambda x: x[0])
            if not (0 <= (cc - r['close_time']).total_seconds() / 86400 <= MAX_GAP): continue
            sub = tr.filter(pl.col('ticker') == t)
            pre = sub.filter(pl.col('created_time') <= r['close_time'])
            post = sub.filter(pl.col('created_time') > r['close_time'])
            if pre.height == 0 or post.height < 3: continue
            out[t] = (float(r['surprise']),
                      float(post['yes_price'][2]) - float(pre['yes_price'][-1]))
        return out
    A, B = tag(sa), tag(sb)
    common = sorted(set(A) & set(B))
    if len(common) < MIN_N: return None
    return (np.array([A[t][0] for t in common]),
            np.array([B[t][0] for t in common]),
            np.array([A[t][1] for t in common]))

def partial(x, y, z):
    """Spearman(x,z) controlling for y, via ranks."""
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    X, Y, Z = rk(x), rk(y), rk(z)
    def resid(u, v):
        v0 = v - v.mean()
        return u - u.mean() - (np.dot(u - u.mean(), v0) / max(np.dot(v0, v0), 1e-9)) * v0
    rx, rz = resid(X, Y), resid(Z, Y)
    d = math.sqrt(np.dot(rx, rx) * np.dot(rz, rz))
    return float(np.dot(rx, rz) / d) if d > 0 else np.nan

TESTS = [('WTI', 'CPI', 'FED', 'any'), ('CPI', 'CPIYOY', 'FED', 'any'),
         ('PAYROLLS', 'U3', 'FED', 'any'), ('WTI', 'CPI', 'FEDDECISION', 'hike'),
         ('CPI', 'PAYROLLS', 'FED', 'any')]
print(f'{"A -> C  (controlling B)":<34}{"n":>5}{"rho(A,C)":>10}{"partial":>10}{"change":>9}')
for a, b, c, side in TESTS:
    j = joint(a, b, c, side)
    if j is None:
        print(f'{a+"->"+c+" | "+b:<34}{"--":>5}   insufficient overlap'); continue
    SA, SB, RC = j
    r0 = spearman(SA, RC); r1 = partial(SA, SB, RC)
    print(f'{a+"->"+c+" | "+b:<34}{len(SA):>5}{r0:>10.3f}{r1:>10.3f}{r1-r0:>+9.3f}')
print('\nA large drop means the apparent A->C edge is routed through B — i.e. the\n'
      'graph has genuine multi-hop structure rather than being a set of\n'
      'independent bilateral effects. That distinction is the case for a graph.')
