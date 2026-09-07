"""Buy and hold to settlement instead of round-tripping.

Round-tripping pays the spread twice and the taker fee twice. Holding to
settlement pays entry only — Kalshi charges no settlement fee — and captures
the FULL repricing to the realised outcome rather than a k-day slice of it.
Both effects favour holding, so the earlier round-trip numbers understate it.

What changes is the payoff, not just the cost: PnL is no longer a price move,
it is (100 if the contract resolves in your favour else 0) minus what you paid.
The signal must therefore predict the OUTCOME, not the 3-day drift.

  A  Total drift to settlement vs the 3-day drift the earlier tests used.
  B  Hold-to-expiry PnL on the 12 SPECS cells, one-way cost.
  C  Variance and capital: per-trade sd, holding period, Sharpe-per-bet.
  D  The unconditional benchmark — is the edge the signal, or just the trade?

Pre-2026 only. Run from event-pred/.
"""
from __future__ import annotations
import sys, math, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, '/Users/caspe2/NUS/FYP/analysis/exploratory_2026_08')
sys.path.insert(0, 'stg_infra')
from sign_diagnostics import surprises, reps_for, SPECS
from liquid_window import mk_all

HALF = 2.35          # measured effective half-spread paid (spread_measurement.py)
def fee(price_c):    # Kalshi taker fee on the contract you buy, at its own price
    p = np.asarray(price_c, float) / 100.0
    return 7.0 * p * (1 - p)
def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

results = (mk_all.select('ticker', 'result')
                 .filter(pl.col('result').is_in(['yes', 'no']))
                 .unique(subset=['ticker']))
RES = dict(zip(results['ticker'].to_list(), results['result'].to_list()))

# ------------------------------------------------------------- assemble ---
rows = []
for trig, tgt, side, sign in SPECS:
    surp = surprises(trig)
    if surp.height == 0: continue
    rep, tr = reps_for(tgt, side)
    reps = list(zip(rep['close'].to_list(), rep['rep'].to_list()))
    for r in surp.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r['close_time']]
        if not nxt: continue
        c, t = min(nxt, key=lambda x: x[0])
        gap = (c - r['close_time']).total_seconds() / 86400
        if not (0 <= gap <= 60): continue
        res = RES.get(t)
        if res is None: continue
        sub = tr.filter(pl.col('ticker') == t)
        pre = sub.filter(pl.col('created_time') <= r['close_time'])
        if pre.height == 0: continue
        p0 = float(pre['yes_price'][-1])
        post = sub.filter(pl.col('created_time') > r['close_time'])
        p3 = float(post['yes_price'][2]) if post.height >= 3 else None
        rows.append({'cell': f'{trig}->{tgt}/{side}', 'event': r['event'],
                     'ticker': t, 'S': float(r['surprise']) * sign, 'p0': p0,
                     'p3': p3, 'res': res, 'hold_days': gap,
                     'settle': 100.0 if res == 'yes' else 0.0})
d = pl.DataFrame(rows).filter(pl.col('S') != 0)
print(f'observations: {d.height}   cells: {d["cell"].n_unique()}   '
      f'events: {d["event"].n_unique()}')

# ------------------------------------------------------- A drift sizes ----
rule('A  HOW MUCH REPRICING IS THERE, 3 DAYS vs ALL THE WAY TO SETTLEMENT?')
dd = d.drop_nulls('p3')
d3 = (dd['p3'] - dd['p0']).to_numpy()
dt_ = (d['settle'] - d['p0']).to_numpy()
print(f'3-trade drift   : n={d3.size:<4} E|move| {np.abs(d3).mean():6.2f}c   '
      f'mean {d3.mean():+.2f}c')
print(f'to settlement   : n={dt_.size:<4} E|move| {np.abs(dt_).mean():6.2f}c   '
      f'mean {dt_.mean():+.2f}c')
print(f'\nHolding to settlement moves {np.abs(dt_).mean()/np.abs(d3).mean():.1f}x further. '
      f'That is the upside\nof the idea: far more to capture, and you pay the spread once.')
print(f'median holding period: {d["hold_days"].median():.0f} days  '
      f'(p90 {d["hold_days"].quantile(.9):.0f})')

# ----------------------------------------------------- B hold-to-expiry ---
rule('B  HOLD-TO-EXPIRY PnL on the theory-signed cells')
print('Signal says up -> buy Yes at p0+half; says down -> buy No at (100-p0)+half.')
print('Cost = entry price + half-spread + taker fee on that price. No exit cost.\n')
sgn = np.sign(d['S'].to_numpy())
p0 = d['p0'].to_numpy(); settle = d['settle'].to_numpy()
entry = np.where(sgn > 0, p0 + HALF, (100 - p0) + HALF)
payoff = np.where(sgn > 0, settle, 100 - settle)
pnl = payoff - entry - fee(entry)
pnl_nocost = payoff - np.where(sgn > 0, p0, 100 - p0)
win = payoff > 0
print(f'{"":<26}{"n":>5}{"win%":>7}{"mean PnL":>11}{"se":>8}{"t":>7}')
def line(nm, v):
    se = v.std(ddof=1) / math.sqrt(len(v))
    print(f'{nm:<26}{len(v):>5}{100*win.mean() if len(v)==len(win) else float("nan"):>7.1f}'
          f'{v.mean():>11.2f}{se:>8.2f}{v.mean()/se:>7.2f}')
line('before costs', pnl_nocost)
line('after entry cost', pnl)
rt = pnl - HALF - fee(entry)          # what a round trip would have cost instead
line('(same, round-tripped)', rt)
print(f'\nper-cell, after entry cost:')
print(f'{"cell":<26}{"n":>4}{"win%":>7}{"mean PnL":>11}{"se":>8}')
dc = d.with_columns(pl.Series('pnl', pnl), pl.Series('win', win))
for cell in sorted(dc['cell'].unique().to_list()):
    s = dc.filter(pl.col('cell') == cell)
    if s.height < 8: continue
    v = s['pnl'].to_numpy()
    print(f'{cell:<26}{s.height:>4}{100*s["win"].mean():>7.1f}{v.mean():>11.2f}'
          f'{v.std(ddof=1)/math.sqrt(len(v)):>8.2f}')

# ------------------------------------------------------- C variance -------
rule('C  VARIANCE AND CAPITAL — what a binary payoff costs you in precision')
sd = pnl.std(ddof=1)
edge = pnl_nocost.mean()          # signal content before execution cost
print(f'per-trade sd            : {sd:.1f}c   (vs {np.nanstd(d3):.1f}c for the 3-day drift trade)')
print(f'signal content (pre-cost): {edge:+.2f}c   info ratio {edge/sd:.3f}')
print(f'trades to detect that at t=2 : {(2*sd/max(abs(edge),1e-9))**2:,.0f}')
print(f'observations available       : {d.height} in four years of in-sample data')
print(f'after entry cost the mean is {pnl.mean():+.2f}c, so there is no positive')
print('edge left to size a sample against.')
cap = d['hold_days'].mean() / 365
print(f'\nmean holding period {d["hold_days"].mean():.0f} days => {365/max(d["hold_days"].mean(),1):.1f} '
      f'turns/year on committed capital.')
print(f'mean entry price {entry.mean():.0f}c, so {entry.mean():.0f}c of capital is tied up '
      f'per contract\nto earn {pnl.mean():+.2f}c => {100*pnl.mean()/entry.mean()/max(cap,1e-9):+.1f}% annualised, before')
print('any allowance for position limits or the fact that these are not independent.')

# --------------------------------------------------- D the benchmark ------
rule('D  IS IT THE SIGNAL, OR JUST BUYING THE FAVOURITE?')
print('A hold-to-expiry rule can look profitable purely by systematically buying\n'
      'the cheap side of a mispriced-looking book. Controls:\n')
alt = {
    'theory-signed (the strategy)': sgn,
    'always buy Yes':               np.ones_like(sgn),
    'always buy No':                -np.ones_like(sgn),
    'buy the favourite (p0>50)':    np.where(p0 > 50, 1, -1),
    'buy the longshot (p0<50)':     np.where(p0 < 50, 1, -1),
    'sign flipped (falsification)': -sgn,
}
print(f'{"rule":<32}{"win%":>7}{"mean PnL":>11}{"se":>8}{"t":>7}')
for nm, s_ in alt.items():
    e = np.where(s_ > 0, p0 + HALF, (100 - p0) + HALF)
    pay = np.where(s_ > 0, settle, 100 - settle)
    v = pay - e - fee(e)
    se = v.std(ddof=1) / math.sqrt(len(v))
    print(f'{nm:<32}{100*(pay>0).mean():>7.1f}{v.mean():>11.2f}{se:>8.2f}{v.mean()/se:>7.2f}')
print('\nIf "always buy No" scores like the strategy, the edge is the base rate of\n'
      'these contracts resolving No, not cross-market information.')
base = (d['res'] == 'no').mean()
print(f'base rate: {100*base:.1f}% of these target contracts resolved NO.')
