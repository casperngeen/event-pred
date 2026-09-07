"""Is the spread too wide to profit even when the signal is surprise-conditioned?

agcrn_complexity.py tested the UNCONDITIONAL universe: every active node at
every snapshot. research_summary.md S6.2 proposes something narrower — trade
only theory-specified trigger->target cells, sized by surprise. That is a
selected subsample and could carry a much larger edge, so the earlier result
does not settle it. This does.

  A  Conditional edge   — signed drift in the 12 SPECS cells vs. round-trip cost.
  B  Surprise scaling   — does edge grow with |surprise| (S6.2's prediction)?
  C  Oracle-in-cell     — perfect foresight on the SAME subsample.
  D  Execution modes    — taker/maker break-evens, and what maker fills cost.

Pre-2026 only. Run from event-pred/.
"""
from __future__ import annotations
import sys, math, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, '/Users/caspe2/NUS/FYP/analysis/exploratory_2026_08')
sys.path.insert(0, 'stg_infra')
from sign_diagnostics import build, SPECS

SPREAD_BOOK = 8.0     # research_log.md S7: order-book ground truth
SPREAD_TRADE = 1.0    # trade-bounce estimate (research_log S7 shows it is 8x low)
def fee(p_c):
    p = np.asarray(p_c, float) / 100.0
    return 7.0 * p * (1 - p)
FEE50 = float(fee(50))          # 1.75c, worst case (fees peak at p=0.5)

def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

# ------------------------------------------------------------- assemble ---
rows = []
for horizon in ('dormant', 'liquid'):
    for trig, tgt, side, sign in SPECS:
        out = build(trig, tgt, side, sign, horizon)
        if out is None:
            continue
        ev, S, R = out
        for e, s_, r_ in zip(ev, S, R):
            rows.append({'horizon': horizon, 'cell': f'{trig}->{tgt}/{side}',
                         'trigger': trig, 'event': e, 'S': float(s_), 'R': float(r_)})
d = pl.DataFrame(rows)
# directional PnL of the theory-signed rule, in cents per contract, pre-cost
d = d.with_columns((np.sign(pl.col('S')) * pl.col('R')).alias('pnl'))
print(f'cells built: {d["cell"].n_unique()}   observations: {d.height}')
print(d.group_by('horizon').agg(pl.len().alias('n'),
                                pl.col('event').n_unique().alias('events')))

# ------------------------------------------------------- A conditional ----
rule('A  CONDITIONAL EDGE — theory-signed trades in the 12 SPECS cells')
print('pnl = sign(aligned surprise) x target response, cents per contract,\n'
      'BEFORE costs. Costs: taker round trip = spread + 2 fees.\n')
print(f'{"horizon":<10}{"n":>5}{"hit%":>7}{"mean pnl":>10}{"se":>7}'
      f'{"E|R|":>7}{"net@1c":>9}{"net@8c":>9}')
for horizon in ('dormant', 'liquid'):
    s = d.filter(pl.col('horizon') == horizon)
    x = s['pnl'].to_numpy()
    hit = (x > 0).mean() * 100
    mu, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    eabs = np.abs(s['R'].to_numpy()).mean()
    print(f'{horizon:<10}{len(x):>5}{hit:>7.1f}{mu:>10.2f}{se:>7.2f}{eabs:>7.2f}'
          f'{mu-SPREAD_TRADE-2*FEE50:>9.2f}{mu-SPREAD_BOOK-2*FEE50:>9.2f}')

print(f'\nper-cell, dormant horizon (the one that survived permutation):')
print(f'{"cell":<26}{"n":>4}{"hit%":>7}{"mean pnl":>10}{"se":>7}{"net@8c":>9}')
sd_ = d.filter(pl.col('horizon') == 'dormant')
for cell in sorted(sd_['cell'].unique().to_list()):
    x = sd_.filter(pl.col('cell') == cell)['pnl'].to_numpy()
    if len(x) < 8: continue
    mu = x.mean(); se = x.std(ddof=1) / math.sqrt(len(x))
    print(f'{cell:<26}{len(x):>4}{(x>0).mean()*100:>7.1f}{mu:>10.2f}{se:>7.2f}'
          f'{mu-SPREAD_BOOK-2*FEE50:>9.2f}')

# ------------------------------------------------- B surprise scaling -----
rule('B  SURPRISE SCALING — does edge grow with |surprise|?  (S6.2 prediction)')
print('If drift magnitude scales with surprise magnitude, the top quantile is\n'
      'where a cost-aware strategy would concentrate. Quartiles of |S| WITHIN\n'
      'each cell, so cross-cell scale differences cannot drive the ordering.\n')
for horizon in ('dormant', 'liquid'):
    s = d.filter(pl.col('horizon') == horizon).with_columns(pl.col('S').abs().alias('aS'))
    s = s.with_columns(
        ((pl.col('aS').rank('average').over('cell') - 1)
         / pl.len().over('cell') * 4).floor().clip(0, 3).alias('q'))
    print(f'\n{horizon}:')
    print(f'{"|S| quartile":<14}{"n":>5}{"hit%":>7}{"mean pnl":>10}{"se":>7}'
          f'{"E|R|":>7}{"net@1c":>9}{"net@8c":>9}')
    for q in range(4):
        x = s.filter(pl.col('q') == q)
        if x.height == 0: continue
        v = x['pnl'].to_numpy()
        mu = v.mean(); se = v.std(ddof=1) / math.sqrt(len(v))
        eabs = np.abs(x['R'].to_numpy()).mean()
        print(f'{"Q"+str(q+1)+(" (largest)" if q==3 else ""):<14}{len(v):>5}'
              f'{(v>0).mean()*100:>7.1f}{mu:>10.2f}{se:>7.2f}{eabs:>7.2f}'
              f'{mu-SPREAD_TRADE-2*FEE50:>9.2f}{mu-SPREAD_BOOK-2*FEE50:>9.2f}')

# --------------------------------------------------- C oracle-in-cell -----
rule('C  ORACLE ON THE SAME SUBSAMPLE — ceiling for any surprise-based model')
print('Perfect foresight restricted to these cells: take |R| every time. This\n'
      'is the most any signal built on these trigger->target pairs can pay.\n')
print(f'{"horizon":<10}{"n":>5}{"E|R|":>8}{"net@1c":>9}{"net@8c":>9}'
      f'{"break-even spread":>20}')
for horizon in ('dormant', 'liquid'):
    s = d.filter(pl.col('horizon') == horizon)
    eabs = np.abs(s['R'].to_numpy()).mean()
    print(f'{horizon:<10}{s.height:>5}{eabs:>8.2f}{eabs-SPREAD_TRADE-2*FEE50:>9.2f}'
          f'{eabs-SPREAD_BOOK-2*FEE50:>9.2f}{eabs-2*FEE50:>20.2f}')
print('\n"break-even spread" = the widest round-trip spread a PERFECT forecaster\n'
      'could survive on these cells. Compare against the measured 8.0c.')

# ------------------------------------------------- D execution modes ------
rule('D  EXECUTION MODES — what each costs, and what edge each needs')
half = SPREAD_BOOK / 2
modes = {
    'taker in, taker out':  (SPREAD_BOOK, 2 * FEE50),
    'maker in, taker out':  (SPREAD_BOOK / 2 - half, FEE50),   # 0 spread net
    'maker in, maker out':  (-SPREAD_BOOK, 0.0),               # earn both halves
    'maker in, hold to settle': (-half, 0.0),
}
print(f'measured book spread {SPREAD_BOOK:.1f}c, half-spread {half:.1f}c, '
      f'taker fee {FEE50:.2f}c at p=50\n')
print(f'{"mode":<28}{"spread c":>10}{"fees c":>8}{"total c":>9}'
      f'{"edge needed":>13}')
for name, (sp, f_) in modes.items():
    tot = sp + f_
    print(f'{name:<28}{sp:>10.2f}{f_:>8.2f}{tot:>9.2f}{tot:>13.2f}')

x = d.filter(pl.col('horizon') == 'dormant')['pnl'].to_numpy()
mu = x.mean()
print(f'\nmeasured dormant edge: {mu:.2f}c/contract (se {x.std(ddof=1)/math.sqrt(len(x)):.2f})')
print(f'{"mode":<28}{"net c/contract":>16}{"verdict":>12}')
for name, (sp, f_) in modes.items():
    net = mu - sp - f_
    print(f'{name:<28}{net:>16.2f}{("PROFITABLE" if net > 0 else "loses"):>12}')

print('\nMaker fills are not free: research_log.md S7 measures fill probability on\n'
      'a resting order at the touch (same contract, 792 placements):')
FILL = {'1 hour': (0.016, 0.003), '1 day': (0.208, 0.119),
        '7 days': (0.549, 0.355), 'rest of window': (0.689, 0.639)}
print(f'\n{"wait":<16}{"P(fill) opt":>13}{"P(fill) cons":>14}'
      f'{"E[net] opt":>13}{"E[net] cons":>13}')
sp, f_ = modes['maker in, taker out']
for w, (o, c) in FILL.items():
    net = mu - sp - f_
    print(f'{w:<16}{o:>13.3f}{c:>14.3f}{net*o:>13.2f}{net*c:>13.2f}')
print('\nE[net] scales the per-fill edge by fill probability. It ignores adverse\n'
      'selection — resting orders fill preferentially when the market moves\n'
      'against you — so these are optimistic even at the conservative column.')
print(f'\nThe dormant signal is defined on the NEXT 3 TRADES after resolution.')
print(f'At 1 hour a resting order fills {FILL["1 hour"][0]*100:.1f}% / '
      f'{FILL["1 hour"][1]*100:.1f}% of the time; waiting 7 days to get filled\n'
      f'means the 3-trade horizon has long since passed.')
