"""Where on the ladder does the edge live, and where does the cost live?

Earlier tests traded ONE strike per event: the ATM contract (agcrn_complexity)
or the most-traded one (surprise_cost_test). ATM is the single most expensive
strike on the ladder — Kalshi's fee is 7c x p(1-p), maximised exactly at 50c.
This evaluates every sufficiently-traded strike instead, bucketed by entry
price, on the hold-to-settlement payoff.

Pre-2026 only. Run from event-pred/.
"""
from __future__ import annotations
import sys, math, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, '/Users/caspe2/NUS/FYP/analysis/exploratory_2026_08')
sys.path.insert(0, 'stg_infra')
from sign_diagnostics import surprises, side_expr, SPECS
from liquid_window import mk_all, tr_all

HALF = 2.35
def fee(p): 
    p = np.asarray(p, float) / 100.0
    return 7.0 * p * (1 - p)
def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

# every strike of every FEDDECISION / FED event, with its result
tgt_specs = {(t, s): sg for tr_, t, s, sg in SPECS for tr_ in [tr_]}
mk = (mk_all.filter(pl.col('series').is_in(['FEDDECISION', 'FED']))
            .filter(pl.col('close_time').is_not_null())
            .filter(pl.col('result').is_in(['yes', 'no']))
            .with_columns(side_expr()))
tk = mk['ticker'].unique().to_list()
tc = (tr_all.filter(pl.col('ticker').is_in(tk)).group_by('ticker')
            .agg(pl.len().alias('ntr')).collect())
mk = mk.join(tc, on='ticker', how='left').filter(pl.col('ntr') >= 5)
trades = (tr_all.filter(pl.col('ticker').is_in(mk['ticker'].unique().to_list()))
                .select('ticker', 'yes_price', 'created_time')
                .sort(['ticker', 'created_time']).collect())
print(f'strikes with >=5 trades: {mk.height} across {mk["event_ticker"].n_unique()} events')

by_ticker = {(t[0] if isinstance(t, tuple) else t): g
              for t, g in trades.group_by('ticker')}
meta = mk.select('ticker', 'event_ticker', 'series', 'side', 'result', 'close_time').to_dicts()
ev_close = {}
for r in meta:
    ev_close.setdefault(r['event_ticker'], r['close_time'])

rows = []
for trig, tgt, side, sign in SPECS:
    surp = surprises(trig)
    if surp.height == 0: continue
    cands = [r for r in meta if r['series'] == tgt and (side == 'any' or r['side'] == side)]
    ev_sorted = sorted({(ev_close[r['event_ticker']], r['event_ticker']) for r in cands})
    for s in surp.iter_rows(named=True):
        nxt = [(c, e) for c, e in ev_sorted if c > s['close_time']]
        if not nxt: continue
        c, ev = min(nxt, key=lambda x: x[0])
        gap = (c - s['close_time']).total_seconds() / 86400
        if not (0 <= gap <= 60): continue
        for r in [x for x in cands if x['event_ticker'] == ev]:
            g = by_ticker.get(r['ticker'])
            if g is None: continue
            pre = g.filter(pl.col('created_time') <= s['close_time'])
            if pre.height == 0: continue
            p0 = float(pre['yes_price'][-1])
            if not (1 <= p0 <= 99): continue
            rows.append({'cell': f'{trig}->{tgt}/{side}', 'ticker': r['ticker'],
                         'event': ev, 'S': float(s['surprise']) * sign, 'p0': p0,
                         'settle': 100.0 if r['result'] == 'yes' else 0.0,
                         'hold': gap})
d = pl.DataFrame(rows).filter(pl.col('S') != 0).unique(subset=['cell', 'ticker', 'event'])
print(f'strike-level observations: {d.height}   (vs 368 one-strike-per-event)')

sgn = np.sign(d['S'].to_numpy()); p0 = d['p0'].to_numpy(); settle = d['settle'].to_numpy()
entry = np.where(sgn > 0, p0, 100 - p0)
payoff = np.where(sgn > 0, settle, 100 - settle)
gross = payoff - entry                       # before any execution cost
cost_taker = HALF + fee(entry)               # hold to settlement: entry only
cost_maker = -HALF                           # rest at the bid, no fee, if filled
d = d.with_columns(pl.Series('gross', gross), pl.Series('entry', entry),
                   pl.Series('net_t', gross - cost_taker),
                   pl.Series('net_m', gross - cost_maker),
                   pl.Series('fee', fee(entry)))

rule('EDGE BY MONEYNESS — sign-invariant bucketing')
print('Bucketing by ENTRY price is circular: the signal picks the side, the side\n'
      'sets the entry price, and high-priced contracts mechanically win more.\n'
      'Bucket by |p0-50| instead, which does not move when the sign flips, and\n'
      'read the edge as (theory-signed - sign-flipped)/2 within each bucket.\n')
dist = np.abs(p0 - 50)
d = d.with_columns(pl.Series('dist', dist))
print(f'{"|p0-50|":<12}{"n":>6}{"signed":>9}{"flipped":>9}{"edge":>8}{"se":>7}'
      f'{"fee":>7}{"net taker":>11}')
dbk = (pl.when(pl.col('dist') < 10).then(pl.lit('0 0-10 (ATM)'))
         .when(pl.col('dist') < 25).then(pl.lit('1 10-25'))
         .when(pl.col('dist') < 40).then(pl.lit('2 25-40'))
         .otherwise(pl.lit('3 40+ (tails)')).alias('dbk'))
dd = d.with_columns(dbk)
for b in sorted(dd['dbk'].unique().to_list()):
    m = (dd['dbk'] == b).to_numpy()
    g = gross[m]
    # flipping the sign negates gross for the same observation set
    edge = g.mean()                     # = (signed - flipped)/2 by construction
    se = g.std(ddof=1) / math.sqrt(m.sum()) if m.sum() > 1 else float('nan')
    f_ = fee(entry[m]).mean()
    print(f'{b[2:]:<12}{m.sum():>6}{g.mean():>9.2f}{-g.mean():>9.2f}{edge:>8.2f}'
          f'{se:>7.2f}{f_:>7.2f}{(g - HALF - fee(entry[m])).mean():>11.2f}')
print('\nsigned and flipped are exact mirrors, so "edge" is just the signed mean —\n'
      'the point is that it is NOT the win rate, which the entry-price cut inflates.')

rule('SELECTIVITY — trade only the strongest surprises')
print('|S| percentile cutoff, all strikes pooled:\n')
print(f'{"keep top":<12}{"n":>6}{"gross":>9}{"se":>7}{"net taker":>11}{"net maker":>11}')
aS = np.abs(d['S'].to_numpy())
for q in (100, 50, 25, 10):
    thr = np.percentile(aS, 100 - q)
    m = aS >= thr
    g = d['gross'].to_numpy()[m]
    print(f'{"all" if q==100 else f"{q}%":<12}{m.sum():>6}{g.mean():>9.2f}'
          f'{g.std(ddof=1)/math.sqrt(m.sum()):>7.2f}'
          f'{(d["net_t"].to_numpy()[m]).mean():>11.2f}'
          f'{(d["net_m"].to_numpy()[m]).mean():>11.2f}')

rule('CONTROL — does the same structure pay without the signal?')
for nm, s_ in (('theory-signed', sgn), ('sign flipped', -sgn),
               ('always favourite', np.where(p0 > 50, 1, -1)),
               ('always cheap side', np.where(p0 < 50, 1, -1))):
    e = np.where(s_ > 0, p0, 100 - p0); pay = np.where(s_ > 0, settle, 100 - settle)
    g = pay - e
    print(f'  {nm:<20} gross {g.mean():+7.2f}c  se {g.std(ddof=1)/math.sqrt(len(g)):.2f}  '
          f'net taker {(g-HALF-fee(e)).mean():+7.2f}c  net maker {(g+HALF).mean():+7.2f}c')
print('\nNOTE: "net maker" credits the 2.35c half-spread on every fill. That credit\n'
      'is market-making revenue, not signal — it accrues to the flipped sign too.\n'
      'Read it as an upper bound that assumes fills with zero adverse selection.')
