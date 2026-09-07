"""How wide is the spread really, and does it narrow toward resolution?

research_log.md S7 reports 8.0c from the order book and concludes trade-based
estimators are "off by 8x". That compared the book against the 1c naive bounce
estimate. This script adds the measurement that was missing: the spread AT THE
INSTANTS TRADES ACTUALLY OCCURRED, by matching each trade to the prevailing
book snapshot. That is the number a taker actually pays.

Also tests whether spreads tighten as a contract approaches resolution, both
within-contract and cross-sectionally.

Inputs
  data/kalshi_orderbooks.jsonl  (11.8GB; pre-filter to macro first, see below)
  data/markets/*.parquet        (one quote per ticker at _fetched_at)

Pre-filter step (once):
  LC_ALL=C grep -E '^\{"market": "(KXFEDDECISION|RECSSNBER)' \
      data/kalshi_orderbooks.jsonl > $SCR/ob_macro.jsonl

OOS discipline: markets quotes are filtered to _fetched_at < 2026-01-01. The
markets archive was fetched through 2026-03-31, so an unfiltered quote panel
would silently pull OOS information into an in-sample estimate.
"""
from __future__ import annotations
import json, datetime as dt
import numpy as np, polars as pl

SCR = '/private/tmp/claude-503/-Users-caspe2-NUS-FYP/a09a383d-5b57-43d1-9091-34a28c3fb458/scratchpad'
IS = dt.datetime(2026, 1, 1)
FEE50 = 1.75
TICKER = 'KXFEDDECISION-26JAN-H0'
CLOSE = dt.datetime(2026, 1, 28)

def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)

# ---------------------------------------------------------- parse book ----
def load_book():
    rows = []
    with open(f'{SCR}/ob_macro.jsonl') as f:
        for line in f:
            try: o = json.loads(line)
            except Exception: continue
            ob = o.get('orderbook') or {}
            y, n = ob.get('yes') or [], ob.get('no') or []
            yb = max((p for p, _ in y), default=None)
            nb = max((p for p, _ in n), default=None)
            if yb is None or nb is None: continue
            rows.append({'ticker': o['market'], 'ts': o['timestamp'],
                         'yes_bid': yb, 'yes_ask': 100 - nb,
                         'yb_sz': sum(s for p, s in y if p == yb),
                         'nb_sz': sum(s for p, s in n if p == nb)})
    return (pl.DataFrame(rows)
            .with_columns(pl.from_epoch(pl.col('ts'), time_unit='ms')
                            .cast(pl.Datetime('us')).alias('t'))
            .with_columns((pl.col('yes_ask') - pl.col('yes_bid')).alias('spread'),
                          ((pl.col('yes_ask') + pl.col('yes_bid')) / 2).alias('mid'))
            .sort('t'))

ob = load_book()
h0 = ob.filter(pl.col('ticker') == TICKER)
rule(f'BOOK COVERAGE — the whole macro sample in kalshi_orderbooks.jsonl')
print(ob.group_by('ticker').agg(pl.len().alias('snapshots'),
                                pl.col('t').min().alias('from'),
                                pl.col('t').max().alias('to')).sort('snapshots', descending=True))
print(f'\nEverything below rests on {TICKER}: one contract, '
      f'{h0.height:,} snapshots,\nand it is FAR-DATED throughout '
      f'({(CLOSE-h0["t"].max()).days}-{(CLOSE-h0["t"].min()).days} days to close).')

# ------------------------------------------------ unconditional spread ----
rule('1  UNCONDITIONAL vs EXECUTABLE SPREAD')
h0 = h0.with_columns((pl.col('t').shift(-1) - pl.col('t')).dt.total_seconds()
                     .clip(0, 3600).fill_null(0).alias('dwell'))
sp_all = h0['spread'].to_numpy().astype(float); w = h0['dwell'].to_numpy()
print(f'all snapshots        : mean {sp_all.mean():.2f}c  median {np.median(sp_all):.1f}c')
print(f'time-weighted        : {(sp_all*w).sum()/w.sum():.2f}c')
print('percentiles          : ' + '  '.join(f'p{q}={np.percentile(sp_all,q):.0f}'
                                            for q in (5, 10, 25, 50, 75, 90, 95)))

tr = (pl.scan_parquet('data/trades/*.parquet')
        .filter(pl.col('ticker') == TICKER)
        .select('created_time', 'yes_price', 'count', 'taker_side').collect()
        .with_columns(pl.col('created_time').dt.replace_time_zone(None)
                        .cast(pl.Datetime('us')).alias('t')).sort('t'))
tr = tr.filter((pl.col('t') >= h0['t'].min()) & (pl.col('t') <= h0['t'].max()))
j = (tr.join_asof(h0.select('t', 'spread', 'mid'), on='t', strategy='backward',
                  tolerance=dt.timedelta(minutes=10)).drop_nulls('spread'))
sp_tr = j['spread'].to_numpy().astype(float); sz = j['count'].to_numpy().astype(float)
print(f'\nAT TRADE MOMENTS (n={j.height} of {tr.height} trades matched):')
print(f'  mean {sp_tr.mean():.2f}c  median {np.median(sp_tr):.1f}c  '
      f'volume-weighted {(sp_tr*sz).sum()/sz.sum():.2f}c')
print('  percentiles        : ' + '  '.join(f'p{q}={np.percentile(sp_tr,q):.0f}'
                                            for q in (10, 25, 50, 75, 90)))
print(f'  P(spread<=2c|trade) {np.mean(sp_tr<=2):.3f}  vs unconditional {np.mean(sp_all<=2):.3f}')
print(f'  P(spread<=4c|trade) {np.mean(sp_tr<=4):.3f}  vs unconditional {np.mean(sp_all<=4):.3f}')
j = j.with_columns((pl.col('yes_price') - pl.col('mid')).abs().alias('half'))
for nm, s in (('taker buys yes', j.filter(pl.col('taker_side') == 'yes')),
              ('taker buys no ', j.filter(pl.col('taker_side') == 'no'))):
    v = s['half'].to_numpy().astype(float)
    if v.size:
        print(f'  effective half-spread paid, {nm}: mean {v.mean():.2f}c  '
              f'median {np.median(v):.1f}c  (n={v.size})')
print(f'\nSelection bias is real but ~1.8x, not 8x: trades cluster at tight moments.')

# ------------------------------------------------------ 2 term structure --
rule('2  DOES THE SPREAD TIGHTEN TOWARD RESOLUTION?')
print('(a) within-contract, KXFEDDECISION-26JAN-H0 (only 56-91 DTC observed):')
b = (h0.with_columns(((pl.lit(CLOSE) - pl.col('t')).dt.total_seconds()/86400).alias('dtc'))
       .with_columns((pl.col('dtc')//5*5).alias('bk'))
       .group_by('bk').agg(pl.len().alias('n'), pl.col('spread').mean().alias('mean'),
                           pl.col('spread').median().alias('med'),
                           pl.col('mid').mean().alias('mid')).sort('bk', descending=True))
print(f'    {"days to close":<16}{"n":>7}{"mean":>8}{"median":>8}{"mid":>7}')
for r in b.iter_rows(named=True):
    print(f'    {int(r["bk"])}-{int(r["bk"])+5:<12}{r["n"]:>7}{r["mean"]:>8.2f}'
          f'{r["med"]:>8.0f}{r["mid"]:>7.0f}')

print('\n(b) cross-section of live quotes, one per ticker, fetched pre-2026:')
PAT = r'^(KX)?(CPI|CPIYOY|CPICORE|COREPCE|PCE|PPI|GDP|U3|PAYROLL|PROLLS|NFP|ADP|FED|JOBLESS|ISMPMI|WTI|RECSSNBER)'
m = (pl.scan_parquet('data/markets/*.parquet')
     .select('ticker','status','yes_bid','yes_ask','close_time','open_interest','_fetched_at')
     .filter(pl.col('_fetched_at') < IS)                 # OOS wall on the QUOTE
     .filter(pl.col('status') == 'active')
     .filter((pl.col('yes_bid') > 0) & (pl.col('yes_ask') > 0)).collect()
     .with_columns(pl.col('ticker').str.contains(PAT).alias('macro'),
                   (pl.col('yes_ask') - pl.col('yes_bid')).alias('spread'))
     .with_columns(((pl.col('close_time').dt.replace_time_zone(None)
                     - pl.col('_fetched_at')).dt.total_seconds()/86400).alias('dtc'))
     .filter((pl.col('spread') > 0) & (pl.col('dtc') > 0)))
bk = (pl.when(pl.col('dtc') < 1).then(pl.lit('0 <1d'))
        .when(pl.col('dtc') < 3).then(pl.lit('1 1-3d'))
        .when(pl.col('dtc') < 7).then(pl.lit('2 3-7d'))
        .when(pl.col('dtc') < 14).then(pl.lit('3 7-14d'))
        .when(pl.col('dtc') < 30).then(pl.lit('4 14-30d'))
        .when(pl.col('dtc') < 90).then(pl.lit('5 30-90d'))
        .otherwise(pl.lit('6 90d+')).alias('bk'))
for label, sub in (('MACRO', m.filter('macro')), ('ALL MARKETS', m)):
    print(f'\n  {label}: n={sub.height}, median spread {sub["spread"].median():.0f}c')
    print(f'    {"bucket":<12}{"n":>7}{"med spr":>9}{"mean spr":>10}{"med OI":>9}')
    for r in (sub.with_columns(bk).group_by('bk')
                 .agg(pl.len().alias('n'), pl.col('spread').median().alias('ms'),
                      pl.col('spread').mean().alias('mn'),
                      pl.col('open_interest').median().alias('oi'))
                 .sort('bk').iter_rows(named=True)):
        print(f'    {r["bk"][2:]:<12}{r["n"]:>7}{r["ms"]:>9.0f}{r["mn"]:>10.2f}{r["oi"]:>9.0f}')
print('\n  Open interest tracks the spread more closely than days-to-close does:')
print('  the tight buckets are the ones with real OI, not simply the near ones.')
print('  Caveat: one moment in time, one quote per ticker, small macro n near close.')

# ---------------------------------------------------------- 3 rebuild -----
rule('3  BREAK-EVENS RECOMPUTED — perfect foresight, single leg')
print('E|R| from surprise_cost_test.py test C: dormant 2.39c, liquid 8.43c\n')
print(f'{"spread basis":<40}{"spread":>8}{"+fees":>8}{"dormant":>10}{"liquid":>9}')
for nm, sp in [('unconditional book', 8.30), ('time-weighted book', 7.18),
               ('at trade moments (mean)', 4.69), ('at trade moments (median)', 4.00),
               ('2x effective half-spread paid', 5.48),
               ('trade-bounce estimate (S4.2.1)', 1.00)]:
    tot = sp + 2 * FEE50
    print(f'{nm:<40}{sp:>8.2f}{tot:>8.2f}{2.39-tot:>10.2f}{8.43-tot:>9.2f}')

rule('4  LADDER WIDTH — how many legs an implied-mean trade actually needs')
mkm = pl.read_parquet(f'{SCR}/macro_markets.parquet')
legs = mkm.group_by('event_ticker').agg(pl.len().alias('legs'), pl.col('series').first())
print(f'median legs/event {legs["legs"].median():.0f}  '
      f'mean {legs["legs"].mean():.1f}  p90 {legs["legs"].quantile(.9):.0f}\n')
for r in (legs.group_by('series').agg(pl.len().alias('events'),
                                      pl.col('legs').median().alias('med'))
              .sort('events', descending=True).head(10).iter_rows(named=True)):
    print(f'  {r["series"]:<14} events {r["events"]:>4}  median legs {r["med"]:.0f}')
print(f'\ncost of expressing one implied-mean view across k legs '
      f'(at 4.69c + 2 fees):')
for k in (1, 3, 5, 8):
    tot = (4.69 + 2 * FEE50) * k
    print(f'  {k} leg(s): {tot:6.2f}c   dormant PF {2.39-tot:+8.2f}   liquid PF {8.43-tot:+8.2f}')
