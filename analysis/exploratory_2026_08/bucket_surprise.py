"""Implied mean / surprise for BUCKET ladders, so WTI can enter the universe.

WTI is 85% bucket contracts and is currently absent from implied-mean output
entirely (research_log.md S8, TODO.md). Bucket legs price P(a <= X <= b)
directly — they ARE a pmf and need normalisation, not the successive
differencing recover_pdf applies to cumulative threshold ladders.

Ladder shape (verified on all 702 WTI events): ~13 unit-width interior buckets
plus 2 open tails ("$61.99 or below" / "$75.0 or above").

Exports build_bucket_surprise(series) with the same (event, close_time,
surprise) contract as liquid_window.pdf_surprise, so it drops into the
existing sign-test machinery.

Pre-2026 only.
"""
from __future__ import annotations
import sys, datetime as dt
import numpy as np, polars as pl

sys.path.insert(0, 'stg_infra')
from stg.events.implied import classify_contract, parse_bucket, parse_threshold, BUCKET, THRESHOLD

IS_CUT = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
MIN_LEGS = 5          # need a real cross-section to call it a distribution
MIN_MASS = 0.5        # normalised mass must not be dominated by missing legs

_mk = (pl.scan_parquet('data/markets/*.parquet')
       .sort('_fetched_at', descending=True).unique(subset=['ticker'], keep='first')
       .filter(pl.col('close_time') < IS_CUT)
       .select('ticker', 'event_ticker', 'yes_sub_title', 'status', 'result', 'close_time')
       .with_columns(pl.col('ticker').str.replace(r'^KX', '')
                       .str.split('-').list.first().alias('series'))
       .collect())
_tr = pl.scan_parquet('data/trades/*.parquet').filter(pl.col('created_time') < IS_CUT)


def _legs(series: str) -> pl.DataFrame:
    m = _mk.filter(pl.col('series') == series)
    cls, lo, hi, thr = [], [], [], []
    for t, s in zip(m['ticker'], m['yes_sub_title']):
        c = classify_contract(t, s)
        cls.append(c)
        b = parse_bucket(t, s) if c == BUCKET else None
        lo.append(b[0] if b else None); hi.append(b[1] if b else None)
        thr.append(parse_threshold(t, s) if c == THRESHOLD else None)
    return m.with_columns(pl.Series('cls', cls), pl.Series('lo', lo, dtype=pl.Float64),
                          pl.Series('hi', hi, dtype=pl.Float64),
                          pl.Series('thr', thr, dtype=pl.Float64))


def build_bucket_surprise(series: str, verbose: bool = True) -> pl.DataFrame:
    m = _legs(series)
    tk = m['ticker'].unique().to_list()
    tr = (_tr.filter(pl.col('ticker').is_in(tk))
             .select('ticker', 'yes_price', 'created_time')
             .with_columns(pl.col('created_time').dt.date().alias('d'))
             .sort(['ticker', 'created_time'])
             .group_by(['ticker', 'd'])
             .agg(pl.col('yes_price').last().alias('px'),
                  pl.len().alias('n')).collect())
    px_by_ticker = {}
    for r in tr.iter_rows(named=True):
        px_by_ticker.setdefault(r['ticker'], []).append((r['d'], r['px']))

    rows, skipped = [], {'no_legs': 0, 'no_trades': 0, 'low_mass': 0, 'no_result': 0}
    for ev in m['event_ticker'].unique().to_list():
        sub = m.filter(pl.col('event_ticker') == ev)
        bk = sub.filter(pl.col('cls') == BUCKET).drop_nulls(['lo', 'hi'])
        if bk.height < MIN_LEGS:
            skipped['no_legs'] += 1; continue
        close = sub['close_time'].min()
        width = float(np.median(bk['hi'].to_numpy() - bk['lo'].to_numpy())) + 0.01

        # last pre-close price for every leg; the day the ladder was most complete
        day_px = {}
        for r in bk.iter_rows(named=True):
            hist = px_by_ticker.get(r['ticker'])
            if not hist: continue
            for d, p in hist:
                day_px.setdefault(d, {})[r['ticker']] = p
        if not day_px:
            skipped['no_trades'] += 1; continue
        best_day = max(day_px, key=lambda d: len(day_px[d]))
        if len(day_px[best_day]) < MIN_LEGS:
            skipped['no_legs'] += 1; continue
        quotes = day_px[best_day]

        mids, probs = [], []
        for r in bk.iter_rows(named=True):
            if r['ticker'] in quotes:
                mids.append((r['lo'] + r['hi']) / 2.0)
                probs.append(quotes[r['ticker']] / 100.0)
        # open tails, if their legs traded that day
        for r in sub.filter(pl.col('cls') == THRESHOLD).drop_nulls('thr').iter_rows(named=True):
            p = day_px[best_day].get(r['ticker'])
            if p is None: continue
            above = 'above' in (r['yes_sub_title'] or '').lower()
            mids.append(r['thr'] + width / 2.0 if above else r['thr'] - width / 2.0)
            probs.append(p / 100.0)
        mids = np.array(mids, float); probs = np.array(probs, float)
        mass = probs.sum()
        if mids.size < MIN_LEGS or mass < MIN_MASS:
            skipped['low_mass'] += 1; continue
        pmf = probs / mass
        implied = float(np.dot(mids, pmf))

        # resolved value: midpoint of the leg that settled yes
        won = sub.filter(pl.col('result') == 'yes')
        if won.height == 0:
            skipped['no_result'] += 1; continue
        w = won.row(0, named=True)
        if w['cls'] == BUCKET and w['lo'] is not None:
            resolved = (w['lo'] + w['hi']) / 2.0
        elif w['thr'] is not None:
            above = 'above' in (w['yes_sub_title'] or '').lower()
            resolved = w['thr'] + width / 2.0 if above else w['thr'] - width / 2.0
        else:
            skipped['no_result'] += 1; continue

        rows.append({'event': ev, 'close_time': close, 'surprise': resolved - implied,
                     'implied': implied, 'resolved': resolved, 'legs': int(mids.size),
                     'mass': float(mass), 'obs_day': best_day})
    out = pl.DataFrame(rows).sort('close_time') if rows else pl.DataFrame()
    if verbose:
        print(f'{series}: {out.height} events with a usable pmf   skipped {skipped}')
    return out


if __name__ == '__main__':
    print('=' * 78); print('BUCKET PMF — construction and validation'); print('=' * 78)
    for series in ('WTI', 'WTIW'):
        s = build_bucket_surprise(series)
        if s.height == 0: continue
        imp = s['implied'].to_numpy(); res = s['resolved'].to_numpy()
        sur = s['surprise'].to_numpy()
        print(f'  legs/event median {s["legs"].median():.0f}   '
              f'normalised mass median {s["mass"].median():.2f} '
              f'(1.0 = ladder prices sum to certainty)')
        print(f'  implied mean range  ${imp.min():.1f} - ${imp.max():.1f}')
        print(f'  resolved range      ${res.min():.1f} - ${res.max():.1f}')
        print(f'  corr(implied, resolved) = {np.corrcoef(imp,res)[0,1]:.3f}   '
              f'(market should broadly know the level)')
        print(f'  surprise: mean {sur.mean():+.3f}  sd {sur.std():.3f}  '
              f'median |s| {np.median(np.abs(sur)):.3f}  ($/bbl)')
        print(f'  |surprise| > $5 : {(np.abs(sur)>5).sum()} events '
              f'(sanity: should be rare)\n')
