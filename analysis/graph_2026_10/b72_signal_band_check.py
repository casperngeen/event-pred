"""What if the signal weight ramped from 8 to 15 cents instead of 5 to 15?  (fixed-fill rescoring of the forward-test cohort)

Current rule (live_trade.size_of): weight = min(1, ((|signal| - 5) / 10)^2) x min(1, width / 10); contracts = round(20 x weight, .01).
Alternative:                       weight = min(1, ((|signal| - 8) /  7)^2) x min(1, width / 10); no order below 8 cents.

Each filled lot of the main cohort (b68_main_all_totals) is joined to the order decision that placed it, its width weight is kept, and
its net cents are scaled by new size / old size (per-contract exits, fees and settlement are unchanged). This is the same fixed-fill
approximation as thesis_sizing20x.py: it ignores that orders below 8 cents would no longer rest (so a ticker could be free for a later order),
and fee rounding. Rescoring the current rule the same way reproduces the replay's own totals (checked below).
"""
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import b61_make_tables as T

SEASON = T.SEASON


def weight(sig, width, lo, span):
    s = min(1.0, ((abs(sig) - lo) / span) ** 2) if abs(sig) >= lo else 0.0
    return s * (1.0 if width is None else min(1.0, max(0.0, width / 10)))


def contracts(w):
    return math.floor(100 * 20 * w + .5) / 100


def orders_of(folder):
    out = []
    with gzip.open(folder / 'decisions.jsonl.gz', 'rt') as f:
        for line in f:
            if '"order"' not in line:
                continue
            x = json.loads(line)
            out += [r for r in (x if isinstance(x, list) else [x]) if r.get('action') == 'order']
    return out


def lot_net(lot, prices):
    g = sum(x['size'] * lot['side'] * (x['price'] - lot['entry']) for x in lot['exits'])
    if lot['remaining'] > 1e-8:
        g += lot['remaining'] * lot['side'] * (prices[lot['ticker']] - lot['entry'])
    gross = g
    fee = lot['entry_fee_c'] + sum(x['fee_c'] for x in lot['exits'])
    return gross, fee


def main():
    games = T.load(T.MAIN)
    rows = {}                                             # (league, kind) -> list of (game index, size, net, gross, fee) per arm
    per_game = {a: {} for a in ('current', 'rescored', 'band8')}
    totals = {a: {} for a in per_game}
    mism = 0
    for (lg, gid), v in sorted(games.items(), key=lambda kv: kv[1]['date']):
        lots = json.loads(v['lots'].read_text())
        if not lots:
            continue
        folder = v['lots'].parent
        orders = orders_of(folder)
        record = json.loads((folder.parent / 'result.json').read_text())
        prices = None
        for lot in lots:
            if lot['kind'] not in T.KINDS:
                continue
            o = orders[lot['parent']]
            assert o['ticker'] == lot['ticker'], (gid, lot['parent'])
            if lot['remaining'] > 1e-8 and prices is None:
                prices = T.settlement_prices(record['root'])
            gross, fee = lot_net(lot, prices)
            old = o['size']                                # a lot is one (partial) fill of this order: scale every lot by the order's size ratio
            res = contracts(weight(o['signal'], o['width'], 5, 10))
            new = contracts(weight(o['signal'], o['width'], 8, 7))
            if abs(res - old) > .011:
                mism += 1
            for arm, size in (('current', old), ('rescored', res), ('band8', new)):
                f = size / old if old else 0.
                size = lot['size'] * f
                key = (lg, lot['kind'])
                d = totals[arm].setdefault(key, dict(q=0., net=0., gross=0., fee=0., fills=0, wins=0))
                d['q'] += size
                d['net'] += (gross - fee) * f
                d['gross'] += gross * f
                d['fee'] += fee * f
                if size > 0:
                    d['fills'] += 1
                    d['wins'] += (gross - fee) * f > 0
                pg = per_game[arm].setdefault((lg, gid), 0.)
                per_game[arm][(lg, gid)] = pg + (gross - fee) * f / 100
    print(f'lots whose recomputed current order size differs from the logged order size by more than 0.01: {mism}')
    allg = {k: 0. for k in games}
    for arm in per_game:
        print(f'\n=== {arm} ===')
        print(f"{'league':6s} {'kind':7s} {'fills':>6s} {'contracts':>10s} {'net $':>9s} {'net c/contract':>15s} {'fill win':>9s}")
        tot = dict(q=0., net=0., fills=0)
        for key in sorted(totals[arm]):
            d = totals[arm][key]
            if not d['q']:
                continue
            print(f"{key[0]:6s} {key[1]:7s} {d['fills']:6d} {d['q']:10.1f} {d['net']/100:9.2f} {100*d['net']/100/d['q']:15.2f} {100*d['wins']/max(d['fills'],1):8.0f}%")
            tot['q'] += d['q']; tot['net'] += d['net']; tot['fills'] += d['fills']
        print(f"{'ALL':6s} {'':7s} {tot['fills']:6d} {tot['q']:10.1f} {tot['net']/100:9.2f} {tot['net']/tot['q']:15.2f}")
        for lg in ('MLB', 'NBA', 'NFL'):
            x = np.array([per_game[arm].get(k, 0.) for k in games if k[0] == lg])
            sd = x.std(ddof=1)
            print(f'   {lg}: {len(x)} games, per-game mean ${x.mean():.3f}, sd ${sd:.3f}, per-game Sharpe {x.mean()/sd:.3f}, t {x.mean()/sd*np.sqrt(len(x)):.2f}')


if __name__ == '__main__':
    main()
