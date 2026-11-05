"""Turn a live runner's own exchange-simulator log into the replay's result files, so live games enter Chapter 6 without a replay.

The live runner (scripts/live_exec.py with --exchange-sim) and the replay (scripts/event_execution.py) emit the same events to execution.jsonl,
so the lots (one per entry fill: size, entry price and fee, then exit fills, then what remains for settlement) can be rebuilt from the log:

  entry fill  -> a lot, numbered in fill order      exit fill -> the earliest lot of the same order that still has contracts left
  anything left after the exits settles at the game's final payoff (b61.settlement_prices, from the recorded final feed)

Output per game, shaped like a b49 replay (b61_make_tables.load reads it):
  <dest>/games/<date>/<league>_<game>/result.json            league, game, date, root, results.host.{by_kind, orders}
  <dest>/games/<date>/<league>_<game>/host/lots.json         lots as event_execution.export_lots writes them
  <dest>/games/<date>/<league>_<game>/host/decisions.jsonl.gz  the live runner's decisions (the guard and block profile reads them)

  b73_live_adapter.py validate [N]        rebuild lots from N replay games' execution.jsonl.gz and compare with their lots.json
  b73_live_adapter.py build SRC DEST      SRC = a pulled sports_live tree (<date>/<league>_<game>/...), DEST = output directory
"""
import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPS = 1e-8


def read_events(path):
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def kind_of(ticker):
    return 'spread' if 'SPREAD' in ticker else 'total' if 'TOTAL' in ticker else 'winner'


def lots_from_events(events):
    orders = {e['id']: e for e in events if e['event'] == 'submitted'}
    lots, by_order = [], {}
    for e in events:
        if e['event'] != 'fill':
            continue
        o = orders[e['id']]
        if e['leg'] == 'entry':
            lot = dict(id=len(lots), parent=e['id'], ticker=o['ticker'], kind=kind_of(o['ticker']), side=o['side'], size=e['size'], remaining=e['size'],
                       entry=e['price'], entry_fee_c=e['fee_c'], entry_fee1c_c=e['fee1c_c'], entry_maker=e['maker'], t_fill=e['exchange_clock'], exits=[])
            lots.append(lot)
            by_order.setdefault(e['id'], []).append(lot)
        else:
            left = e['size']
            for lot in by_order.get(e['id'], []):
                if left <= EPS:
                    break
                take = min(left, lot['remaining'])
                if take <= EPS:
                    continue
                share = take / e['size']
                lot['exits'].append(dict(size=take, price=e['price'], fee_c=e['fee_c'] * share, fee1c_c=e['fee1c_c'] * share,
                                         t=e['exchange_clock'], maker=e['maker']))
                lot['remaining'] -= take
                left -= take
    for lot in lots:
        if lot['remaining'] <= EPS:
            lot['remaining'] = 0.
    return lots


def results_from_lots(lots, payoff, n_orders):
    by = {k: dict(entry_contracts=0., fills=0, pnl_c=0., fees_c=0., pnl_1c_c=0., fees_1c_c=0., settled_contracts=0., filled_orders=set()) for k in ('spread', 'total')}
    for lot in lots:
        if lot['kind'] not in by:
            continue
        a = by[lot['kind']]
        a['entry_contracts'] += lot['size']
        a['fills'] += 1
        a['filled_orders'].add(lot['parent'])
        gross = sum(x['size'] * lot['side'] * (x['price'] - lot['entry']) for x in lot['exits'])
        if lot['remaining'] > EPS:
            a['settled_contracts'] += lot['remaining']
            gross += lot['remaining'] * lot['side'] * (payoff(lot['ticker']) - lot['entry'])
        for tag, entry, fees_k, pnl_k in (('fee_c', 'entry_fee_c', 'fees_c', 'pnl_c'), ('fee1c_c', 'entry_fee1c_c', 'fees_1c_c', 'pnl_1c_c')):
            fees = lot[entry] + sum(x[tag] for x in lot['exits'])
            a[fees_k] += fees
            a[pnl_k] += gross - fees
    for a in by.values():
        a['filled_orders'] = len(a['filled_orders'])
        a['net_per_contract_c'] = a['pnl_c'] / a['entry_contracts'] if a['entry_contracts'] else None
        a['gross_c'] = a['pnl_c'] + a['fees_c']
    return dict(orders=n_orders, by_kind=by)


def validate(n=40):
    import glob
    bad = ok = 0
    files = sorted(glob.glob(str(HERE / 'out/b55_main_20x/games/*/*/host/execution.jsonl.gz')))
    files = files[::max(1, len(files) // n)][:n]
    for f in files:
        events = read_events(f)
        mine = lots_from_events(events)
        ref = json.loads((Path(f).parent / 'lots.json').read_text())
        def key(lots):
            return [(l['parent'], l['ticker'], l['side'], round(l['size'], 4), l['entry'], round(l['remaining'], 4), round(l['entry_fee_c'], 4),
                     [(round(x['size'], 4), x['price'], round(x['fee_c'], 4)) for x in l['exits']]) for l in lots]
        if key(mine) == key(ref):
            ok += 1
        else:
            bad += 1
            print('MISMATCH', f, len(mine), len(ref))
    print(f'{ok} identical, {bad} mismatching of {len(files)} replay games')
    return bad


def build(src, dest):
    sys.path.insert(0, str(HERE))
    import b61_make_tables as T
    src, dest = Path(src).resolve(), Path(dest).resolve()
    n = 0
    for gdir in sorted(src.glob('*/*_*')):
        if not (gdir / 'execution.jsonl').exists() or not (gdir / 'summary.json').exists():
            continue
        league, game = gdir.name.split('_', 1)
        date = gdir.parent.name
        events = read_events(gdir / 'execution.jsonl')
        lots = lots_from_events(events)
        n_orders = sum(e['event'] == 'submitted' for e in events)
        game_root = gdir.resolve()                                          # settlement prices come from the recorded final feed (game.json + feed/)
        prices = T.settlement_prices(str(game_root.relative_to(ROOT))) if any(l['remaining'] > EPS for l in lots) else {}
        res = results_from_lots(lots, lambda tk: prices[tk], n_orders)
        out = dest / 'games' / date / gdir.name
        (out / 'host').mkdir(parents=True, exist_ok=True)
        (out / 'host/lots.json').write_text(json.dumps(lots, indent=1))
        dec = gdir / 'decisions.jsonl'
        if dec.exists():
            with dec.open('rb') as a, gzip.open(out / 'host/decisions.jsonl.gz', 'wb') as b:
                shutil.copyfileobj(a, b)
        (out / 'result.json').write_text(json.dumps(dict(date=date, league=league, game=game, root=str(game_root.relative_to(ROOT)) if game_root.is_relative_to(ROOT) else str(game_root), source='live',
                                                         results=dict(host=res))))
        n += 1
    print(n, 'games written to', dest)


def compare(dest):
    """Live (this adapter's output) against the replay (b68_main_all_totals) on the games both have, per league and contract type."""
    sys.path.insert(0, str(HERE))
    import b61_make_tables as T
    live, rep = T.load(Path(dest).name), T.load(T.MAIN)
    common = sorted(set(live) & set(rep))
    print(f'{len(live)} live games, {len(rep)} replayed games, {len(common)} in both')
    same = sum(abs(sum(live[k]['by'][x]['pnl_c'] for x in T.KINDS) - sum(rep[k]['by'][x]['pnl_c'] for x in T.KINDS)) < 0.01 for k in common)
    print(f'games with identical net P&L: {same} of {len(common)}')
    print(f"{'':12s} {'live $':>9s} {'replay $':>9s} {'live contr':>11s} {'replay contr':>12s} {'live fills':>10s} {'replay fills':>12s}")
    for lg in ('MLB', 'NBA', 'NFL'):
        for kind in T.KINDS:
            ks = [k for k in common if k[0] == lg]
            if not ks:
                continue
            f = lambda d, key: sum(d[k]['by'][kind][key] for k in ks)
            print(f"{lg + ' ' + kind:12s} {f(live,'pnl_c')/100:9.2f} {f(rep,'pnl_c')/100:9.2f} {f(live,'entry_contracts'):11.1f} {f(rep,'entry_contracts'):12.1f} "
                  f"{f(live,'fills'):10d} {f(rep,'fills'):12d}")


if __name__ == '__main__':
    if sys.argv[1] == 'compare':
        compare(sys.argv[2]); sys.exit(0)
    if sys.argv[1] == 'validate':
        sys.exit(1 if validate(int(sys.argv[2]) if len(sys.argv) > 2 else 40) else 0)
    build(sys.argv[2], sys.argv[3])
