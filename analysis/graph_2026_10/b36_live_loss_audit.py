"""Audit recorded live paper orders; no inference, network, or changes to source logs.

Run: venv/bin/python analysis/graph_2026_10/b36_live_loss_audit.py
Outputs one CSV per fill/game and descriptive grouped results in JSON.
Different sizing regimes are reported separately. Filter comparisons retain existing
fills: they are attribution, not executable strategy backtests.
"""
from __future__ import annotations

import bisect
import collections
import csv
import gzip
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('out') / 'b36_live_loss_audit'


def rows(path):
    if not path.exists():
        path = Path(str(path) + '.gz')
    if path.exists():
        with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def fee(rate, price):
    return rate * price / 100 * (1 - price / 100)


def aggregate(rs):
    rs = list(rs)
    n = sum(r['size'] for r in rs)
    pnl = sum(r['pnl_c'] for r in rs)
    return dict(fills=len(rs), games=len({r['root'] for r in rs}), contracts=n,
                pnl_c=pnl, net_per_contract_c=pnl / n if n else None)


def group(rs, keys):
    gs = collections.defaultdict(list)
    for r in rs:
        gs['|'.join(str(r[k]) for k in keys)].append(r)
    return {k: aggregate(v) for k, v in sorted(gs.items())}


def scores(doc, league):
    if league == 'MLB':
        final = doc['feed']['gameData']['status']['abstractGameState'] == 'Final'
        t = doc['feed']['liveData']['linescore']['teams']
        return final, float(t['home']['runs']), float(t['away']['runs'])
    c = doc['header']['competitions'][0]
    s = {x['homeAway']: float(x['score']) for x in c['competitors']}
    return c['status']['type']['completed'], s['home'], s['away']


def payoff(m, league, home, away, hs, aws):
    # Same strict > strike convention as live_model.contract_meta and training.
    ticker = m['ticker']
    if 'TOTAL' in ticker.split('-')[0]:
        quantity = hs + aws
    else:
        team = re.match(r'[A-Z]+', ticker.split('-')[-1]).group()
        assert team in (home, away), (ticker, home, away)
        quantity = hs - aws if team == home else aws - hs
    if 'GAME' in ticker.split('-')[0]:
        strike = 0
    else:
        strike = None
        for txt, pat, offset in ((m.get('yes_sub_title'), r'(?i)over ([\d.]+)', 0),
                                 (m.get('title'), r'(?i)over ([\d.]+)', 0),
                                 (m.get('title'), r'(?i)more than ([\d.]+)', 0),
                                 (m.get('yes_sub_title'), r'(\d+)\+', -.5),
                                 (m.get('title'), r'(?i)at least ([\d.]+)', -.5)):
            match = re.search(pat, txt or '')
            if match:
                strike = float(match.group(1)) + offset
                break
        assert strike is not None, m
        if m.get('floor_strike') is not None:
            assert abs(strike - float(m['floor_strike'])) < 1e-6, m
    return 100.0 if quantity > strike else 0.0


def audit(root, logroot=None):
    logroot = logroot or root
    g = json.loads((root / 'game.json').read_text())
    files = sorted((root / 'feed').glob('*.json.gz'))
    if not files:
        return [], {'root': str(root.relative_to(ROOT)), 'excluded': 'no feed'}
    doc = json.loads(gzip.decompress(files[-1].read_bytes()))
    final, hs, aws = scores(doc, g['league'])
    orders = {}
    active_ids = {}
    for e in rows(logroot / 'orders.jsonl'):
        if e['event'] == 'placed':
            old = active_ids.get(e['id'])
            if old is not None and e == orders[old]['placed']:
                continue
            if old is not None:
                assert any(k in orders[old] for k in ('expired','closed','to settlement','cancelled')), ('ambiguous restart',root,e)
            serial = len(orders)
            orders[serial] = {'placed': e, 'serial': serial}
            active_ids[e['id']] = serial
        elif e.get('id') in active_ids:
            orders[active_ids[e['id']]][e['event']] = e
    if not final and any('filled' in o and 'closed' not in o for o in orders.values()):
        return [], {'root': str(root.relative_to(ROOT)), 'excluded': 'unfinished feed with unresolved fills'}
    placed = {(o['placed']['t0'], o['placed']['ticker']): o for o in orders.values()}
    danger = collections.defaultdict(list)
    decisions = collections.defaultdict(list)
    first = {}
    for d in rows(logroot / 'decisions.jsonl'):
        if d.get('event') == 'first play':
            first = d
        if 'ticker' not in d:
            continue
        decisions[d['ticker']].append(d)
        if (d['t'], d['ticker']) in placed:
            placed[d['t'], d['ticker']]['decision'] = d
        if d.get('action', '').startswith('skip:'):
            danger[d['ticker']].append((d['t'], d['action']))
    fractional = any(o['placed']['size'] % 1 for o in orders.values())
    regime = 'fractional' if fractional else 'whole'
    meta = {m['ticker']: m for m in g['contracts']}
    for ds in decisions.values():
        ds.sort(key=lambda d: d['t'])
    out = []
    for o in orders.values():
        if 'filled' not in o:
            continue
        p, f, d = o['placed'], o['filled'], o.get('decision', {})
        if p['size'] <= 0:
            continue  # Old engines logged fills even for orders rounded to zero.
        entry = f['price']
        entry_fee = fee(7 if f.get('how') == 'crossed' else 1.75, entry)
        if 'closed' in o:
            c = o['closed']
            net, how, end = c['net'], c['how'], c['t']
            expect = p['side'] * (c['price'] - entry) - entry_fee - fee(1.75 if how == 'limit filled' else 7, c['price'])
            assert abs(net - expect) < 1e-6, (root, o)
            assert abs(c['pnl'] - net * p['size']) < 1e-6
        else:
            net = p['side'] * (payoff(meta[p['ticker']], g['league'], g['home_k'], g['away_k'], hs, aws) - entry) - entry_fee
            how, end = 'settlement', None
        skips = [reason for t, reason in danger[p['ticker']] if p['t0'] < t < f['t']]
        ds = decisions[p['ticker']]
        at_fill = bisect.bisect_right([d['t'] for d in ds], f['t']) - 1
        df = ds[at_fill] if at_fill >= 0 else {}
        sf = df.get('signal')
        level = df.get('level')
        # Evaluate from our limit rather than the possibly moved reference print.
        edge_fill = p['side'] * (level - entry) if level is not None else None
        invalid = next((x['t'] for x in ds if p['t0'] < x['t'] < f['t'] - .5
                        and x.get('level') is not None and p['side'] * (x['level'] - p['limit']) <= 0), None)
        sig = abs(p['signal'])
        width = d.get('width')
        out.append(dict(root=str(root.relative_to(ROOT)), date=root.parent.name, league=g['league'],
                        game=g['game_id'], matchup=g['away'] + '@' + g['home'], regime=regime,
                        id=o['serial'], logged_id=p['id'], ticker=p['ticker'], kind=p['kind'], side=p['side'], size=p['size'],
                        t0=p['t0'], t_fill=f['t'], t_exit=end, entry=entry, signal=p['signal'],
                        signal_bin='<8' if sig < 8 else '8-10' if sig < 10 else '10-15' if sig < 15 else '15+',
                        width=width, width_bin='missing' if width is None else '<5' if width < 5 else '5-10' if width < 10 else '10+',
                        late=bool(d.get('late')), how=how, net_c=net, pnl_c=net * p['size'],
                        risk_c=(entry if p['side'] == 1 else 100-entry) * p['size'],
                        fill_wait_s=f['t'] - p['t0'], guard_before_fill=bool(skips),
                        fill_signal=sf, fill_model_edge_c=edge_fill,
                        fill_edge_nonpositive=edge_fill <= 0 if edge_fill is not None else None,
                        invalidated_before_fill=invalid is not None, invalid_at=invalid,
                        guard_reason='; '.join(sorted(set(skips))),
                        exit_limit=o.get('exit posted', {}).get('price'),
                        fill_has_decision=bool(d)))
    s = aggregate(out)
    summary = json.loads((logroot / 'summary.json').read_text()) if (logroot / 'summary.json').exists() else {}
    s.update(root=str(root.relative_to(ROOT)), date=root.parent.name, league=g['league'], matchup=g['away'] + '@' + g['home'],
             regime=regime, orders=len(orders), closed_c=sum(r['pnl_c'] for r in out if r['how'] != 'settlement'),
             final_feed=final,
             reported_closed_c=summary.get('pnl_closed_c'), late_block=first.get('late_block'),
             score=f'{aws:g}-{hs:g}', summary_orders=summary.get('orders'))
    return out, s


def write_csv(path, rs):
    fields = list(dict.fromkeys(k for r in rs for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rs)


def main():
    import numpy as np
    fills, games = [], []
    paths = sorted((ROOT / 'data/sports_live/vm').glob('2026*/*/game.json'))
    paths += sorted((ROOT / 'data/sports_live/sandbox_archive').glob('2026*/*/game.json'))
    # Older sandbox live runs only; exclude missed-game replays and preseason recordings.
    paths += sorted((ROOT / 'data/sports_live/sandbox').glob('2026*/*/game.json'))
    seen = set()
    for p in paths:
        if not (p.parent / 'orders.jsonl').exists() and not (p.parent / 'orders.jsonl.gz').exists():
            continue
        g = json.loads(p.read_text())
        # This archive interleaves concurrent runners using the same ids. The
        # earlier local copy was cleaned in the documented Oct 15 investigation.
        if 'sandbox_archive' in p.parts and g['league'] == 'MLB' and g['game_id'] == '849810':
            continue
        if p.parent.parent.name == '2026-10-17' and g['league'] == 'MLB' and g['game_id'] == '849816':
            games.append(dict(root=str(p.parent.relative_to(ROOT)), league='MLB', date='2026-10-17',
                              excluded='wrong-game contracts after rainout; forward_test_log incident'))
            continue
        key = (g['league'], g['game_id'])
        if key in seen:
            continue
        seen.add(key)
        fs, gs = audit(p.parent)
        fills += fs
        games.append(gs)
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'fills.csv', fills)
    write_csv(OUT / 'games.csv', games)
    groups = {}
    for keys in [('league','regime'), ('league','regime','date'), ('league','regime','kind'),
                 ('league','regime','kind','how'), ('league','regime','kind','side'),
                 ('league','regime','kind','signal_bin'), ('league','regime','kind','width_bin'),
                 ('league','regime','kind','late'), ('league','regime','guard_before_fill')]:
        groups['/'.join(keys)] = group(fills, keys)
    (OUT / 'groups.json').write_text(json.dumps(groups, indent=2) + '\n')
    # Cluster on games, including games with zero fills. These intervals do not
    # account for shared nightly conditions or selection of the current rules.
    rng = np.random.default_rng(36)
    intervals = {}
    for league, regime, kind in sorted({(r['league'], r['regime'], r['kind']) for r in fills}):
        rs = [r for r in fills if (r['league'], r['regime'], r['kind']) == (league, regime, kind)]
        roots = sorted({g['root'] for g in games if not g.get('excluded') and g['league'] == league and g['regime'] == regime})
        values = np.array([[sum(r['pnl_c'] for r in rs if r['root'] == root),
                            sum(r['size'] for r in rs if r['root'] == root)] for root in roots])
        samples = values[rng.integers(0, len(roots), (10000, len(roots)))].sum(1)
        valid = samples[:, 1] > 0
        bounds = np.percentile(samples[valid, 0] / samples[valid, 1], [2.5, 97.5])
        intervals['|'.join((league, regime, kind))] = dict(games=len(roots), lo95_c=float(bounds[0]), hi95_c=float(bounds[1]))
    (OUT / 'bootstrap.json').write_text(json.dumps(intervals, indent=2) + '\n')
    replay_fills, replay_games = [], []
    for p in sorted((ROOT / 'data/sports_live/replay_hist').glob('2026*/*/game.json')):
        fs, gs = audit(p.parent, p.parent / 'rerun/backtest')
        replay_fills += fs
        replay_games.append(gs)
    write_csv(OUT / 'historical_replay_fills.csv', replay_fills)
    write_csv(OUT / 'historical_replay_games.csv', replay_games)
    replay_groups = { '/'.join(keys):group(replay_fills, keys)
                      for keys in [('league','kind'), ('league','date','kind'), ('league','kind','how')] }
    (OUT / 'historical_replay_groups.json').write_text(json.dumps(replay_groups, indent=2) + '\n')
    print(json.dumps({'games':len(games), 'fills':len(fills), 'summary':groups['league/regime'],
                      'by_kind':groups['league/regime/kind'], 'by_exit':groups['league/regime/kind/how']}, indent=2))


if __name__ == '__main__':
    main()
