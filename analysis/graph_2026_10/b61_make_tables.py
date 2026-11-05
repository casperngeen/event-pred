"""Build the live-evaluation tables from replay outputs; rerun after each batch of games.

  python3 b61_make_tables.py            writes thesis/generated/results_tables.tex and analysis/graph_2026_10/out/b61_summary.json

Per-game P&L = sum over contract types of net cents / 100 (20x sizing, fees on, host latency). Games with no fill count as 0.
Annualised Sharpe = mean(per-game P&L) / sd(per-game P&L) * sqrt(games per season) with MLB 2430, NBA 1230, NFL 272;
no results are pooled across leagues.
"""
import glob
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
GEN = HERE.parents[1] / 'thesis/generated'
SEASON = {'MLB': 2430, 'NBA': 1230, 'NFL': 272}
KINDS = ('spread', 'total')
MAIN = 'b68_main_all_totals'
RNG = np.random.default_rng(0)


def load(name):
    """{(league, game): dict(date, kinds={kind: by_kind}, lots=[...])} for every scored game of an output directory."""
    games = {}
    for f in sorted(glob.glob(str(OUT / name / 'games/*/*/result.json'))):
        r = json.load(open(f))
        res = (r.get('results') or {}).get('host')
        if not res:
            continue
        games[(r['league'], r['game'])] = dict(date=r['date'], by=res['by_kind'], orders=res['orders'], lots=Path(f).parent / 'host/lots.json')
    return games


def load_arms(name):
    """{(league, game): {arm: record}} for a multi-arm output directory (records shaped like load()'s)."""
    games = {}
    for f in sorted(glob.glob(str(OUT / name / 'games/*/*/result.json'))):
        r = json.load(open(f))
        arms = r.get('results') or {}
        games[(r['league'], r['game'])] = {a: dict(date=r['date'], by=v['by_kind'], orders=v['orders'], lots=Path(f).parent / a / 'lots.json') for a, v in arms.items()}
    return games


def merge(*dirs):
    """First directory is the base; later ones override game by game (variants carry over untouched games)."""
    out = {}
    for d in dirs:
        out.update(load(d))
    return out


@lru_cache(maxsize=None)
def settlement_prices(root):
    """Use the same final recorded game outcome as the replay's settlement ledger."""
    from b36_live_loss_audit import scores, payoff
    root = HERE.parents[1] / root
    game = json.loads((root / 'game.json').read_text())
    feed = sorted((root / 'feed').glob('*.json.gz'))[-1]
    doc = json.loads(gzip.decompress(feed.read_bytes()))
    final, home, away = scores(doc, game['league'])
    assert final, f'Final score unavailable for {root}'
    return {m['ticker']: payoff(m, game['league'], game['home_k'], game['away_k'], home, away)
            for m in game['contracts']}


def lots_pnl(path, kinds):
    """Per-entry-lot net cents, including exits, fees and remaining settlement."""
    out = []
    record = None
    prices = None
    for lot in json.loads(path.read_text()):
        if lot['kind'] not in kinds:
            continue
        g = sum(x['size'] * lot['side'] * (x['price'] - lot['entry']) for x in lot['exits'])
        if lot['remaining'] > 1e-8:
            if prices is None:
                record = json.loads((path.parent.parent / 'result.json').read_text())
                prices = settlement_prices(record['root'])
            g += lot['remaining'] * lot['side'] * (prices[lot['ticker']] - lot['entry'])
        fee = lot['entry_fee_c'] + sum(x['fee_c'] for x in lot['exits'])
        out.append((g - fee, lot['size'] * 0 + g - fee > 0))
    return out


def maximum_drawdown(profits):
    """Peak-to-trough loss in cumulative P&L, including the initial zero balance."""
    curve = np.r_[0., np.cumsum(profits)]
    return float((np.maximum.accumulate(curve)-curve).max())


def stats(games, league=None, kinds=KINDS):
    rows = [(k, v) for k, v in sorted(games.items(), key=lambda kv: kv[1]['date']) if league in (None, k[0])]
    pg, n_ord, n_fill, contracts, gross, fees, net1c = [], 0, 0, 0., 0., 0., 0.
    fills = []
    daily = {}
    for (lg, gid), v in rows:
        pnl = sum(v['by'][k]['pnl_c'] for k in kinds) / 100
        pg.append(pnl)
        daily[v['date']] = daily.get(v['date'], 0.) + pnl
        n_ord += sum(v['by'][k]['filled_orders'] for k in kinds)
        n_fill += sum(v['by'][k]['fills'] for k in kinds)
        contracts += sum(v['by'][k]['entry_contracts'] for k in kinds)
        gross += sum(v['by'][k]['gross_c'] for k in kinds) / 100
        fees += sum(v['by'][k]['fees_c'] for k in kinds) / 100
        net1c += sum(v['by'][k]['pnl_1c_c'] for k in kinds) / 100
        if any(v['by'][k]['fills'] for k in kinds):
            fill_values = [p for p, _ in lots_pnl(v['lots'], kinds)]
            assert abs(sum(fill_values)-pnl*100) < .01, f'Lot/ledger mismatch: {v["lots"]}'
            fills += fill_values
    pg = np.array(pg)
    n = len(pg)
    sd = pg.std(ddof=1) if n > 1 else float('nan')
    net = pg.sum()
    s = pg.mean() / sd if sd and sd > 0 else float('nan')
    boot = [(lambda x: x.mean() / x.std(ddof=1))(RNG.choice(pg, n)) for _ in range(2000)] if n > 2 and sd > 0 else [float('nan')]
    f = np.array(fills) if fills else np.array([0.])
    wins, losses = f[f > 0], f[f < 0]
    return dict(games=n, games_traded=int((pg != 0).sum()), orders=n_ord, fills=n_fill, contracts=contracts, gross=gross, fees=fees, net=net, net_1c=net1c,
                net_c_per_contract=100 * net / contracts if contracts else float('nan'),
                game_win=float((pg > 0).sum() / max((pg != 0).sum(), 1)), fill_win=float((f > 0).mean()),
                profit_factor=float(wins.sum() / -losses.sum()) if losses.size and wins.size else float('nan'),
                avg_win_c=float(wins.mean()) if wins.size else float('nan'), avg_loss_c=float(losses.mean()) if losses.size else float('nan'),
                mean_game=float(pg.mean()) if n else float('nan'), sd_game=float(sd), t_stat=float(s * math.sqrt(n)) if n > 1 else float('nan'),
                sharpe_per_game=float(s), max_drawdown=maximum_drawdown([daily[d] for d in sorted(daily)]),
                lag1=float(np.corrcoef(pg[:-1], pg[1:])[0, 1]) if n > 3 and pg.std() > 0 else float('nan'),
                boot_lo=float(np.nanpercentile(boot, 2.5)), boot_hi=float(np.nanpercentile(boot, 97.5)), per_game=pg.tolist(), season=SEASON.get(league))


def sharpe(st, league):
    return st['sharpe_per_game'] * math.sqrt(SEASON[league])


def f1(x, d=1):
    return '--' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:,.{d}f}'.replace(',', '\\,')


def money(x):
    return ('$-$' if x < 0 else '') + '\\$' + f1(abs(x), 2) if x == x else '--'


def table(caption, label, head, rows, note='', fit_width=True, placement='t'):
    if 'Quantity' in head:
        note = (note + ' Quantity counts entry units.').strip()
    cols = 'l' + 'r' * (len(head) - 1)
    start = f'\\begin{{tabular}}{{{cols}}}\\toprule'
    end = '\\bottomrule\\end{tabular}'
    if fit_width:
        start = '\\resizebox{\\textwidth}{!}{' + start
        end += '}'
    t = [f'\\begin{{table}}[{placement}]\\centering\\small', f'\\caption{{{caption}}}\\label{{{label}}}', start,
         ' & '.join(head) + ' \\\\ \\midrule']
    t += [' & '.join(r) + ' \\\\' for r in rows]
    t += [end, f'\\tablenote{{{note}}}' if note else '', '\\end{table}']
    return '\n'.join(t)


def highlight_best(rows, columns, minimize=()):
    """Bold the best displayed metrics within each league; never assume a winner."""
    start = 0
    while start < len(rows):
        stop = start + 1
        while stop < len(rows) and not rows[stop][0]:
            stop += 1
        for column in columns:
            missing = math.inf if column in minimize else -math.inf
            values = [float(row[column].replace('\\%', '')) if row[column] != '--' else missing
                      for row in rows[start:stop]]
            best = min(values) if column in minimize else max(values)
            for i, value in enumerate(values):
                if value == best and math.isfinite(value):
                    rows[start+i][column] = '\\textbf{' + rows[start+i][column] + '}'
        start = stop
    return rows


def plot_baseline_comparison(main_g, base, summary):
    """Running per-unit returns on the table's matched games and decision times."""
    import datetime as dt
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'event_pred_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1), sharex='col')
    arms = [('model', 'Model', '#222222', '-'),
            ('sym_model', 'Symmetric', '#4477aa', '--'),
            ('micro_model', 'Micro-price', '#ee7733', ':'),
            ('flow_model', 'Trade flow', '#228833', '-.')]
    for column, league in enumerate(SEASON):
        keys = sorted((k for k in base if k[0] == league and k in main_g),
                      key=lambda k: (main_g[k]['date'], k[1]))
        if not keys:
            for ax in axes[:, column]:
                ax.set_visible(False)
            continue
        endpoints = {arm: np.zeros(2) for arm, *_ in arms}
        for row, kind in enumerate(KINDS):
            ax = axes[row, column]
            for arm, label, color, style in arms:
                daily = {}
                for key in keys:
                    record = main_g[key] if arm == 'model' else base[key][arm]
                    day = record['date']
                    totals = daily.setdefault(day, [0., 0.])
                    totals[0] += record['by'][kind]['pnl_c']
                    totals[1] += record['by'][kind]['entry_contracts']
                days = sorted(daily)
                cumul_pnl = np.cumsum([daily[d][0] for d in days])
                cumul_qty = np.cumsum([daily[d][1] for d in days])
                curve = np.divide(cumul_pnl, cumul_qty, out=np.full(len(days), np.nan), where=cumul_qty > 0)
                detail = summary.get(f'base/{league}/{kind}/{arm}')
                if detail is not None:
                    assert abs(curve[-1] - detail['net_c_per_contract']) < 1e-8, (league, kind, arm)
                endpoints[arm] += (cumul_pnl[-1], cumul_qty[-1])
                ax.plot([dt.date.fromisoformat(d) for d in days], curve,
                        color=color, linestyle=style, linewidth=1.9 if arm == 'model' else 1.3, label=label)
            ax.set_title(f'{league} {"spreads" if kind == "spread" else "totals"}')
            ax.axhline(0, color='#777777', linewidth=.6)
            ax.grid(axis='y', alpha=.2)
            ticks = sorted({days[0], days[len(days)//2], days[-1]})
            ax.set_xticks([dt.date.fromisoformat(day) for day in ticks])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            ax.tick_params(axis='x', labelsize=8, rotation=25)
        for arm, (pnl, quantity) in endpoints.items():
            expected = summary[f'base/{league}/{arm}']['net_c_per_contract']
            assert abs(pnl / quantity - expected) < 1e-8, (league, arm)
    for ax in axes[:, 0]:
        ax.set_ylabel('Running net return\n(cents per unit)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, 1))
    GEN.mkdir(parents=True, exist_ok=True)
    fig.savefig(GEN / 'results_baseline_comparison.pdf', bbox_inches='tight')
    fig.savefig(Path(tempfile.gettempdir()) / 'thesis_baseline_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def baseline_comparison_table(main_g, base, summary):
    """Compare strategies within each league and contract type on matched games."""
    arms = [('model', 'Model'), ('sym_model', 'Symmetric'),
            ('micro_model', 'Micro-price'), ('flow_model', 'Trade flow')]
    rows = []
    for league in SEASON:
        keys = {k for k in base if k[0] == league and k in main_g}
        if not keys:
            continue
        records = {arm: {k: main_g[k] if arm == 'model' else base[k][arm] for k in keys}
                   for arm, _ in arms}
        for arm, _ in arms:
            pooled_key = f'base/{league}/{arm}'
            if pooled_key not in summary:
                summary[pooled_key] = stats(records[arm], league)
        for kind in KINDS:
            for arm, label in arms:
                st = stats(records[arm], league, (kind,))
                summary[f'base/{league}/{kind}/{arm}'] = st
                rows.append([f'{league} {kind}' if arm == 'model' else '', label,
                             str(st['games']), f1(st['net_c_per_contract'], 2),
                             f1(100 * st['fill_win'], 0) + '\\%', f1(st['profit_factor'], 2),
                             f1(sharpe(st, league), 2), f1(st['max_drawdown'], 2)])
    rows = highlight_best(rows, (3, 4, 5, 6, 7), minimize=(7,))
    head = ['Contract', 'Strategy', 'Games', 'Net c/unit', 'Win', 'PF', 'Sharpe (ann.)', 'Max DD (\\$)']
    note = ('Bold marks the best value within each league and contract type. Drawdown is in dollars; '
            'model sizing differs from the one-contract baselines. PF: profit factor.')
    return table('Model and Baseline Performance', 'tab:baselines', head, rows, note, placement='H')


def main():
    from b68_all_totals_reporting import build
    pending = build()
    if any(pending.values()):
        raise RuntimeError('Complete b68_all_totals_reporting.py --refresh before generating unrestricted totals')
    main_g = load(MAIN)
    summary, tex = {}, []
    # 1. main table, league x contract type
    rows, per_league = [], {}
    for lg in ('MLB', 'NBA', 'NFL'):
        for kinds, name in ((('spread',), 'Spread'), (('total',), 'Total'), (KINDS, 'All')):
            s = stats(main_g, lg, kinds)
            summary[f'main/{lg}/{name}'] = s
            if name == 'All':
                per_league[lg] = s
                continue
            rows.append([lg if name == 'Spread' else '', name, str(s['games']), str(s['fills']), f1(s['contracts'], 0), money(s['gross']),
                         money(s['fees']), money(s['net']), f1(s['net_c_per_contract'], 2)])
    tex.append(table('Forward-Test Trading Results',
                     'tab:live', ['League', 'Type', 'Games', 'Fills', 'Quantity', 'Gross', 'Fees', 'Net', 'Net c/contract'], rows,
                     '$20\\times$ sizing; fees included. Fills count entry lots.'))
    # 3. guard on/off
    ng = merge(MAIN, 'b70_noguard_all_totals')
    ng_games = set(load('b70_noguard_all_totals'))
    rows = []
    for lg in ('MLB', 'NBA', 'NFL'):
        for name, g in (('Guard on', main_g), ('Guard off', ng)):
            s = stats(g, lg)
            summary[f'guard/{lg}/{name}'] = s
            rows.append([lg if name == 'Guard on' else '', name, str(s['games']), str(s['fills']), f1(s['contracts'], 0), money(s['net']),
                         f1(s['net_c_per_contract'], 2), f1(100 * s['fill_win'], 0) + '\\%'])
    tex.append(table('Feed-Guard Comparison',
                     'tab:guard', ['League', 'Arm', 'Games', 'Fills', 'Quantity', 'Net', 'Net c/contract', 'Fill win'], rows,
                     'League-specific guards; other conditions are fixed.'))
    summary['guard/meta'] = dict(games=len(ng_games))
    # 5. baselines: one replay holds the six arms; the model row uses the same games
    base = load_arms('b69_baselines_all_totals')
    nb = {lg: len([k for k in base if k[0] == lg and k in main_g]) for lg in SEASON}
    nm = {lg: len([k for k in main_g if k[0] == lg]) for lg in SEASON}
    tex.append(baseline_comparison_table(main_g, base, summary))
    GEN.mkdir(parents=True, exist_ok=True)
    plot_baseline_comparison(main_g, base, summary)
    for name, t in zip(('main', 'guard', 'baselines'), tex):     # one file per table, in creation order
        (GEN / f'results_tables_{name}.tex').write_text('% generated by analysis/graph_2026_10/b61_make_tables.py: do not edit\n' + t + '\n')
    (OUT / 'b61_summary.json').write_text(json.dumps({k: ({a: b for a, b in v.items() if a != 'per_game'} if isinstance(v, dict) else v) for k, v in summary.items()}, indent=1))
    # every summary number as a LaTeX macro (\\resmainNBAAllnet ...), so prose can quote refreshed values
    import re
    mac = []
    for k, v in summary.items():
        items = {'': v} if not isinstance(v, dict) else {a: b for a, b in v.items() if isinstance(b, (int, float)) and a != 'season'}
        for a, b in items.items():
            name = re.sub('[^A-Za-z]', '', k) + re.sub('[^A-Za-z]', '', a)
            mac.append('\\newcommand{\\res%s}{%s}' % (name, ('%.2f' % b).replace('-', '$-$') if isinstance(b, float) else str(b)))
    (GEN / 'results_macros.tex').write_text('% generated by b61_make_tables.py: do not edit\n' + '\n'.join(sorted(set(mac))) + '\n')
    print('games', {lg: per_league[lg]['games'] for lg in per_league})


if __name__ == '__main__':
    main()
