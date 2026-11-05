"""Build tables and a report from completed standardized replay jobs."""
from __future__ import annotations

import collections
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('out')/'b37_standard_replay'


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def rows(path):
    if not path.exists():
        path = Path(str(path)+'.gz')
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        yield from map(json.loads,f)


def coverage(result):
    if result['mode']=='reconstructed':
        return dict(coverage='reconstructed',initial_feed_plays=None)
    root = ROOT/result['root']
    candidates = []
    for a in rows(root/'feed_arrivals.jsonl'):
        p = root/'feed'/(a['stamp']+'.json.gz')
        if p.exists():
            candidates.append((a['received_at'],p))
    _,p = min(candidates)
    d = json.loads(gzip.decompress(p.read_bytes()))
    if result['league']=='MLB':
        n = sum(e.get('isPitch',False) for pa in d['feed']['liveData']['plays']['allPlays'] for e in pa['playEvents'])
    else:
        # A recording starting with a sizeable play history cannot recreate the
        # missed live period, even when it later reaches the final score.
        n = len(d.get('plays') or [])
        if not n and d.get('drives'):
            n = sum(len(x.get('plays') or []) for x in (d['drives'].get('previous') or []))
            n += len((d['drives'].get('current') or {}).get('plays') or [])
    label = 'starts_midgame' if n>5 else 'recorded_from_start'
    if not result['final_score_available']:
        label += ';unfinished_feed'
    if result.get('missing_feed_files'):
        label += ';missing_snapshots'
    return dict(coverage=label,initial_feed_plays=n)


def main():
    manifest = json.loads((OUT/'rule.json').read_text())
    manifest_hash = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    files = [p for p in sorted(OUT.glob('*/*/*/result.json')) if json.loads(p.read_text()).get('manifest_hash')==manifest_hash]
    extension = json.loads((OUT/'mlb_reconstruction_rule.json').read_text()) if (OUT/'mlb_reconstruction_rule.json').exists() else {}
    files = [p for p in files if p.parent.name.startswith('NBA_') or p.parent.parent.parent.name!='reconstructed'
             or json.loads(p.read_text()).get('extension_sha256')==extension.get('source_sha256')]
    records, fills = [], []
    results = [json.loads(p.read_text()) for p in files]
    for p,r in zip(files,results):
        c = coverage(r)
        g = json.loads((ROOT/r['root']/'game.json').read_text())
        for arm,stats in r['results'].items():
            common = dict(mode=r['mode'],phase=r['phase'],date=r['date'],league=r['league'],game=r['game'],
                          matchup=g['away']+'@'+g['home'],arm=arm,source=r['root'],final_score_available=r['final_score_available'],**c)
            for kind,s in stats['by_kind'].items():
                records.append(dict(common,kind=kind,**s,orders_all_kinds=stats['orders'],
                                    unresolved_fills_all_kinds=stats['unresolved_fills']))
            for o in json.loads((p.parent/arm/'fills.json').read_text()):
                fills.append(dict(common,kind=o['kind'],id=o['id'],ticker=o['ticker'],side=o['side'],size=o['size'],
                                  entry=o.get('entry',o['limit']),signal=o['signal'],t0=o['t0'],t_fill=o['t_fill'],
                                  t_exit=o.get('t_exit'),stage=o['stage'],how=o.get('how','settlement'),net_c=o['net'],
                                  pnl_c=o['pnl_c'],resolved=o['resolved']))
    csv_write(OUT/'games.csv',records)
    csv_write(OUT/'fills.csv',fills)
    groups = {}
    rng = np.random.default_rng(37)
    for keys in [('mode','phase','arm','league','kind'),('mode','phase','arm','league','date','kind'),
                 ('mode','phase','arm','league','kind','coverage')]:
        grouping = collections.defaultdict(list)
        for r in records:
            grouping['|'.join(str(r[k]) for k in keys)].append(r)
        out = {}
        for key,gs in sorted(grouping.items()):
            valid = [r for r in gs if r['final_score_available']]
            n = sum(r['contracts'] for r in valid)
            pnl = sum(r['pnl_c'] for r in valid)
            est = dict(games=len(gs),settled_games=len(valid),fills=sum(r['fills'] for r in valid),contracts=n,pnl_c=pnl,
                       net_per_contract_c=pnl/n if n else None)
            if n and len(valid)>1 and len(keys)==5:
                a = np.array([[r['pnl_c'],r['contracts']] for r in valid])
                bs = a[rng.integers(0,len(a),(10000,len(a)))].sum(1)
                bs = bs[bs[:,1]>0]
                est['lo95_c'],est['hi95_c'] = np.percentile(bs[:,0]/bs[:,1],[2.5,97.5]).tolist()
            out[key] = est
        groups['/'.join(keys)] = out
    (OUT/'groups.json').write_text(json.dumps(groups,indent=2)+'\n')
    inventory = json.loads((OUT/'inventory.json').read_text())
    regular = groups['mode/phase/arm/league/kind']
    report = [
        'STANDARDIZED REPLAY — 29 OCTOBER 2026',
        '',
        f'Completed game/scenario jobs: {len(results)}. Main inventory: {len(inventory["entries"])}; reconstructed MLB supplements are separate jobs.',
        '',
        'Rule: current local live_trade.py and live_model.py, fingerprinted with model weights in rule.json.',
        'MLB: five-seed d10 pitch ensemble, delay 10 s. NBA/NFL: five-seed d60 ensemble, delay 60 s.',
        'Both spread and total markets; decisions every 5 s; NBA real-stamp agreement gate and late totals block on;',
        'max-of-last-five feed-lag guard, NBA reviews/placeholders and MLB pending-ball/15–60 s guards as in current code.',
        'Minimum signal 5c, squared signal times side-split print-width sizing; one order/position per ticker.',
        'Strict through-print fills, 60 s entry, 120 s hold, 120 s maker exit then 120 s opposite-print fallback.',
        'Entry/exit prices and all stage transitions are resimulated; these are not resized historical fills.',
        'Primary latest arm: fractional weights rounded to 0.01, matching current deployed default.',
        'Companion whole5 arm: round(5 * weight), half up; zero-size orders omitted. Both arms use identical observations.',
        'Maker/taker fee coefficients are the engine\'s 1.75 / 7 cents times p(1-p).',
        '',
        'RESULTS: available recorded segments, regular season/postseason, final-score games only',
        'Arm         League Kind       Games Fills Contracts       Net c   Net c/contract',
    ]
    for key,s in regular.items():
        mode,phase,arm,league,kind = key.split('|')
        if mode!='recorded' or phase!='regular_or_postseason':
            continue
        edge = f'{s["net_per_contract_c"]:.3f}' if s['net_per_contract_c'] is not None else 'n/a'
        report.append(f'{arm:11s} {league:6s} {kind:10s} {s["settled_games"]:5d} {s["fills"]:5d} {s["contracts"]:9.3f} {s["pnl_c"]:11.3f} {edge:>16s}')
    report += ['', 'COVERAGE AND INTERPRETATION',
        'Recorded mode preserves recorded feed snapshots and the original trade/candle arrival timestamps.',
        'Main sources are VM recordings, downloaded sandbox replay_inputs, then earlier local sandbox/local recordings.',
        'One source per game: duplicate sandbox/VM runs are never pooled as independent games.',
        'Preseason NBA recordings and reconstructed feeds are separate groups in groups.json and games.csv.',
        'Reconstructed NBA: finished feeds with d60 gating, trades/candles at event time; no historical live feed revisions.',
        'Reconstructed MLB supplements: archived feed cut at each pitch end, arriving +10 s; tape/candles at event time.',
        'These reconstructed results are idealized-input counterfactuals, not additional live results.',
        'A recording that starts midgame remains a partial recording; see the coverage column.',
        'The main table includes those partial segments; mode/phase/arm/league/kind/coverage gives a separate breakdown.',
        'Unfinished feeds cannot supply reliable settlement outcomes. Their resolved-only amounts remain in game files,',
        'but those games are excluded from pooled settled results above.',
        'Retain the current engine\'s behavior after a final-status feed rather than adding a new trading rule.',
        'Only received observations enter the market and play history. Cached GRU encodings are exact functions of received play tensors.',
        'The two sizing arms have separate paper books. Identical forecasts are reused; forecasts are independent of paper positions.',
        'Validation: latest-arm order events for 27 Oct SAC–SA are exactly equal to the existing full replay (14 events, six orders, one fill).',
        'This replay models no queue priority, partial fills, impact or order/cancellation latency beyond the existing engine.',
        'Archived order books were not used for the primary rule, which fills on strict trade-throughs.',
        'Game-bootstrap intervals are descriptive; they do not account for shared nightly conditions or selection of the rule.',
        'Some current guards were chosen after observing these dates. Standardization does not turn retrospective replays into a fresh out-of-sample test.',
        'No live trading or deployment code/configuration was changed.',
        '',
        'FILES',
        'rule.json: frozen parameters and code/weight hashes; inventory.json: input selection and exclusions.',
        'games.csv: per-game/type/arm accounting; fills.csv: every resimulated positive-size fill.',
        'groups.json: league/type/night/coverage totals; recorded/ and reconstructed/: per-game event logs and results.',
        'errors.json and progress.json: batch completion state. Incomplete batches must not be reported as complete.',
        '',
        'REPRODUCE',
        'venv/bin/python -u -W ignore analysis/graph_2026_10/b37_standard_replay.py --workers 4',
        'venv/bin/python -u -W ignore analysis/graph_2026_10/b37_reconstruct_mlb.py',
        'venv/bin/python analysis/graph_2026_10/b37_summarise.py',
    ]
    (OUT/'report.txt').write_text('\n'.join(report)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/b37_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [(l,k) for l in ['NBA','NFL','MLB'] for k in ['spread','total']]
    fig,ax = plt.subplots(figsize=(9,4.7))
    x = np.arange(len(labels))
    for arm,shift,color,label in [('latest',-.18,'#276B9B','Latest: fractional'),('whole5',.18,'#E39C35','Whole contracts: K=5')]:
        values = [regular.get('|'.join(('recorded','regular_or_postseason',arm,l,k)),{}).get('net_per_contract_c') for l,k in labels]
        ax.bar(x+shift,[np.nan if v is None else v for v in values],width=.34,color=color,label=label)
    ax.axhline(0,color='#444',linewidth=.8)
    ax.set_xticks(x,[l+'\n'+k for l,k in labels])
    ax.set_ylabel('Net cents per filled contract')
    ax.set_title('Same current rule and models across recorded runs')
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.08,.015,'Final-score games; recorded segments only. Partial coverage is flagged in games.csv. NBA preseason and reconstructed feeds excluded.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(OUT/'league_results.png',dpi=180)
    plt.close(fig)
    print('\n'.join(report[:35]))


if __name__=='__main__':
    main()
