"""Chronological comparison of the fixed b38 strategy families."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import os
from pathlib import Path

OUT=Path(__file__).with_name('out')/'b38_strategy_study'


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def aggregate(games,name):
    xs=[(g,g['results'][name]) for g in games]
    n=sum(s['contracts'] for g,s in xs);p=sum(s['pnl_c'] for g,s in xs)
    daily=collections.defaultdict(float)
    for g,s in xs:daily[g['date']]+=s['pnl_c']
    return dict(games=len(xs),fills=sum(s['fills'] for g,s in xs),contracts=n,pnl_c=p,
                pnl_stress_c=sum(s['pnl_stress_c'] for g,s in xs),pnl_stress_half_c=sum(s['pnl_stress_half_c'] for g,s in xs),
                net_per_contract_c=p/n if n else None,peak_single_game_risk_c=max((s['peak_principal_risk_c'] for g,s in xs),default=0),
                worst_game_c=min((s['pnl_c'] for g,s in xs),default=0),
                best_game_c=max((s['pnl_c'] for g,s in xs),default=0),
                without_best_game_c=p-max((s['pnl_c'] for g,s in xs),default=0),
                positive_nights=sum(v>0 for v in daily.values()),negative_nights=sum(v<0 for v in daily.values()))


def main():
    rule=json.loads((OUT/'study_rule.json').read_text())
    h=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()
    all_results=[json.loads(p.read_text()) for p in sorted(OUT.glob('games/*/*/result.json'))]
    results=[r for r in all_results if r.get('study_hash')==h and not r.get('excluded')]
    excluded=[r for r in all_results if r.get('study_hash')==h and r.get('excluded')]
    names=[c['name'] for c in rule['candidates']]
    rows=[];groups={};selected={};daily=[];walk=[]
    for league in ['ALL','NBA','NFL','MLB']:
        gs=[g for g in results if league=='ALL' or g['league']==league]
        subsets={'design':[g for g in gs if g['date']<='2026-10-25'],
                 'check':[g for g in gs if g['date']>='2026-10-26'],'all':gs}
        for split,gg in subsets.items():
            for name in names:
                s=aggregate(gg,name);rows.append(dict(league=league,split=split,candidate=name,**s));groups[(league,split,name)]=s
        ranked=sorted(names,key=lambda name:groups[league,'design',name]['pnl_stress_c'],reverse=True)
        winner=ranked[0] if ranked and groups[league,'design',ranked[0]]['pnl_stress_c']>0 else 'cash'
        selected[league]=dict(candidate=winner,design=groups.get((league,'design',winner)),
                              check=groups.get((league,'check',winner)),baseline_check=groups[league,'check','baseline'])
        for day in sorted({g['date'] for g in gs}):
            past=[g for g in gs if g['date']<day];today=[g for g in gs if g['date']==day]
            if sum(g['results']['baseline']['fills'] for g in past)<20:
                choice='baseline'
            else:
                choice=max(names,key=lambda name:aggregate(past,name)['pnl_stress_c'])
                if aggregate(past,choice)['pnl_stress_c']<=0:choice='cash'
            s=aggregate(today,choice) if choice!='cash' else dict(pnl_c=0,pnl_stress_c=0,fills=0,contracts=0)
            b=aggregate(today,'baseline')
            walk.append(dict(league=league,date=day,choice=choice,pnl_c=s['pnl_c'],pnl_stress_c=s['pnl_stress_c'],
                             fills=s['fills'],contracts=s['contracts'],baseline_c=b['pnl_c'],baseline_stress_c=b['pnl_stress_c']))
        for day in sorted({g['date'] for g in gs}):
            gg=[g for g in gs if g['date']==day]
            for name in names:daily.append(dict(league=league,date=day,candidate=name,**aggregate(gg,name)))
    write_csv(OUT/'comparisons.csv',rows);write_csv(OUT/'daily.csv',daily);write_csv(OUT/'walk_forward.csv',walk)
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    selected_name=selected['ALL']['candidate']
    report=['STRATEGY LOSS-CONTROL STUDY — 29 OCTOBER 2026','',
            f'Completed eligible game segments: {len(results)}; excluded unfinished feeds: {len(excluded)}.',
            'Twenty fixed strategy variants, K=5, with latest model forecasts and existing live guards held constant.',
            'Every order, entry fill, exit and subsequent opportunity is resimulated. Baseline fill paths and net P&L',
            'must exactly reproduce the standardized replay in each eligible game before variants are accepted.',
            '',
            'DESIGN: dates through 25 October. CHRONOLOGICAL CHECK: 26–28 October.',
            'Selection: greatest design-period P&L after adding 1c per contract per executed leg to costs.',
            'If no candidate is positive after that stress, the selected alternative is cash.',
            'No later-period outcome enters that selection. These are retrospective dates that have already been',
            'viewed, and some base guards were developed on them: this is not an untouched holdout.',
            'Preseason and reconstructed-feed replays are excluded from the strategy-selection sample.',
            '',f'Portfolio candidate selected on design dates: {selected_name}',
            '', 'Candidate           Design stress c   Later stress c   Later raw c   Later fills   Later worst game c']
    ranked=sorted(names,key=lambda name:groups['ALL','design',name]['pnl_stress_c'],reverse=True)
    for name in ranked:
        d=groups['ALL','design',name];c=groups['ALL','check',name]
        report.append(f'{name:21s} {d["pnl_stress_c"]:14.2f} {c["pnl_stress_c"]:16.2f} {c["pnl_c"]:13.2f} {c["fills"]:13d} {c["worst_game_c"]:20.2f}')
    report+=['','LEAGUE-SPECIFIC DESIGN SELECTIONS']
    for league in ['NBA','NFL','MLB']:
        s=selected[league];c=s['check'];b=s['baseline_check']
        report.append(f'{league}: {s["candidate"]}; later games {b["games"]}; '+
                      (f'later net {c["pnl_c"]:.2f}c, stress {c["pnl_stress_c"]:.2f}c; baseline {b["pnl_c"]:.2f}c.' if c else 'cash, no exposure.'))
    report+=['','LIMITS AND EXECUTION CHECKS',
             'Higher paper profit does not establish a live edge. Strategy selection searches 20 alternatives;',
             'do not read an unadjusted confidence interval or the best retrospective variant as confirmation.',
             'Check per-night stability, results without the best game, fills and exposure in comparisons.csv.',
             'The later NFL sample contains only the recorded games after 25 October, not another full Sunday.',
             'walk_forward.csv additionally chooses each day using prior dates only (baseline until 20 prior fills).',
             'This study initially retains the current strict through-print fills and backdated paper exit behavior.',
             'b39_execution_checks separately tests causal exit posting and sufficient limit-fill print volume.',
             'Those checks still do not reproduce full queue priority, market impact or a real order lifecycle.',
             'Book-quote reconstruction is used to check entry/exit liquidity and whether print prices were executable.',
             'Costs: current engine maker/taker formulas plus 0.5c and 1c per contract per executed leg sensitivity.',
             'Per-series multipliers and account rounding must match the actual exchange account before live use.',
             'Official fee schedule: https://kalshi.com/docs/kalshi-fee-schedule.pdf',
             'No live strategy changes or trades were made. Any candidate needs forward paper validation.',
             '', 'FILES: comparisons.csv, daily.csv, walk_forward.csv, selection.json, study_rule.json, games/.']
    (OUT/'report.txt').write_text('\n'.join(report)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/b37_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    display=ranked[:8]
    if 'baseline' not in display:display+=['baseline']
    fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)
    for ax,split,title in zip(axes,['design','check'],['Design dates through 25 Oct','Later dates 26–28 Oct']):
        values=[groups['ALL',split,n]['pnl_stress_c']/100 for n in display]
        ax.barh(display,values,color=['#277A61' if v>=0 else '#BD5B4A' for v in values]);ax.axvline(0,color='#444',lw=.8)
        ax.set_title(title);ax.set_xlabel('Net dollars after additional 1c/contract/leg');ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle('Candidates ranked using design-period results only')
    fig.text(.05,.015,'Retrospective chronological check; K=5. Paper execution assumptions require separate validation.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(OUT/'candidate_results.png',dpi=180);plt.close(fig)
    print('\n'.join(report[:39]))


if __name__=='__main__':main()
