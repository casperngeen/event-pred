"""Finish local reports without keeping an assistant turn active.

Reads existing replay/study checkpoints, waits for pending local jobs, and runs
the summary writers. No trading, network calls or original-data mutations.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=HERE/'out/b37_standard_replay'
STUDY=HERE/'out/b38_strategy_study'


def comparison():
    reference={};section=None
    for line in (HERE/'out/b34_integer_sizing.txt').read_text().splitlines():
        if line.startswith('================ '):section=line.removeprefix('================ ')
        if '│ frozen (fractional)' in line or '│ round, K=5 ' in line:
            parts=[x.strip() for x in line.strip('│').split('┆')]
            arm='latest' if parts[0]=='frozen (fractional)' else 'whole5'
            league,kind=section.split()
            reference[league,kind,arm]=dict(backtest_net_c=float(parts[-4]),backtest_fills=int(parts[1]),backtest_contracts=float(parts[2]))
    results=[json.loads(p.read_text()) for p in BASE.glob('recorded/*/*/result.json')]
    rows=[]
    for cohort in ['all_recorded','recent_24_28','preseason']:
        gs=[r for r in results if r['final_score_available'] and
            ((cohort=='preseason' and r['phase']=='preseason') or
             (cohort!='preseason' and r['phase']!='preseason' and (cohort=='all_recorded' or r['root'].startswith('data/sports_live/vm'))))]
        for league,kind,arm in reference:
            xs=[(r,r['results'][arm]['by_kind'][kind]) for r in gs if r['league']==league]
            if not xs:continue
            n=sum(x['contracts'] for r,x in xs);p=sum(x['pnl_c'] for r,x in xs)
            rows.append(dict(cohort=cohort,league=league,kind=kind,arm=arm,
                             approved_keep=not(league=='NBA' and kind=='total'),
                             **reference[league,kind,arm],forward_games=len(xs),forward_fills=sum(x['fills'] for r,x in xs),
                             forward_contracts=n,forward_pnl_c=p,forward_net_c=p/n if n else None,
                             without_best_game_c=p-max((x['pnl_c'] for r,x in xs),default=0)))
    with (BASE/'backtest_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=['BACKTEST VS STANDARDIZED FORWARD REPLAY','',
            'Reference: b34_integer_sizing.txt, deployed d60 NBA/NFL models and d10 MLB pitch model.',
            'All recorded regular/postseason segments with final scores, latest-rule replay. NBA totals are marked for exclusion.',
            'The original rule was amended on 8 October to include all six spread/total market types.',
            '', 'Market        Sizing     Backtest c/contract  Forward c/contract  Forward fills']
    for r in rows:
        if r['cohort']!='all_recorded' or not r['approved_keep']:continue
        edge=f'{r["forward_net_c"]:.2f}' if r['forward_net_c'] is not None else 'n/a'
        report.append(f'{r["league"]+" "+r["kind"]:13s} {r["arm"]:10s} {r["backtest_net_c"]:19.2f} {edge:>19s} {r["forward_fills"]:14d}')
    report+=['','Interpretation limits:',
             'Positive historical and forward estimates support the same direction; few nights/games and correlated rungs limit confidence.',
             'Standardized retrospective replays are counterfactuals, not all original live orders.',
             'Some live guards and the NBA-total exclusion were chosen after seeing these dates.',
             'The K=5 backtest reference resizes the historical fills; the forward K=5 arm reruns the full order path.',
             'Compare the fractional arm as well, and do not attribute sizing/path differences to model skill alone.',
             'The latest guard behavior and real feed revisions cannot be identically reproduced by idealized backtest feeds.',
             'Execution sensitivities are in b39_execution_checks. These are not full order-book queue or market-impact validations.',
             'NBA preseason and reconstructed feeds remain separate cohorts; they are not pooled into regular recorded forward results.',
             'The previously opened large holdout report (b17_holdout_report.txt) is additional evidence, with different timing/model settings:',
             'MLB spread +5.43c (924 games), MLB total +5.00c (1106 games), NFL spread +4.61c (64 games), NFL total +0.10c (64 games).',
             'Do not pool those holdout figures with the current d60 live replay. They particularly caution against treating NFL totals as confirmed.',
             'No live deployment was changed. Suggested retained NBA universe: spread only.']
    (BASE/'backtest_comparison.txt').write_text('\n'.join(report)+'\n')


def summarize(script):
    with (BASE/(script.stem+'_finalize.log')).open('w') as f:
        subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    completed=set();started=time.time();last=None
    while time.time()-started<6*3600:
        inv=json.loads((BASE/'inventory.json').read_text())['entries']
        expected=len(inv)+5
        rule=json.loads((BASE/'rule.json').read_text());h=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()
        rs=[json.loads(p.read_text()) for p in BASE.glob('*/*/*/result.json')]
        rs=[r for r in rs if r.get('manifest_hash')==h]
        regular=sum(r['mode']=='recorded' and r['phase']!='preseason' for r in rs)
        reg_expected=sum(e['mode']=='recorded' and e['phase']!='preseason' for e in inv)
        sr=json.loads((STUDY/'study_rule.json').read_text())
        sh=hashlib.sha256(json.dumps(sr,sort_keys=True).encode()).hexdigest()
        nstudy=sum(json.loads(p.read_text()).get('study_hash')==sh for p in STUDY.glob('games/*/*/result.json'))
        state=dict(replays_complete=len(rs),replays_expected=expected,recorded_regular_complete=regular,
                   recorded_regular_expected=reg_expected,strategy_games_complete=nstudy,
                   reports=list(sorted(completed)),remote_book_artifacts='/tmp/event-pred-b38-l1 on Lumid',
                   note='Remote book extraction is separate. No assistant tokens are used by these processes.')
        (BASE/'background_status.json').write_text(json.dumps(state,indent=2)+'\n')
        key=(len(rs),nstudy)
        if key!=last:print(json.dumps(state),flush=True);last=key
        if regular>=reg_expected and 'forward_comparison' not in completed:
            comparison();summarize(HERE/'b37_summarise.py');completed.add('forward_comparison')
        if nstudy>=reg_expected and 'strategy' not in completed:
            summarize(HERE/'b38_summarise.py');completed.add('strategy')
        if len(rs)>=expected:
            summarize(HERE/'b37_summarise.py');comparison();completed.add('all_replays')
        if {'all_replays','strategy'}<=completed:
            state.update(reports=sorted(completed),status='local reports complete')
            (BASE/'background_status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True);return
        time.sleep(15)
    raise TimeoutError('Background jobs did not finish within six hours; inspect their logs.')


if __name__=='__main__':main()
