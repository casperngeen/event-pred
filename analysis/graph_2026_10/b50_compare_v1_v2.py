"""Event replay v2 (exchange-clock book, Kalshi fees) against the print-fill standardized replay (b37) on the same games.

b37: strict through-print fills, flat 1.75 / 7 fee coefficients, 'latest' sizing (fractional weights to 0.01).
b49: same engine decisions, L2 book on the exchange clock, per-series Kalshi fees; arms host / slow / stress latency.
"""
import collections
import json
from pathlib import Path

import pandas as pd

OUT=Path(__file__).with_name('out')
B37=OUT/'b37_standard_replay'/'games.csv'
B49=OUT/'b49_event_replay'


def v2_rows():
    rows=[]
    for p in sorted(B49.glob('games/*/*/result.json')):
        r=json.loads(p.read_text())
        for arm,s in r.get('results',{}).items():
            for kind,a in s['by_kind'].items():
                rows.append(dict(date=r['date'],league=r['league'],game=int(r['game']),kind=kind,arm=arm,coverage=r['coverage'],host=r.get('host'),
                                 orders=s['orders'],fills=a['fills'],filled_orders=a.get('filled_orders'),contracts=a['entry_contracts'],
                                 gross_c=a['gross_c'],net_c=a['pnl_c'],net1c_c=a['pnl_1c_c'],fees_c=a['fees_c'],settled=a['settled_contracts']))
    return pd.DataFrame(rows)


def main():
    v2=v2_rows()
    old=pd.read_csv(B37)
    old=old[(old['mode']=='recorded')&(old['arm']=='latest')&(old['phase']=='regular_or_postseason')]
    old=old.rename(columns={'fills':'fills_v1','contracts':'contracts_v1','pnl_c':'net_v1_c'})[['date','league','game','kind','coverage','fills_v1','contracts_v1','net_v1_c']].rename(columns={'coverage':'coverage_v1'})
    lines=['EVENT REPLAY v2 vs PRINT-FILL REPLAY (b37), same games','',
           'v1 = strict through-print fills, flat fee coefficients. v2 = exchange-clock L2 book, queue, per-series Kalshi fees (direct-member precision; net$0.01 = FCM-cleared).',
           'ASSUMPTION (decided 2026-10-30): direct member, $0.0001 balance precision; not confirmed with Kalshi. net$0.01 is the FCM-cleared sensitivity.',
           'Orders are decided afresh in each run; fills and exits are resimulated. Contracts are filled contracts.','']
    both=v2[v2.arm=='host'].merge(old,on=['date','league','game','kind'],how='inner')
    lines.append(f'Games in both: {both.game.nunique()} (v2 host arm {v2[v2.arm=="host"].game.nunique()} games, v1 {old.game.nunique()}).')
    lines.append('')
    lines.append('league kind | games | v1 fills | v1 contracts | v1 net c/ct || v2 fills | v2 contracts | v2 gross c/ct | v2 net c/ct | v2 net$0.01 c/ct')
    def row(name,d):
        c1=d.contracts_v1.sum();c2=d.contracts.sum()
        f=lambda x,n:f'{x/n:7.2f}' if n else '    n/a'
        return f'{name:<12}| {d.game.nunique():5d} | {int(d.fills_v1.sum()):8d} | {c1:12.2f} | {f(d.net_v1_c.sum(),c1)} || {int(d.fills.sum()):8d} | {c2:12.2f} | {f(d.gross_c.sum(),c2)} | {f(d.net_c.sum(),c2)} | {f(d.net1c_c.sum(),c2)}'
    for (lg,k),d in both.groupby(['league','kind']):lines.append(row(f'{lg} {k}',d))
    lines.append(row('ALL',both))
    clean=both[(both.coverage=='recorded_from_start')&(both.coverage_v1=='recorded_from_start')]
    lines+=['','CLEAN SUBSET: recorded from the first play, no book-state errors or truncation in v2 (partial and flagged recordings removed)',
            'NOTE sizing differs: v1 = 1x normalized weight rounded to 0.01 contracts, v2 = 5x weight rounded to 0.01; c/contract partly absorbs this, the set of orders does not match.','',
            'league kind | games | v1 fills | v1 contracts | v1 net c/ct || v2 fills | v2 contracts | v2 gross c/ct | v2 net c/ct | v2 net$0.01 c/ct']
    for (lg,k),d in clean.groupby(['league','kind']):lines.append(row(f'{lg} {k}',d))
    lines.append(row('ALL',clean))
    lines+=['','LATENCY ARMS (v2, all games, per league and kind: net c/contract after Kalshi fees)','league kind | host | slow | stress | contracts (host)']
    pv=v2.groupby(['league','kind','arm']).agg(c=('contracts','sum'),n=('net_c','sum')).reset_index()
    for (lg,k),d in pv.groupby(['league','kind']):
        g={r.arm:(r.n/r.c if r.c else float('nan')) for r in d.itertuples()};c=d[d.arm=='host'].c.sum()
        lines.append(f'{lg} {k:<7}| {g.get("host",float("nan")):6.2f} | {g.get("slow",float("nan")):6.2f} | {g.get("stress",float("nan")):6.2f} | {c:8.2f}')
    lines+=['','BY RECORDING HOST (v2 host arm): net c/contract','host league kind | games | contracts | net c/ct']
    for (h,lg,k),d in v2[v2.arm=='host'].groupby(['host','league','kind']):
        c=d.contracts.sum();lines.append(f'{h} {lg} {k:<7}| {d.game.nunique():5d} | {c:8.2f} | {d.net_c.sum()/c if c else float("nan"):6.2f}')
    (B49/'comparison.txt').write_text('\n'.join(lines)+'\n');both.to_csv(B49/'comparison_by_game.csv',index=False)
    print('\n'.join(lines))


if __name__=='__main__':main()
