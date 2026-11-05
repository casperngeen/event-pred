"""Compare 30/60/120-second historical strategies on development folds only.
Run from repository root: venv/bin/python -W ignore analysis/graph_2026_10/thesis_horizon_comparison.py
Generate 30-second logs first with BACKTEST_HOLDS=30s and b33_ablation_trades.sh
for each league, gru, the league's feed delay (MLB 10 s, NBA/NFL 60 s), and its final-model tag (MLB includes _pitch).
Uses the five-seed final-model logs (MLB 10 s, NBA/NFL 60 s delay) and historical normalised
fractional sizing (no twenty-times rounding). Matching forecast/holding horizons;
entry/exit resting deadlines remain fixed. No model retraining or holdout access.
Writes out/thesis_horizon_comparison.csv. Paired t uses per-game total net P&L
on the union of games traded by either horizon, with zeros for absent trades.
"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, str(Path('analysis/graph_2026_10').resolve()))
from b33_ablation_table import FULL, OUT, SIZE, liquidity, with_net, per_game, tstat
rows=[]
for league in ('MLB','NBA','NFL'):
    original=sorted(OUT.glob(f'{FULL.at(league).log_stem(league)}_fold*.parquet'))
    short=sorted(OUT.glob(f'{FULL.at(league).log_stem(league)}_holds30s_fold*.parquet'))
    assert len(original)==len(short)==5, f'Missing development fold logs for {league}'
    files=original+short
    f=pl.concat([pl.read_parquet(p) for p in files],how='vertical_relaxed').filter(pl.col('k')==1,pl.col('through')==0)
    f=liquidity(with_net(f)).with_columns(SIZE.alias('size')).filter(pl.col('size')>0).with_columns((pl.col('size')*pl.col('net')).alias('pnl'))
    for kind in ('spread','total'):
        g=f.filter(pl.col('kind')==kind)
        parts={h:g.filter(pl.col('hold')==h) for h in ('30s','60s','120s')}
        def paired(shorter):
            games=pl.concat([parts['120s']['game_id'],parts[shorter]['game_id']]).unique()
            return tstat(per_game(parts['120s'],games)-per_game(parts[shorter],games))
        pair_t=paired('60s')
        pair30=paired('30s')
        for h,x in parts.items():
            size=x['size'].sum()
            rows.append(dict(league=league,kind=kind,horizon=h,trades=x.height,contracts=round(size,2),gross_c=round((x['gross']*x['size']).sum()/size,2),net_c=round(x['pnl'].sum()/size,2),net_usd=round(x['pnl'].sum()/100,2),paired_t_120_minus_60=pair_t,paired_t_120_minus_30=pair30))
    print('Completed',league,flush=True)
result=pl.DataFrame(rows)
result.write_csv(OUT/'thesis_horizon_comparison.csv')
print(result)
