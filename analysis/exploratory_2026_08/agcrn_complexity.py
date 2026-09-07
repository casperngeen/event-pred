"""Does the CA report's AGCRN earn its complexity?

Five cheap tests, all IN-SAMPLE ONLY (< 2026-01-01, per stg.splits.OOS_START).
The 2026 OOS block is not read by this script.

  T1  Capacity accounting   — AGCRN parameter count vs. supervisable labels.
  T2  Effective sample size — how many *independent* shocks the panel holds.
  T3  Edge vs. cost         — break-even hit rate given the measured spread.
  T4  Perfect-foresight PnL — an oracle upper bound on ANY model, after costs.
  T5  Complexity ladder     — zero / own / +graph, walk-forward within IS.

Panel: node = macro event, node value = ATM ticker daily close (report S4.3.1),
snapshot = resolution date of a finalised macro event (report S4.2).

Nodes whose OWN event resolves inside the label window are dropped: their move
is own-event news, not cross-market propagation, and including it flatters
every number below.

Run from event-pred/:  venv/bin/python ../analysis/exploratory_2026_08/agcrn_complexity.py
"""
from __future__ import annotations
import datetime as dt, math
import numpy as np, polars as pl

SCR = '/private/tmp/claude-503/-Users-caspe2-NUS-FYP/a09a383d-5b57-43d1-9091-34a28c3fb458/scratchpad'
IS_CUT = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
HORIZONS = (1, 2, 3)
LONG_H = 14
BUFFER = 2            # extra days of clearance from the node's own resolution
SPREAD_BOOK = 8.0     # research_log.md S7, order-book ground truth (cents)
SPREAD_TRADE = 1.0    # naive trade-bounce estimate (biased lower bound)
rng = np.random.default_rng(0)

def rule(t): print('\n' + '=' * 78 + f'\n{t}\n' + '=' * 78)
def fee(p_c):         # Kalshi taker fee, cents/contract: 7% * P * (1-P)
    p = np.asarray(p_c, float) / 100.0
    return 7.0 * p * (1 - p)

# ----------------------------------------------------------------- panel ---
mk = pl.read_parquet(f'{SCR}/macro_markets.parquet')
td = pl.read_parquet(f'{SCR}/tickerday.parquet')
assert mk['close_time'].max() < IS_CUT, 'OOS leak in markets'

keep = (mk.group_by('series').agg(pl.col('event_ticker').n_unique().alias('ev'))
          .filter(pl.col('ev') >= 5)['series'].to_list())
mk = mk.filter(pl.col('series').is_in(keep))

# ATM ticker chosen per (event, day); the SAME ticker is then tracked forward,
# as in report S4.3.1. Letting the ATM ticker switch between t and t+k compares
# two different contracts and manufactures spurious mean reversion (see note).
node = (td.join(mk.select('ticker', 'event_ticker', 'series', 'close_time'), on='ticker')
          .with_columns((pl.col('close') - 50).abs().alias('dist'))
          .sort(['event_ticker', 'd', 'dist'])
          .group_by(['event_ticker', 'd'])
          .agg(pl.col('ticker').first().alias('atm'),
               pl.col('series').first(), pl.col('close').first().alias('p'),
               pl.col('close_time').first().alias('res_time'),
               pl.col('vol').sum(), pl.col('ntrades').sum())
          .with_columns(pl.col('res_time').dt.date().alias('res_date'))
          .sort(['event_ticker', 'd']))

tick_px = td.select('ticker', 'd', pl.col('close').alias('px'))
for k in (*HORIZONS, LONG_H):
    fwd = (tick_px.with_columns((pl.col('d') - dt.timedelta(days=k)).alias('d'))
                  .rename({'px': f'p_fwd{k}', 'ticker': 'atm'}))
    node = (node.join(fwd, on=['atm', 'd'], how='left')
                .with_columns((pl.col(f'p_fwd{k}') - pl.col('p')).alias(f'y{k}')))

trig = (mk.filter(pl.col('status') == 'finalized')
          .group_by('event_ticker').agg(pl.col('series').first(),
                                        pl.col('close_time').min())
          .with_columns(pl.col('close_time').dt.date().alias('d')))
snap_dates = sorted(trig['d'].unique().to_list())

act = node.filter(pl.col('d').is_in(snap_dates) & (pl.col('res_date') > pl.col('d')))
# clearance: node's own resolution must sit beyond the longest label window
clean = act.filter(pl.col('res_date') >
                   pl.col('d') + dt.timedelta(days=LONG_H + BUFFER))

rule('PANEL (in-sample only, < 2026-01-01)')
print(f'series in universe                : {len(keep)}')
print(f'resolved macro events             : {trig.height}')
print(f'snapshot dates (resolution days)  : {len(snap_dates)}')
print(f'active node-snapshots             : {act.height}')
print(f'  mean nodes/snapshot             : {act.height/len(snap_dates):.1f}  (report: ~33)')
print(f'after own-resolution clearance    : {clean.height} '
      f'({100*clean.height/act.height:.0f}% kept)')
print(f'  mean nodes/snapshot             : '
      f'{clean.height/clean["d"].n_unique():.1f}')
for k in (*HORIZONS, LONG_H):
    print(f'  labelled at k={k:<2}                : {clean[f"y{k}"].drop_nulls().len()}')

# ------------------------------------------------------- T1 capacity -------
rule('T1  CAPACITY ACCOUNTING — AGCRN parameters vs. supervisable labels')

def agcrn_params(c_in, d_emb, hid, n_h=3, mlp_hidden=64):
    """Report eq. 4.1-4.6: shared MLP -> E; weight pool W_hat in R^{d x c x c'};
    3 gate transforms over [h_{t-1}, x_t] (width hid+c_in); linear head."""
    shared_mlp = c_in * mlp_hidden + mlp_hidden + mlp_hidden * d_emb + d_emb
    gate_in = hid + c_in
    pool = 3 * (d_emb * gate_in * hid)
    bias = 3 * (d_emb * hid)
    head = hid * n_h + n_h
    return dict(shared_mlp=shared_mlp, weight_pools=pool, bias_pools=bias,
                head=head, total=shared_mlp + pool + bias + head)

n_nodes = min(clean[f'y{k}'].drop_nulls().len() for k in HORIZONS)
n_lab = n_nodes * len(HORIZONS)
print(f'node-snapshots with all 3 horizons labelled : {n_nodes}')
print(f'supervisable scalar labels (x3 horizons)    : {n_lab}\n')
print(f'{"c_in":>5}{"d_emb":>7}{"hidden":>8}{"params":>12}{"params/label":>14}')
for c_in, d, hid in [(10, 2, 16), (10, 10, 64), (10, 10, 128), (32, 10, 64)]:
    p = agcrn_params(c_in, d, hid)
    print(f'{c_in:>5}{d:>7}{hid:>8}{p["total"]:>12,}{p["total"]/n_lab:>14.1f}')
p = agcrn_params(10, 10, 64)
print('\nbreakdown at the Bai et al. (2020) default (c=10, d=10, hidden=64):')
for kk, v in p.items():
    print(f'   {kk:<14}{v:>10,}')
smallest = agcrn_params(10, 2, 16)['total']
print(f'\nSmallest config above still carries {smallest:,} parameters.')
print(f'A linear AR(1) on the same target carries 2.')

# --------------------------------------------- T2 effective sample size ----
rule('T2  EFFECTIVE SAMPLE SIZE — labels are not independent observations')
rep = clean.group_by('event_ticker').agg(pl.len().alias('n_snaps'))
print(f'distinct event nodes ever active        : {rep.height}')
print(f'median re-labellings per node           : {rep["n_snaps"].median():.0f}'
      f'   (max {rep["n_snaps"].max()})')
sd = clean.filter(pl.col('y3').is_not_null())
per_day = (sd.group_by('d').agg(pl.col('y3').mean().alias('m'),
                                pl.col('y3').std().alias('s'), pl.len().alias('n'))
             .filter(pl.col('n') >= 3))
within = float(np.nanmean(per_day['s'].to_numpy() ** 2))
between = float(np.nanvar(per_day['m'].to_numpy()))
icc = between / (between + within)
nbar = float(per_day['n'].mean()); deff = 1 + (nbar - 1) * icc
print(f'\nintra-snapshot clustering of the k=3 label '
      f'({per_day.height} snapshots with >=3 nodes):')
print(f'   ICC                                  : {icc:.3f}')
print(f'   design effect 1+(nbar-1)*ICC         : {deff:.2f}  (nbar={nbar:.1f})')
print(f'   effective n vs nominal               : '
      f'{sd.height/deff:,.0f} vs {sd.height:,}')
print(f'   effective labels vs AGCRN params     : '
      f'{3*sd.height/deff:,.0f} vs {p["total"]:,}'
      f'  ({p["total"]/(3*sd.height/deff):.0f} params per effective label)')

# --------------------------------------------------- T3 edge vs cost -------
rule('T3  EDGE vs. COST — is the cross-market drift big enough to pay for it?')
for k in (*HORIZONS, LONG_H):
    s = clean.filter(pl.col(f'y{k}').is_not_null())
    y = s[f'y{k}'].to_numpy().astype(float); px = s['p'].to_numpy().astype(float)
    e_abs = np.abs(y).mean(); f = fee(px).mean()
    print(f'\nk={k:<2} n={y.size:<5} E|drift|={e_abs:5.2f}c  '
          f'median|drift|={np.median(np.abs(y)):5.2f}c  '
          f'mean drift={y.mean():+5.2f}c  fee={f:4.2f}c/side')
    for name, sp in [('book spread 8.0c ', SPREAD_BOOK),
                     ('trade-bounce 1.0c', SPREAD_TRADE)]:
        cost = sp + 2 * f
        be = 0.5 + cost / (2 * e_abs)
        print(f'     {name}  round-trip {cost:5.2f}c  '
              f'break-even hit rate {be*100:6.1f}%'
              f'{"   IMPOSSIBLE" if be > 1 else ""}')

# ------------------------------------------------- T4 oracle backtest ------
rule('T4  PERFECT-FORESIGHT ORACLE — upper bound on ANY model, after costs')
print('Trade every active node at every snapshot in the *realised* direction of\n'
      'the k-day move: 100% hit rate. No model, AGCRN included, can beat this.\n')
print(f'{"k":>3}{"trades":>8}{"gross":>10}{"fees":>9}{"net@1c":>10}{"net@8c":>10}'
      f'{"c/trade@1c":>12}{"c/trade@8c":>12}')
for k in (*HORIZONS, LONG_H):
    s = clean.filter(pl.col(f'y{k}').is_not_null())
    y = np.abs(s[f'y{k}'].to_numpy().astype(float)); px = s['p'].to_numpy().astype(float)
    n = y.size; gross = y.sum(); fees = 2 * fee(px).sum()
    n1 = gross - fees - SPREAD_TRADE * n; n8 = gross - fees - SPREAD_BOOK * n
    print(f'{k:>3}{n:>8}{gross:>10,.0f}{fees:>9,.0f}{n1:>10,.0f}{n8:>10,.0f}'
          f'{n1/n:>+12.2f}{n8/n:>+12.2f}')

print('\nSame oracle degraded to attainable hit rates. 65.6% is the pooled dormant\n'
      'sign accuracy from research_log.md S3 — the best in-sample directional\n'
      'result this project has produced, and itself a specification search.\n')
print(f'{"hit":>7}{"k":>4}{"net c/trade @1c":>18}{"net c/trade @8c":>18}')
for hit in (1.00, 0.656, 0.60, 0.55):
    for k in HORIZONS:
        s = clean.filter(pl.col(f'y{k}').is_not_null())
        y = np.abs(s[f'y{k}'].to_numpy().astype(float))
        px = s['p'].to_numpy().astype(float)
        edge = (2 * hit - 1) * y.mean() - 2 * fee(px).mean()
        print(f'{hit:>7.3f}{k:>4}{edge-SPREAD_TRADE:>+18.2f}{edge-SPREAD_BOOK:>+18.2f}')

# --------------------------------------------------- T5 ladder -------------
rule('T5  COMPLEXITY LADDER — does graph structure beat trivial baselines?')
lad = clean.filter(pl.col('y3').is_not_null()).sort('d')
lag = (tick_px.rename({'ticker': 'atm', 'px': 'p_lag'})
              .with_columns((pl.col('d') + dt.timedelta(days=3)).alias('d')))
lad = (lad.join(lag, on=['atm', 'd'], how='left')
          .with_columns((pl.col('p') - pl.col('p_lag')).alias('mom3'))
          .drop_nulls('mom3'))
dm = lad.group_by('d').agg(pl.col('mom3').sum().alias('sm'), pl.len().alias('nn'))
lad = (lad.join(dm, on='d').filter(pl.col('nn') >= 3)
          .with_columns(((pl.col('sm') - pl.col('mom3')) / (pl.col('nn') - 1))
                        .alias('nb_loo')))
# same-series neighbour mean (the strongest plausible "edge" the graph could learn)
ds = lad.group_by('d', 'series').agg(pl.col('mom3').sum().alias('ss'),
                                     pl.len().alias('sn'))
lad = (lad.join(ds, on=['d', 'series'])
          .with_columns(pl.when(pl.col('sn') > 1)
                          .then((pl.col('ss') - pl.col('mom3')) / (pl.col('sn') - 1))
                          .otherwise(0.0).alias('nb_series')))

dates = sorted(lad['d'].unique().to_list())
n_folds, start_frac = 8, 0.4
cuts = [dates[min(len(dates) - 1,
                  int(len(dates) * (start_frac + (1 - start_frac) * i / n_folds)))]
        for i in range(n_folds)] + [dates[-1] + dt.timedelta(days=1)]
print(f'walk-forward: {len(dates)} snapshot dates, {n_folds} expanding folds\n'
      f'first train ends {cuts[0]}, last fold ends {cuts[-1]}; 2026 never touched\n')

def ridge(X, y, lam=10.0):
    X1 = np.column_stack([np.ones(len(X)), X])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1]); A[0, 0] -= lam
    return np.linalg.solve(A, X1.T @ y)

RUNGS = {
    'zero (predict no change)': None,
    'train mean':               [],
    'own state: p':             ['p'],
    'own: p, mom3':             ['p', 'mom3'],
    '+ graph: all-node nbr':    ['p', 'mom3', 'nb_loo'],
    '+ graph: same-series nbr': ['p', 'mom3', 'nb_loo', 'nb_series'],
    '+ graph + activity':       ['p', 'mom3', 'nb_loo', 'nb_series', 'ntrades'],
}
preds = {r: [] for r in RUNGS}; truth = []
for i in range(n_folds):
    tr = lad.filter(pl.col('d') < cuts[i])
    te = lad.filter((pl.col('d') >= cuts[i]) & (pl.col('d') < cuts[i + 1]))
    if te.height == 0 or tr.height < 50: continue
    ytr = tr['y3'].to_numpy().astype(float); yte = te['y3'].to_numpy().astype(float)
    truth.append(yte)
    for name, cols in RUNGS.items():
        if cols is None: pr = np.zeros_like(yte)
        elif not cols:   pr = np.full_like(yte, ytr.mean())
        else:
            Xtr = tr.select(cols).to_numpy().astype(float)
            Xte = te.select(cols).to_numpy().astype(float)
            mu, sdv = Xtr.mean(0), Xtr.std(0) + 1e-9
            pr = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sdv]) @ \
                 ridge((Xtr - mu) / sdv, ytr)
        preds[name].append(pr)

yte = np.concatenate(truth); base_sse = float((yte ** 2).sum())
print(f'pooled holdout rows: {yte.size}\n')
print(f'{"rung":<28}{"MAE":>8}{"RMSE":>9}{"R2 vs zero":>12}{"dir acc %":>11}'
      f'{"net c/trade@1c":>16}')
res = {}
for name in RUNGS:
    pr = np.concatenate(preds[name])
    mae = np.abs(yte - pr).mean(); rmse = math.sqrt(((yte - pr) ** 2).mean())
    r2 = 1 - float(((yte - pr) ** 2).sum()) / base_sse
    nz = pr != 0
    da = (np.sign(pr[nz]) == np.sign(yte[nz])).mean() if nz.any() else float('nan')
    pnl = (np.sign(pr[nz]) * yte[nz]).mean() - 2 * fee(50) - SPREAD_TRADE \
          if nz.any() else float('nan')
    res[name] = (r2, da, pr)
    print(f'{name:<28}{mae:>8.2f}{rmse:>9.2f}{r2:>12.4f}{da*100:>11.1f}{pnl:>+16.2f}')

own = res['own: p, mom3'][0]
for g in ('+ graph: all-node nbr', '+ graph: same-series nbr', '+ graph + activity'):
    print(f'\nincremental R2 from graph terms, {g:<26}: {res[g][0]-own:+.5f}')
print('\nR2 is against predicting zero: negative means worse than doing nothing.')
print('net c/trade uses the OPTIMISTIC 1c spread and a 50c fee; at the measured')
print(f'{SPREAD_BOOK:.0f}c book spread every rung is {SPREAD_BOOK-SPREAD_TRADE:.0f}c/trade worse.')

# ------------------------------------- T6 the "22 relationships" screen ----
rule('T6  THE SCREENING RULE behind the report\'s 22 relationships')
print("Report S4.3.1 calls a pair significant when the 95% CI on the mean\n"
      "[-3,+14] price change excludes zero for >=3 CONSECUTIVE days. The paths\n"
      "are cumulative sums normalised to zero at day 0, so consecutive days are\n"
      "near-perfectly autocorrelated: the consecutive-day rule is not a\n"
      "multiple-comparison guard. Monte Carlo under a pure random walk:\n")

def screen_fpr(n_paths, n_days=14, n_sims=4000, min_run=3, sd=1.0):
    hits = 0
    for _ in range(n_sims):
        steps = rng.normal(0, sd, size=(n_paths, n_days))
        paths = np.cumsum(steps, axis=1)              # normalised to 0 at day 0
        m = paths.mean(0); se = paths.std(0, ddof=1) / math.sqrt(n_paths)
        sig = np.abs(m) > 1.96 * se
        run = best = 0
        for s_ in sig:
            run = run + 1 if s_ else 0
            best = max(best, run)
        hits += best >= min_run
    return hits / n_sims

print(f'{"pairs n":>9}{"P(false positive) per pair":>30}')
for n_paths in (5, 10, 28, 100, 811):
    print(f'{n_paths:>9}{screen_fpr(n_paths):>30.3f}')

# how many hypotheses does the exhaustive grid actually contain?
ev_by_series = (trig.group_by('series').agg(pl.len().alias('n_ev'))
                    .filter(pl.col('n_ev') >= 5).sort('n_ev', descending=True))
S = ev_by_series.height
grid = S * (S - 1)
print(f'\nseries with >=5 resolved events in the IS panel : {S}')
print(f'ordered trigger->target pairs in the grid       : {grid}')
fpr = screen_fpr(28)
print(f'expected false positives at the measured per-pair rate {fpr:.3f} : '
      f'{grid*fpr:.0f}')
print(f'the report reports                                              : 22')
print('\nBenjamini-Hochberg on the same grid would need a per-pair p below')
print(f'   0.05 * 22 / {grid} = {0.05*22/grid:.5f} for the 22nd-ranked pair to survive.')
