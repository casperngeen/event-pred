"""Does profitability improve if we condition on the entry price?

Two reasons to expect it might, pulling the same way:

  1. The fee is 0.07 x P x (1-P) -- maximal at 50c, near-zero at the extremes.
     A trade entered at 90c pays ~0.63c; the same trade at 50c pays 1.75c.
  2. Whelan et al. (2026) document a favourite-longshot bias on Kalshi:
     low-price contracts win far less often than break-even after fees,
     high-price contracts yield small positive returns (research_summary.md §7.4).

Reason 2 is also the confound: if high-price entries look profitable, that may
be the base rate, not the signal. So every bucket carries the same "buy the
favourite" benchmark used in hold_to_expiry.py §D, computed WITHIN the bucket.

Buckets are on the price actually PAID (p_entry if buying Yes, 100-p_entry if
buying No), because that is what sets the fee.

In-sample only. Run from the repo root after horizon_sweep.py.
"""
from __future__ import annotations
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, "stg_infra")
from stg.direction import SignRule, fit_structure, gate_coverage, predict_oof, walk_forward

FEE_RATE, HALF = 0.07, 2.35
CONTRACTS = int(os.environ.get("CONTRACTS", "100"))
def fee(c, C=CONTRACTS):
    p = np.asarray(c, float)/100.0
    return np.ceil(FEE_RATE*C*p*(1-p)*100)/100.0*100.0/C

panel = pl.read_parquet("artifacts/horizon_panel.parquet")
folds = walk_forward(panel, n_folds=8, start_frac=0.4)
st = [(f, fit_structure(panel.filter(pl.Series(f.train)))) for f in folds]
p_up = predict_oof(panel, SignRule(), st)
cov = np.zeros(panel.height, bool)
for f in folds: cov |= f.test
direction = np.where(p_up >= 0.5, 1, -1)

BUCKETS = [(0,35),(35,50),(50,65),(65,80),(80,101)]

for gate in ("bh","p05"):
    mask = cov & gate_coverage(panel, st, gate)
    d = panel.filter(pl.Series(mask)).with_columns(pl.Series("dir", direction[mask].astype(float)))
    d = d.filter(pl.col("settle_yes").is_not_null() & pl.col("p_entry").is_not_null())
    pe = d["p_entry"].to_numpy().astype(float)
    won = d["settle_yes"].to_numpy() > 0
    buy_yes = d["dir"].to_numpy() > 0
    paid = np.where(buy_yes, pe, 100-pe)

    def settle(by):
        pr = np.where(by, pe, 100-pe)
        return np.where(by == won, 100.0, 0.0) - (pr + HALF + fee(pr))
    strat, favr = settle(buy_yes), settle(pe > 50)

    # 1-day round trip, same rows
    px = d["p_1d"].to_numpy().astype(float)
    dirm = d["dir"].to_numpy()
    rt = dirm*(px-pe) - 1.0 - fee(pe) - fee(px)

    print("\n" + "="*94)
    print(f"gate={gate}   {len(pe)} OOF signals   fees at C={CONTRACTS}")
    print("="*94)
    print(f"{'price paid':<13}{'n':>4}{'win%':>7}{'fee':>7}  |{'SETTLE net':>12}"
          f"{'favourite':>11}{'edge':>8}  |{'1d round trip':>15}")
    for lo,hi in BUCKETS:
        m = (paid>=lo)&(paid<hi)
        if m.sum() < 4:
            print(f"  {f'{lo}-{hi}c':<11}{m.sum():>4}{'':>7}{'':>7}  |{'-- too few --':>12}")
            continue
        okr = m & np.isfinite(px)
        rtv = f"{rt[okr].mean():>15.2f}" if okr.sum()>=4 else f"{'--':>15}"
        print(f"  {f'{lo}-{hi}c':<11}{m.sum():>4}{np.mean(buy_yes[m]==won[m])*100:>6.0f}%"
              f"{fee(paid[m]).mean():>7.2f}  |{strat[m].mean():>12.2f}{favr[m].mean():>11.2f}"
              f"{strat[m].mean()-favr[m].mean():>8.2f}  |{rtv}")
    se = strat.std(ddof=1)/np.sqrt(len(strat))
    print(f"  {'ALL':<11}{len(pe):>4}{np.mean(buy_yes==won)*100:>6.0f}%{fee(paid).mean():>7.2f}"
          f"  |{strat.mean():>12.2f}{favr.mean():>11.2f}{strat.mean()-favr.mean():>8.2f}"
          f"  |{rt[np.isfinite(px)].mean():>15.2f}")
    print(f"  (settle t on ALL = {strat.mean()/se:.2f})")

# --- significance, with the multiple-comparison problem made explicit ------
print("\n" + "="*94)
print("SIGNIFICANCE PER BUCKET (gate=p05) -- 5 buckets tested, so read accordingly")
print("="*94)
mask = cov & gate_coverage(panel, st, "p05")
d = panel.filter(pl.Series(mask)).with_columns(pl.Series("dir", direction[mask].astype(float)))
d = d.filter(pl.col("settle_yes").is_not_null() & pl.col("p_entry").is_not_null())
pe = d["p_entry"].to_numpy().astype(float); won = d["settle_yes"].to_numpy()>0
by = d["dir"].to_numpy()>0; paid = np.where(by, pe, 100-pe)
def settle(b):
    pr=np.where(b,pe,100-pe)
    return np.where(b==won,100.0,0.0)-(pr+HALF+fee(pr))
strat = settle(by)
rng = np.random.default_rng(0)
print(f"{'bucket':<12}{'n':>5}{'net':>9}{'sd':>8}{'t':>7}{'boot P(>0)':>12}{'perm p':>9}")
for lo,hi in BUCKETS:
    m=(paid>=lo)&(paid<hi)
    if m.sum()<4: continue
    x=strat[m]; se=x.std(ddof=1)/np.sqrt(len(x))
    bs=np.array([x[rng.integers(0,len(x),len(x))].mean() for _ in range(10000)])
    # permutation: shuffle which side the signal picks, within the bucket
    idx=np.where(m)[0]
    perm=[]
    for _ in range(10000):
        b2=by.copy(); b2[idx]=rng.permutation(by[idx])
        perm.append(settle(b2)[m].mean())
    perm=np.array(perm)
    print(f"  {f'{lo}-{hi}c':<10}{len(x):>5}{x.mean():>9.2f}{x.std(ddof=1):>8.1f}"
          f"{x.mean()/se:>7.2f}{(bs>0).mean():>12.3f}{(perm>=x.mean()).mean():>9.4f}")
print("\n  Bonferroni threshold for 5 buckets at 0.05 -> each needs p < 0.010")
