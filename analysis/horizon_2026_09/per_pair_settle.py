"""B: per-pair hold-to-settlement, with the 'buy the favourite' control.

hold_to_expiry.py §D showed the pooled hold-to-settlement strategy can look
profitable purely by buying the cheap side of a book that mostly resolves one
way. Any per-pair result therefore has to clear its own base rate, not zero.
The decisive column is `disagree_edge`: the strategy's PnL minus the favourite
rule's, restricted to the events where the two rules take opposite sides. That
is the only place the signal can add anything.

In-sample only. Run from the repo root after horizon_sweep.py.
"""
from __future__ import annotations
import os
import sys
import numpy as np, polars as pl
sys.path.insert(0, "stg_infra")
from stg.direction import SignRule, fit_structure, gate_coverage, predict_oof, walk_forward
from stg.splits import BURN_IN_END

FEE_RATE, HALF = 0.07, 2.35
CONTRACTS = int(os.environ.get("CONTRACTS", "1"))
def fee(c, C=None):
    """Cents per contract for an order of C. See horizon_sweep.py::fee."""
    C = CONTRACTS if C is None else C
    p = np.asarray(c, float)/100.0
    return np.ceil(FEE_RATE*C*p*(1-p)*100)/100.0*100.0/C

panel = pl.read_parquet("artifacts/horizon_panel.parquet")
folds = walk_forward(panel, n_folds=8, start_frac=0.4)
structures = [(f, fit_structure(panel.filter(pl.Series(f.train)))) for f in folds]
p_up = predict_oof(panel, SignRule(), structures)
covered = np.zeros(panel.height, bool)
for f in folds: covered |= f.test
direction = np.where(p_up >= 0.5, 1, -1)

def settle_pnl(buy_yes, pe, won):
    price = np.where(buy_yes, pe, 100-pe)
    return np.where(buy_yes == won, 100.0, 0.0) - (price + HALF + fee(price))

print("="*100)
print("B  PER-PAIR HOLD-TO-SETTLEMENT (walk-forward OOF direction)")
print("="*100)
for gate in ("bh","p05"):
    mask = covered & gate_coverage(panel, structures, gate)
    d = panel.filter(pl.Series(mask)).with_columns(
        pl.Series("dir", direction[mask].astype(float)))
    d = d.filter(pl.col("settle_yes").is_not_null() & pl.col("p_entry").is_not_null())
    print(f"\ngate={gate}   {d.height} settled OOF signals")
    print(f"  {'pair':<30}{'n':>4}{'win%':>7}{'strat':>8}{'fav':>8}{'diff':>8}"
          f"{'dis_n':>7}{'dis_strat':>10}{'dis_fav':>9}")
    rows=[]
    for (pair,), g in d.group_by("pair", maintain_order=True):
        if g.height < 8: continue
        pe = g["p_entry"].to_numpy().astype(float)
        won = g["settle_yes"].to_numpy() > 0
        sig = g["dir"].to_numpy() > 0
        fav = pe > 50
        s, f_ = settle_pnl(sig, pe, won), settle_pnl(fav, pe, won)
        dis = sig != fav
        rows.append(dict(pair=pair, n=g.height, win=float((sig==won).mean()),
                         strat=float(s.mean()), fav=float(f_.mean()),
                         dis_n=int(dis.sum()),
                         dis_s=float(s[dis].mean()) if dis.sum() else np.nan,
                         dis_f=float(f_[dis].mean()) if dis.sum() else np.nan))
    for r in sorted(rows, key=lambda x:-x["strat"]):
        print(f"  {r['pair']:<30}{r['n']:>4}{r['win']*100:>6.0f}%{r['strat']:>8.1f}"
              f"{r['fav']:>8.1f}{r['strat']-r['fav']:>8.1f}{r['dis_n']:>7}"
              f"{r['dis_s']:>10.1f}{r['dis_f']:>9.1f}")
    # pooled
    pe = d["p_entry"].to_numpy().astype(float); won = d["settle_yes"].to_numpy()>0
    sig = d["dir"].to_numpy()>0; fav = pe>50
    s, f_ = settle_pnl(sig,pe,won), settle_pnl(fav,pe,won)
    se = s.std(ddof=1)/np.sqrt(len(s))
    print(f"  {'POOLED':<30}{len(s):>4}{(sig==won).mean()*100:>6.0f}%{s.mean():>8.1f}"
          f"{f_.mean():>8.1f}{s.mean()-f_.mean():>8.1f}  t={s.mean()/se:.2f}")
    dis = sig!=fav
    sd = s[dis]; sde = sd.std(ddof=1)/np.sqrt(len(sd))
    print(f"    on the {dis.sum()} disagreement events: strategy {sd.mean():+.2f}c "
          f"(t={sd.mean()/sde:.2f})  favourite {f_[dis].mean():+.2f}c")
