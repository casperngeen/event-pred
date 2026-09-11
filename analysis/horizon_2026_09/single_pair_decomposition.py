"""Why is the tradable edge so much smaller than the measured one?

Decomposes the strongest Stage-1 edge (PAYROLLS->FED, rho=0.565, p=0.00029)
into the part a trader can reach and the part the estimator sees. Backs
research_log.md §13.1.

Run from the repo root.
"""
from __future__ import annotations
import os, sys
import numpy as np, polars as pl
sys.path.insert(0, "stg_infra")
from stg.splits import BURN_IN_END
from stg.structure.stats import spearman, spearman_p

TRIG, TGT = "PAYROLLS", "FED"
FEE_RATE = 0.07
CONTRACTS = int(os.environ.get("CONTRACTS", "100"))
def fee(c, C=CONTRACTS):
    p = np.asarray(c, float)/100.0
    return np.ceil(FEE_RATE*C*p*(1-p)*100)/100.0*100.0/C

p = pl.read_parquet("artifacts/panels/pair_panel_dormant.parquet")
g = p.filter((pl.col("trigger")==TRIG) & (pl.col("target")==TGT)
             & (pl.col("t0")>=BURN_IN_END) & pl.col("p_entry").is_not_null())
S = g["surprise"].to_numpy().astype(float)
R = g["response"].to_numpy().astype(float)
p0 = g["p0"].to_numpy().astype(float)
pe = g["p_entry"].to_numpy().astype(float)
px = p0 + R
d = np.sign(S); d[d == 0] = 1

rho = spearman(S, R)
print(f"{TRIG}->{TGT}, n={g.height}\n")
print("A  WHAT THE STATISTIC MEASURES vs WHAT THE TRADE NEEDS")
print(f"  Spearman rho(surprise, response)     {rho:+.3f}  p={spearman_p(rho,len(S)):.5f}")
print(f"  sign agreement measured from p0      {np.mean(np.sign(S)==np.sign(R)):.3f}")
print(f"  sign agreement at executable entry   {np.mean(d*(px-pe)>0):.3f}")
print("  => rho ranks magnitudes; the trade needs the sign to still be")
print("     predictable at a price you can transact at. They come apart.\n")

jump, drift = pe - p0, px - pe
tot = np.mean(d*jump) + np.mean(d*drift)
print("B  WHERE THE MOVE HAPPENS")
print(f"  signed jump   p0 -> first print     {np.mean(d*jump):+.2f}c  (untradable)")
print(f"  signed drift  first print -> exit   {np.mean(d*drift):+.2f}c  (tradable)")
print(f"  jump share of the total             {np.mean(d*jump)/tot:.0%}")
print(f"  median lag, resolution -> 1st print "
      f"{np.median((g['t_entry']-g['t0']).dt.total_seconds()/60):.1f} min\n")

print("C  THE COST FLOOR")
print(f"  mean entry price {pe.mean():.1f}c (the p(1-p) fee peaks at 50c)")
print(f"  round-trip fee   {(fee(pe)+fee(px)).mean():.2f}c at C={CONTRACTS}")
print(f"  tradable signal {np.mean(d*drift):+.2f}c vs floor "
      f"{(fee(pe)+fee(px)).mean()+1.0:.2f}c (spread 1.0c + fees)\n")

print("D  DOES A BIGGER SURPRISE HELP?")
z = np.abs(g["z_surprise"].to_numpy().astype(float))
for lo in (0.0, 0.5, 1.0, 1.5):
    m = z >= lo
    if m.sum() < 4: continue
    net = d*drift - 1.0 - (fee(pe)+fee(px))
    print(f"  |z|>={lo}: n={m.sum():>3}  drift {np.mean((d*drift)[m]):+.2f}c  "
          f"net {net[m].mean():+.2f}c  win {np.mean((d*drift)[m]>0):.2f}")
