"""CPI and CPIYOY resolve from the SAME BLS release on the SAME day.
If both surprise measures are real, they must agree. Internal consistency check."""
import sys, math, polars as pl
sys.path.insert(0,"/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
from liquid_window import pdf_surprise

a = pdf_surprise("CPI").select("event","close_time","surprise","implied_mean","implied_std","resolved")
b = pdf_surprise("CPIYOY").select("event","close_time","surprise","implied_mean","implied_std","resolved")
a = a.with_columns(pl.col("close_time").dt.date().alias("d"))
b = b.with_columns(pl.col("close_time").dt.date().alias("d"))
j = a.join(b, on="d", suffix="_yoy")
print(f"CPI events: {a.height}  CPIYOY events: {b.height}  same-day matched: {j.height}")
if j.height>=8:
    for c1,c2,lbl in [("surprise","surprise_yoy","signed surprise"),
                      ("implied_std","implied_std_yoy","implied_std")]:
        s=j.filter(pl.col(c1).is_not_null()&pl.col(c2).is_not_null())
        r=s.select(pl.corr(c1,c2)).item(); n=s.height
        t=r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r*r))
        print(f"  corr({lbl:<16}) = {r:>6.3f}  t={t:>5.2f}  n={n}")
    s=j.with_columns(pl.col("surprise").abs().alias("A"), pl.col("surprise_yoy").abs().alias("B"))
    r=s.select(pl.corr("A","B")).item(); n=s.height
    t=r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r*r))
    print(f"  corr(|surprise| CPI vs CPIYOY) = {r:>6.3f}  t={t:>5.2f}  n={n}")
    print("\n  sample (same-day pairs):")
    print(j.select("d","surprise","surprise_yoy","implied_std","implied_std_yoy").head(10))
