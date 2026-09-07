import sys
sys.path.insert(0, "/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
from comovement import responses_for_trigger
import polars as pl, math

r_fedd = responses_for_trigger("CPI","FEDDECISION").rename({f"resp_{k}":f"fedd_{k}" for k in (1,3,5,10)}).drop("surprise")
r_fed  = responses_for_trigger("CPI","FED").rename({f"resp_{k}":f"fed_{k}" for k in (1,3,5,10)})
joined = r_fed.join(r_fedd, on="event", how="inner")

for k in (3,5):
    c1,c2=f"fedd_{k}",f"fed_{k}"
    sub = joined.filter(pl.col(c1).is_not_null()&pl.col(c2).is_not_null()).with_columns(
        (pl.col(c1).abs()+pl.col(c2).abs()).alias("magsum"))
    sub = sub.sort("magsum", descending=True)
    print(f"\n=== horizon {k}: top 5 by combined |response| ===")
    print(sub.select("event",c1,c2).head(5))
    n = sub.height
    r_full = sub.select(pl.corr(c1,c2)).item()
    for drop in (1,2,3):
        trimmed = sub.slice(drop, n-drop)  # drop the `drop` largest-magnitude events
        r_trim = trimmed.select(pl.corr(c1,c2)).item()
        print(f"  drop top {drop} outlier(s): n={trimmed.height}, r={r_trim:.3f}  (full-sample r={r_full:.3f})")
