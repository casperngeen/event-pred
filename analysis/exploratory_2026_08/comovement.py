import sys
sys.path.insert(0, "/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
import polars as pl, math
from pairwise_sweep import build_surprise, representative_tickers, IS_CUTOFF, trades_lf

def responses_for_trigger(trigger, target, max_gap_days=45, horizons=(1,3,5,10)):
    surp = build_surprise(trigger)
    rep = representative_tickers(target)
    meeting_map = list(zip(rep["meeting_time"].to_list(), rep["rep_ticker"].to_list()))
    def next_meeting(t):
        cands=[(m,tk) for m,tk in meeting_map if m>t]
        return min(cands, key=lambda x:x[0]) if cands else (None,None)
    recs=[]
    for row in surp.iter_rows(named=True):
        m,tk = next_meeting(row["close_time"])
        recs.append({**row,"target_meeting":m,"target_ticker":tk})
    pairs = pl.DataFrame(recs).filter(pl.col("target_ticker").is_not_null())
    pairs = pairs.with_columns(((pl.col("target_meeting")-pl.col("close_time")).dt.total_hours()/24).alias("gap"))
    pairs = pairs.filter((pl.col("gap")<=max_gap_days)&(pl.col("gap")>=0))
    target_tickers = pairs["target_ticker"].unique().to_list()
    tt = (trades_lf.filter(pl.col("ticker").is_in(target_tickers)).select("ticker","yes_price","created_time")
          .sort(["ticker","created_time"]).collect())
    out=[]
    for row in pairs.iter_rows(named=True):
        tk=row["target_ticker"]; t0=row["close_time"]
        sub = tt.filter(pl.col("ticker")==tk)
        pre = sub.filter(pl.col("created_time")<=t0)
        if pre.height==0: continue
        p0 = pre["yes_price"][-1]
        post = sub.filter(pl.col("created_time")>t0)
        rec={"event":row["event"],"surprise":row["surprise"]}
        for k in horizons:
            rec[f"resp_{k}"] = (post["yes_price"][k-1]-p0) if post.height>=k else None
        out.append(rec)
    return pl.DataFrame(out)

# Same trigger (CPI), two different targets: FEDDECISION and FED
r_fedd = responses_for_trigger("CPI","FEDDECISION").rename({f"resp_{k}":f"fedd_{k}" for k in (1,3,5,10)})
r_fed  = responses_for_trigger("CPI","FED").rename({f"resp_{k}":f"fed_{k}" for k in (1,3,5,10)})
r_fedd = r_fedd.drop("surprise")
joined = r_fed.join(r_fedd, on="event", how="inner")
print(f"CPI trigger events with BOTH FEDDECISION and FED response computed: {joined.height}")

print(f"\n{'horizon':<10}{'n':>4}{'corr(FEDDECISION resp, FED resp)':>36}{'t':>8}")
for k in (1,3,5,10):
    c1,c2 = f"fedd_{k}", f"fed_{k}"
    sub = joined.filter(pl.col(c1).is_not_null() & pl.col(c2).is_not_null())
    n = sub.height
    if n<8: continue
    r = sub.select(pl.corr(c1,c2)).item()
    t = r*math.sqrt(n-2)/math.sqrt(max(1e-9,1-r*r))
    print(f"{k:<10}{n:>4}{r:>36.3f}{t:>8.2f}")

# does |surprise| jointly explain BOTH responses (co-movement conditional on the trigger, not just raw co-movement)?
print("\n--- do both targets respond in the SAME direction as each other, sign-wise? ---")
for k in (1,3,5,10):
    c1,c2=f"fedd_{k}",f"fed_{k}"
    sub = joined.filter(pl.col(c1).is_not_null() & pl.col(c2).is_not_null())
    if sub.height<8: continue
    same_sign = sub.filter((pl.col(c1)*pl.col(c2))>0).height
    print(f"  horizon {k}: same-sign fraction = {same_sign}/{sub.height} ({100*same_sign/sub.height:.0f}%)  [50% expected under independence]")
