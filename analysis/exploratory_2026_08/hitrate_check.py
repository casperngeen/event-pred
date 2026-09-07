"""A raw sign-agreement rate is only comparable to 50% if BOTH variables are
sign-balanced. If surprises skew negative (disinflation) and hike-contract
responses skew negative (decaying to zero), high agreement is pure marginal
bias, not a relationship. Check the marginals."""
import sys, datetime as dt, polars as pl
sys.path.insert(0,"/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
sys.path.insert(0,"stg_infra")
from signed_test import rep, trades, LIQUID_DAYS, MIN_TRADES
from liquid_window import pdf_surprise

def collect(trigger, side, horizon):
    surp = pdf_surprise(trigger)
    rs = rep.filter(pl.col("side")==side).sort("close")
    reps=list(zip(rs["close"].to_list(), rs["rep"].to_list()))
    rows=[]
    for r in surp.iter_rows(named=True):
        nxt=[(c,t) for c,t in reps if c>r["close_time"]]
        if not nxt: continue
        c,t=min(nxt,key=lambda x:x[0])
        gap=(c-r["close_time"]).total_seconds()/86400
        if not (0<=gap<=60): continue
        sub=trades.filter(pl.col("ticker")==t)
        pre=sub.filter(pl.col("created_time")<=r["close_time"])
        if pre.height==0: continue
        p0=pre["yes_price"][-1]
        if horizon=="dormant":
            post=sub.filter(pl.col("created_time")>r["close_time"])
            p1=post["yes_price"][2] if post.height>=3 else None
        else:
            lw=sub.filter((pl.col("created_time")>=c-dt.timedelta(days=LIQUID_DAYS))&(pl.col("created_time")<=c))
            p1=lw["yes_price"].mean() if lw.height>=MIN_TRADES else None
        if p1 is None: continue
        rows.append({"S":r["surprise"],"R":p1-p0})
    return pl.DataFrame(rows)

print(f"{'trigger':<8}{'side':<6}{'horizon':<9}{'n':>4}{'S>0':>7}{'R>0':>7}{'chance':>9}{'actual':>9}{'excess':>9}")
for trig in ["CPI","CPIYOY"]:
    for side,exp in [("cut",-1.0),("hike",1.0)]:
        for hz in ["dormant","liquid"]:
            d=collect(trig,side,hz)
            d=d.filter((pl.col("S")!=0)&(pl.col("R")!=0))
            n=d.height
            if n<8: continue
            ps=d.filter(pl.col("S")>0).height/n
            pr=d.filter(pl.col("R")>0).height/n
            # chance-level agreement given marginals
            chance = ps*pr + (1-ps)*(1-pr)
            if exp<0: chance = ps*(1-pr) + (1-ps)*pr
            actual=d.filter((pl.col("S")*pl.col("R")*exp)>0).height/n
            print(f"{trig:<8}{side:<6}{hz:<9}{n:>4}{100*ps:>6.0f}%{100*pr:>6.0f}%{100*chance:>8.0f}%{100*actual:>8.0f}%{100*(actual-chance):>+8.0f}%")
