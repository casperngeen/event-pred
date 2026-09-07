"""Sign-aligned directional test.

Theory gives the sign in advance: a hawkish CPI surprise (print above the
market's implied mean) should RAISE P(hike) and LOWER P(cut). Testing that is
falsifiable in a way the magnitude-only test was not.

Pre-2026 only.
"""
import sys, math, datetime as dt, polars as pl
sys.path.insert(0, "/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
sys.path.insert(0, "stg_infra")
from liquid_window import pdf_surprise, mk_all, tr_all, IS_CUT

LIQUID_DAYS = 7
MIN_TRADES = 3

fed = mk_all.filter(pl.col("series") == "FEDDECISION").filter(pl.col("close_time").is_not_null())
# classify strike direction from the ticker suffix
# Match H0 (hold) first so the hike branch needs no negative lookahead —
# the Rust regex engine polars uses does not support look-around.
fed = fed.with_columns(
    pl.when(pl.col("ticker").str.contains(r"-H0$")).then(pl.lit("hold"))
     .when(pl.col("ticker").str.contains(r"-H\d")).then(pl.lit("hike"))
     .when(pl.col("ticker").str.contains(r"-C\d")).then(pl.lit("cut"))
     .otherwise(pl.lit("other")).alias("side"))
print("FEDDECISION strike sides:", fed.group_by("side").len().sort("len", descending=True).to_dicts())

tkc = (tr_all.filter(pl.col("ticker").is_in(fed["ticker"].unique().to_list()))
       .group_by("ticker").agg(pl.len().alias("n")).collect())
fed = fed.join(tkc, on="ticker", how="left").with_columns(pl.col("n").fill_null(0)).filter(pl.col("n") > 0)

# most-traded contract per (event, side)
rep = (fed.sort("n", descending=True).group_by(["event_ticker", "side"])
       .agg(pl.col("ticker").first().alias("rep"),
            pl.col("close_time").first().alias("close"),
            pl.col("n").first().alias("n")))

trades = (tr_all.filter(pl.col("ticker").is_in(rep["rep"].unique().to_list()))
          .select("ticker", "yes_price", "created_time").sort(["ticker", "created_time"]).collect())


def run(trigger, side, horizon):
    surp = pdf_surprise(trigger)
    r_side = rep.filter(pl.col("side") == side).sort("close")
    reps = list(zip(r_side["close"].to_list(), r_side["rep"].to_list()))
    rows = []
    for r in surp.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r["close_time"]]
        if not nxt:
            continue
        c, t = min(nxt, key=lambda x: x[0])
        gap = (c - r["close_time"]).total_seconds() / 86400
        if not (0 <= gap <= 60):
            continue
        sub = trades.filter(pl.col("ticker") == t)
        pre = sub.filter(pl.col("created_time") <= r["close_time"])
        if pre.height == 0:
            continue
        p0 = pre["yes_price"][-1]
        if horizon == "dormant":
            post = sub.filter(pl.col("created_time") > r["close_time"])
            p1 = post["yes_price"][2] if post.height >= 3 else None
        else:
            lw = sub.filter((pl.col("created_time") >= c - dt.timedelta(days=LIQUID_DAYS)) &
                            (pl.col("created_time") <= c))
            p1 = lw["yes_price"].mean() if lw.height >= MIN_TRADES else None
        if p1 is None:
            continue
        rows.append({"surprise": r["surprise"], "resp": p1 - p0, "p0": p0})
    if len(rows) < 8:
        return None
    d = pl.DataFrame(rows)
    r_ = d.select(pl.corr("surprise", "resp")).item()
    n = d.height
    t_ = r_ * math.sqrt(n - 2) / math.sqrt(max(1e-9, 1 - r_ * r_)) if r_ is not None else float("nan")
    # directional hit rate: does response sign match predicted sign?
    expected = 1.0 if side == "hike" else (-1.0 if side == "cut" else 0.0)
    hits = d.filter((pl.col("surprise") * pl.col("resp") * expected) > 0).height
    tot = d.filter((pl.col("surprise") != 0) & (pl.col("resp") != 0)).height
    return n, r_, t_, (hits / tot if tot else float("nan")), tot


print(f"\n{'trigger':<8}{'side':<7}{'horizon':<10}{'n':>4}{'corr(S,resp)':>14}{'t':>7}{'hit%':>8}  predicted sign")
for trigger in ["CPI", "CPIYOY"]:
    for side, pred in [("cut", "negative"), ("hike", "positive"), ("hold", "n/a")]:
        for horizon in ["dormant", "liquid"]:
            out = run(trigger, side, horizon)
            if out is None:
                continue
            n, r_, t_, hit, tot = out
            print(f"{trigger:<8}{side:<7}{horizon:<10}{n:>4}{r_:>14.3f}{t_:>7.2f}{100*hit:>7.0f}%  {pred}")
