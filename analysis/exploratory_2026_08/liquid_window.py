"""Does trigger surprise get priced when the TARGET enters its own liquid window,
rather than immediately after the trigger resolves (while the target is dormant)?

Pre-2026 only (OOS wall enforced explicitly).
"""
import polars as pl, sys, datetime as dt, numpy as np, math
sys.path.insert(0, "stg_infra")
from stg.io.kalshi import KalshiOHLCV
from stg.events.implied import (
    parse_threshold, parse_threshold_from_subtitle, recover_pdf,
    pdf_implied_stats, resolved_value,
)

IS_CUT = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
LIQUID_DAYS = 7           # target's "awake" window = final N days before its own close
MIN_TRADES_LIQUID = 3     # require the target actually traded in that window

mk_all = (pl.read_parquet("data/markets/*.parquet")
          .sort("_fetched_at", descending=True).unique(subset=["ticker"], keep="first")
          .filter(pl.col("close_time") < IS_CUT))
mk_all = mk_all.with_columns(
    pl.col("ticker").str.replace(r"^KX", "").str.split("-").list.first().alias("series"))
tr_all = pl.scan_parquet("data/trades/*.parquet").filter(pl.col("created_time") < IS_CUT)


def pdf_surprise(series):
    """Proper PDF-based surprise: resolved value minus pre-resolution implied mean."""
    mk = mk_all.filter(pl.col("series") == series)
    tk = mk["ticker"].unique().to_list()
    tr = tr_all.filter(pl.col("ticker").is_in(tk)).collect()
    if tr.height == 0:
        return pl.DataFrame()
    d = KalshiOHLCV.build_daily(tr, mk).join(
        mk.select(["ticker", "yes_sub_title"]).unique(subset=["ticker"]), on="ticker", how="left")
    d = d.with_columns(
        pl.struct(["ticker", "yes_sub_title"]).map_elements(
            lambda r: parse_threshold(r["ticker"], r["yes_sub_title"]),
            return_dtype=pl.Float64).alias("threshold"),
        pl.col("yes_sub_title").map_elements(
            lambda s: parse_threshold_from_subtitle(s)[1], return_dtype=pl.Utf8
        ).alias("conv"),
    ).filter(pl.col("threshold").is_not_null())
    # restrict to fresh-ish days so the ladder is a real cross-section
    d = d.filter(pl.col("trade_count") > 0)

    rows = []
    for ev in d["event_ticker"].unique().to_list():
        sub = d.filter(pl.col("event_ticker") == ev)
        ct = sub["close_time"].min()
        if ct is None:
            continue
        # last day with >=3 fresh legs before resolution
        cand = (sub.group_by("date").agg(pl.len().alias("legs"))
                .filter(pl.col("legs") >= 3).sort("date"))
        if cand.height == 0:
            continue
        last_day = cand["date"][-1]
        lad = sub.filter(pl.col("date") == last_day).sort("threshold")
        thr = lad["threshold"].to_numpy(); prb = lad["close"].to_numpy() / 100.0
        order = np.argsort(thr); thr, prb = thr[order], prb[order]
        if len(thr) < 3:
            continue
        mid, p = recover_pdf(thr, prb)
        st = pdf_implied_stats(mid, p)
        rv = resolved_value(mk_all.filter(pl.col("event_ticker") == ev))
        if rv is None:
            continue
        rows.append({"event": ev, "close_time": ct, "implied_mean": st["mean"],
                     "implied_std": st["std"], "resolved": rv,
                     "surprise": rv - st["mean"], "snap_date": last_day})
    return pl.DataFrame(rows).sort("close_time") if rows else pl.DataFrame()


def target_windows(series):
    mk = mk_all.filter(pl.col("series") == series).select(
        "event_ticker", "ticker", "close_time").filter(pl.col("close_time").is_not_null())
    tk = mk["ticker"].unique().to_list()
    tc = tr_all.filter(pl.col("ticker").is_in(tk)).group_by("ticker").agg(
        pl.len().alias("n")).collect()
    mk = mk.join(tc, on="ticker", how="left").with_columns(pl.col("n").fill_null(0))
    rep = (mk.sort("n", descending=True).group_by("event_ticker").agg(
        pl.col("ticker").first().alias("rep"), pl.col("close_time").first().alias("close"),
        pl.col("n").first()))
    return rep.filter(pl.col("n") > 0).sort("close")


def run(trigger, target):
    surp = pdf_surprise(trigger)
    if surp.height == 0:
        return
    rep = target_windows(target)
    pairs = []
    reps = list(zip(rep["close"].to_list(), rep["rep"].to_list()))
    for r in surp.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r["close_time"]]
        if not nxt:
            continue
        c, t = min(nxt, key=lambda x: x[0])
        pairs.append({**r, "tgt_close": c, "tgt": t,
                      "gap": (c - r["close_time"]).total_seconds() / 86400})
    pairs = pl.DataFrame(pairs).filter((pl.col("gap") >= 0) & (pl.col("gap") <= 60))
    if pairs.height < 8:
        print(f"{trigger}->{target}: only {pairs.height} pairs"); return

    tks = pairs["tgt"].unique().to_list()
    tt = (tr_all.filter(pl.col("ticker").is_in(tks))
          .select("ticker", "yes_price", "created_time").sort(["ticker", "created_time"]).collect())

    out = []
    for r in pairs.iter_rows(named=True):
        sub = tt.filter(pl.col("ticker") == r["tgt"])
        pre = sub.filter(pl.col("created_time") <= r["close_time"])
        if pre.height == 0:
            continue
        p0 = pre["yes_price"][-1]
        # (a) dormant-window response: next few trades right after the trigger
        post = sub.filter(pl.col("created_time") > r["close_time"])
        p_imm = post["yes_price"][2] if post.height >= 3 else None
        # (b) liquid-window response: target's final LIQUID_DAYS before its own close
        lw_start = r["tgt_close"] - dt.timedelta(days=LIQUID_DAYS)
        lw = sub.filter((pl.col("created_time") >= lw_start) &
                        (pl.col("created_time") <= r["tgt_close"]))
        p_liq = lw["yes_price"].mean() if lw.height >= MIN_TRADES_LIQUID else None
        out.append({"event": r["event"], "surprise": r["surprise"], "gap": r["gap"],
                    "p0": p0, "n_post": post.height, "n_liquid": lw.height,
                    "resp_dormant": (p_imm - p0) if p_imm is not None else None,
                    "resp_liquid": (p_liq - p0) if p_liq is not None else None})
    res = pl.DataFrame(out).with_columns(pl.col("surprise").abs().alias("abs_s"))

    print(f"\n=== {trigger} -> {target} ===")
    print(f"  pairs={res.height}  median trigger->target gap={res['gap'].median():.0f}d"
          f"  median trades in target liquid window={res['n_liquid'].median():.0f}")
    for col, lbl in [("resp_dormant", "dormant (next 3 trades)"),
                     ("resp_liquid", f"liquid (final {LIQUID_DAYS}d mean)")]:
        s = res.filter(pl.col(col).is_not_null()).with_columns(pl.col(col).abs().alias("ar"))
        n = s.height
        if n < 8:
            print(f"  {lbl:<28} n={n} (too few)"); continue
        r_ = s.select(pl.corr("abs_s", "ar")).item()
        t_ = r_ * math.sqrt(n - 2) / math.sqrt(max(1e-9, 1 - r_ * r_))
        print(f"  {lbl:<28} n={n:<4} corr(|surp|,|resp|)={r_:>6.3f}  t={t_:>5.2f}"
              f"   mean|resp|={s['ar'].mean():.1f}c")


for trig, tgt in [("CPI", "FEDDECISION"), ("CPIYOY", "FEDDECISION"), ("U3", "FEDDECISION")]:
    run(trig, tgt)
