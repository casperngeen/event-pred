"""Sign/rank-based diagnostics with a permutation null.

Every test so far has been magnitude-weighted (Pearson on |surprise| or raw
surprise). The CPI/CPIYOY consistency check showed sign is the reliable
component (r=0.686) and magnitude is not (r=0.242), so test the sign directly.

Design
------
* "Aligned surprise" = signed surprise x theory's predicted direction, so the
  prediction becomes uniformly POSITIVE across every pair.
* Statistics: pooled sign-agreement rate, and Spearman rank correlation
  (rank-based => magnitude-robust).
* Significance: permutation. Shuffle surprise values WITHIN each trigger
  series, keeping every target response fixed. This preserves the marginal
  distribution of surprises and the full dependence structure among targets
  (shared contracts, overlapping windows) and breaks only the trigger->target
  pairing -- which is exactly the null of interest, and is why this is valid
  where treating the 8 cells as independent binomials was not.

Pre-2026 only.
"""
import sys, datetime as dt, numpy as np, polars as pl
sys.path.insert(0, "/private/tmp/claude-503/-Users-caspe2-NUS-FYP/11bb2d79-5115-41c9-bb2a-12cdfe19da73/scratchpad/a2")
sys.path.insert(0, "stg_infra")
from liquid_window import pdf_surprise, mk_all, tr_all

LIQUID_DAYS = 7
MIN_TRADES = 3
N_PERM = 5000
RNG = np.random.default_rng(0)

# (trigger, target_series, side, expected_sign)
# Hawkish surprise (higher inflation / stronger jobs) -> hike up, cut down.
# Higher unemployment is DOVISH, so its signs flip.
SPECS = [
    ("CPI",      "FEDDECISION", "hike", +1), ("CPI",      "FEDDECISION", "cut", -1),
    ("CPIYOY",   "FEDDECISION", "hike", +1), ("CPIYOY",   "FEDDECISION", "cut", -1),
    ("PAYROLLS", "FEDDECISION", "hike", +1), ("PAYROLLS", "FEDDECISION", "cut", -1),
    ("U3",       "FEDDECISION", "hike", -1), ("U3",       "FEDDECISION", "cut", +1),
    ("CPI",      "FED",         "any",  +1),
    ("CPIYOY",   "FED",         "any",  +1),
    ("PAYROLLS", "FED",         "any",  +1),
    ("U3",       "FED",         "any",  -1),
]

_surp_cache = {}
def surprises(series):
    if series not in _surp_cache:
        _surp_cache[series] = pdf_surprise(series)
    return _surp_cache[series]

def side_expr():
    return (pl.when(pl.col("ticker").str.contains(r"-H0$")).then(pl.lit("hold"))
             .when(pl.col("ticker").str.contains(r"-H\d")).then(pl.lit("hike"))
             .when(pl.col("ticker").str.contains(r"-C\d")).then(pl.lit("cut"))
             .otherwise(pl.lit("any")).alias("side"))

_rep_cache = {}
def reps_for(target, side):
    key = (target, side)
    if key in _rep_cache:
        return _rep_cache[key]
    mk = mk_all.filter(pl.col("series") == target).filter(pl.col("close_time").is_not_null())
    mk = mk.with_columns(side_expr())
    if side != "any":
        mk = mk.filter(pl.col("side") == side)
    tk = mk["ticker"].unique().to_list()
    tc = tr_all.filter(pl.col("ticker").is_in(tk)).group_by("ticker").agg(pl.len().alias("n")).collect()
    mk = mk.join(tc, on="ticker", how="left").with_columns(pl.col("n").fill_null(0)).filter(pl.col("n") > 0)
    rep = (mk.sort("n", descending=True).group_by("event_ticker")
           .agg(pl.col("ticker").first().alias("rep"), pl.col("close_time").first().alias("close")))
    rep = rep.sort("close")
    tr = (tr_all.filter(pl.col("ticker").is_in(rep["rep"].unique().to_list()))
          .select("ticker", "yes_price", "created_time").sort(["ticker", "created_time"]).collect())
    _rep_cache[key] = (rep, tr)
    return _rep_cache[key]

def build(trigger, target, side, sign, horizon):
    """Return (event_ids, aligned_surprise_array, response_array)."""
    surp = surprises(trigger)
    if surp.height == 0:
        return None
    rep, tr = reps_for(target, side)
    reps = list(zip(rep["close"].to_list(), rep["rep"].to_list()))
    ev, S, R = [], [], []
    for r in surp.iter_rows(named=True):
        nxt = [(c, t) for c, t in reps if c > r["close_time"]]
        if not nxt:
            continue
        c, t = min(nxt, key=lambda x: x[0])
        gap = (c - r["close_time"]).total_seconds() / 86400
        if not (0 <= gap <= 60):
            continue
        sub = tr.filter(pl.col("ticker") == t)
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
        ev.append(r["event"]); S.append(r["surprise"] * sign); R.append(p1 - p0)
    if not ev:
        return None
    return ev, np.array(S, float), np.array(R, float)


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else np.nan


def pooled_stats(datasets):
    """datasets: list of (aligned_S, R). Returns (sign_rate, mean_spearman, n)."""
    hits = tot = 0
    rhos = []
    for S, R in datasets:
        mask = (S != 0) & (R != 0)
        hits += int((S[mask] * R[mask] > 0).sum()); tot += int(mask.sum())
        rho = spearman(S, R)
        if not np.isnan(rho):
            rhos.append(rho)
    return (hits/tot if tot else np.nan), (float(np.mean(rhos)) if rhos else np.nan), tot


for horizon in ["dormant", "liquid"]:
    built = []
    print(f"\n{'='*74}\nHORIZON: {horizon}\n{'='*74}")
    print(f"{'trigger->target/side':<34}{'n':>4}{'sign%':>8}{'spearman':>10}")
    per_trigger = {}
    for trig, tgt, side, sign in SPECS:
        out = build(trig, tgt, side, sign, horizon)
        if out is None or len(out[0]) < 8:
            continue
        ev, S, R = out
        built.append((trig, S, R))
        per_trigger.setdefault(trig, []).append((ev, S, R))
        mask = (S != 0) & (R != 0)
        sr = (S[mask]*R[mask] > 0).mean() if mask.sum() else np.nan
        print(f"{trig+'->'+tgt+'/'+side:<34}{len(S):>4}{100*sr:>7.0f}%{spearman(S,R):>10.3f}")

    datasets = [(S, R) for _, S, R in built]
    obs_sign, obs_rho, n_tot = pooled_stats(datasets)
    print(f"\n  POOLED: n={n_tot}  sign-agreement={100*obs_sign:.1f}%  mean Spearman={obs_rho:.3f}")

    # Permutation: shuffle surprises within each trigger series, keeping the
    # event->surprise map consistent across that trigger's target cells.
    trig_events = {}
    for trig, cells in per_trigger.items():
        allev = sorted({e for ev, _, _ in cells for e in ev})
        trig_events[trig] = allev

    null_sign, null_rho = [], []
    for _ in range(N_PERM):
        perm_map = {}
        for trig, allev in trig_events.items():
            vals = []
            seen = set()
            for ev, S, R in per_trigger[trig]:
                for e, s in zip(ev, S):
                    if e not in seen:
                        seen.add(e); vals.append(s)
            shuffled = RNG.permutation(vals)
            perm_map[trig] = dict(zip(sorted(seen), shuffled))
        pdatasets = []
        for trig, cells in per_trigger.items():
            for ev, S, R in cells:
                Sp = np.array([perm_map[trig][e] for e in ev], float)
                pdatasets.append((Sp, R))
        s_, r_, _ = pooled_stats(pdatasets)
        null_sign.append(s_); null_rho.append(r_)
    null_sign = np.array(null_sign); null_rho = np.array(null_rho)
    p_sign = float((null_sign >= obs_sign).mean())
    p_rho = float((null_rho >= obs_rho).mean())
    print(f"  permutation null ({N_PERM} draws): sign mean={100*null_sign.mean():.1f}% "
          f"sd={100*null_sign.std():.1f}%   rho mean={null_rho.mean():.3f} sd={null_rho.std():.3f}")
    print(f"  ONE-SIDED p:  sign-agreement p={p_sign:.4f}   Spearman p={p_rho:.4f}")
