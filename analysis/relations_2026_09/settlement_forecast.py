#!/usr/bin/env python
"""Does the channel signal forecast the target's SETTLEMENT better than its price?

    venv/bin/python analysis/relations_2026_09/settlement_forecast.py

Every previous test in this project predicted a *price change*. Those are dead:
granting perfect foresight over direction and strike, the dormant trade nets
+0.09c and the pre-resolution jump +0.37c (``strike_selection.py``), because the
median leg moves 0.00c against a 1.14c round trip.

Holding to settlement changes the payoff object to 0/100c with no exit cost, and
its oracle is +52c -- so the structure is wide open and the whole question
becomes forecasting: can the signal beat the market's probability by more than
the ~2.5pp cost hurdle?

SPECIFICATION
-------------
unit           one (trigger event, target leg). The target event is the first
               closing strictly after the trigger resolves, within MAX_GAP_DAYS.
decision time  t_res = trigger's close_time.
price          p_entry = the leg's first trade STRICTLY AFTER t_res -- the
               earliest executable price. Deliberately not p0: once the trigger
               has resolved p0 is unobtainable, and using it would import the
               untradable jump research_log.md §13 documents.
signal         d = sign(surprise) * hawkish(trigger) * target_sign(leg),
               theory-fixed, zero fitted parameters (same table as
               channel_pooling.py).
outcome        y = 1 if the leg settled YES (markets.result), else 0.
forecasts      market q = p_entry/100  vs  updated clip(q + delta*d, .01, .99).
               delta on a fixed grid, and fitted WALK-FORWARD (prior years only).
metrics        Brier and log loss (market vs updated); net P&L per contract with
               cost = fee(price) + spread/2 charged once (settlement is free).
dependence     all legs of one target event share a settlement and many triggers
               map to the same event -- everything is bootstrapped clustered on
               target_event.
controls       WTI as trigger (no scheduled release: must show nothing);
               direction permuted within trigger series.
guards         target closes strictly after t_res; p_entry strictly after t_res;
               settlement strictly after close; delta fitted on prior data only;
               no leg selected using its own outcome.

In-sample only (pre-2026).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release
from stg.panel.targets import MAX_GAP_DAYS, target_frames, _group_times
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/relations_2026_09/out")
FEE_RATE, CONTRACTS = 0.07, 100

HAWKISH = {"CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1,
           "PCECORE": +1, "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1,
           "CPIFOOD": +1, "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1,
           "U3": -1, "JOBLESSCLAIMS": -1, "GDP": +1, "ISMPMI": +1,
           "WTI": +1, "WTIW": +1}          # WTI kept only as the control
# A hawkish surprise raises the policy path, so it raises P(rate above k) for
# every "Above k" leg on FED, and P(hike); it lowers P(cut).
TARGETS = {("FED", "any"): +1, ("FEDDECISION", "hike"): +1,
           ("FEDDECISION", "cut"): -1}


def fee_cents(p) -> np.ndarray:
    x = np.asarray(p, dtype=float) / 100.0
    return np.ceil(FEE_RATE * CONTRACTS * x * (1 - x) * 100) / 100.0 * 100.0 / CONTRACTS


def spread_cents(p) -> np.ndarray:
    o = np.full(np.shape(p), 2.0)
    o[np.abs(np.asarray(p, dtype=float) - 50.0) > 40.0] = 1.0
    return o


def build(sp: pl.DataFrame, mk: pl.DataFrame, tr: pl.LazyFrame) -> pl.DataFrame:
    rows = []
    for (target, side), tsign in TARGETS.items():
        legs, tt = target_frames(target, side, markets=mk, trades=tr)
        if legs.height == 0:
            continue
        settled = dict(zip(mk["ticker"].to_list(), mk["result"].to_list()))
        times, prices = {}, {}
        if tt.height:
            for tk, g in zip(*_group_times(tt)):
                times[tk] = g
        for tk, g in tt.group_by("ticker"):
            key = tk[0] if isinstance(tk, tuple) else tk
            prices[key] = g.sort("created_time")["yes_price"].to_numpy()
        ev_legs, ev_close = {}, {}
        for e, c, t in zip(legs["event_ticker"].to_list(), legs["close_time"].to_list(),
                           legs["ticker"].to_list()):
            ev_legs.setdefault(e, []).append(t)
            ev_close[e] = c
        order = sorted({(c, e) for e, c in ev_close.items()})

        for r in sp.iter_rows(named=True):
            trig = r["series"]
            if trig == target or trig not in HAWKISH or is_same_release(trig, target):
                continue
            t_res = r["close_time"]
            d = int(np.sign(r["surprise"])) * HAWKISH[trig] * tsign
            if d == 0:
                continue
            pick = next(((c, e) for c, e in order if c > t_res), None)   # GUARD: strictly after
            if pick is None:
                continue
            c_close, c_event = pick
            gap = (c_close - t_res).total_seconds() / 86400
            if gap > MAX_GAP_DAYS:
                continue
            t0 = np.datetime64(t_res.replace(tzinfo=None), "ns").astype("int64")
            for tk in ev_legs[c_event]:
                res = settled.get(tk)
                if res not in ("yes", "no"):
                    continue
                ts, px = times.get(tk), prices.get(tk)
                if ts is None or px is None:
                    continue
                i = int(np.searchsorted(ts, t0, side="right"))           # GUARD: strictly after
                if i >= len(px):
                    continue
                p_entry = float(px[i])
                if not 1 <= p_entry <= 99:
                    continue
                rows.append(dict(trigger=trig, target=target, side=side,
                                 trigger_event=r["event_ticker"], target_event=c_event,
                                 ticker=tk, t_res=t_res, gap_days=gap,
                                 d=d, p_entry=p_entry, y=1 if res == "yes" else 0))
    return pl.DataFrame(rows)


def brier(q, y):
    return float(np.mean((np.asarray(q) - np.asarray(y)) ** 2))


def logloss(q, y):
    q = np.clip(np.asarray(q, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def cluster_ci(values: np.ndarray, clusters: np.ndarray, n_boot=10000, seed=0):
    """Bootstrap the mean, resampling whole target events."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([idx[c] for c in rng.choice(uniq, uniq.size, replace=True)])
        out[b] = values[take].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)

    d = build(sp, mk, tr)
    assert_no_oos(d, time_col="t_res")
    real = d.filter(pl.col("trigger") != "WTI")
    print(f"rows {d.height}  (non-WTI {real.height})   "
          f"target events {d['target_event'].n_unique()}   "
          f"trigger events {d['trigger_event'].n_unique()}")
    print(f"base rate P(settle yes) = {real['y'].mean():.3f}   "
          f"mean entry price = {real['p_entry'].mean():.1f}c")

    q = real["p_entry"].to_numpy() / 100.0
    y = real["y"].to_numpy().astype(float)
    dd = real["d"].to_numpy().astype(float)
    cl = real["target_event"].to_numpy()

    print("\n=== (a) is the market's own price already well calibrated? ===")
    print(f"market Brier {brier(q, y):.4f}   log loss {logloss(q, y):.4f}   "
          f"base-rate Brier {brier(np.full_like(q, y.mean()), y):.4f}")

    print("\n=== (b) does adding delta*d improve the forecast? ===")
    print("delta | Brier(upd)  dBrier   logloss(upd)  dLogLoss")
    base_b, base_l = brier(q, y), logloss(q, y)
    for delta in (0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10):
        qu = np.clip(q + delta * dd, 0.01, 0.99)
        print(f"{delta:5.2f} | {brier(qu,y):9.4f} {brier(qu,y)-base_b:+8.4f}   "
              f"{logloss(qu,y):10.4f} {logloss(qu,y)-base_l:+9.4f}")

    print("\n=== (c) walk-forward delta (fit on strictly prior years) ===")
    yr = np.array([t.year for t in real["t_res"].to_list()])
    grid = np.arange(0.0, 0.101, 0.005)
    qu_wf = q.copy()
    for Y in sorted(set(yr)):
        prior = yr < Y
        if prior.sum() < 200:
            continue
        best = min(grid, key=lambda g: brier(np.clip(q[prior] + g * dd[prior], .01, .99), y[prior]))
        qu_wf[yr == Y] = np.clip(q[yr == Y] + best * dd[yr == Y], .01, .99)
        print(f"  {Y}: fitted delta = {best:.3f} on {prior.sum()} prior rows, "
              f"applied to {(yr==Y).sum()}")
    ok = qu_wf != q
    if ok.sum():
        print(f"  walk-forward rows {ok.sum()}: Brier {brier(qu_wf[ok],y[ok]):.4f} "
              f"vs market {brier(q[ok],y[ok]):.4f} "
              f"(delta {brier(qu_wf[ok],y[ok])-brier(q[ok],y[ok]):+.4f})")

    print("\n=== (d) net P&L: trade every signal in direction d, hold to settlement ===")
    def pnl(frame: pl.DataFrame) -> np.ndarray:
        p = frame["p_entry"].to_numpy(); yy = frame["y"].to_numpy().astype(float)
        s = frame["d"].to_numpy()
        cost = fee_cents(p) + spread_cents(p) / 2.0
        buy_yes = 100.0 * yy - p
        buy_no = 100.0 * (1 - yy) - (100.0 - p)
        return np.where(s > 0, buy_yes, buy_no) - cost

    rows = []
    for lab, f in [("ALL (non-WTI)", real),
                   ("inflation triggers", real.filter(pl.col("trigger").is_in(
                       ["CPI", "CPICORE", "CPIYOY", "CPICOREYOY", "PCECORE"]))),
                   ("labour triggers", real.filter(pl.col("trigger").is_in(
                       ["PAYROLLS", "U3", "JOBLESSCLAIMS", "ADP"]))),
                   ("target FED", real.filter(pl.col("target") == "FED")),
                   ("target FEDDECISION", real.filter(pl.col("target") == "FEDDECISION")),
                   ("WTI trigger (CONTROL)", d.filter(pl.col("trigger") == "WTI"))]:
        if f.height < 20:
            continue
        v = pnl(f)
        lo, hi, p0 = cluster_ci(v, f["target_event"].to_numpy())
        rows.append(dict(subset=lab, n=len(v), events=f["target_event"].n_unique(),
                         hit=float((v > 0).mean()), mean_net=float(v.mean()),
                         ci_lo=lo, ci_hi=hi, p_le0=p0))
    with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=220):
        print(pl.DataFrame(rows))

    print("\n=== (e) control: direction permuted within trigger series ===")
    rng = np.random.default_rng(0)
    v_obs = pnl(real)
    null = []
    for _ in range(2000):
        perm = real["d"].to_numpy().copy()
        for t in real["trigger"].unique().to_list():
            m = (real["trigger"].to_numpy() == t)
            perm[m] = rng.permutation(perm[m])
        f = real.with_columns(pl.Series("d", perm))
        null.append(pnl(f).mean())
    null = np.array(null)
    print(f"observed {v_obs.mean():+.3f}c   null {null.mean():+.3f} +/- {null.std():.3f}   "
          f"one-sided p = {(1 + (null >= v_obs.mean()).sum()) / (1 + len(null)):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT / "settlement_forecast.parquet")


if __name__ == "__main__":
    main()
