#!/usr/bin/env python
"""Does trading the *best strike* beat trading the most-traded leg?

    venv/bin/python analysis/relations_2026_09/strike_selection.py

The pipeline collapses every target event to ONE contract -- the most-traded leg
at the trigger instant (``panel/targets.py::_best_leg``). That is a measurement
convenience, not a trading decision. A surprise shifts the target's whole
implied distribution, so every strike reprices, each by a different amount and
at a different cost: the Kalshi fee ``0.07*p(1-p)`` is maximal at the money and
the measured spread *narrows* away from it (research_log.md §11.2).

So: given the same signal, is there a strike-selection rule that beats the
default? ``research_log.md`` §11.2 answered this with a *model* -- an assumed
Gaussian shift against assumed costs -- and concluded there is no cost-optimal
strike. This measures it instead, on realised strike-level moves.

**The signal is the zero-parameter channel rule**, not the per-pair sign rule:
``direction = sign(surprise) * hawkish(trigger) * target_sign(leg)``. Nothing is
fitted, so no fold machinery is needed and there is no in-sample edge selection
contaminating the comparison. It is also the strongest signal in the project
(63.2% aligned, p=0.0004 clustered; see channel_pooling.py).

**Read the ``oracle`` row first.** It picks the leg that actually moved most in
the predicted direction -- not a strategy, an upper bound. If the
best-leg-in-hindsight still loses after costs, no selection rule can work and
the question is closed regardless of predictive skill.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release
from stg.panel.targets import target_frames, _group_times
from stg.splits import assert_no_oos

PANELS = Path("artifacts/panels")
OUT = Path("analysis/relations_2026_09/out")

HAWKISH = {"CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1,
           "PCECORE": +1, "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1,
           "CPIFOOD": +1, "CPIAPPAREL": +1, "PAYROLLS": +1, "ADP": +1,
           "U3": -1, "JOBLESSCLAIMS": -1, "GDP": +1, "ISMPMI": +1}
TARGET_SIGN = {("FED", "any"): +1, ("FEDDECISION", "hike"): +1,
               ("FEDDECISION", "cut"): -1}

FEE_RATE = 0.07
CONTRACTS = 100


def fee_cents(price_cents: np.ndarray) -> np.ndarray:
    """Kalshi fee in cents per contract, matching direction/tradability.py."""
    p = np.asarray(price_cents, dtype=float) / 100.0
    total = np.ceil(FEE_RATE * CONTRACTS * p * (1 - p) * 100) / 100.0
    return total * 100.0 / CONTRACTS


def spread_cents(price_cents: np.ndarray) -> np.ndarray:
    """Effective spread by moneyness -- the measured medians in §11.2.

    Spreads *narrow* away from the money: 2.0c median inside 40c of the money,
    1.0c beyond. This is the half of the cost structure that works in favour of
    an off-ATM strike; the fee is the other half.
    """
    out = np.full(np.shape(price_cents), 2.0)
    out[np.abs(np.asarray(price_cents) - 50.0) > 40.0] = 1.0
    return out


def build_legs(sp: pl.DataFrame, mk, tr) -> pl.DataFrame:
    """Every (signal, leg) with a reference price and a dormant exit."""
    rows = []
    for (target, side), tsign in TARGET_SIGN.items():
        legs, tt = target_frames(target, side, markets=mk, trades=tr)
        if legs.height == 0:
            continue
        times = {}
        if tt.height:
            for tk, g in zip(*_group_times(tt)):
                times[tk] = g
        prices = {}
        for tk, g in tt.group_by("ticker"):
            gg = g.sort("created_time")
            prices[tk[0] if isinstance(tk, tuple) else tk] = gg["yes_price"].to_numpy()
        ev_close = dict(zip(legs["event_ticker"].to_list(), legs["close_time"].to_list()))
        ev_legs: dict = {}
        for c_ev, c_tk in zip(legs["event_ticker"].to_list(), legs["ticker"].to_list()):
            ev_legs.setdefault(c_ev, []).append(c_tk)
        order = sorted({(c, e) for e, c in ev_close.items()})

        for r in sp.iter_rows(named=True):
            trig = r["series"]
            if trig == target or trig not in HAWKISH or is_same_release(trig, target):
                continue
            t_res = r["close_time"]
            t0 = np.datetime64(t_res.replace(tzinfo=None), "ns").astype("int64")
            pick = next(((c, e) for c, e in order if c > t_res), None)
            if pick is None:
                continue
            c_close, c_event = pick
            if (c_close - t_res).total_seconds() / 86400 > 45:
                continue
            direction = int(np.sign(r["surprise"])) * HAWKISH[trig] * tsign
            if direction == 0:
                continue
            for tk in ev_legs.get(c_event, []):
                ts, px = times.get(tk), prices.get(tk)
                if ts is None or px is None or len(ts) == 0:
                    continue
                i = int(np.searchsorted(ts, t0, side="right"))
                if i == 0 or len(ts) - i < 3:
                    continue
                p0, p_entry, p1 = float(px[i - 1]), float(px[i]), float(px[i + 2])
                rows.append(dict(
                    trigger=trig, target=target, side=side,
                    trigger_event=r["event_ticker"], target_event=c_event,
                    ticker=tk, direction=direction, n_pre=i,
                    p0=p0, p_entry=p_entry, p1=p1,
                    moneyness=abs(p_entry - 50.0),
                ))
    d = pl.DataFrame(rows)
    if d.height == 0:
        return d
    pe, p1 = d["p_entry"].to_numpy(), d["p1"].to_numpy()
    gross = d["direction"].to_numpy() * (p1 - pe)
    cost = fee_cents(pe) + fee_cents(p1) + spread_cents(pe)
    spread = spread_cents(pe)
    fees = fee_cents(pe) + fee_cents(p1)
    return d.with_columns(
        pl.Series("gross", gross), pl.Series("cost", cost),
        pl.Series("net", gross - cost),
        # §6.4's maker bound: the spread becomes a credit rather than a cost.
        # Fill risk is ignored, so this is an upper bound on an upper bound.
        pl.Series("net_maker", gross + spread - fees))


def select(d: pl.DataFrame, rule: str) -> pl.DataFrame:
    """One leg per signal, by the named rule."""
    key = ["trigger_event", "target_event"]
    if rule == "most_traded":
        e = pl.col("n_pre")                       # current pipeline behaviour
    elif rule == "atm":
        e = -pl.col("moneyness")
    elif rule == "cheapest":
        e = -pl.col("cost")                       # furthest out: lowest fee+spread
    elif rule == "oracle":
        e = pl.col("net")                         # upper bound, not a strategy
    elif rule == "worst":
        e = -pl.col("net")                        # lower bound, for the range
    else:
        raise ValueError(rule)
    return (d.with_columns(e.alias("_s"))
             .sort("_s", "ticker", descending=[True, False])
             .group_by(key, maintain_order=True).first().drop("_s"))


def summarise(d: pl.DataFrame, label: str) -> dict:
    net = d["net"].to_numpy()
    t = float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 1 else float("nan")
    return dict(rule=label, n=len(net), mean_price=float(d["p_entry"].mean()),
                mean_gross=float(d["gross"].mean()), mean_cost=float(d["cost"].mean()),
                mean_net=float(net.mean()), median_net=float(np.median(net)),
                mean_net_maker=float(d["net_maker"].mean()),
                hit=float((d["gross"].to_numpy() > 0).mean()), t=t)


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    mk, tr = load_markets(is_only=True), scan_trades(is_only=True)

    d = build_legs(sp, mk, tr)
    print(f"(signal, leg) rows: {d.height}   "
          f"signals: {d.select(['trigger_event','target_event']).n_unique()}")
    print(f"legs per signal: median "
          f"{d.group_by(['trigger_event','target_event']).len()['len'].median()}")

    print("\n=== selection rules, cents per trade, taker both ways ===")
    rows = [summarise(select(d, r), r) for r in
            ("most_traded", "atm", "cheapest", "oracle", "worst")]
    rows.append(summarise(d, "ALL legs (trade every strike)"))
    with pl.Config(tbl_rows=20, float_precision=3, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    print("\n=== by moneyness band (every leg, not selected) ===")
    b = d.with_columns(
        pl.when(pl.col("moneyness") <= 15).then(pl.lit("ATM +/-15c"))
         .when(pl.col("moneyness") <= 30).then(pl.lit("15-30c out"))
         .when(pl.col("moneyness") <= 40).then(pl.lit("30-40c out"))
         .otherwise(pl.lit(">40c out")).alias("band"))
    with pl.Config(tbl_rows=10, float_precision=3):
        print(b.group_by("band").agg(
            pl.len().alias("n"),
            pl.col("gross").mean().alias("gross"),
            pl.col("cost").mean().alias("cost"),
            pl.col("net").mean().alias("net"),
            (pl.col("gross") > 0).mean().alias("hit")).sort("net", descending=True))

    OUT.mkdir(parents=True, exist_ok=True)
    d.write_parquet(OUT / "strike_legs.parquet")


if __name__ == "__main__":
    main()
