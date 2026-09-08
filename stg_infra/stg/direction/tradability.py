"""From a directional edge to a trade: can the signal be executed, and does it
survive the cost of executing it?

A 63% hit rate is a statistical result. Three things stand between it and an
economic one, and this module measures each:

1. **Entry price.** The structure estimator measures the response from ``p0``,
   the last trade *before* the trigger resolved. Nobody can transact there once
   the trigger has resolved — the first post-resolution print has already moved.
   ``update_2026_08.md`` §5 puts ~48% of the signed move in that first print, so
   the gap between ``p0`` and ``p_entry`` is the part of the effect a trader
   pays for rather than earns.
2. **Spread.** Kalshi retains no historical quotes, but ``taker_side`` labels
   each print as the bid or the ask, so pairing temporally adjacent
   opposite-direction trades recovers an effective spread
   (``research_summary.md`` §4.2.1). Promoted here from the ad-hoc script.
3. **Fees.** Kalshi charges per contract on a ``p(1-p)`` schedule, which is
   worst exactly where these markets sit — near 50c.

Nothing here is a backtest. It is a per-signal ledger with costs attached, so
the economic claim is bounded rather than asserted.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Kalshi trading fee, per contract, rounded up to the cent:
#   fee = ceil(FEE_RATE * price * (1 - price))       price in dollars
# 1.75c at 50c, the project's own working figure (see
# analysis/exploratory_2026_08/spread_measurement.py::FEE50). Verify against the
# live schedule before quoting a net number in the thesis.
FEE_RATE = 0.07
DEFAULT_MAX_GAP_S = 60


def effective_spread(
    trades: pl.DataFrame,
    *,
    max_gap_s: int = DEFAULT_MAX_GAP_S,
    by: str = "ticker",
) -> pl.DataFrame:
    """Effective spread in cents from labelled taker direction.

    A yes-taker print lifts the ask; a no-taker print hits the bid. Adjacent
    opposite-direction prints within ``max_gap_s`` therefore bracket the book,
    and their price difference is an effective spread. Same-timestamp fills are
    collapsed to the first, because a sweep walks the book and only the first
    fill sits at the best quote.

    ``trades`` needs ``ticker, created_time, yes_price, taker_side`` and a
    grouping column if ``by`` is not ``ticker``. Returns median/mean spread and
    the pair count per group.

    Validity check (``research_summary.md`` §4.2.1): widening ``max_gap_s`` 60x
    must not move the estimate. If it does, mid-price drift is contaminating it.
    """
    need = {"ticker", "created_time", "yes_price", "taker_side", by}
    missing = need - set(trades.columns)
    if missing:
        raise ValueError(f"effective_spread needs columns {sorted(missing)}")

    t = (trades.sort("ticker", "created_time")
         .unique(subset=["ticker", "created_time"], keep="first", maintain_order=True))
    t = t.with_columns(
        prev_side=pl.col("taker_side").shift(1).over("ticker"),
        prev_price=pl.col("yes_price").shift(1).over("ticker"),
        prev_time=pl.col("created_time").shift(1).over("ticker"),
    ).drop_nulls(["prev_side", "prev_price", "prev_time"])
    t = t.filter(
        (pl.col("taker_side") != pl.col("prev_side"))
        & ((pl.col("created_time") - pl.col("prev_time")).dt.total_seconds() <= max_gap_s)
    )
    if t.is_empty():
        return pl.DataFrame()
    # yes-taker = ask, no-taker = bid; the signed difference is the spread
    t = t.with_columns(
        spread=pl.when(pl.col("taker_side") == "yes")
        .then(pl.col("yes_price") - pl.col("prev_price"))
        .otherwise(pl.col("prev_price") - pl.col("yes_price"))
    ).filter(pl.col("spread") >= 0)
    return (t.group_by(by)
            .agg(pl.col("spread").median().alias("spread_median"),
                 pl.col("spread").mean().alias("spread_mean"),
                 pl.len().alias("n_pairs"))
            .sort(by))


def trade_ledger(
    panel: pl.DataFrame,
    predictions: np.ndarray,
    mask: np.ndarray,
    *,
    spreads: dict[str, float] | None = None,
    fee_rate: float = FEE_RATE,
) -> pl.DataFrame:
    """One row per executed signal, with gross and net cents.

    ``predictions`` is +1/-1 per panel row (out-of-fold), ``mask`` selects the
    rows actually traded. Two gross measures are reported:

    ``gross_p0``      direction x (p1 - p0)      what the estimator measures
    ``gross_entry``   direction x (p1 - p_entry) what a trader could capture

    Costs are reported under two execution assumptions:

    ``net``        taker — cross to enter and to exit, paying one effective
                   spread per round trip, plus the fee on both legs.
    ``net_maker``  the most favourable case — both legs filled by resting
                   orders, *earning* the spread instead of paying it, fees
                   still charged. Fill risk is not modelled, so this is an
                   upper bound, not a strategy.

    ``research_summary.md`` §6.4 argues for making rather than taking, on the
    grounds that a multi-day drift needs no immediacy. That argument does not
    transfer to this horizon: the dormant window closes in well under an hour,
    so a resting order that goes unfilled misses the move entirely. The taker
    figure is the honest central case here and the maker figure brackets it.
    """
    df = panel.filter(pl.Series(mask)).with_columns(
        pl.Series("direction", predictions[mask].astype(float)))
    df = df.filter(pl.col("p_entry").is_not_null())
    if df.is_empty():
        return df

    sp = spreads or {}
    spread_col = pl.Series(
        "spread", [float(sp.get(t, np.nan)) for t in df["target"].to_list()])
    p_exit = df["p0"] + df["response"]

    def fee(price_cents: pl.Series) -> pl.Series:
        p = price_cents.to_numpy().astype(float) / 100.0
        return pl.Series(np.ceil(fee_rate * p * (1 - p) * 100) / 100.0 * 100)

    df = df.with_columns(
        spread_col,
        pl.Series("p_exit", p_exit),
        gross_p0=pl.col("direction") * pl.col("response"),
        gross_entry=pl.col("direction") * (pl.Series(p_exit) - pl.col("p_entry")),
        fees=pl.Series(fee(df["p_entry"]).to_numpy() + fee(p_exit).to_numpy()),
    )
    return df.with_columns(
        net=pl.col("gross_entry") - pl.col("spread") - pl.col("fees"),
        net_maker=pl.col("gross_entry") + pl.col("spread") - pl.col("fees"),
        hit=(pl.col("gross_entry") > 0).cast(pl.Int8),
        hold_hours=(pl.col("t_exit") - pl.col("t_entry")).dt.total_seconds() / 3600,
        entry_lag_min=(pl.col("t_entry") - pl.col("t0")).dt.total_seconds() / 60,
    )


def ledger_summary(ledger: pl.DataFrame) -> dict:
    """Aggregate the ledger into the numbers a tradability claim needs.

    Cost aggregates are taken over the rows that *have* a spread estimate. A
    target too thin to yield one adjacent opposite-direction trade pair (which
    is itself a tradability finding) would otherwise turn every mean into NaN.
    """
    if ledger.is_empty():
        return {}
    n = ledger.height
    priced = ledger.filter(pl.col("spread").is_not_nan()
                           & pl.col("spread").is_not_null())

    def num(v, default=float("nan")):
        return default if v is None else float(v)

    def cost(col, agg="mean"):
        if priced.is_empty():
            return float("nan")
        return num(getattr(priced[col], agg)())

    span_years = max(
        (ledger["t0"].max() - ledger["t0"].min()).total_seconds() / (365.25 * 86400),
        1e-9)
    unpriced = sorted(set(ledger["target"].to_list())
                      - set(priced["target"].to_list())) if not priced.is_empty() \
        else sorted(set(ledger["target"].to_list()))
    return dict(
        n_trades=n,
        n_priced=priced.height,
        unpriced_targets=unpriced,
        per_year=n / span_years,
        hit_rate=float(ledger["hit"].mean()),
        gross_p0=float(ledger["gross_p0"].mean()),
        gross_entry=float(ledger["gross_entry"].mean()),
        entry_slippage=float((ledger["gross_p0"] - ledger["gross_entry"]).mean()),
        spread=cost("spread"),
        fees=cost("fees"),
        net=cost("net"),
        net_maker=cost("net_maker"),
        net_median=cost("net", "median"),
        net_sd=cost("net", "std"),
        net_total=cost("net", "sum"),
        entry_lag_min_median=float(ledger["entry_lag_min"].median()),
        hold_hours_median=float(ledger["hold_hours"].median()),
    )


STALENESS_BUCKETS = ((0.0, 1.0, "<1h"), (1.0, 24.0, "1-24h"), (24.0, 1e9, ">24h"))


def decompose_move(panel: pl.DataFrame, direction: np.ndarray,
                   mask: np.ndarray) -> tuple[pl.DataFrame, dict]:
    """Split the signed response into the part before and after the first
    executable price, and bucket it by how stale the reference price was.

        jump   p0 -> first post-resolution print   (nobody can trade this)
        drift  first print -> third print          (the executable part)

    ``p0`` is the last trade before the trigger resolved, which in these books
    can be hours or days old. If the measured edge were an artifact of that
    staleness (``research_summary.md`` §4.1's central threat), accuracy would
    *rise* with staleness. The returned table is that test.
    """
    d = panel.filter(pl.Series(mask)).with_columns(
        pl.Series("dir", direction[mask].astype(float)))
    d = d.with_columns(
        stale_h=(pl.col("t0") - pl.col("t_p0")).dt.total_seconds() / 3600,
        jump=pl.col("p_entry") - pl.col("p0"),
        drift=(pl.col("p0") + pl.col("response")) - pl.col("p_entry"),
    ).with_columns(
        s_jump=pl.col("dir") * pl.col("jump"),
        s_drift=pl.col("dir") * pl.col("drift"),
        hit=(pl.col("dir") == pl.col("y")).cast(pl.Int8),
    )
    bucket = pl.lit(STALENESS_BUCKETS[-1][2])
    for lo, hi, name in reversed(STALENESS_BUCKETS[:-1]):
        bucket = pl.when(pl.col("stale_h") < hi).then(pl.lit(name)).otherwise(bucket)
    d = d.with_columns(bucket=bucket)

    order = {name: i for i, (_, _, name) in enumerate(STALENESS_BUCKETS)}
    by_stale = (d.group_by("bucket").agg(
        pl.len().alias("n"),
        pl.col("hit").mean().alias("acc"),
        pl.col("s_jump").mean(), pl.col("s_drift").mean(),
        pl.col("stale_h").median().alias("stale_median"))
        .with_columns(o=pl.col("bucket").replace_strict(order, default=99))
        .sort("o").drop("o"))

    z = d["z_surprise"].abs().to_numpy()
    summary = dict(
        n=d.height,
        stale_median_h=float(d["stale_h"].median()),
        stale_p75_h=float(d["stale_h"].quantile(0.75)),
        frac_over_6h=float((d["stale_h"] > 6).mean()),
        s_jump=float(d["s_jump"].mean()), s_drift=float(d["s_drift"].mean()),
        hit_total=float(d["hit"].mean()),
        hit_jump=float((d["s_jump"] > 0).mean()),
        hit_drift=float((d["s_drift"] > 0).mean()),
        abs_jump_median=float(d["jump"].abs().median()),
        abs_drift_median=float(d["drift"].abs().median()),
        corr_stale_absjump=float(np.corrcoef(d["stale_h"], d["jump"].abs())[0, 1]),
        corr_z_absjump=float(np.corrcoef(z, d["jump"].abs())[0, 1]),
        corr_z_absdrift=float(np.corrcoef(z, d["drift"].abs())[0, 1]),
    )
    return by_stale, summary
