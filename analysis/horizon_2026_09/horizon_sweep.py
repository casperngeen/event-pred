"""Does a longer holding horizon make these edges pay net of cost?

    venv/bin/python analysis/horizon_2026_09/horizon_sweep.py

Motivation. The dormant round trip fails structurally, not marginally
(research_log.md §11.2, and the single-pair decomposition on PAYROLLS->FED):
~65% of the signed move happens in the first post-resolution print, the
capturable remainder averages ~1c, and a taker round trip costs ~3.7c. Costs
are charged per round trip regardless of holding time, so a longer hold
amortises a fixed toll against a larger move. `hold_to_expiry.py` tested the
two endpoints; this sweeps the horizon between them.

Three parts:

  A  Horizon sweep, walk-forward. The OOF sign rule's direction is held fixed
     and evaluated at 1h / 6h / 1d / 3d / 7d / settlement, net of costs.
  B  Per-pair breakdown at the best horizon, with the "buy the favourite"
     control that hold_to_expiry.py §D showed the pooled strategy needs.
  C  Long/short constructions. A single binary contract is already two-sided
     (buying No IS shorting Yes), so "long-short" here means neutralising a
     COMMON factor, not gaining access to the short side.

In-sample only. Run from the repo root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.direction import (
    SignRule, fit_structure, gate_coverage, predict_oof, walk_forward,
)
from stg.panel._io import load_markets, scan_trades
from stg.splits import BURN_IN_END

PANEL = Path("artifacts/panels/pair_panel_dormant.parquet")
SPREADS = Path("artifacts/effective_spreads.parquet")
FEE_RATE = 0.07

# Horizons measured from t_entry (the first post-resolution print). "settle"
# is the binary payoff at the target's own resolution.
HORIZONS = [("1h", 1/24), ("6h", 0.25), ("1d", 1.0), ("3d", 3.0),
            ("7d", 7.0), ("14d", 14.0)]


CONTRACTS = int(os.environ.get("CONTRACTS", "1"))


def fee(price_c: np.ndarray, contracts: int = None) -> np.ndarray:
    """Kalshi taker fee, in cents **per contract**, for an order of ``C``.

    Official schedule (verified 2026-09-11 against Kalshi's published formula,
    last updated 2026-07-07):

        fee = round_up(0.07 * C * P * (1 - P))      P in dollars

    ``C`` sits *inside* the round-up, so the single rounding is amortised
    across the order. At P=0.50 that is 2.00c/contract at C=1 falling to a
    floor of 1.75c by C~100 -- a 12.5% saving and no more. The fee is
    otherwise strictly proportional to size: there is no volume tier.
    """
    C = CONTRACTS if contracts is None else contracts
    p = np.asarray(price_c, float) / 100.0
    total = np.ceil(FEE_RATE * C * p * (1 - p) * 100) / 100.0   # dollars
    return total * 100.0 / C                                    # cents/contract


def build_horizon_prices(panel: pl.DataFrame) -> pl.DataFrame:
    """Price of each row's target ticker at t_entry + h, for each horizon.

    Last trade at or before the cutoff, and only trades strictly after
    ``t_entry`` count -- otherwise the entry print itself would be returned and
    every horizon would read as a zero move.
    """
    tickers = panel["target_ticker"].unique().to_list()
    tr = (scan_trades(is_only=True)
          .filter(pl.col("ticker").is_in(tickers))
          .select("ticker", "created_time", "yes_price")
          .sort("ticker", "created_time").collect())
    print(f"  loaded {tr.height:,} trades for {len(tickers)} target tickers")

    by_ticker: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (tk,), g in tr.group_by("ticker", maintain_order=True):
        by_ticker[tk] = (g["created_time"].to_numpy(),
                         g["yes_price"].to_numpy().astype(float))

    t_entry = panel["t_entry"].to_numpy()
    out: dict[str, list[float]] = {name: [] for name, _ in HORIZONS}
    for i, tk in enumerate(panel["target_ticker"].to_list()):
        times, prices = by_ticker.get(tk, (np.array([]), np.array([])))
        for name, days in HORIZONS:
            if len(times) == 0:
                out[name].append(np.nan)
                continue
            cutoff = t_entry[i] + np.timedelta64(int(days * 86400), "s")
            m = (times > t_entry[i]) & (times <= cutoff)
            out[name].append(prices[m][-1] if m.any() else np.nan)
    return panel.with_columns(
        [pl.Series(f"p_{name}", out[name]) for name, _ in HORIZONS])


def sweep(panel: pl.DataFrame, direction: np.ndarray, mask: np.ndarray,
          spreads: dict[str, float], label: str) -> pl.DataFrame:
    """Net cents per trade at each horizon, for the masked rows."""
    d = panel.filter(pl.Series(mask))
    dir_m = direction[mask].astype(float)
    pe = d["p_entry"].to_numpy().astype(float)
    sp = np.array([spreads.get(t, np.nan) for t in d["target"].to_list()])
    rows = []

    # round-trip horizons
    for name, days in HORIZONS:
        px = d[f"p_{name}"].to_numpy().astype(float)
        ok = np.isfinite(px)
        if ok.sum() < 5:
            continue
        gross = dir_m[ok] * (px[ok] - pe[ok])
        costs = sp[ok] + fee(pe[ok]) + fee(px[ok])
        net = gross - costs
        rows.append(dict(horizon=name, days=days, n=int(ok.sum()),
                         gross=float(gross.mean()), cost=float(np.nanmean(costs)),
                         net=float(np.nanmean(net)),
                         t=float(np.nanmean(net) / (np.nanstd(net, ddof=1)
                                                    / np.sqrt(ok.sum()))),
                         win=float(np.mean(net > 0))))

    # settlement: binary payoff, entry cost only, no exit fee
    res = d["settle_yes"].to_numpy()
    ok = np.isfinite(res.astype(float))
    if ok.sum() >= 5:
        buy_yes = dir_m[ok] > 0
        price = np.where(buy_yes, pe[ok], 100 - pe[ok])
        cost = price + np.nan_to_num(sp[ok], nan=np.nanmean(sp)) / 2 + fee(price)
        payoff = np.where(buy_yes == (res[ok] > 0), 100.0, 0.0)
        net = payoff - cost
        rows.append(dict(horizon="settle", days=float(np.median(
            d["days_to_close"].to_numpy()[ok])), n=int(ok.sum()),
            gross=float((payoff - price).mean()), cost=float((cost - price).mean()),
            net=float(net.mean()),
            t=float(net.mean() / (net.std(ddof=1) / np.sqrt(ok.sum()))),
            win=float(np.mean(net > 0))))
    return pl.DataFrame(rows).with_columns(pl.lit(label).alias("set"))


def main() -> None:
    panel = pl.read_parquet(PANEL).filter(pl.col("t0") >= BURN_IN_END)

    mk = load_markets(is_only=True)
    res = (mk.select("ticker", "result").filter(pl.col("result").is_in(["yes", "no"]))
           .unique(subset=["ticker"]))
    rmap = {t: (1.0 if r == "yes" else 0.0)
            for t, r in zip(res["ticker"].to_list(), res["result"].to_list())}
    panel = panel.with_columns(
        pl.col("target_ticker").replace_strict(rmap, default=None).alias("settle_yes"))

    print("building horizon prices...")
    panel = build_horizon_prices(panel)

    sp_cache = pl.read_parquet(SPREADS).filter(pl.col("window_s") == 60)
    spreads = dict(zip(sp_cache["series"].to_list(),
                       sp_cache["spread_median"].to_list()))

    folds = walk_forward(panel, n_folds=8, start_frac=0.4)
    structures = [(f, fit_structure(panel.filter(pl.Series(f.train)))) for f in folds]
    p_up = predict_oof(panel, SignRule(), structures)
    covered = np.zeros(panel.height, bool)
    for f in folds:
        covered |= f.test
    direction = np.where(p_up >= 0.5, 1, -1)

    print("\n" + "=" * 78)
    print("A  HORIZON SWEEP -- walk-forward OOF sign rule, net of costs")
    print("=" * 78)
    for gate in ("bh", "p05"):
        mask = covered & gate_coverage(panel, structures, gate)
        tab = sweep(panel, direction, mask, spreads, gate)
        print(f"\ngate={gate}  ({mask.sum()} OOF signals)")
        print(f"  {'horizon':<9}{'n':>5}{'gross':>9}{'cost':>8}{'net':>9}{'t':>7}{'win':>7}")
        for r in tab.iter_rows(named=True):
            print(f"  {r['horizon']:<9}{r['n']:>5}{r['gross']:>9.2f}{r['cost']:>8.2f}"
                  f"{r['net']:>9.2f}{r['t']:>7.2f}{r['win']:>7.2f}")
    panel.write_parquet("artifacts/horizon_panel.parquet")
    print("\nwrote artifacts/horizon_panel.parquet")


if __name__ == "__main__":
    main()
