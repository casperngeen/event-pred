from __future__ import annotations

from typing import List, Tuple

import numpy as np
import polars as pl

from stg_infra.stg.pairs.config import EventPairsConfig
from stg_infra.stg.pairs.stats import half_life, pairwise_corr, rolling_beta


def select_event_universe(
    eprice: pl.DataFrame,
    *,
    min_event_days: int,
    top_events: int,
) -> List[str]:
    estats = (
        eprice.group_by("event_ticker")
        .agg([
            pl.len().alias("n_days"),
            pl.col("log_volume_norm").mean().alias("avg_liq_proxy"),
        ])
        .filter(pl.col("n_days") >= min_event_days)
        .sort("avg_liq_proxy", descending=True)
        .head(top_events)
    )
    return estats["event_ticker"].to_list()


def build_event_panel(
    eprice: pl.DataFrame,
    events: List[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    wide_e = (
        eprice.filter(pl.col("event_ticker").is_in(events))
        .pivot(index="date", on="event_ticker", values="implied_mean_raw", aggregate_function="first")
        .sort("date")
    )
    dates = wide_e["date"].to_numpy()
    ecols = [c for c in wide_e.columns if c != "date"]
    EX = wide_e.select(ecols).to_numpy()
    return dates, ecols, EX


def select_event_pairs(
    dates: np.ndarray,
    ecols: List[str],
    EX: np.ndarray,
    cfg: EventPairsConfig,
) -> Tuple[List[Tuple[str, str]], pl.DataFrame]:
    if len(dates) < cfg.lookback_sel + 2:
        raise RuntimeError(f"Not enough dates: have {len(dates)} need ~{cfg.lookback_sel}+")

    EXs = EX[-cfg.lookback_sel:]

    candidates = []
    n = len(ecols)

    min_corr_overlap = max(8, cfg.lookback_sel // 2)
    min_level_overlap = max(12, cfg.lookback_sel // 2 + 2)

    for i in range(n):
        for j in range(i + 1, n):
            a = EXs[:, i]
            b = EXs[:, j]

            c = pairwise_corr(a, b, min_overlap=min_corr_overlap)
            if not np.isfinite(c):
                continue
            if c < cfg.corr_min or c > cfg.corr_max:
                continue

            m = np.isfinite(a) & np.isfinite(b)
            a2 = a[m]
            b2 = b[m]
            overlap = int(len(a2))
            if overlap < min_level_overlap:
                continue

            beta = rolling_beta(a2, b2, min(cfg.beta_lookback, overlap))
            if not np.isfinite(beta):
                continue

            spread = a2 - beta * b2
            hl = half_life(spread)
            if not np.isfinite(hl) or hl < cfg.half_life_min or hl > cfg.half_life_max:
                continue

            candidates.append(
                {
                    "event_A": ecols[i],
                    "event_B": ecols[j],
                    "corr": float(c),
                    "half_life": float(hl),
                    "overlap_n": overlap,
                }
            )

    if not candidates:
        diag = pl.DataFrame({"event_A": [], "event_B": [], "corr": [], "half_life": [], "overlap_n": []})
        return [], diag

    diag = pl.DataFrame(candidates).sort("corr", descending=True)
    top = diag.head(cfg.top_pairs)

    pairs = list(zip(top["event_A"].to_list(), top["event_B"].to_list()))
    return pairs, diag
