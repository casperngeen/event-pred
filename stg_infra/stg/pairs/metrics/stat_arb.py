from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import adfuller


def _clean(spread) -> pd.Series:
    if isinstance(spread, pd.Series):
        return spread.dropna()
    return pd.Series(spread).dropna()


def adf_pvalue(spread, min_obs: int = 20) -> float:
    s = _clean(spread)
    if len(s) < min_obs:
        return float("nan")
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return float("nan")


def stationary_ratio(pair_to_spread: dict[str, pd.Series], p_threshold: float = 0.05) -> float:
    pvals = []
    for _, s in pair_to_spread.items():
        p = adf_pvalue(s)
        if not np.isnan(p):
            pvals.append(p)
    if len(pvals) == 0:
        return float("nan")
    pvals = np.array(pvals)
    return float((pvals < p_threshold).mean())


def zero_crossing_rate(spread, periods_per_week: float = 5.0) -> float:
    """
    Count sign changes around mean per week.
    """
    s = _clean(spread)
    if len(s) < 2:
        return float("nan")
    z = s - s.mean()
    signs = np.sign(z)
    crossings = int((signs[1:] * signs[:-1] < 0).sum())
    weeks = len(s) / periods_per_week
    if weeks <= 0:
        return float("nan")
    return float(crossings / weeks)


def zscore_tail_stats(spread) -> dict:
    s = _clean(spread)
    if len(s) < 3:
        return {"z_skew": float("nan"), "z_kurtosis": float("nan")}
    std = s.std(ddof=0)
    if std == 0:
        return {"z_skew": float("nan"), "z_kurtosis": float("nan")}
    z = (s - s.mean()) / std
    return {
        "z_skew": float(skew(z, bias=False)),
        "z_kurtosis": float(kurtosis(z, fisher=False, bias=False)),
    }


def pair_stat_arb_metrics(
    pair_to_spread: dict[str, pd.Series],
    periods_per_week: float = 5.0,
    p_threshold: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      pair_df columns: pair, adf_pvalue, zero_cross_rate_week, z_skew, z_kurtosis
      summary dict with stationary_ratio and aggregate stats
    """
    rows = []
    for pair, spread in pair_to_spread.items():
        p = adf_pvalue(spread)
        zcr = zero_crossing_rate(spread, periods_per_week=periods_per_week)
        tails = zscore_tail_stats(spread)
        rows.append(
            {
                "pair": pair,
                "adf_pvalue": p,
                "zero_cross_rate_week": zcr,
                "z_skew": tails["z_skew"],
                "z_kurtosis": tails["z_kurtosis"],
            }
        )

    pair_df = pd.DataFrame(rows)

    if len(pair_df) == 0:
        summary = {
            "num_pairs": 0,
            "stationary_ratio_p_lt_0.05": float("nan"),
            "median_adf_pvalue": float("nan"),
            "median_zero_cross_rate_week": float("nan"),
            "median_z_kurtosis": float("nan"),
        }
        return pair_df, summary

    valid_p = pair_df["adf_pvalue"].dropna()
    summary = {
        "num_pairs": int(len(pair_df)),
        f"stationary_ratio_p_lt_{p_threshold}": float((valid_p < p_threshold).mean()) if len(valid_p) else float("nan"),
        "median_adf_pvalue": float(pair_df["adf_pvalue"].median(skipna=True)),
        "median_zero_cross_rate_week": float(pair_df["zero_cross_rate_week"].median(skipna=True)),
        "median_z_kurtosis": float(pair_df["z_kurtosis"].median(skipna=True)),
    }
    return pair_df, summary
