from __future__ import annotations
import numpy as np
import pandas as pd


def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x.dropna()
    return pd.Series(x).dropna()


def annualized_return(daily_returns, ann_factor: int = 252) -> float:
    """
    CAGR-style annualized return from daily arithmetic returns.
    """
    r = _to_series(daily_returns)
    if len(r) == 0:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / ann_factor
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annualized_volatility(daily_returns, ann_factor: int = 252) -> float:
    r = _to_series(daily_returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(ann_factor))


def sharpe_ratio(daily_returns, risk_free_rate: float = 0.0, ann_factor: int = 252) -> float:
    """
    risk_free_rate is annual (e.g. 0.03 for 3% annually).
    """
    r = _to_series(daily_returns)
    if len(r) < 2:
        return float("nan")
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / ann_factor) - 1.0
    excess = r - rf_daily
    denom = excess.std(ddof=0)
    if denom == 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * excess.mean() / denom)


def sortino_ratio(daily_returns, risk_free_rate: float = 0.0, ann_factor: int = 252) -> float:
    r = _to_series(daily_returns)
    if len(r) < 2:
        return float("nan")
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / ann_factor) - 1.0
    excess = r - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * excess.mean() / downside_std)


def max_drawdown(equity_curve) -> float:
    """
    Returns minimum drawdown in decimal (e.g. -0.35).
    """
    eq = _to_series(equity_curve)
    if len(eq) == 0:
        return float("nan")
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    return float(dd.min())


def calmar_ratio(daily_returns, ann_factor: int = 252) -> float:
    r = _to_series(daily_returns)
    if len(r) == 0:
        return float("nan")
    eq = (1.0 + r).cumprod()
    mdd = max_drawdown(eq)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    ann_ret = annualized_return(r, ann_factor=ann_factor)
    return float(ann_ret / abs(mdd))


def profit_factor(trade_pnls) -> float:
    p = _to_series(trade_pnls)
    if len(p) == 0:
        return float("nan")
    gross_profit = float(p[p > 0].sum())
    gross_loss = float(p[p < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return gross_profit / abs(gross_loss)


def win_rate(pnls) -> float:
    p = _to_series(pnls)
    if len(p) == 0:
        return float("nan")
    return float((p > 0).mean())


def risk_metrics_summary(
    daily_returns,
    trade_pnls=None,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
) -> dict:
    r = _to_series(daily_returns)
    eq = (1.0 + r).cumprod() if len(r) else pd.Series(dtype=float)

    summary = {
        "annualized_return": annualized_return(r, ann_factor=ann_factor),
        "annualized_volatility": annualized_volatility(r, ann_factor=ann_factor),
        "sharpe_ratio": sharpe_ratio(r, risk_free_rate=risk_free_rate, ann_factor=ann_factor),
        "sortino_ratio": sortino_ratio(r, risk_free_rate=risk_free_rate, ann_factor=ann_factor),
        "max_drawdown": max_drawdown(eq) if len(eq) else float("nan"),
        "calmar_ratio": calmar_ratio(r, ann_factor=ann_factor),
    }

    if trade_pnls is None:
        trade_pnls = r

    summary["profit_factor"] = profit_factor(trade_pnls)
    summary["win_rate"] = win_rate(trade_pnls)
    return summary
