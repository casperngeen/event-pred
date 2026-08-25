from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .risk import risk_metrics_summary
from .stat_arb import pair_stat_arb_metrics
from .predictive import predictive_metrics_summary
from .graph import graph_metrics_over_time, average_edge_persistence


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if pd.isna(x):
        return None
    return x


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).dropna()
    return (1.0 + r).cumprod()


def _drawdown_curve(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _save_basic_plots(
    out_dir: Path,
    baseline_returns: pd.Series | None,
    stg_returns: pd.Series | None,
    pair_df_baseline: pd.DataFrame | None,
    pair_df_stg: pd.DataFrame | None,
    graph_ts_baseline: pd.DataFrame | None,
    graph_ts_stg: pd.DataFrame | None,
):
    # 1) Equity + Drawdown
    if baseline_returns is not None or stg_returns is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        if baseline_returns is not None:
            eq_b = _equity_curve(baseline_returns)
            dd_b = _drawdown_curve(eq_b)
            axes[0].plot(eq_b.index, eq_b.values, label="Baseline")
            axes[1].plot(dd_b.index, dd_b.values, label="Baseline")
        if stg_returns is not None:
            eq_s = _equity_curve(stg_returns)
            dd_s = _drawdown_curve(eq_s)
            axes[0].plot(eq_s.index, eq_s.values, label="STG")
            axes[1].plot(dd_s.index, dd_s.values, label="STG")
        axes[0].set_title("Equity Curve")
        axes[1].set_title("Drawdown")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        fig.savefig(out_dir / "equity_drawdown.png", dpi=150)
        plt.close(fig)

    # 2) ADF p-value histogram
    if pair_df_baseline is not None or pair_df_stg is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        if pair_df_baseline is not None and "adf_pvalue" in pair_df_baseline:
            ax.hist(pair_df_baseline["adf_pvalue"].dropna(), bins=30, alpha=0.6, label="Baseline")
        if pair_df_stg is not None and "adf_pvalue" in pair_df_stg:
            ax.hist(pair_df_stg["adf_pvalue"].dropna(), bins=30, alpha=0.6, label="STG")
        ax.set_title("ADF p-value Distribution")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "adf_pvalue_hist.png", dpi=150)
        plt.close(fig)

    # 3) Edge turnover time-series
    if graph_ts_baseline is not None or graph_ts_stg is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        if graph_ts_baseline is not None and "turnover" in graph_ts_baseline:
            ax.plot(graph_ts_baseline["date"], graph_ts_baseline["turnover"], label="Baseline")
        if graph_ts_stg is not None and "turnover" in graph_ts_stg:
            ax.plot(graph_ts_stg["date"], graph_ts_stg["turnover"], label="STG")
        ax.set_title("Edge Turnover Over Time")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "edge_turnover.png", dpi=150)
        plt.close(fig)


def build_full_metrics_report(
    out_dir: str | Path,
    # --- baseline inputs ---
    baseline_daily_returns: pd.Series | None = None,
    baseline_trade_pnls: pd.Series | None = None,
    baseline_pair_to_spread: dict[str, pd.Series] | None = None,
    baseline_y_true=None,
    baseline_y_pred=None,
    baseline_dated_edge_sets: list[tuple[pd.Timestamp, set[tuple]]] | None = None,
    # --- stg inputs ---
    stg_daily_returns: pd.Series | None = None,
    stg_trade_pnls: pd.Series | None = None,
    stg_pair_to_spread: dict[str, pd.Series] | None = None,
    stg_y_true=None,
    stg_y_pred=None,
    stg_dated_edge_sets: list[tuple[pd.Timestamp, set[tuple]]] | None = None,
    # shared
    node_to_category: dict | None = None,
    risk_free_rate: float = 0.0,
    ann_factor: int = 252,
    periods_per_week: float = 5.0,
    adf_p_threshold: float = 0.05,
    save_plots: bool = True,
) -> dict:
    """
    Produces:
      - metrics_summary.json
      - pair_level_metrics_baseline.csv / pair_level_metrics_stg.csv
      - graph_timeseries_baseline.csv / graph_timeseries_stg.csv
      - comparative_metrics.csv
      - optional plots
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    summary = {"baseline": {}, "stg": {}, "comparison": {}}

    # ---------------- baseline ----------------
    pair_df_baseline = None
    graph_ts_baseline = None

    if baseline_daily_returns is not None:
        summary["baseline"]["risk"] = risk_metrics_summary(
            baseline_daily_returns,
            trade_pnls=baseline_trade_pnls if baseline_trade_pnls is not None else baseline_daily_returns,
            risk_free_rate=risk_free_rate,
            ann_factor=ann_factor,
        )

    if baseline_pair_to_spread is not None:
        pair_df_baseline, stat_summary_b = pair_stat_arb_metrics(
            baseline_pair_to_spread,
            periods_per_week=periods_per_week,
            p_threshold=adf_p_threshold,
        )
        summary["baseline"]["stat_arb"] = stat_summary_b
        pair_df_baseline.to_csv(out_dir / "pair_level_metrics_baseline.csv", index=False)

    if baseline_y_true is not None and baseline_y_pred is not None:
        summary["baseline"]["predictive"] = predictive_metrics_summary(
            baseline_y_true, baseline_y_pred, ic_method="spearman"
        )

    if baseline_dated_edge_sets is not None:
        graph_ts_baseline = graph_metrics_over_time(
            baseline_dated_edge_sets, node_to_category=node_to_category
        )
        summary["baseline"]["graph"] = {
            "avg_turnover": float(graph_ts_baseline["turnover"].dropna().mean()) if "turnover" in graph_ts_baseline else float("nan"),
            "avg_homophily": float(graph_ts_baseline["homophily"].dropna().mean()) if "homophily" in graph_ts_baseline else float("nan"),
            "avg_edge_persistence": average_edge_persistence([x[1] for x in baseline_dated_edge_sets]),
        }
        graph_ts_baseline.to_csv(out_dir / "graph_timeseries_baseline.csv", index=False)

    # ---------------- stg ----------------
    pair_df_stg = None
    graph_ts_stg = None

    if stg_daily_returns is not None:
        summary["stg"]["risk"] = risk_metrics_summary(
            stg_daily_returns,
            trade_pnls=stg_trade_pnls if stg_trade_pnls is not None else stg_daily_returns,
            risk_free_rate=risk_free_rate,
            ann_factor=ann_factor,
        )

    if stg_pair_to_spread is not None:
        pair_df_stg, stat_summary_s = pair_stat_arb_metrics(
            stg_pair_to_spread,
            periods_per_week=periods_per_week,
            p_threshold=adf_p_threshold,
        )
        summary["stg"]["stat_arb"] = stat_summary_s
        pair_df_stg.to_csv(out_dir / "pair_level_metrics_stg.csv", index=False)

    if stg_y_true is not None and stg_y_pred is not None:
        summary["stg"]["predictive"] = predictive_metrics_summary(
            stg_y_true, stg_y_pred, ic_method="spearman"
        )

    if stg_dated_edge_sets is not None:
        graph_ts_stg = graph_metrics_over_time(
            stg_dated_edge_sets, node_to_category=node_to_category
        )
        summary["stg"]["graph"] = {
            "avg_turnover": float(graph_ts_stg["turnover"].dropna().mean()) if "turnover" in graph_ts_stg else float("nan"),
            "avg_homophily": float(graph_ts_stg["homophily"].dropna().mean()) if "homophily" in graph_ts_stg else float("nan"),
            "avg_edge_persistence": average_edge_persistence([x[1] for x in stg_dated_edge_sets]),
        }
        graph_ts_stg.to_csv(out_dir / "graph_timeseries_stg.csv", index=False)

    # ---------------- comparison table ----------------
    comp_rows = []
    def add_comp(metric_name, base_val, stg_val, higher_is_better=True):
        if base_val is None or stg_val is None:
            return
        try:
            b = float(base_val)
            s = float(stg_val)
        except Exception:
            return
        rel = np.nan
        if np.isfinite(b) and b != 0:
            rel = (s - b) / abs(b)
        better = None
        if np.isfinite(b) and np.isfinite(s):
            better = "stg" if ((s > b) if higher_is_better else (s < b)) else "baseline"
        comp_rows.append(
            {
                "metric": metric_name,
                "baseline": b,
                "stg": s,
                "relative_change_vs_baseline": rel,
                "better": better,
            }
        )

    # risk comparisons
    for m, hib in [
        ("annualized_return", True),
        ("annualized_volatility", False),
        ("sharpe_ratio", True),
        ("sortino_ratio", True),
        ("max_drawdown", True),  # less negative is better
        ("calmar_ratio", True),
        ("profit_factor", True),
        ("win_rate", True),
    ]:
        b = summary.get("baseline", {}).get("risk", {}).get(m)
        s = summary.get("stg", {}).get("risk", {}).get(m)
        if m == "max_drawdown":
            add_comp(m, b, s, higher_is_better=True)
        else:
            add_comp(m, b, s, higher_is_better=hib)

    # stat-arb comparisons
    stat_key = f"stationary_ratio_p_lt_{adf_p_threshold}"
    for m, hib in [
        (stat_key, True),
        ("median_adf_pvalue", False),
        ("median_zero_cross_rate_week", True),
        ("median_z_kurtosis", False),
    ]:
        b = summary.get("baseline", {}).get("stat_arb", {}).get(m)
        s = summary.get("stg", {}).get("stat_arb", {}).get(m)
        add_comp(m, b, s, higher_is_better=hib)

    # predictive comparisons
    for m, hib in [("ic", True), ("rmse", False), ("mae", False), ("mape", False)]:
        b = summary.get("baseline", {}).get("predictive", {}).get(m)
        s = summary.get("stg", {}).get("predictive", {}).get(m)
        add_comp(m, b, s, higher_is_better=hib)

    # graph comparisons
    for m, hib in [
        ("avg_turnover", False),
        ("avg_homophily", True),
        ("avg_edge_persistence", True),
    ]:
        b = summary.get("baseline", {}).get("graph", {}).get(m)
        s = summary.get("stg", {}).get("graph", {}).get(m)
        add_comp(m, b, s, higher_is_better=hib)

    comp_df = pd.DataFrame(comp_rows)
    if len(comp_df):
        comp_df.to_csv(out_dir / "comparative_metrics.csv", index=False)
    summary["comparison"]["num_compared_metrics"] = int(len(comp_df))

    # ---------------- save summary ----------------
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # ---------------- plots ----------------
    if save_plots:
        _save_basic_plots(
            out_dir=out_dir,
            baseline_returns=baseline_daily_returns,
            stg_returns=stg_daily_returns,
            pair_df_baseline=pair_df_baseline,
            pair_df_stg=pair_df_stg,
            graph_ts_baseline=graph_ts_baseline,
            graph_ts_stg=graph_ts_stg,
        )

    return summary
