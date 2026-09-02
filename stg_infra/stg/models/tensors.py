"""Node panel -> dense (T, N, F) tensors for the sequence models.

The node panel is one row per (series, snapshot-date). Here it becomes a dense
tensor padded over the *union* node set, with a boolean mask marking which
(t, node) cells are real (node active that snapshot and its ladder was deep
enough to recover a belief). Everything downstream — AGCRN, the linear
baselines, the walk-forward harness — consumes these arrays.

In-sample only: ``assert_no_oos`` runs on the snapshot dates at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from stg.nodes.kalshi import FEATURE_ORDER
from stg.splits import assert_no_oos

LabelKind = Literal["belief_z", "atm_cents"]

# features fed to the model, in fixed order (subset of the panel columns)
MODEL_FEATURES = list(FEATURE_ORDER)


@dataclass
class PanelTensors:
    X: np.ndarray            # (T, N, F) float32, nan->0
    mask: np.ndarray         # (T, N) bool  — node active & belief present
    dates: np.ndarray        # (T,) datetime64[D]
    nodes: list[str]         # length N, deterministic order
    implied_mean: np.ndarray  # (T, N) float — raw, for label construction
    atm_price: np.ndarray | None = None   # (T, N) — representative yes_price, cents

    @property
    def T(self) -> int: return self.X.shape[0]

    @property
    def N(self) -> int: return self.X.shape[1]

    @property
    def F(self) -> int: return self.X.shape[2]


def representative_price_panel(node_panel: pl.DataFrame) -> pl.DataFrame:
    """(series, date, atm_price) — the most-traded contract of each node's
    nearest event, last yes_price on/before the snapshot date.

    Used for the ``atm_cents`` robustness label. One targeted trades scan over
    ~1k representative tickers.
    """
    from stg.panel._io import load_markets, scan_trades
    from stg.panel.registry import series_filter_expr

    mk = load_markets(is_only=True)
    names = sorted(node_panel["series"].unique().to_list())
    sel = pl.any_horizontal([series_filter_expr(n) for n in names])
    universe_tickers = mk.filter(sel)["ticker"].unique().to_list()

    # one scan: daily last-price + per-ticker trade count for every universe ticker
    daily = (scan_trades(is_only=True)
             .filter(pl.col("ticker").is_in(universe_tickers))
             .select("ticker", "yes_price", "created_time")
             .with_columns(pl.col("created_time").dt.date().alias("d"))
             .sort("created_time")
             .group_by("ticker", "d")
             .agg(pl.col("yes_price").last().alias("px"),
                  pl.len().alias("n"))
             .collect())
    tc = dict(daily.group_by("ticker").agg(pl.col("n").sum())
              .iter_rows())
    # most-traded ticker per event
    ev_meta = (mk.filter(sel).select("ticker", "event_ticker", "series_raw")
               .with_columns(pl.col("ticker").map_elements(
                   lambda t: tc.get(t, 0), return_dtype=pl.Int64).alias("nt")))
    ev_to_ticker: dict[str, str] = {}
    for ev, g in ev_meta.filter(pl.col("nt") > 0).group_by("event_ticker"):
        ev_to_ticker[ev[0]] = g.sort("nt", descending=True)["ticker"][0]
    px = {(r["ticker"], r["d"]): r["px"] for r in daily.iter_rows(named=True)}
    hist: dict[str, list] = {}
    for (tk, d), p in sorted(px.items(), key=lambda x: x[0][1]):
        hist.setdefault(tk, []).append((d, p))

    rows = []
    for r in node_panel.iter_rows(named=True):
        tk = ev_to_ticker.get(r["event_ticker"])
        if tk is None:
            continue
        seq = [p for d, p in hist.get(tk, []) if d <= r["date"]]
        if seq:
            rows.append(dict(series=r["series"], date=r["date"], atm_price=float(seq[-1])))
    return pl.DataFrame(rows)


def build_tensor(node_panel: pl.DataFrame,
                 atm_price: pl.DataFrame | None = None) -> PanelTensors:
    """Dense tensors over the union node set.

    ``atm_price`` (optional): long frame with ``series, date, atm_price`` for the
    ``atm_cents`` label. See :func:`representative_price_panel`.
    """
    assert_no_oos(node_panel, time_col="date")
    nodes = sorted(node_panel["series"].unique().to_list())
    dates = np.array(sorted(node_panel["date"].unique().to_list()), dtype="datetime64[D]")
    ni = {n: i for i, n in enumerate(nodes)}
    di = {d: i for i, d in enumerate(dates.tolist())}
    T, N, Fn = len(dates), len(nodes), len(MODEL_FEATURES)

    X = np.zeros((T, N, Fn), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)
    im = np.full((T, N), np.nan, dtype=np.float64)

    have = [c for c in MODEL_FEATURES if c in node_panel.columns]
    for r in node_panel.iter_rows(named=True):
        t, i = di[r["date"]], ni[r["series"]]
        for f, col in enumerate(MODEL_FEATURES):
            v = r.get(col) if col in have else None
            X[t, i, f] = 0.0 if v is None else float(v)
        if r.get("implied_mean") is not None:
            im[t, i] = float(r["implied_mean"])
            mask[t, i] = True

    ap = None
    if atm_price is not None:
        ap = np.full((T, N), np.nan, dtype=np.float64)
        for r in atm_price.iter_rows(named=True):
            if r["date"] in di and r["series"] in ni:
                ap[di[r["date"]], ni[r["series"]]] = float(r["atm_price"])

    return PanelTensors(X=X, mask=mask, dates=dates, nodes=nodes,
                        implied_mean=im, atm_price=ap)


def build_labels(pt: PanelTensors, k: int, kind: LabelKind = "belief_z"):
    """(Y, label_mask) at horizon ``k`` snapshots.

    ``belief_z``  — raw Δ implied_mean (per-series z-scoring is done per fold in
                    the harness, not here).
    ``atm_cents`` — Δ of the representative contract's yes_price, in cents.

    ``label_mask[t, i]`` is True iff node i is real at both t and t+k.
    """
    T, N = pt.T, pt.N
    Y = np.full((T, N), np.nan, dtype=np.float64)
    lm = np.zeros((T, N), dtype=bool)
    src = pt.implied_mean if kind == "belief_z" else pt.atm_price
    if src is None:
        raise ValueError(f"label kind {kind!r} needs data not present in the tensor")
    ok = np.isfinite(src)
    for t in range(T - k):
        both = ok[t] & ok[t + k]
        Y[t, both] = src[t + k, both] - src[t, both]
        lm[t, both] = True
    return Y, lm


def sequence_windows(pt: PanelTensors, Y: np.ndarray, label_mask: np.ndarray,
                     L: int = 12):
    """Sliding windows.

    Returns a dict of arrays:
        Xs     (n, L, N, F)   input feature sequences
        Ms     (n, L, N)      input masks
        y      (n, N)         target at the window's last step
        ym     (n, N)         target mask
        t_idx  (n,)           snapshot index of the target (into pt.dates)
    """
    T = pt.T
    idx = [t for t in range(L - 1, T) if label_mask[t].any()]
    Xs = np.stack([pt.X[t - L + 1: t + 1] for t in idx]).astype(np.float32)
    Ms = np.stack([pt.mask[t - L + 1: t + 1] for t in idx])
    y = np.stack([Y[t] for t in idx])
    ym = np.stack([label_mask[t] for t in idx])
    return {"Xs": Xs, "Ms": Ms, "y": y, "ym": ym,
            "t_idx": np.array(idx), "dates": pt.dates[idx]}


# ---- per-fold standardisation -------------------------------------------
def fit_feature_scaler(Xs: np.ndarray, Ms: np.ndarray):
    """Per-(node, feature) mean/std over active cells of the training windows.

    Xs: (n, L, N, F), Ms: (n, L, N). Returns (mu, sd) each (N, F).
    """
    n, L, N, F = Xs.shape
    flat = Xs.reshape(-1, N, F)
    m = Ms.reshape(-1, N)
    mu = np.zeros((N, F))
    sd = np.ones((N, F))
    for i in range(N):
        sel = flat[m[:, i], i, :]
        if sel.shape[0] > 1:
            mu[i] = sel.mean(0)
            s = sel.std(0)
            sd[i] = np.where(s > 1e-8, s, 1.0)
    return mu, sd


def apply_feature_scaler(Xs: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((Xs - mu[None, None]) / sd[None, None]).astype(np.float32)


def fit_label_scaler(y: np.ndarray, ym: np.ndarray, kind: LabelKind,
                     fallback: np.ndarray | None = None):
    """Per-node label std (the difference has mean ~0). Identity for atm_cents.

    A per-series label scale is a *units* choice, not a learned parameter, so it
    is legitimately computed once over the whole IS panel and passed in as
    ``fallback`` — this avoids degenerate std=1 for nodes with <2 labelled rows
    in an early fold, which would leave PAYROLLS (Δ in thousands) unscaled.
    """
    N = y.shape[1]
    sd = np.ones(N) if fallback is None else np.asarray(fallback, float).copy()
    if kind == "belief_z":
        for i in range(N):
            v = y[ym[:, i], i]
            if v.size > 5:
                s = v.std()
                if s > 1e-9:
                    sd[i] = s
    return sd


def global_label_sd(Y: np.ndarray, label_mask: np.ndarray, kind: LabelKind) -> np.ndarray:
    """Per-node label std over every labelled cell (the units choice)."""
    N = Y.shape[1]
    sd = np.ones(N)
    if kind == "belief_z":
        for i in range(N):
            v = Y[label_mask[:, i], i]
            if v.size > 1 and v.std() > 1e-9:
                sd[i] = v.std()
    return sd


def apply_label_scaler(y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return y / sd[None]
