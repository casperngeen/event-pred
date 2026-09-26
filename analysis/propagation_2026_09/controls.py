#!/usr/bin/env python
"""Is what survives in ``profile.py`` A's information, or news that came later?

    venv/bin/python analysis/propagation_2026_09/controls.py
    (needs build_panel.py to have written out/depth_legs.parquet)

A payrolls surprise and a Fed meeting three months out have a lot in between:
two more Fed meetings, two more payrolls prints. ``profile.py``'s aligned
residual at long horizons could be slow propagation of A -- or it could be:

* **own_pre** -- the target's own latest surprise before entry. Public at
  entry, so if the market underreacts to the target's *own* prints and A
  happens to correlate with them, A gets credit for C's momentum.
* **own_mid** -- the target's own surprises between entry and settlement (the
  intervening Fed meetings, say). Not known at entry. Controlling for it asks
  whether A predicts C_k *only through* C's intermediate prints -- the
  mediation question. If A's coefficient survives it, A carries information
  about C_k beyond what C's next prints reveal.
* **trig_mid** -- the trigger series' own *next* surprises in the same window,
  aligned by the relation's sign. If payrolls surprises are persistent, the
  next payrolls print moves the Fed path the same way and the original
  surprise looks like it propagated slowly when really it recurred.

Each control is the sum of the clipped (|z| <= 3) ``z_surprise`` values of the
releases in its window, zero when there are none; the target's own sign needs
no alignment (a higher C print raises every C leg's YES probability).

Per cell, OLS with an intercept::

    resid = a + b * sign(signal) + c . controls

``b`` is reported four ways -- no controls, + own_pre, + own_mid, + trig_mid --
each adding to the last. ``b`` with no controls differs from ``profile.py``'s
aligned residual only by the intercept. Clustered bootstrap on
``target_event`` for the CI; block-permutation p on ``b`` with the controls
held fixed (Frisch-Waugh: permute A's sign, project out the controls).

Only uses the target's and trigger's surprise-panel rows, so a target with no
surprise panel (ADP has 8 rows) is controlled only as far as the panel goes;
``cov_*`` columns report the share of rows with a non-empty window.

In-sample only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.splits import assert_no_oos
from stg.structure.stats import benjamini_hochberg

OUT = Path("analysis/propagation_2026_09/out")
PANELS = Path("artifacts/panels")
N_PERM = 2000
N_BOOT = 2000
MIN_EVENTS = 6
Z_CLIP = 3.0
BANDS = [(0, 30, "0-1m"), (30, 60, "1-2m"), (60, 90, "2-3m"),
         (90, 120, "3-4m"), (120, 181, "4-6m")]
MID = (10.0, 75.0)
SPECS = [("b0", []), ("b_pre", ["own_pre"]), ("b_mid", ["own_pre", "own_mid"]),
         ("b_all", ["own_pre", "own_mid", "trig_mid"])]


def release_index(sp: pl.DataFrame) -> dict:
    """series -> (sorted close epoch-ns, clipped z)."""
    out = {}
    for s in sp["series"].unique().to_list():
        g = sp.filter(pl.col("series") == s).sort("close_time")
        out[s] = (g["close_time"].dt.epoch("ns").to_numpy(),
                  np.clip(g["z_surprise"].to_numpy(), -Z_CLIP, Z_CLIP))
    return out


def window_sum(idx, series, lo, hi, lo_open=False):
    """Sum of z over releases of ``series`` with close in [lo, hi) (or (lo, hi))."""
    if series not in idx:
        return 0.0, False
    ns, z = idx[series]
    i = np.searchsorted(ns, lo, side="right" if lo_open else "left")
    j = np.searchsorted(ns, hi, side="left")
    return (float(z[i:j].sum()), j > i) if j > i else (0.0, False)


def add_controls(d: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    idx = release_index(sp)
    hour = 3_600 * 1_000_000_000
    t_entry = d["t_entry"].dt.epoch("ns").to_numpy()
    close = d["close_time"].dt.epoch("ns").to_numpy()
    tgt, trg = d["target"].to_numpy(), d["trigger"].to_numpy()
    direction = d["direction"].to_numpy().astype(float)

    pre, mid, tm = np.zeros(d.height), np.zeros(d.height), np.zeros(d.height)
    c_pre, c_mid, c_tm = (np.zeros(d.height, dtype=bool) for _ in range(3))
    for i in range(d.height):
        # own_pre: target's latest release strictly before entry
        if tgt[i] in idx:
            ns, z = idx[tgt[i]]
            j = np.searchsorted(ns, t_entry[i], side="left")
            if j > 0:
                pre[i], c_pre[i] = z[j - 1], True
        # own_mid: target releases in [entry, settlement), excluding the target
        # event itself (it closes at `close`; one hour of slack for ties)
        mid[i], c_mid[i] = window_sum(idx, tgt[i], t_entry[i], close[i] - hour)
        # trig_mid: the trigger series' next releases, same window, aligned
        s, c_tm[i] = window_sum(idx, trg[i], t_entry[i], close[i] - hour, lo_open=True)
        tm[i] = direction[i] * s
    return d.with_columns(pl.Series("own_pre", pre), pl.Series("own_mid", mid),
                          pl.Series("trig_mid", tm), pl.Series("cov_pre", c_pre),
                          pl.Series("cov_mid", c_mid), pl.Series("cov_trig", c_tm))


def perm_signs(d: pl.DataFrame, seed: int = 0):
    """``profile.py``'s block permutation: z shuffled within trigger series."""
    direction = d["direction"].to_numpy().astype(float)
    trig_ev = d["trigger_event"].to_numpy()
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    first = {}
    for i, te in enumerate(trig_ev):
        first.setdefault(te, i)
    zs, sr = d["z_surprise"].to_numpy(), d["trigger"].to_numpy()
    te_z = np.array([zs[first[te]] for te in uniq_te])
    te_series = np.array([sr[first[te]] for te in uniq_te], dtype=object)
    groups = [np.where(te_series == s)[0] for s in np.unique(te_series)]
    rng = np.random.default_rng(seed)
    perm = np.empty((N_PERM, d.height), dtype=np.int8)
    for b in range(N_PERM):
        zp = te_z.copy()
        for g in groups:
            zp[g] = te_z[rng.permutation(g)]
        perm[b] = np.sign(direction * zp[row_te]).astype(np.int8)
    return np.sign(direction * te_z[row_te]), perm


def fit_b(y, s, C, ev, perm_s, seed=0):
    """OLS b on s with intercept + controls C; cluster-bootstrap CI; perm p."""
    n = len(y)
    if C.size:                        # a control with an empty window everywhere
        C = C[:, C.std(axis=0) > 0]   # in this cell carries nothing; drop it
    X = np.column_stack([np.ones(n), s] + ([C] if C.size else []))
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        return np.nan, np.nan, np.nan, np.nan
    b = np.linalg.solve(XtX, X.T @ y)[1]

    # perm p via Frisch-Waugh: residualise y and each permuted s on [1, C]
    Z = np.column_stack([np.ones(n)] + ([C] if C.size else []))
    P = np.linalg.pinv(Z)
    y_r = y - Z @ (P @ y)
    S = perm_s.astype(float)
    S_r = S - (S @ P.T) @ Z.T
    b_null = (S_r @ y_r) / np.maximum((S_r * S_r).sum(axis=1), 1e-12)
    p = float((b_null >= b - 1e-12).mean())

    # cluster bootstrap on per-cluster sufficient statistics
    uniq, inv = np.unique(ev, return_inverse=True)
    k, q = len(uniq), X.shape[1]
    XtX_c = np.zeros((k, q, q))
    Xty_c = np.zeros((k, q))
    np.add.at(XtX_c, inv, X[:, :, None] * X[:, None, :])
    np.add.at(Xty_c, inv, X * y[:, None])
    rng = np.random.default_rng(seed)
    W = np.zeros((N_BOOT, k))
    pick = rng.integers(0, k, size=(N_BOOT, k))
    np.add.at(W, (np.repeat(np.arange(N_BOOT), k), pick.ravel()), 1.0)
    A = np.einsum("bk,kij->bij", W, XtX_c)
    r = W @ Xty_c
    ok = np.linalg.matrix_rank(A) == q
    bb = np.linalg.solve(A[ok], r[ok][..., None])[:, 1, 0]
    return float(b), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5)), p


def cell_table(d: pl.DataFrame, obs_sign, perm, label: str) -> pl.DataFrame:
    y = d["win"].to_numpy() - d["p_entry"].to_numpy() / 100.0
    ev = d["target_event"].to_numpy()
    keys = ["group", "horizon", "delay_d"]
    cells = d.select(keys).with_row_index("_r").group_by(keys).agg(pl.col("_r")).sort(keys)
    rows = []
    for c in cells.iter_rows(named=True):
        m = np.asarray(c["_r"])
        if len(np.unique(ev[m])) < MIN_EVENTS:
            continue
        r = {k: c[k] for k in keys}
        r.update(legs=len(m), tgt_ev=len(np.unique(ev[m])),
                 cov_mid=float(d["cov_mid"].to_numpy()[m].mean()),
                 cov_trig=float(d["cov_trig"].to_numpy()[m].mean()))
        for name, cols in SPECS:
            C = d.select(cols).to_numpy()[m] if cols else np.empty((len(m), 0))
            b, lo, hi, p = fit_b(y[m], obs_sign[m], C, ev[m], perm[:, m])
            r[name], r[f"{name}_p"] = 100 * b, p
            if name in ("b0", "b_all"):
                r[f"{name}_lo"], r[f"{name}_hi"] = 100 * lo, 100 * hi
        rows.append(r)
    t = pl.DataFrame(rows)
    for name, _ in SPECS:
        t = t.with_columns(pl.Series(f"{name}_bh", benjamini_hochberg(
            t[f"{name}_p"].fill_nan(1.0).to_numpy(), q=0.10), dtype=pl.Boolean))

    print(f"\n=== {label} ===")
    print(f"{t.height} cells;  b in pp;  BH-FDR q=0.10 within column.  "
          f"* BH survivor, + perm p < 0.05")

    def fmt(name):
        return pl.format("{}{}", pl.col(name).round(1),
                         pl.when(pl.col(f"{name}_bh")).then(pl.lit("*"))
                         .when(pl.col(f"{name}_p") < 0.05).then(pl.lit("+"))
                         .otherwise(pl.lit(""))).alias(name)
    show = t.select(*keys, "tgt_ev", "cov_mid", "cov_trig",
                    *[fmt(n) for n, _ in SPECS], "b_all_lo", "b_all_hi")
    with pl.Config(tbl_rows=200, tbl_cols=-1, float_precision=2, tbl_width_chars=260):
        print(show)
    return t


def main() -> None:
    sp = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(sp, time_col="close_time")
    sp = (sp.with_columns((pl.col("surprise") / pl.col("implied_std")).alias("z_surprise"))
          .filter(pl.col("z_surprise").is_finite()))

    d = pl.read_parquet(OUT / "depth_legs.parquet")
    band = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, lab in reversed(BANDS):
        band = (pl.when((pl.col("gap_days") >= lo) & (pl.col("gap_days") < hi))
                .then(pl.lit(lab)).otherwise(band))
    d = d.with_columns(band.alias("horizon")).filter(pl.col("horizon").is_not_null())
    d = add_controls(d, sp)

    print(f"rows {d.height}")
    print("control coverage (share of rows with a non-empty window), by target:")
    with pl.Config(tbl_rows=20, float_precision=2):
        print(d.group_by("target").agg(pl.col("cov_pre").mean(), pl.col("cov_mid").mean(),
                                       pl.col("cov_trig").mean(), pl.len()).sort("target"))
    print("\ncorrelation of sign(signal) with each control (all rows):")
    s = np.sign(d["signal"].to_numpy())
    for c in ("own_pre", "own_mid", "trig_mid"):
        print(f"  {c:<9} {np.corrcoef(s, d[c].to_numpy())[0, 1]:+.3f}")

    obs_sign, perm = perm_signs(d)
    t_all = cell_table(d, obs_sign, perm, "all legs")
    mid = ((d["p_entry"] >= MID[0]) & (d["p_entry"] < MID[1])).to_numpy()
    t_mid = cell_table(d.filter(pl.Series(mid)), obs_sign[mid], perm[:, mid],
                       f"{MID[0]:.0f}-{MID[1]:.0f}c legs only (not pre-specified)")

    d.write_parquet(OUT / "depth_legs_controls.parquet")
    t_all.write_parquet(OUT / "controls_group.parquet")
    t_mid.write_parquet(OUT / "controls_group_mid.parquet")
    print(f"\nwrote {OUT}/depth_legs_controls.parquet, controls_group*.parquet")


if __name__ == "__main__":
    main()
