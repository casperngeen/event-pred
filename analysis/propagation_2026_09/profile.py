#!/usr/bin/env python
"""How long does a trigger's surprise stay unpriced in its target? No controls.

    venv/bin/python analysis/propagation_2026_09/profile.py
    (needs build_panel.py to have written out/depth_legs.parquet)

The grid
--------
* **horizon** -- days from the trigger's resolution to the target event's
  settlement, banded in months. ``0-1m`` is roughly the leadlag panel's
  next-event row; ``2-3m`` and beyond is the "A reaches C in months" case.
* **delay d** -- entry at the first print after ``t_res + d`` days.

In each (group, horizon, delay) cell the statistic is ``relations.py``'s
**aligned residual**, ``mean(sign(signal) * (win - p_entry))`` in pp: when the
relation says the YES leg is underpriced *at the price you actually pay*, how
underpriced was it. Anything the market absorbed before entry is already in
``p_entry``, so a positive cell means A's information was still unpriced at
that delay, in that horizon.

What slow propagation looks like: aligned residual that is still positive at
long horizons, and that survives delay. What fast, efficient incorporation
looks like: signal at ``0-1m, d = 0`` at most, and nothing anywhere else.

Inference is ``relations.py``'s, unchanged, so the numbers are comparable:
clustered bootstrap on ``target_event``; block permutation that shuffles
``z_surprise`` among trigger events within each trigger series; BH-FDR at
q = 0.10 across every reported cell of a table. Cells overlap -- one entry is
often the same across delays when the leg had not traded yet -- so the cells
are not independent tests and BH is only a rough guard.

A second pass restricts to 10-75c legs, where ``leadlag_findings.md`` §2 found
all of the pooled signal's power. Reported, not pre-specified.

In-sample only. No controls: ``controls.py`` asks whether what survives here is
A's information or the target's own intervening prints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.structure.stats import benjamini_hochberg

OUT = Path("analysis/propagation_2026_09/out")
N_PERM = 2000
N_BOOT = 4000
MIN_EVENTS = 6
BANDS = [(0, 30, "0-1m"), (30, 60, "1-2m"), (60, 90, "2-3m"),
         (90, 120, "3-4m"), (120, 181, "4-6m")]
MID = (10.0, 75.0)


def cluster_boot(vals, groups, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s = np.bincount(inv, weights=vals, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load() -> pl.DataFrame:
    d = pl.read_parquet(OUT / "depth_legs.parquet")
    band = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, lab in reversed(BANDS):
        band = pl.when((pl.col("gap_days") >= lo) & (pl.col("gap_days") < hi)).then(pl.lit(lab)).otherwise(band)
    return d.with_columns(band.alias("horizon")).filter(pl.col("horizon").is_not_null())


def perm_signs(d: pl.DataFrame, seed: int = 0):
    """Observed signs and N_PERM block-permuted sign vectors for ``d``'s rows."""
    direction = d["direction"].to_numpy().astype(float)
    trig_ev = d["trigger_event"].to_numpy()
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    te_z = np.zeros(len(uniq_te))
    te_series = np.empty(len(uniq_te), dtype=object)
    first = {}
    for i, te in enumerate(trig_ev):
        first.setdefault(te, i)
    zs, sr = d["z_surprise"].to_numpy(), d["trigger"].to_numpy()
    for i, te in enumerate(uniq_te):
        te_z[i], te_series[i] = zs[first[te]], sr[first[te]]
    groups = [np.where(te_series == s)[0] for s in np.unique(te_series)]
    rng = np.random.default_rng(seed)
    perm = np.empty((N_PERM, d.height), dtype=np.int8)
    for b in range(N_PERM):
        zp = te_z.copy()
        for g in groups:
            zp[g] = te_z[rng.permutation(g)]
        perm[b] = np.sign(direction * zp[row_te]).astype(np.int8)
    return np.sign(direction * te_z[row_te]), perm


def cell_table(d: pl.DataFrame, obs_sign, perm, keys: list[str], label: str) -> pl.DataFrame:
    resid = d["win"].to_numpy() - d["p_entry"].to_numpy() / 100.0
    ev = d["target_event"].to_numpy()
    kd = d.select(keys).with_row_index("_r")
    cells = kd.group_by(keys, maintain_order=False).agg(pl.col("_r")).sort(keys)

    codes = np.empty(d.height, dtype=np.int64)
    for i, rows in enumerate(cells["_r"]):
        codes[np.asarray(rows)] = i
    k = cells.height
    cnt = np.bincount(codes, minlength=k).astype(float)
    obs = np.bincount(codes, weights=obs_sign * resid, minlength=k) / cnt
    ge = np.zeros(k)
    for b in range(N_PERM):
        ge += np.bincount(codes, weights=perm[b] * resid, minlength=k) / cnt >= obs - 1e-12
    pval = ge / N_PERM

    out = []
    for i in range(k):
        m = codes == i
        n_ev = len(np.unique(ev[m]))
        lo, hi = cluster_boot((obs_sign * resid)[m], ev[m])
        out.append(dict(**{kk: cells[kk][i] for kk in keys},
                        legs=int(m.sum()), tgt_ev=n_ev,
                        trig_ev=len(np.unique(d["trigger_event"].to_numpy()[m])),
                        deferred=float(d["deferred"].to_numpy()[m].mean()),
                        lag_med=float(np.median(d["entry_lag_d"].to_numpy()[m])),
                        aligned_pp=100 * obs[i], ci_lo=100 * lo, ci_hi=100 * hi,
                        perm_p=pval[i]))
    t = pl.DataFrame(out).filter(pl.col("tgt_ev") >= MIN_EVENTS)
    bh = benjamini_hochberg(t["perm_p"].to_numpy(), q=0.10) if t.height else []
    t = t.with_columns(pl.Series("bh", bh, dtype=pl.Boolean))
    print(f"\n=== {label} ===")
    print(f"cells with >= {MIN_EVENTS} target events: {t.height};  BH-FDR q=0.10 within "
          f"this table: {int(t['bh'].sum())} survive.  lag_med = median days from the "
          f"cut to the entry print.")
    with pl.Config(tbl_rows=200, tbl_cols=-1, float_precision=2, tbl_width_chars=260):
        print(t)
    return t


def main() -> None:
    d = load()
    obs_sign, perm = perm_signs(d)
    print(f"rows {d.height}   trigger events {d['trigger_event'].n_unique()}   "
          f"target events {d['target_event'].n_unique()}   pairs {d['pair'].n_unique()}")
    print("aligned_pp > 0: the relation's sign was right about the YES leg at the "
          "price paid, i.e. A's information was not yet priced at entry.")

    pooled = cell_table(d, obs_sign, perm, ["horizon", "delay_d"],
                        "POOLED over all pairs: horizon x delay")
    by_src = cell_table(d, obs_sign, perm, ["source", "horizon", "delay_d"],
                        "by source (identified vs theory)")
    by_grp = cell_table(d, obs_sign, perm, ["group", "horizon", "delay_d"],
                        "by group: horizon x delay")

    mid = ((d["p_entry"] >= MID[0]) & (d["p_entry"] < MID[1])).to_numpy()
    dm = d.filter(pl.Series(mid))
    grp_mid = cell_table(dm, obs_sign[mid], perm[:, mid], ["group", "horizon", "delay_d"],
                         f"by group, {MID[0]:.0f}-{MID[1]:.0f}c legs only (not pre-specified)")

    # compact pivot of the by-group table: aligned_pp, * = BH survivor
    def pivot(t: pl.DataFrame, label: str) -> None:
        t = t.with_columns(pl.format("{}{}", pl.col("aligned_pp").round(1),
                                     pl.when(pl.col("bh")).then(pl.lit("*"))
                                     .when(pl.col("perm_p") < 0.05).then(pl.lit("+"))
                                     .otherwise(pl.lit(""))).alias("v"))
        pv = t.pivot(on="delay_d", index=["group", "horizon"], values="v",
                     sort_columns=True).sort("group", "horizon")
        pv = pv.rename({c: f"d={c}" for c in pv.columns if c not in ("group", "horizon")})
        print(f"\n--- {label}: aligned_pp by delay (* BH, + perm p<0.05) ---")
        with pl.Config(tbl_rows=60, tbl_width_chars=200):
            print(pv)

    pivot(by_grp, "all legs")
    pivot(grp_mid, f"{MID[0]:.0f}-{MID[1]:.0f}c legs")

    for nm, t in (("pooled", pooled), ("source", by_src), ("group", by_grp), ("group_mid", grp_mid)):
        t.write_parquet(OUT / f"profile_{nm}.parquet")
    print(f"\nwrote {OUT}/profile_*.parquet")


if __name__ == "__main__":
    main()
