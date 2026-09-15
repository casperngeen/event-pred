#!/usr/bin/env python
"""Which specific trigger->target relations carry the settlement signal, and
does *learning* them beat *imposing* them?

    venv/bin/python analysis/leadlag_2026_09/relations.py
    (needs build_panel.py to have written out/leadlag_legs.parquet)

Everything so far pooled 135 ordered pairs under one economically imposed sign
(``HAWKISH[trigger] * HAWKISH[target]``, zero fitted parameters) and reported a
single number. Two questions that leaves open:

1. **Is the pooled result carried by a few sensible relations, or is it spread
   thin?** A pooled effect assembled from cells that individually make no
   economic sense is a different object from one driven by, say, CPI->FED.
2. **Would learning the relation per pair do better than imposing it?** This is
   ``relations_study_plan`` §3.4's Stage-1-versus-block-model question, moved
   onto a settlement target. Stage 1 spends one parameter per ordered pair --
   135 of them here -- and the block model spends none.

The statistic
-------------
Per cell, the **aligned residual**::

    mean over legs of  sign(signal) * (win - p_entry)

in percentage points. It reads directly: when the relation says the YES leg is
underpriced, how underpriced was it? Zero parameters, no tercile cut, no
fitting. Positive means the relation works in the direction economics predicts.

The null is the same block permutation used throughout: ``z_surprise`` shuffled
among trigger events **within each trigger series**, which preserves the
ladder, the outcomes, the prices and the CPI/CPIYOY dependence, and destroys
only the trigger->target pairing. Multiplicity is handled with BH-FDR at
q = 0.10, and the search size is reported, because 135 cells is a large search
and ``relations_findings.md`` item 1b already warned the BH gate is coarse.

The honest comparison
---------------------
Walk-forward. For each year, estimate whatever the variant needs on **prior
years only**, then score the held-out year:

* ``imposed``    -- the HAWKISH sign. Zero parameters.
* ``learn_pair`` -- the sign of the aligned residual per (trigger, target),
                    fitted on prior years. 135 parameters.
* ``learn_chan`` -- the same, per channel (type -> type). 15 parameters.
* ``flip``       -- the HAWKISH sign, reversed. A control: if this scores as
                    well as ``imposed``, the sign is not doing any work.

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.structure.stats import benjamini_hochberg

OUT = Path("analysis/leadlag_2026_09/out")
N_PERM = 4000
N_BOOT = 10000
MIN_EVENTS = 6          # target events a cell needs before it is reported


def cluster_boot(vals, groups, seed=0):
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    s = np.bincount(inv, weights=vals, minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(N_BOOT, k))
    boot = s[pick].sum(axis=1) / np.maximum(c[pick].sum(axis=1), 1e-9)
    return (float(vals.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), float((boot <= 0).mean()))


def group_means(vals, codes, k):
    s = np.bincount(codes, weights=vals, minlength=k)
    c = np.bincount(codes, minlength=k).astype(float)
    return s / np.maximum(c, 1e-9)


def main() -> None:
    d = pl.read_parquet(OUT / "leadlag_legs.parquet")
    resid = d["win"].to_numpy().astype(float) - d["p_entry"].to_numpy() / 100.0
    direction = d["direction"].to_numpy().astype(float)
    ev = d["target_event"].to_numpy()
    yr = d["yr"].to_numpy()

    z_raw = d["z_surprise"].to_numpy()
    lim = float(np.percentile(np.abs(z_raw), 99))
    z = np.clip(z_raw, -lim, lim)
    sig = direction * z

    # ---- block-permutation machinery (shared by every test below)
    trig_ev = d["trigger_event"].to_numpy()
    trig_sr = d["trigger"].to_numpy()
    uniq_te, row_te = np.unique(trig_ev, return_inverse=True)
    te_z = np.zeros(len(uniq_te))
    te_series = np.empty(len(uniq_te), dtype=object)
    for i, te in enumerate(uniq_te):
        j = int(np.argmax(trig_ev == te))
        te_z[i] = z[j]
        te_series[i] = trig_sr[j]
    series_groups = [np.where(te_series == s)[0] for s in np.unique(te_series)]

    rng = np.random.default_rng(0)
    perm_sign = np.empty((N_PERM, len(resid)), dtype=np.int8)
    for b in range(N_PERM):
        zp = te_z.copy()
        for g in series_groups:
            zp[g] = te_z[rng.permutation(g)]
        perm_sign[b] = np.sign(direction * zp[row_te]).astype(np.int8)
    obs_sign = np.sign(sig)

    def cell_table(keys: np.ndarray, label: str) -> pl.DataFrame:
        uniq_k, codes = np.unique(keys, return_inverse=True)
        k = len(uniq_k)
        obs = group_means(obs_sign * resid, codes, k)
        null = np.empty((N_PERM, k))
        for b in range(N_PERM):
            null[b] = group_means(perm_sign[b] * resid, codes, k)
        pvals = (null >= obs[None, :]).mean(axis=0)

        n_legs = np.bincount(codes, minlength=k)
        n_ev = np.array([len(np.unique(ev[codes == i])) for i in range(k)])
        n_trig = np.array([len(np.unique(trig_ev[codes == i])) for i in range(k)])
        keep = n_ev >= MIN_EVENTS
        sel = np.where(keep)[0]
        bh = np.zeros(k, dtype=bool)
        if len(sel):
            bh[sel] = benjamini_hochberg(pvals[sel], q=0.10)

        ci = []
        for i in range(k):
            m = codes == i
            _, lo, hi, _ = cluster_boot((obs_sign * resid)[m], ev[m])
            ci.append((lo, hi))
        out = pl.DataFrame(dict(
            cell=[str(x) for x in uniq_k], legs=n_legs, tgt_events=n_ev,
            trig_events=n_trig, aligned_pp=100 * obs,
            ci_lo=[100 * c[0] for c in ci], ci_hi=[100 * c[1] for c in ci],
            perm_p=pvals, bh_survivor=bh, reported=keep,
        )).filter(pl.col("reported")).drop("reported").sort("aligned_pp", descending=True)
        print(f"\n=== {label} ===")
        print(f"cells tested (>= {MIN_EVENTS} target events): {int(keep.sum())} "
              f"of {k};  BH-FDR q = 0.10;  aligned_pp > 0 means the relation "
              f"works in the predicted direction")
        with pl.Config(tbl_rows=60, float_precision=2, tbl_width_chars=230):
            print(out)
        return out

    pair_key = np.array([f"{a}->{b}" for a, b in
                         zip(d["trigger"].to_numpy(), d["target"].to_numpy())])
    chan_key = d["channel"].to_numpy()

    print(f"rows {len(resid)}   pairs {len(np.unique(pair_key))}   "
          f"channels {len(np.unique(chan_key))}   target events {len(np.unique(ev))}")
    print(f"pooled aligned residual: {100 * float((obs_sign * resid).mean()):+.2f}pp")

    chan = cell_table(chan_key, "by channel (type -> type)")
    pair = cell_table(pair_key, "by ordered pair")

    print("\nBH survivors:")
    for nm, t in (("channel", chan), ("pair", pair)):
        s = t.filter(pl.col("bh_survivor"))
        print(f"  {nm}: {s.height} of {t.height}"
              + (f"  -> {', '.join(s['cell'].to_list())}" if s.height else ""))

    # ------------------------------------------------------------- learned
    print("\n\n=== does LEARNING the relation beat IMPOSING it? ===")
    print("Walk-forward: the sign is estimated on prior years only, then")
    print("scored on the held-out year. `imposed` fits nothing; `learn_pair`")
    print("fits 135 signs; `learn_chan` fits 15; `flip` is the reversed")
    print("imposed sign, a control that should lose.\n")

    years = sorted(np.unique(yr))
    uniq_p, code_p = np.unique(pair_key, return_inverse=True)
    uniq_c, code_c = np.unique(chan_key, return_inverse=True)

    variants: dict[str, np.ndarray] = {}
    for name in ("imposed", "flip", "learn_pair", "learn_chan"):
        scored = np.full(len(resid), np.nan)
        for Y in years[1:]:
            tr, te = yr < Y, yr == Y
            if te.sum() == 0 or tr.sum() < 200:
                continue
            if name == "imposed":
                s_te = obs_sign[te]
            elif name == "flip":
                s_te = -obs_sign[te]
            else:
                codes = code_p if name == "learn_pair" else code_c
                k = len(uniq_p) if name == "learn_pair" else len(uniq_c)
                # sign of the aligned residual on training years; a cell with
                # no training data falls back to the imposed sign rather than
                # to zero, so the variants are scored on the same legs.
                fit = np.zeros(k)
                seen = np.zeros(k, dtype=bool)
                tr_codes = codes[tr]
                m = group_means((obs_sign * resid)[tr], tr_codes, k)
                cnt = np.bincount(tr_codes, minlength=k)
                fit[cnt > 0] = np.sign(m[cnt > 0])
                seen[cnt > 0] = True
                learned = np.where(seen[codes[te]] & (fit[codes[te]] != 0),
                                   fit[codes[te]], 1.0)
                s_te = obs_sign[te] * learned
            scored[te] = s_te * resid[te]
        variants[name] = scored

    rows = []
    ok_all = ~np.isnan(variants["imposed"])
    for name, v in variants.items():
        ok = ~np.isnan(v)
        assert (ok == ok_all).all()
        obs, lo, hi, pneg = cluster_boot(v[ok], ev[ok])
        rows.append(dict(variant=name, params={"imposed": 0, "flip": 0,
                                               "learn_pair": len(uniq_p),
                                               "learn_chan": len(uniq_c)}[name],
                         n_scored=int(ok.sum()),
                         n_ev=int(len(np.unique(ev[ok]))),
                         aligned_pp=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi,
                         p_le0=pneg))
    with pl.Config(float_precision=2, tbl_width_chars=210):
        print(pl.DataFrame(rows))

    print("\nby year (aligned residual, pp):")
    rows = []
    for Y in years[1:]:
        m = (yr == Y) & ok_all
        if m.sum() < 100:
            continue
        r = dict(year=int(Y), n=int(m.sum()))
        for name, v in variants.items():
            r[name] = 100 * float(v[m].mean())
        rows.append(r)
    with pl.Config(float_precision=2, tbl_width_chars=200):
        print(pl.DataFrame(rows))

    pair.write_parquet(OUT / "relations_pair.parquet")
    chan.write_parquet(OUT / "relations_channel.parquet")
    print(f"\nwrote {OUT / 'relations_pair.parquet'}, "
          f"{OUT / 'relations_channel.parquet'}")


if __name__ == "__main__":
    main()
