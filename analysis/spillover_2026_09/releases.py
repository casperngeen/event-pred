#!/usr/bin/env python
"""More scheduled shocks: releases with a consensus but no Kalshi market.

    venv/bin/python -W ignore analysis/spillover_2026_09/releases.py > analysis/spillover_2026_09/out/releases.txt

The release studies use the ~10 macro series Kalshi trades (154 release
instants). The economic calendar (lum.id findata, cached by
``relations_2026_09/consensus_surprise.py`` at
``data/external/econ_calendar_us_2021q4_2025.parquet``; 2021-10 → 2025-12,
nothing from 2026) has actual and consensus for ~140 US releases. Those with
no Kalshi market of their own, a clear economic direction, and a release time
more than 1 h from any Kalshi-traded release become extra triggers:

  surprise   z = (actual − consensus) / sd of that release's (actual − consensus),
             clipped at ±3; HAWKISH = +1 if a higher value is hawkish, −1 for
             continuing claims
  targets    the 17 Kalshi threshold series' lead contracts (``_tape.py``),
             repricing from the release to +1 h / +4 h / +24 h (a print after the
             release is required)
  signed     s = sign(z)·HAWKISH[trigger]·HAWKISH[target]; response = s·ΔP

Null: flip each release's surprise sign at random (2,000 times), keeping every
release's size, timing and target set; p = share of flips with a mean signed
response ≥ the real one. Reported pooled, by trigger family and target type,
and by trigger (BH-FDR over triggers, q = 0.10).
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")
from stg.structure.stats import benjamini_hochberg

sys.path.insert(0, "analysis/spillover_2026_09")
from _tape import H, HAWKISH, LEGS, REL, TYPE, lead_contract, response  # noqa: E402

OUT = "analysis/spillover_2026_09/out"
CAL = "data/external/econ_calendar_us_2021q4_2025.parquet"
N_PERM = 2000
rng = np.random.default_rng(0)

from calendar_triggers import TRIGGERS  # noqa: E402

TARGET_TYPE = {s: ("policy" if s == "FED" else TYPE[s]) for s in LEGS}

cal = (pl.read_parquet(CAL)
       .filter((pl.col("country") == "US") & pl.col("estimate").is_not_null() & pl.col("actual").is_not_null())
       .with_columns(pl.col("event").str.replace(r"\s*\([^)]*\)\s*$", "").str.strip_chars().alias("ev"),
                     pl.col("date").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
                     .cast(pl.Datetime("us")).alias("t"))
       .filter(pl.col("ev").is_in(list(TRIGGERS)))
       .unique(["ev", "t"], keep="first")
       .with_columns((pl.col("actual") - pl.col("estimate")).alias("surp")))
cal = cal.with_columns((pl.col("surp") / pl.col("surp").std().over("ev")).clip(-3, 3).alias("z")) \
         .filter(pl.col("z").is_finite() & (pl.col("z") != 0))


def near_kalshi_release(t):
    i = np.searchsorted(REL, t)
    near = [abs(t - REL[j]) for j in (i - 1, i) if 0 <= j < len(REL)]
    return min(near) <= H


keep = [not near_kalshi_release(np.datetime64(t, "us")) for t in cal["t"]]
n0 = cal.height
cal = cal.filter(pl.Series(keep)).sort("t")
print(f"calendar releases of the {len(TRIGGERS)} chosen triggers with a consensus: {n0}; "
      f"{cal.height} are > 1 h from any Kalshi-traded release and kept; "
      f"{cal['t'].n_unique()} distinct release instants (vs 154 Kalshi release instants)")

rows = []
for r in cal.iter_rows(named=True):
    t = np.datetime64(r["t"], "us")
    fam, hk = TRIGGERS[r["ev"]]
    for B in LEGS:
        tb = lead_contract(B, t)
        if tb is None:
            continue
        _, after = response(tb, t, t)
        rows.append(dict(ev=r["ev"], family=fam, t=r["t"], z=r["z"], rel=f"{r['ev']}@{r['t']}",
                         target=B, ttype=TARGET_TYPE[B], hs=hk * HAWKISH[B],
                         **{h: v for h, v in after.items()}))
R = pl.DataFrame(rows)
R.write_parquet(f"{OUT}/releases.parquet")


def test(d, h):
    """Mean theory-signed response and a sign-flip permutation p (flip per release)."""
    d = d.filter(pl.col(h).is_finite())
    if d.height < 20:
        return None
    rel = d["rel"].to_numpy()
    u, inv = np.unique(rel, return_inverse=True)
    base = (d["hs"] * d[h]).to_numpy()                     # response signed by theory, before sign(z)
    sz = np.sign(d["z"].to_numpy())
    obs = (sz * base).mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(u)))
    null = (flips[:, inv] * sz[None, :] * base[None, :]).mean(1)
    nz = (sz * base)[base != 0]
    return dict(n=d.height, rel=len(u), mean=obs, p=(1 + (null >= obs).sum()) / (1 + N_PERM),
                agree=float((nz > 0).mean()) if len(nz) else np.nan)


def show(lab, c):
    print(f"{lab:44} {c['n']:>6} {c['rel']:>5} {c['mean']:>+8.3f} {c['p']:>7.3f} {c['agree']:>6.3f}")


hdr = f"{'':44} {'n':>6} {'rel.':>5} {'mean ¢':>8} {'p':>7} {'agree':>6}"
for h in ("+1h", "+4h", "+24h"):
    print(f"\n{'=' * 84}\nhorizon {h}: theory-signed response of Kalshi targets (¢); p from sign flips\n{'=' * 84}")
    print(hdr)
    c = test(R, h)
    if c:
        show("ALL extra triggers × all targets", c)
    for fam in ("inflation", "activity", "labour", "sentiment"):
        for tt in ("inflation", "labour", "growth", "policy"):
            c = test(R.filter((pl.col("family") == fam) & (pl.col("ttype") == tt)), h)
            if c:
                show(f"  {fam} release → {tt} targets", c)

print(f"\n{'=' * 84}\nBY TRIGGER, +4h, all targets; BH-FDR over triggers (q = 0.10)\n{'=' * 84}")
per = []
for ev in sorted(TRIGGERS):
    c = test(R.filter(pl.col("ev") == ev), "+4h")
    if c:
        per.append(dict(ev=ev, **c))
P = pl.DataFrame(per).sort("p")
P = P.with_columns(pl.Series("bh", benjamini_hochberg(P["p"].to_numpy(), 0.10)))
print(hdr + "  BH")
for r in P.iter_rows(named=True):
    print(f"{r['ev']:44} {r['n']:>6} {r['rel']:>5} {r['mean']:>+8.3f} {r['p']:>7.3f} {r['agree']:>6.3f}"
          f"  {'yes' if r['bh'] else ''}")
print(f"{P.height} triggers tested, {int(P['bh'].sum())} survive BH at q = 0.10")
