#!/usr/bin/env python
"""Does a jump in A predict the direction of B's next move, with no theory imposed?

    venv/bin/python -W ignore analysis/spillover_2026_09/predictive.py \
        > analysis/spillover_2026_09/out/predictive.txt      # needs jumps.py first

``jumps.py`` signs each response by theory (HAWKISH), which tests one
pre-specified direction. This asks the agnostic question instead: predict
whether B's lead contract rises or falls after A's jump, with the direction
*learned from that pair's earlier jumps only* (walk-forward), and score it on
the later ones. Nothing about the direction is imposed.

For each jump threshold and horizon, three predictors of sign(ΔB):
  learned, per pair      sign(ΔA) × sign of the mean of sign(ΔA)·ΔB over the
                         pair's earlier jumps (≥ 5 earlier non-zero responses)
  learned, per channel   the same, pooled over the type → type channel
                         (more data per estimate, coarser)
  theory                 sign(ΔA)·HAWKISH[A]·HAWKISH[B] (the jumps.py direction)
Scored on responses with ΔB ≠ 0: accuracy, and balanced accuracy (mean of the
hit rates on B-up and B-down cells, so B's own drift cannot score). CI:
bootstrap over source jumps. The whole procedure is repeated on the placebo
times, where every predictor should score 0.5.
Co-jumps and same-release pairs are excluded.
"""
from __future__ import annotations

import numpy as np
import polars as pl

OUT = "analysis/spillover_2026_09/out"
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1, "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1, "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1, "FED": +1,
}
HORIZONS = ["+1h", "+4h", "+24h", "+72h"]
MIN_HIST, N_BOOT = 5, 2000
rng = np.random.default_rng(0)

R = pl.read_parquet(f"{OUT}/jumps.parquet").filter(~pl.col("same_release"))
# undo the theory signing: raw ΔB and raw direction of A's jump
R = R.with_columns(
    pl.struct("A", "B").map_elements(lambda r: HAWKISH[r["A"]] * HAWKISH[r["B"]],
                                     return_dtype=pl.Int64).alias("hh"))
R = R.with_columns((pl.col("sign") * pl.col("hh")).alias("dA"))


def walk_forward(d: pl.DataFrame, h: str, key: list[str]) -> pl.DataFrame:
    """Per group in ``key``, predict sign(ΔB) from the group's earlier rows."""
    d = d.filter(pl.col(h).is_finite() & ~pl.col("cojump")).with_columns(
        (pl.col(h) * pl.col("sign")).alias("dB")).sort("t0")          # raw ΔB
    out = []
    for _, g in d.group_by(key, maintain_order=True):
        dA, dB, t = g["dA"].to_numpy(), g["dB"].to_numpy(), g["t0"].to_numpy()
        prod = np.sign(dA) * dB
        for i in range(len(g)):
            hist = prod[:i][(t[:i] < t[i]) & (dB[:i] != 0)]
            if len(hist) < MIN_HIST or dB[i] == 0 or hist.mean() == 0:
                continue
            out.append(dict(jump=g["jump"][i], pred=np.sign(dA[i]) * np.sign(hist.mean()),
                            theory=float(g["sign"][i]), up=dB[i] > 0))
    return pl.DataFrame(out) if out else pl.DataFrame()


def scores(p: pl.DataFrame, col: str) -> tuple:
    if p.is_empty():
        return (0, np.nan, np.nan, np.nan, np.nan)
    up = p["up"].cast(pl.Boolean).to_numpy().astype(bool)
    hit = (p[col].to_numpy() > 0) == up
    jumps = p["jump"].to_numpy()
    u, inv = np.unique(jumps, return_inverse=True)

    def bal(ix):
        hu, hd = hit[ix][up[ix]], hit[ix][~up[ix]]
        return np.mean([hu.mean() if len(hu) else np.nan, hd.mean() if len(hd) else np.nan])
    groups = [np.nonzero(inv == k)[0] for k in range(len(u))]
    bs = []
    for _ in range(N_BOOT):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(u), len(u))]).astype(int)
        bs.append(bal(ix))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return (len(hit), hit.mean(), bal(np.arange(len(hit))), lo, hi)


for thresh in sorted(R["thresh"].unique()):
    print("\n" + "#" * 100 + f"\nJUMP THRESHOLD {thresh:.0f}c\n" + "#" * 100)
    print(f"{'horizon':7} {'kind':8} {'predictor':22} {'n':>6} {'acc':>6} {'bal acc [95% CI]':>24}")
    for h in HORIZONS:
        for kind in ("real", "placebo"):
            d = R.filter((pl.col("thresh") == thresh) & (pl.col("kind") == kind))
            per_pair = walk_forward(d, h, ["A", "B"])
            per_chan = walk_forward(d, h, ["channel"])
            for name, p, col in (("learned, per pair", per_pair, "pred"),
                                 ("learned, per channel", per_chan, "pred"),
                                 ("theory", per_chan, "theory")):
                n, acc, b, lo, hi = scores(p, col)
                if n:
                    print(f"{h:7} {kind:8} {name:22} {n:>6} {acc:>6.3f} {b:>7.3f} [{lo:.3f}, {hi:.3f}]",
                          flush=True)
        print()
