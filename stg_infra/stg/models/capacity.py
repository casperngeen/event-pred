"""Capacity accounting for AGCRN — parameters vs. the supervision the panel
actually supplies.

Promotes ``analysis/exploratory_2026_08/agcrn_complexity.py`` T1 (parameter
count) and T2 (effective sample size). The point: at the Bai et al. (2020)
default configuration the model carries ~15 parameters per label and ~36 per
*effective* label once within-snapshot clustering is accounted for.
"""

from __future__ import annotations

import numpy as np
import polars as pl


K_SUPPORT = 2   # [I, A]


def agcrn_param_count(c_in: int, d_emb: int, hidden: int,
                      n_horizons: int = 1, mlp_hidden: int = 64,
                      embedding: str = "learned", n_nodes: int = 19) -> dict:
    """Exact parameter count for ``stg.models.agcrn.AGCRN``.

    Per AVWGCN(c, c'): weight pool ``d*K*c*c'`` + bias pool ``d*c'``.
    The AGCRN cell has a gate conv (c_in = F+H, c_out = 2H) and an update conv
    (c_in = F+H, c_out = H); plus a linear head and the node embedding source.
    """
    def avwgcn(ci, co):
        return d_emb * K_SUPPORT * ci * co + d_emb * co

    gate = avwgcn(c_in + hidden, 2 * hidden)
    update = avwgcn(c_in + hidden, hidden)
    head = hidden * n_horizons + n_horizons
    if embedding == "shared_mlp":
        emb = c_in * mlp_hidden + mlp_hidden + mlp_hidden * d_emb + d_emb
    else:  # learned E in R^{N x d}
        emb = n_nodes * d_emb
    total = gate + update + head + emb
    return dict(embedding=emb, gate=gate, update=update, head=head, total=total)


CONFIGS = {
    "minimal": dict(d_emb=2, hidden=16),
    "default": dict(d_emb=10, hidden=64),   # Bai et al. default
}


def within_snapshot_icc(label_panel: pl.DataFrame,
                        value_col: str = "y", group_col: str = "t_idx",
                        min_group: int = 3) -> dict:
    """ICC of the label within a snapshot, design effect, effective n.

    label_panel: long frame with one row per (snapshot, node) labelled cell.
    """
    g = (label_panel.group_by(group_col)
         .agg(pl.col(value_col).mean().alias("m"),
              pl.col(value_col).std().alias("s"),
              pl.len().alias("n"))
         .filter(pl.col("n") >= min_group))
    within = float(np.nanmean(g["s"].to_numpy() ** 2))
    between = float(np.nanvar(g["m"].to_numpy()))
    icc = between / (between + within) if (between + within) > 0 else 0.0
    nbar = float(g["n"].mean())
    deff = 1 + (nbar - 1) * icc
    n_total = label_panel.height
    return dict(icc=icc, nbar=nbar, design_effect=deff,
                n_nominal=n_total, n_effective=n_total / deff)


def capacity_table(label_panel: pl.DataFrame, c_in: int, n_horizons: int = 3,
                   n_nodes: int = 19) -> pl.DataFrame:
    eff = within_snapshot_icc(label_panel)
    n_lab = label_panel.height * n_horizons
    n_eff_lab = eff["n_effective"] * n_horizons
    rows = []
    for name, cfg in CONFIGS.items():
        for emb in ("learned", "shared_mlp"):
            p = agcrn_param_count(c_in, cfg["d_emb"], cfg["hidden"],
                                  n_horizons, embedding=emb, n_nodes=n_nodes)
            rows.append(dict(
                config=name, embedding=emb, d_emb=cfg["d_emb"],
                hidden=cfg["hidden"], params=p["total"],
                params_per_label=p["total"] / n_lab,
                params_per_eff_label=p["total"] / n_eff_lab,
            ))
    return pl.DataFrame(rows), eff
