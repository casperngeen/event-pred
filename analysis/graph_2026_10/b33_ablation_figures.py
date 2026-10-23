"""Figures for the report from the ablation table (out/b33_ablation_table.csv, b33_ablation_table.py):
    out/b33_fig_delay.png       forecast skill and trading profit against the feed delay, per contract type, with the
                                live feeds' delays marked
    out/b33_fig_components.png  each removed part / input: change in R^2 and the paired per-game t against the full model

    venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CONTRACTS = ["MLB spread", "MLB total", "NBA spread", "NBA total", "NFL spread", "NFL total"]
LIVE = {"MLB": (7, 15), "NBA": (45, 45), "NFL": (43, 43)}      # measured live delays (s): MLB cache-busted to default


def delay_of(version: str) -> float | None:
    if version == "full model (frozen)":
        return 10.0
    if version.startswith("delay "):
        return float(version.split()[1])
    return None


def main() -> None:
    df = pl.read_csv(OUT / "b33_ablation_table.csv").with_columns(
        (pl.col("contracts") * pl.col("net_c") / 100).alias("usd"))
    # --- delay
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharex="col")
    for j, lg in enumerate(["MLB", "NBA", "NFL"]):
        for kind, mk in (("spread", "o-"), ("total", "s--")):
            d = (df.filter(pl.col("contract") == f"{lg} {kind}")
                 .with_columns(pl.col("version").map_elements(delay_of, return_dtype=pl.Float64).alias("delay"))
                 .filter(pl.col("delay").is_not_null()).sort("delay"))
            ax[0, j].plot(d["delay"], d["r2_120s"], mk, label=kind)
            ax[1, j].plot(d["delay"], d["usd"], mk, label=kind)
        for i in range(2):
            lo, hi = LIVE[lg]
            ax[i, j].axvspan(lo - 0.5, hi + 0.5, color="tab:red", alpha=0.15, lw=0)
            ax[i, j].axvline(10, color="grey", lw=0.8, ls=":")
            ax[i, j].grid(alpha=0.3)
        ax[1, j].axhline(0, color="black", lw=0.8)
        ax[0, j].set_title(f"{lg}  (red: live feed delay; dotted: frozen model, 10 s)", fontsize=9)
        ax[1, j].set_xlabel("feed delay the model is trained and tested at (s)")
    ax[0, 0].set_ylabel("R² of the 120-s price change (%)")
    ax[1, 0].set_ylabel("test-fold profit after fees ($)")
    ax[0, 0].legend()
    fig.suptitle("In-game model: forecast skill and profit against how late the plays arrive (test folds, 5 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "b33_fig_delay.png", dpi=160)
    # --- components and inputs
    rows = ["no graph (no message passing)", "no play memory", "neither (step and contract features only)",
            "no play types", "no win probability (and its baseline)", "no structural baseline",
            "no recent trades / flow", "no quotes"]
    full = {r["contract"]: r for r in df.filter(pl.col("version") == "full model (frozen)").iter_rows(named=True)}
    dr2 = np.full((len(rows), len(CONTRACTS)), np.nan)
    tt = np.full_like(dr2, np.nan)
    for r in df.filter(pl.col("version").is_in(rows)).iter_rows(named=True):
        i, j = rows.index(r["version"]), CONTRACTS.index(r["contract"])
        dr2[i, j] = r["r2_120s"] - full[r["contract"]]["r2_120s"]
        tt[i, j] = r["vs_full_t"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for a, m, title, lim, fmt in ((ax[0], dr2, "change in R² (percentage points)", 6, "{:+.1f}"),
                                  (ax[1], tt, "profit vs full model: paired per-game t", 6, "{:+.1f}")):
        im = a.imshow(m, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                a.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=8,
                       fontweight="bold" if abs(m[i, j]) >= 2 and a is ax[1] else "normal")
        a.set_xticks(range(len(CONTRACTS)), CONTRACTS, rotation=30, ha="right", fontsize=8)
        a.set_yticks(range(len(rows)), rows if a is ax[0] else [""] * len(rows), fontsize=8)
        a.set_title(title, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.04)
    fig.suptitle("What each part of the in-game model contributes (red: removing it hurts; bold: |t| >= 2)")
    fig.tight_layout()
    fig.savefig(OUT / "b33_fig_components.png", dpi=160)
    print("wrote", OUT / "b33_fig_delay.png", OUT / "b33_fig_components.png")


if __name__ == "__main__":
    main()
