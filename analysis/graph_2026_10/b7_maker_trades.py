"""Per-trade log of the signal-filtered maker strategy with the fixed exit (b6_maker_exit "limit"), for
robustness checks (b7_maker_robust.py). Priced only from trade prints.

Entry: every 60 s of game time, for each winner / spread / total contract with a print in the last
60 s (ref), only on the side the model favours by >= MIN_SIGNAL at the hold's horizon: buy limit at
ref - k or sell limit at ref + k, resting ENTRY_S seconds. Exit at fill + hold: rest at the last print
+- k, else take (the next taker on our exit side within EXIT_S), else settlement.

A fill needs a print beyond our price by more than `through` cents: through = 0 is the usual rule
(strictly through); through = 1 is stricter (at least 2c through), a check on queue-position luck.
Variants: (k, through) in VARIANTS.

    RUN_TAG=_h30-60-120-1800_tpf venv/bin/python -W ignore analysis/graph_2026_10/b7_maker_trades.py \
        results/ingame NBA 10 data/sports/derived/ingame/NBA/panel_v3f
    -> out/b7_maker_trades_<L>_d<delay><RUN_TAG>[_sa][_none][_seed<s>].parquet   (SIGNAL_FROM=anchor: signal from the model's anchor;
       MEMORY=none / SEED=1 pick the ablation or second-seed runs)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b5_longshort import LEAGUE, DELAY, MEMORY, PANEL, RUN_TAG, SEED, Panel, attach_tickers, rows   # noqa: E402
from b6_maker_exit import ENTRY_S, EXIT_S, REF_MAX_S, first_taker                    # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades                              # noqa: E402
from stg.splits import sports_holdout_unsealed                                       # noqa: E402

HOLDS = {"60s": 60, "120s": 120}
MIN_SIGNAL = float(os.environ.get("MIN_SIGNAL", 5.0))   # smallest |signal| logged (lower for models with smaller forecasts)
VARIANTS = [(1, 0), (1, 1), (2, 0)]
SIGNAL_FROM = os.environ.get("SIGNAL_FROM", "ref")     # "ref" (last print) or "anchor" (the model's value now)


def first_beyond(px, yes, a, b, taker_yes: bool, price, above: bool, through: int):
    """Index of the first print in [a, b) by the given taker side beyond price by more than `through`."""
    if b <= a:
        return -1
    w = yes[a:b] if taker_yes else ~yes[a:b]
    hit = w & ((px[a:b] > price + through) if above else (px[a:b] < price - through))
    return a + int(np.argmax(hit)) if hit.any() else -1


def simulate(r: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    out = []
    by_t = {t: d for (t,), d in tr.partition_by("ticker", as_dict=True).items()}
    for (tk,), d in r.partition_by("ticker", as_dict=True).items():
        p = by_t.get(tk)
        if p is None:
            continue
        pt, px = p["t"].to_numpy(), p["yes_price"].to_numpy()
        yes = (p["taker_side"] == "yes").to_numpy()
        T, rid, ys, refs = d["t"].to_numpy(), d["row"].to_numpy(), d["y"].to_numpy(), d["ref"].to_numpy()
        sig = {hn: d[f"sig_{hn}"].to_numpy() for hn in HOLDS}
        a0, b0 = np.searchsorted(pt, T, side="right"), np.searchsorted(pt, T + ENTRY_S, side="right")
        for j in range(len(T)):
            ref = refs[j]
            for hn, hs in HOLDS.items():
                side = 1 if sig[hn][j] > 0 else -1
                if abs(sig[hn][j]) < MIN_SIGNAL:
                    continue
                for k, thr in VARIANTS:
                    lim = ref - k if side == 1 else ref + k
                    if lim <= 0 or lim >= 100:
                        continue
                    # long: a NO-taker prints below our bid; short: a YES-taker prints above our ask
                    f = first_beyond(px, yes, a0[j], b0[j], taker_yes=(side == -1), price=lim, above=(side == -1), through=thr)
                    if f < 0:
                        continue
                    te = pt[f] + hs
                    ae, be = np.searchsorted(pt, te, side="right"), np.searchsorted(pt, te + EXIT_S, side="right")
                    ref_e = px[ae - 1] if ae > 0 else ref
                    lx = ref_e + k if side == 1 else ref_e - k
                    fx = first_beyond(px, yes, ae, be, taker_yes=(side == 1), price=lx, above=(side == 1), through=thr) \
                        if 0 < lx < 100 else -1
                    if fx >= 0:
                        ex, how = float(lx), "limit filled"
                    else:
                        b2 = np.searchsorted(pt, te + 2 * EXIT_S, side="right")
                        x2 = first_taker(yes, be, b2, taker_yes=(side == -1))
                        ex, how = (float(px[x2]), "taken") if x2 >= 0 else (100.0 * ys[j], "settled")
                    out.append((int(rid[j]), hn, k, thr, side, float(sig[hn][j]), float(lim), ex, how, float(pt[f])))
    return pl.DataFrame(out, schema=["row", "hold", "k", "through", "side", "signal", "entry", "exit", "how", "t_fill"],
                        orient="row")


def main() -> None:
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    r = attach_tickers(rows(pn, games)).filter(pl.col("y").is_finite(), pl.col("kind").is_in(["winner", "spread", "total"])).with_row_index("row")
    tr = (scan_trades(r["series_ticker"].unique().to_list()).filter(pl.col("ticker").is_in(r["ticker"].unique().implode()))
          .select("ticker", (pl.col("known_at").dt.epoch("us").cast(pl.Float64) / 1e6).alias("t"), "yes_price", "taker_side")
          .collect().sort("ticker", "t"))
    r = (r.sort("t").join_asof(tr.select("ticker", "t", pl.col("yes_price").alias("ref"), pl.col("t").alias("t_ref")).sort("t"),
                               on="t", by="ticker", strategy="backward")
         .filter((pl.col("t") - pl.col("t_ref")) <= REF_MAX_S))
    # signal: the predicted level minus the last print (default), or minus the model's own anchor (SIGNAL_FROM=anchor:
    # for models whose anchor is the current mid, so the gap between the last print and the mid is not a signal)
    base = pl.col("anchor") if SIGNAL_FROM == "anchor" else pl.col("ref")
    r = r.with_columns([pl.col(f"level_{hn}").sub(base).alias(f"sig_{hn}") for hn in HOLDS])
    dec = r.group_by("kind").len().rename({"len": "decisions"})
    r = r.filter(pl.max_horizontal(*[pl.col(f"sig_{hn}").abs() for hn in HOLDS]) >= MIN_SIGNAL)
    g = pl.read_parquet(PANEL.parent / "games.parquet").select("game_id", "fold", "first_play")
    f = (simulate(r, tr).join(r.select("row", "kind", "game_id", "ticker", "t", "ref", "strike", "y"), on="row")
         .join(g, on="game_id", how="left").join(dec, on="kind"))
    # the final test's trade logs stay inside the holdout folder (and never overwrite the test folds' logs)
    base_dir = derived_dir() / "ingame" / "results" if sports_holdout_unsealed() else Path(__file__).resolve().parent / "out"
    out = base_dir / (f"b7_maker_trades_{LEAGUE}_d{DELAY:g}{RUN_TAG}{'_sa' if SIGNAL_FROM == 'anchor' else ''}"
                                                     f"{'' if MEMORY == 'gru' else '_' + MEMORY}{f'_seed{SEED}' if SEED else ''}.parquet")
    f.write_parquet(out)
    print(f"{LEAGUE} d{DELAY:g}: {f.height:,} trades -> {out}")


if __name__ == "__main__":
    main()
