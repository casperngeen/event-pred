"""Were the backtest's entry prices really available? Checks each taker entry of b5_backtest against
the trades that actually printed in that contract right after it.

The entry quote is rebuilt from 1-min candle closes and trade prints (stg.sports.prices): a side is
the last price seen on it, so right after a big play it can be a price the book has already left.
For a simulated buy at ask A at time T, someone else buying YES at <= A within the next WINDOW_S
seconds shows A was still there; YES buys only at prices above A show it was gone. Same for sells
(NO-taker prints at >= the bid). Entries with no print in the window are "unknown".

Trades: as b5_backtest (hold 60 s, edge >= MARGIN, quotes <= 30 s old), spread and total contracts.
Each is matched to its contract by game, table step, kind and the quoted bid / ask (ambiguous matches
take the first ticker; same quotes, so the check is the same).

    venv/bin/python -W ignore analysis/graph_2026_10/b5_fills.py results/ingame NFL 10 [panel_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stg.models.ingame import KIND_NAMES, Panel           # noqa: E402
from stg.sports.loaders import derived_dir, scan_trades    # noqa: E402
from b5_backtest import clustered_t, fee                   # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ingame")
LEAGUE = sys.argv[2] if len(sys.argv) > 2 else "NFL"
DELAY = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
TAB = derived_dir() / "ingame" / LEAGUE
PANEL = Path(sys.argv[4]) if len(sys.argv) > 4 else TAB / "panel_v2"
HOLD, EVERY, MARGIN, MAX_AGE, WINDOW_S, STEP = "60s", 12, 5.0, 30.0, 10.0, 5


def sim_trades(pn: Panel, pr: dict, hz: list[str]) -> pl.DataFrame:
    h = hz.index(HOLD)
    step_k = pr["step"] - pn.game_step_ptr[pr["game"]]
    lvl = pr["value"] + pr["delta"][:, h]
    bid, ask = pr["entry_bid"], pr["entry_ask"]
    ok = (step_k % EVERY == 0) & np.isfinite(pr["y"]) & (pr["quote_age_s"] <= MAX_AGE) & (pr["kind"] > 0)
    le = np.where(pr["has_ask"], lvl - ask, -np.inf)
    se = np.where(pr["has_bid"], bid - lvl, -np.inf)
    side = np.where((le >= MARGIN) & (le >= se), 1, np.where(se >= MARGIN, -1, 0))
    m = np.flatnonzero(ok & (side != 0))
    s = side[m]
    entry = np.where(s == 1, ask[m], bid[m])
    exit_px = np.where(s == 1, pr["lo"][m, h], pr["hi"][m, h]).astype(float)
    quoted = np.isfinite(exit_px) & np.where(s == 1, exit_px > 0, exit_px < 100)
    exit_val = np.where(quoted, exit_px, 100 * pr["y"][m])
    gross = s * (exit_val - entry)
    return pl.DataFrame({"g": pr["game"][m], "t": pn.steps_t[pr["step"][m]].astype(float), "kind": pr["kind"][m],
                         "side": s, "bid": np.round(bid[m]), "ask": np.round(ask[m]), "entry": entry,
                         "gross": gross, "net": gross - fee(entry) - np.where(quoted, fee(exit_val), 0)})


def main() -> None:
    pl.Config.set_tbl_width_chars(200)
    pn = Panel(PANEL)
    games = json.loads((PANEL / "manifest.json").read_text())["games"]
    prs, hz = [], None
    for f in sorted(RES.glob(f"{LEAGUE}_fold*_gru_d{DELAY:g}_seed0.npz")):
        hz = json.loads(f.with_suffix(".json").read_text())["horizons"]
        prs.append(dict(np.load(f)))
    pr = {k: np.concatenate([p[k] for p in prs]) for k in prs[0]}
    t = sim_trades(pn, pr, hz).with_columns(pl.col("g").map_elements(lambda g: games[g], return_dtype=pl.String).alias("game_id"))
    gm = pl.read_parquet(TAB / "games.parquet").select("game_id", pl.col("first_play").dt.epoch("us") / 1e6)
    con = pl.read_parquet(TAB / "contracts.parquet").select("ticker", "series_ticker", "kind")
    kind_id = {n: i for i, n in enumerate(KIND_NAMES)}
    t = t.join(gm, on="game_id").with_columns(((pl.col("t") - pl.col("first_play")) / STEP).round().cast(pl.Int64).alias("k"))
    parts = []
    for gid in t["game_id"].unique().to_list():
        cs = pl.read_parquet(TAB / "contract_steps" / f"{gid}.parquet", columns=["k", "ticker", "bid", "ask"])
        cs = cs.join(con, on="ticker").with_columns(pl.col("kind").replace_strict(kind_id).alias("kind"), pl.lit(gid).alias("game_id"))
        parts.append(t.filter(pl.col("game_id") == gid).join(cs, on=["game_id", "k", "kind", "bid", "ask"], how="left")
                     .unique(["game_id", "k", "kind", "bid", "ask", "side"], keep="first", maintain_order=True))
    t = pl.concat(parts)
    print(f"{LEAGUE}, delay {DELAY:g} s: {t.height} spread/total entries, matched to a ticker: {t['ticker'].is_not_null().mean():.3f}")
    t = t.filter(pl.col("ticker").is_not_null()).with_columns(pl.from_epoch((pl.col("t") * 1e6).cast(pl.Int64), "us")
                                                            .dt.replace_time_zone("UTC").alias("ts"))
    series = con.filter(pl.col("ticker").is_in(t["ticker"].implode()))["series_ticker"].unique().to_list()
    tr = (scan_trades(series).filter(pl.col("ticker").is_in(t["ticker"].unique().implode()))
          .select("ticker", "known_at", "yes_price", "taker_side").collect())
    # prints in (T, T + WINDOW_S] on the entry's side
    w = (t.with_row_index("i").join(tr, on="ticker", how="inner")
         .filter(pl.col("known_at") > pl.col("ts"), pl.col("known_at") <= pl.col("ts") + pl.duration(seconds=WINDOW_S),
                 pl.col("taker_side") == pl.when(pl.col("side") == 1).then(pl.lit("yes")).otherwise(pl.lit("no"))))
    agg = w.group_by("i").agg(pl.len().alias("prints"),
                              ((pl.col("side") * (pl.col("entry") - pl.col("yes_price"))) >= 0).any().alias("fill_seen"),
                              (pl.col("side") * (pl.col("yes_price").first() - pl.col("entry"))).first().alias("first_vs_entry"))
    t = t.with_row_index("i").join(agg, on="i", how="left").with_columns(
        pl.when(pl.col("prints").is_null()).then(pl.lit("no print"))
        .when(pl.col("fill_seen")).then(pl.lit("filled at <= entry")).otherwise(pl.lit("only worse prints")).alias("status"))
    out = (t.group_by("kind", "status").agg(pl.len().alias("trades"), pl.col("game_id").n_unique().alias("games"),
                                            pl.col("gross").mean().round(2), pl.col("net").mean().round(2),
                                            pl.col("first_vs_entry").median().alias("median_first_print_worse_by"))
           .with_columns(pl.col("kind").replace_strict(dict(enumerate(KIND_NAMES)))).sort("kind", "status"))
    print(f"prints on the entry's side within {WINDOW_S:g} s after entry:")
    print(out)
    ok = t.filter(pl.col("status") == "filled at <= entry")
    if ok.height:
        print(f"entries confirmed by a print: gross {ok['gross'].mean():.2f}, net {ok['net'].mean():.2f} "
              f"(t {clustered_t(ok['net'].to_numpy(), ok['game_id'].to_numpy()):.1f}), {ok.height} trades")


if __name__ == "__main__":
    main()
