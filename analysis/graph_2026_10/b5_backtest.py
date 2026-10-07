"""Taker backtest of the in-game model's forecasts on its out-of-sample test folds (gross first).

Inputs: the test predictions written by cluster/train_ingame.py --save-preds (one npz per fold and
variant) and the panel (for step positions). Rules, per contract at a step:

    holding period h           30 s, 60 s, 120 s or 300 s (each a separate strategy, on the model's own
                               forecast for that horizon)
    predicted level at +h      L = value + predicted change(h)
    buy YES at the ask         if L - ask >= M   (needs a quoted ask)
    sell YES at the bid        if bid - L >= M   (needs a quoted bid)
    exit after h               sell at the bid then / buy back at the ask then; if that side is not
                               quoted, hold to settlement (100 or 0)
    also: hold to settlement   for every trade, as a second exit rule

Entries only every h (one per contract per holding period), so trades in the same contract never
overlap. Runs for every feed delay found (the delay with which plays reached the model). One contract per trade. Gross P&L in cents; net subtracts a taker fee of
FEE_RATE x p (1 - p) per contract on every trade at price p (entry, and the exit unless settled;
Kalshi base rate unverified, see kalshi-fees). t-statistics cluster by game. Split by contract kind,
margin M, and the entry quote's age (stale quotes may not be executable).

    venv/bin/python -W ignore analysis/graph_2026_10/b5_backtest.py results/ingame NFL
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stg_infra"))
from stg.models.ingame import KIND_NAMES, Panel           # noqa: E402
from stg.sports.loaders import derived_dir                 # noqa: E402

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ingame")
LEAGUE = sys.argv[2] if len(sys.argv) > 2 else "NFL"
HOLDS = {"30s": 6, "60s": 12, "120s": 24, "300s": 60}       # holding period -> entry spacing in 5-s steps
MARGINS = [0.0, 2.0, 3.0, 5.0]
AGES = {"any age": np.inf, "<= 30 s": 30.0, "<= 10 s": 10.0}
PANEL = Path(sys.argv[3]) if len(sys.argv) > 3 else None   # optional panel path (default: the league's panel_v2)
FEE_RATE = 7.0                   # cents x p (1 - p), taker, multiplier 1 (NFL game series)


def fee(p: np.ndarray) -> np.ndarray:
    q = np.clip(p / 100, 0, 1)
    return FEE_RATE * q * (1 - q)


def clustered_t(x: np.ndarray, g: np.ndarray) -> float:
    if len(x) < 20:
        return float("nan")
    d = pl.DataFrame({"x": x - x.mean(), "g": g}).group_by("g").agg(pl.col("x").sum())
    se = float(np.sqrt((d["x"] ** 2).sum())) / len(x)
    return float(x.mean() / se) if se > 0 else float("nan")


def trades(pr: dict, step_k: np.ndarray, margin: float, max_age: float, h: int, every: int) -> pl.DataFrame:
    lvl = pr["value"] + pr["delta"][:, h]
    bid, ask = pr["entry_bid"], pr["entry_ask"]
    ok = (step_k % every == 0) & np.isfinite(pr["y"]) & (pr["quote_age_s"] <= max_age)
    long_edge = np.where(pr["has_ask"], lvl - ask, -np.inf)
    short_edge = np.where(pr["has_bid"], bid - lvl, -np.inf)
    side = np.where((long_edge >= margin) & (long_edge >= short_edge), 1, np.where(short_edge >= margin, -1, 0))
    m = ok & (side != 0)
    s, y = side[m], 100 * pr["y"][m]
    entry = np.where(s == 1, ask[m], bid[m])
    lo, hi = pr["lo"][m, h].astype(float), pr["hi"][m, h].astype(float)   # bid / ask after h (0 / 100 = side missing)
    exit_px = np.where(s == 1, lo, hi)
    quoted = np.isfinite(exit_px) & np.where(s == 1, exit_px > 0, exit_px < 100)
    exit_val = np.where(quoted, exit_px, y)
    g60 = s * (exit_val - entry)
    gst = s * (y - entry)
    f60 = fee(entry) + np.where(quoted, fee(exit_val), 0)
    return pl.DataFrame({"kind": pr["kind"][m], "game": pr["game"][m], "side": s, "entry": entry, "edge": np.where(s == 1, long_edge[m], short_edge[m]),
                         "gross_60": g60, "net_60": g60 - f60, "gross_settle": gst, "net_settle": gst - fee(entry),
                         "exit_quoted": quoted})


def summarise(t: pl.DataFrame) -> dict:
    out = dict(trades=t.height, games=t["game"].n_unique())
    for c in ("gross_60", "net_60", "gross_settle", "net_settle"):
        v = t[c].to_numpy()
        out[c] = round(float(v.mean()), 3) if len(v) else None
        out[f"t_{c}"] = round(clustered_t(v, t["game"].to_numpy()), 1) if len(v) else None
    out["win_60"] = round(float((t["gross_60"] > 0).mean()), 3) if t.height else None
    out["exit_quoted"] = round(float(t["exit_quoted"].mean()), 3) if t.height else None
    return out


def main() -> None:
    import json
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(240)
    pn = Panel(PANEL or derived_dir() / "ingame" / LEAGUE / "panel_v2")
    gsp = pn.game_step_ptr
    rows = []
    files = sorted(RES.glob(f"{LEAGUE}_fold*_d*_seed0.npz"))
    tag_of = lambda f: f.stem.split("_d", 1)[1].split("_seed")[0].partition("_")[2] or "all"   # noqa: E731
    groups = {}
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        groups.setdefault((meta["memory"], meta.get("delay_s", 0.0), tag_of(f)), []).append((f, meta))
    for (memory, delay, tag), fl in sorted(groups.items()):
        prs = [dict(np.load(f)) for f, _ in fl]
        hz = fl[0][1]["horizons"]
        pr = {k: np.concatenate([p[k] for p in prs]) for k in prs[0]}
        step_k = pr["step"] - gsp[pr["game"]]
        for hold, every in HOLDS.items():
            if hold not in hz:
                continue
            h = hz.index(hold)
            for age_name, age in AGES.items():
                for margin in MARGINS:
                    t = trades(pr, step_k, margin, age, h, every)
                    for kind in range(3):
                        rows.append(dict(memory=memory, horizons=tag, delay=delay, hold=hold, age=age_name, margin=margin,
                                         kind=KIND_NAMES[kind], **summarise(t.filter(pl.col("kind") == kind))))
        print(f"{memory}, horizons {tag}, feed delay {delay:g} s: {len(fl)} folds, {len(pr['value']):,} contract-steps")
    r = pl.DataFrame(rows)
    r.write_parquet(Path(__file__).resolve().parent / "out" / f"b5_backtest_{LEAGUE}{'_' + RES.name if RES.name != 'ingame' else ''}.parquet")
    cols = ["horizons", "delay", "hold", "margin", "trades", "games", "gross_60", "t_gross_60", "net_60", "t_net_60", "win_60",
            "gross_settle", "t_gross_settle", "exit_quoted"]
    print("\n######## gross / net P&L per trade (cents, one contract), exit after the holding period (columns *_60 "
          "= that exit); t clustered by game; entry quotes <= 30 s old")
    for kind in KIND_NAMES:
        for memory in r["memory"].unique().sort():
            print(f"--- {kind}, memory {memory}")
            print(r.filter(pl.col("kind") == kind, pl.col("memory") == memory, pl.col("age") == "<= 30 s")
                  .select(cols).sort("horizons", "hold", "margin", "delay"))


if __name__ == "__main__":
    main()
