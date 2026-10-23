"""The in-game model's ablation table for the report: each version of the model against the full one, on the test
folds, by forecast skill and by trading P&L under the frozen rule. Versions not run yet are listed as missing, so
this can be rerun after each step of reports/ablation_runsheet.md.

Every version is the average of 5 seeds (seed noise alone moved a contract's P&L about as much as the old
single-seed ablations did, out/b11_ablation_pnl.txt). Columns:
    R2 120s     R^2 (%) of the 120-s change in the traded price against no change, spread / total rows, the mean over
                the 5 seeds (per-seed .json files), folds weighted by rows; "ens" = the 5-seed average's R^2 where
                cluster/average_seeds.py recorded it
    P&L         the frozen strategy on the seed-averaged forecasts (b33_ablation_trades.sh -> b7_maker_trades.py):
                every 5 s, one at a time (MLB: skip 15-60 s after the last known pitch), offset 1c, strictly-through
                fills, fixed exit at 120 s, frozen sizing; net cents per contract after fees, game-level t
    vs full     per-game P&L minus the full model's, summed over every game either traded ($), with the paired t;
                negative = the removed part was earning money

    venv/bin/python -W ignore analysis/graph_2026_10/b33_ablation_table.py
    -> out/b33_ablation_table.txt, out/b33_ablation_table.csv
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

sys.argv = sys.argv[:1] + ["MLB"]       # b1_response (imported via b17) reads a league from argv
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b17_holdout_report import SIZE                       # noqa: E402
from b7_maker_liquidity import liquidity                 # noqa: E402
from b7_maker_robust import with_net                       # noqa: E402

RES = Path("results/ingame")
OUT = Path(__file__).resolve().parent / "out"
LEAGUES = ("MLB", "NBA", "NFL")
KINDS = ("spread", "total")


@dataclass
class V:
    group: str
    name: str
    memory: str = "gru"
    delay: float = 10
    rounds: int = 2
    drop: tuple[str, ...] = ()
    dpfix: bool = False
    pitch: bool = True                     # MLB: every pitch is a play (False: plate appearances, the older model)
    rule_mlb: str = "_dec5_one_skip15-60"
    leagues: tuple[str, ...] = LEAGUES
    note: str = field(default="")

    def tag(self, L: str) -> str:
        t = "_h30-60-120-1800_tp" + (f"_r{self.rounds}" if self.rounds != 2 else "")
        t += "_pitch" if L == "MLB" and self.pitch else ""
        t += "_dpfix" if self.dpfix else ""
        return t + "".join(f"_no-{d}" for d in self.drop)

    def run(self, L: str) -> str:          # the training run's name between fold<k>_ and _seed<s>
        return f"{self.memory}_d{self.delay:g}{self.tag(L)}"

    def log_stem(self, L: str) -> str:     # b7_maker_trades.py's log name without _fold<k>
        rule = self.rule_mlb if L == "MLB" else "_dec5_one"
        return f"b7_maker_trades_{L}_d{self.delay:g}{self.tag(L)}_ens5{'' if self.memory == 'gru' else '_' + self.memory}{rule}"


FULL = V("A. components", "full model (frozen)")
VERSIONS = [
    FULL,
    V("A. components", "no graph (no message passing)", rounds=0),
    V("A. components", "no play memory", memory="none"),
    V("A. components", "neither (step and contract features only)", memory="none", rounds=0),
    V("B. inputs", "no play types", drop=("play_type",)),
    V("B. inputs", "no win probability (and its baseline)", drop=("wp",)),
    V("B. inputs", "no structural baseline", drop=("baseline",)),
    V("B. inputs", "no recent trades / flow", drop=("prints",)),
    V("B. inputs", "no quotes", drop=("quotes",)),
    V("C. feed delay", "delay 0 s", delay=0),
    V("C. feed delay", "delay 5 s", delay=5, leagues=("MLB",)),
    V("C. feed delay", "delay 15 s", delay=15, leagues=("MLB",)),
    V("C. feed delay", "delay 30 s", delay=30, leagues=("NBA", "NFL")),
    V("C. feed delay", "delay 45 s (ESPN live)", delay=45, leagues=("NBA", "NFL")),
    V("C. feed delay", "delay 60 s", delay=60, leagues=("NBA", "NFL")),
    V("D. MLB play unit", "every pitch, no skip window", rule_mlb="_dec5_one", leagues=("MLB",)),
    V("D. MLB play unit", "plate appearances, no skip window", pitch=False, rule_mlb="_dec5_one", leagues=("MLB",)),
    V("D. MLB play unit", "double plays coded as such", dpfix=True, leagues=("MLB",)),
]


def r2(v: V, L: str, kind: str) -> tuple[float | None, float | None, int]:
    """Mean over seeds of the row-weighted R^2 (%) at 120 s; the ensemble's where recorded; seeds found."""
    per_seed = []
    for s in range(5):
        fs = [RES / f"{L}_fold{k}_{v.run(L)}_seed{s}.json" for k in range(1, 6)]
        if not all(f.exists() for f in fs):
            continue
        ms = [json.loads(f.read_text())["metrics"][kind]["120s"] for f in fs]
        per_seed.append(sum(m["r2_model"] * m["rows"] for m in ms) / sum(m["rows"] for m in ms))
    ens = []
    for k in range(1, 6):
        f = RES / "ens5" / f"{L}_fold{k}_{v.run(L)}_ens5_seed0.json"
        if f.exists() and "r2_ens5" in (j := json.loads(f.read_text())):
            ens.append((j["r2_ens5"][kind]["120s"], j["metrics"][kind]["120s"]["rows"]))
    e = sum(a * n for a, n in ens) / sum(n for _, n in ens) if len(ens) == 5 else None
    return (100 * float(np.mean(per_seed)) if per_seed else None), e, len(per_seed)


def trades(v: V, L: str) -> pl.DataFrame | None:
    files = sorted(OUT.glob(f"{v.log_stem(L)}_fold*.parquet")) or sorted(OUT.glob(f"{v.log_stem(L)}.parquet"))
    if len(files) not in (1, 5):
        return None
    f = pl.concat([pl.read_parquet(x) for x in files], how="vertical_relaxed").filter(
        pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")
    return (liquidity(with_net(f)).with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
            .with_columns((pl.col("size") * pl.col("net")).alias("pnl")))


def per_game(g: pl.DataFrame, games: pl.Series) -> np.ndarray:
    return (pl.DataFrame({"game_id": games}).join(g.group_by("game_id").agg(pl.col("pnl").sum()), on="game_id", how="left")
            .fill_null(0).sort("game_id")["pnl"].to_numpy())


def tstat(x: np.ndarray) -> float | None:
    return round(float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), 1) if len(x) > 1 and x.std() > 0 else None


def main() -> None:
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_hide_column_data_types(True)
    pl.Config.set_fmt_str_lengths(45)
    rows, cache = [], {}
    for L in LEAGUES:
        full = trades(FULL, L)
        for v in VERSIONS:
            if L not in v.leagues:
                continue
            t = full if v is FULL else trades(v, L)
            for kind in KINDS:
                r_seed, r_ens, n_seeds = r2(v, L, kind)
                row = dict(contract=f"{L} {kind}", group=v.group, version=v.name, seeds=n_seeds,
                           r2_120s=None if r_seed is None else round(r_seed, 2), r2_ens=None if r_ens is None else round(r_ens, 2))
                if t is None:
                    rows.append(row | dict(status="no trade logs yet"))
                    continue
                g = t.filter(pl.col("kind") == kind)
                fg = full.filter(pl.col("kind") == kind) if full is not None else None
                games = g["game_id"].unique() if fg is None else pl.concat([g["game_id"], fg["game_id"]]).unique()
                pg = per_game(g, games)
                c = g["size"].sum()
                folds = g.group_by("fold").agg(pl.col("pnl").sum()) if "fold" in g.columns else None
                row |= dict(status="ok", contracts=round(c), net_c=round(g["pnl"].sum() / c, 2) if c else None, t_game=tstat(pg),
                            folds_up=f"{(folds['pnl'] > 0).sum()}/{folds.height}" if folds is not None else None)
                if fg is not None and v is not FULL:
                    d = pg - per_game(fg, games)
                    row |= dict(vs_full_usd=round(d.sum() / 100, 1), vs_full_t=tstat(d))
                rows.append(row)
        print(L, "done", flush=True)
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(OUT / "b33_ablation_table.csv")
    lines = [__doc__.split("\n\n")[0], "", __doc__.split("\n\n")[1].rstrip(), ""]
    for contract in df["contract"].unique(maintain_order=True):
        lines += [f"================ {contract}", str(df.filter(pl.col("contract") == contract).drop("contract")), ""]
    (OUT / "b33_ablation_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
