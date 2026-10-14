"""NFL holdout: is the drop a latency problem? The same strategy (every 5 s, one at a time, fixed exit, frozen
sizing) and forecast checks at feed delays of 0, 3, 5 and 10 s. Descriptive only (spec section 5).

    d10            the frozen models, the holdout result
    d<x>_w10       the frozen 10 s models fed plays x s after they are logged (a mismatch: trained for 10 s)
    d<x>           models trained at a x s delay (cluster/train_final.sbatch DELAY=x), when present

    STG_SPORTS_UNSEAL=final-test venv/bin/python -W ignore analysis/graph_2026_10/b31_nfl_latency.py
    -> out/b31_nfl_latency.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b17_holdout_report import SIZE, exits, summary      # noqa: E402
from b7_maker_liquidity import liquidity                # noqa: E402
from b7_maker_robust import OUT, with_net                # noqa: E402
from stg.sports.loaders import derived_dir               # noqa: E402
from stg.splits import sports_holdout_unsealed           # noqa: E402

KINDS = {1: "spread", 2: "total"}
HZ = ["30s", "60s", "120s"]


def main() -> None:
    assert sports_holdout_unsealed(), "reads the holdout: set STG_SPORTS_UNSEAL=final-test"
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_hide_column_data_types(True)
    res = derived_dir() / "ingame" / "results"
    runs = [("d10 (frozen)", 10, "")] + [(f"d{d}_w10 (10 s weights)", d, "_w10") for d in (5, 3, 0)] + \
           [(f"d{d} (trained at {d} s)", d, "") for d in (5, 3, 0)]
    fc, tr = [], []
    for name, d, w in runs:
        f = res / f"NFL_fold99_gru_d{d}_h30-60-120-1800_tp{w}_ens5_seed0.npz"
        log = res / f"b7_maker_trades_NFL_d{d}_h30-60-120-1800_tp{w}_ens5_dec5_one.parquet"
        if not f.exists() or not log.exists():
            continue
        z = np.load(f)
        tgt = z["value_h"][:, :3] - z["value"][:, None]
        for kc, kn in KINDS.items():
            r = dict(run=name, kind=kn)
            for i, h in enumerate(HZ):
                ok = (z["kind"] == kc) & np.isfinite(tgt[:, i])
                y, dd = tgt[ok, i], z["delta"][ok, i]
                r[f"R2_{h}"] = round(100 * (1 - ((y - dd) ** 2).sum() / (y ** 2).sum()), 2)
                if h == "60s":
                    r["slope_60s"] = round(float((y * dd).sum() / (dd ** 2).sum()), 2)
            fc.append(r)
        t = (liquidity(with_net(pl.read_parquet(log).filter(pl.col("k") == 1, pl.col("through") == 0, pl.col("hold") == "120s")))
             .with_columns(SIZE.alias("size")).filter(pl.col("size") > 0)
             .with_columns(pl.col("first_play").dt.convert_time_zone("America/New_York").dt.date().alias("day")))
        for kn in KINDS.values():
            x = t.filter(pl.col("kind") == kn)
            s = summary(exits(x)["fixed exit"], x["game_id"].unique().sort(), x.select("day").unique())
            tr.append(dict(run=name, kind=kn) | {k: s[k] for k in ("contracts", "gross_c", "NET_c", "t_game")})
    lines = [__doc__.split("\n\n")[0], "", "-- forecasts (average of 5 seeds): R^2 (%) against no change; slope of the realised move on the forecast",
             str(pl.DataFrame(fc).sort("kind")), "", "-- trades, fixed exit (net = after fees, c per contract)", str(pl.DataFrame(tr).sort("kind"))]
    (OUT / "b31_nfl_latency.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
