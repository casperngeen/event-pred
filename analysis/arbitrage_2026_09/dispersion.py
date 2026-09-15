#!/usr/bin/env python
"""Direction 2: index-vs-components dispersion, on the CPI basket.

    venv/bin/python analysis/arbitrage_2026_09/dispersion.py

Headline CPI is a *known weighted sum* of components, several of which trade as
their own ladders. So the variance of the headline is not free::

    Var(CPI) = Sum_i Sum_j w_i w_j Cov(comp_i, comp_j)

which is the index-vs-single-stock dispersion relation from options, with BLS
publishing the weights instead of an index provider. Two consequences, neither
needing any macro forecast:

1. With headline and core both priced, the headline ladder implies a variance
   for the **non-core** remainder (food and energy). That implied remainder can
   be compared with what the component ladders themselves say.
2. With headline, core and gasoline priced together, the system is
   over-determined and the implied **correlation** can be backed out. A
   correlation outside [-1, 1] is not a mispricing, it is an impossibility --
   the hardest evidence available short of a locked arbitrage.

Weights
-------
BLS relative importances are published, but not in this repo (`data/` has no
weights file; `research_summary.md` §3 asset 2 lists fetching them as an
outstanding item). The values below are the standard published shares for the
2022-25 period, and **every result is reported across a sensitivity band**
rather than at a point, precisely because they are entered by hand here. Replace
`W` with the exact BLS relative-importance table before citing any of this.

Decomposition used::

    CPI = w_core * CORE + w_food * FOOD + w_energy * ENERGY
    core ~ 0.79,  food ~ 0.135,  energy ~ 0.069,  gasoline ~ 0.034 (subset of energy)

In-sample only.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

PANELS = Path("artifacts/panels")
OUT = Path("analysis/arbitrage_2026_09/out")

W_CORE = 0.79
W_GAS = 0.034
CORE_BAND = (0.75, 0.83)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_month(t: str):
    x = t.split("-", 1)[1] if "-" in t else ""
    if len(x) < 5 or not x[:2].isdigit() or x[2:5].upper() not in MONTHS:
        return None
    return dt.date(2000 + int(x[:2]), MONTHS[x[2:5].upper()], 1)


def series_sd(n: pl.DataFrame, s: str, nm: str) -> pl.DataFrame:
    return (n.filter(pl.col("series") == s)
            .select("ref", "date",
                    pl.col("implied_std").alias(nm),
                    pl.col("n_fresh_legs").alias(f"legs_{nm}")))


def main() -> None:
    n = pl.read_parquet(PANELS / "node_panel_event.parquet")
    n = (n.with_columns(pl.col("event_ticker")
                        .map_elements(parse_month, return_dtype=pl.Date).alias("ref"))
         .drop_nulls("ref")
         .filter(pl.col("implied_std").is_not_null() & (pl.col("implied_std") > 0)))

    j = (series_sd(n, "CPI", "head")
         .join(series_sd(n, "CPICORE", "core"), on=["ref", "date"], how="inner")
         .drop_nulls(["head", "core"]))
    print(f"CPI + CPICORE, same reference month, same day: {j.height} days, "
          f"{j['ref'].n_unique()} months\n")

    # ---------------------------------------------------------------- §1
    print("=== 1. what non-core volatility does the headline ladder imply? ===")
    print("Var(CPI) = w^2 Var(core) + (1-w)^2 Var(rest) + 2 w (1-w) rho sd sd.")
    print("Setting rho = 0 gives the implied sd of the food+energy remainder.")
    print("rho = 0 is the GENEROUS case: any positive correlation would force")
    print("the implied remainder lower still.\n")
    rows = []
    for w in (CORE_BAND[0], W_CORE, CORE_BAND[1]):
        vh = j["head"].to_numpy() ** 2
        vc = j["core"].to_numpy() ** 2
        resid = vh - (w ** 2) * vc
        sd_rest = np.sqrt(np.clip(resid, 0, None)) / (1 - w)
        rows.append(dict(w_core=w, n=len(vh),
                         frac_negative=float((resid < 0).mean()),
                         implied_sd_rest_p50=float(np.median(sd_rest)),
                         implied_sd_rest_p90=float(np.percentile(sd_rest, 90)),
                         head_p50=float(np.median(j["head"].to_numpy())),
                         core_p50=float(np.median(j["core"].to_numpy()))))
    with pl.Config(float_precision=4, tbl_width_chars=220):
        print(pl.DataFrame(rows))
    print("\n`frac_negative` is the share of days where the headline ladder is")
    print("too narrow for its own core ladder even at rho = 0 -- i.e. days on")
    print("which no non-negative remainder variance can reconcile the two.")

    # ---------------------------------------------------------------- §2
    print("\n=== 2. is that remainder plausible next to the gasoline ladder? ===")
    g = series_sd(n, "CPIGAS", "gas")
    k = j.join(g, on=["ref", "date"], how="inner").drop_nulls("gas")
    print(f"days with CPI + CPICORE + CPIGAS all priced: {k.height}, "
          f"{k['ref'].n_unique()} months")
    if k.height >= 8:
        vh = k["head"].to_numpy() ** 2
        vc = k["core"].to_numpy() ** 2
        sg = k["gas"].to_numpy()
        resid = vh - (W_CORE ** 2) * vc
        sd_rest = np.sqrt(np.clip(resid, 0, None)) / (1 - W_CORE)
        # gasoline alone contributes w_gas * sd_gas to headline volatility;
        # gasoline is a SUBSET of the remainder, so the remainder's own
        # volatility must be at least that of gasoline scaled by weight share
        floor = sg * W_GAS / (1 - W_CORE)
        print(f"\nimplied sd(remainder)  median {np.median(sd_rest):.3f}pp")
        print(f"gasoline-implied floor  median {np.median(floor):.3f}pp")
        print(f"days where implied remainder < gasoline floor: "
              f"{int((sd_rest < floor).sum())} of {len(sd_rest)} "
              f"({(sd_rest < floor).mean():.0%})")
        print("\nThe remainder contains gasoline plus food plus the rest of")
        print("energy, so its volatility cannot be below gasoline's contribution")
        print("unless the components offset -- which needs negative correlation")
        print("between food and energy.")

        # over-determined: back out implied rho(core, gasoline-proxy remainder)
        print("\n=== 3. the over-determined system: implied correlation ===")
        rows = []
        for w in (CORE_BAND[0], W_CORE, CORE_BAND[1]):
            wr = 1 - w
            num = vh - (w ** 2) * vc - (wr ** 2) * (sg ** 2)
            den = 2 * w * wr * k["core"].to_numpy() * sg
            rho = num / den
            rows.append(dict(w_core=w, n=len(rho),
                             rho_p10=float(np.percentile(rho, 10)),
                             rho_p50=float(np.median(rho)),
                             rho_p90=float(np.percentile(rho, 90)),
                             frac_below_m1=float((rho < -1).mean()),
                             frac_above_p1=float((rho > 1).mean())))
        with pl.Config(float_precision=3, tbl_width_chars=220):
            print(pl.DataFrame(rows))
        print("\nTreating the remainder as gasoline-like. rho outside [-1, 1] is")
        print("impossible, so any such fraction is a hard inconsistency -- but")
        print("read it against the weight sensitivity, not at a point.")

    # ---------------------------------------------------------------- §4
    print("\n=== 4. how does the head/core width ratio behave? ===")
    print("Headline carries food and energy on top of core, so its ladder must")
    print("be WIDER. A ratio at or below 1 is the inconsistency.\n")
    j = j.with_columns((pl.col("head") / pl.col("core")).alias("ratio"))
    r = j["ratio"].to_numpy()
    print(f"head/core implied-sd ratio:  p10 {np.percentile(r,10):.3f}   "
          f"p50 {np.median(r):.3f}   p90 {np.percentile(r,90):.3f}")
    print(f"days with ratio <= 1 (headline no wider than core): "
          f"{(r <= 1).mean():.0%}")
    print(f"days with ratio <= {W_CORE:.2f} (below even the core contribution "
          f"alone): {(r <= W_CORE).mean():.0%}")
    gated = j.filter((pl.col("legs_head") >= 4) & (pl.col("legs_core") >= 4))
    if gated.height >= 30:
        rg = gated["ratio"].to_numpy()
        print(f"\nboth ladders >= 4 fresh legs ({gated.height} days): "
              f"p50 {np.median(rg):.3f}, share <= 1: {(rg <= 1).mean():.0%}")

    OUT.mkdir(parents=True, exist_ok=True)
    j.write_parquet(OUT / "dispersion.parquet")
    print(f"\nwrote {OUT / 'dispersion.parquet'}")


if __name__ == "__main__":
    main()
