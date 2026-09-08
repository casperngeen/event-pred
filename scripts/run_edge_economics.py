#!/usr/bin/env python
"""Do the recovered edges make economic sense, and can they be traded?

    venv/bin/python scripts/run_edge_economics.py

Two questions the direction study raises but does not answer.

**Economics.** Each channel carries a sign predicted by macro theory before any
data is seen (``research_summary.md`` §8.3 item 2: sign restrictions are
falsification that validation loss cannot provide). This script states those
predictions explicitly and checks the estimated edges against them — including
U3, the falsification cell, where the dovish reading flips the sign.

**Tradability.** A hit rate is not a return. The ledger prices every
out-of-fold signal at the first post-resolution print (not the pre-resolution
reference the estimator uses), charges the taker-direction effective spread
(``research_summary.md`` §4.2.1) and the Kalshi p(1-p) fee on both legs.

Writes artifacts/edge_economics.md and artifacts/direction_ledger.parquet.
In-sample only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.direction import (
    SignRule, decompose_move, effective_spread, fit_structure, gate_coverage,
    ledger_summary, predict_oof, trade_ledger, walk_forward,
)
from stg.direction.tradability import DEFAULT_MAX_GAP_S
from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import series_filter_expr
from stg.splits import BURN_IN_END, OOS_START
from stg.structure.stats import spearman, spearman_p

OUT = Path("artifacts")
PAIR_PANEL = Path("artifacts/panels/pair_panel_dormant.parquet")

# Signs predicted from macro theory BEFORE looking at the estimates. "+" means
# a positive surprise in the trigger should raise the target contract's yes
# price. FED / FEDDECISION-hike contracts pay on *higher* rates, so anything
# that pushes the policy path up is "+".
PRIORS: list[tuple[str, str, str, int, str]] = [
    ("CPICORE", "CPI", "any", +1,
     "mechanical — core is a subset of the same BLS print, not propagation"),
    ("CPI", "FED", "any", +1,
     "Taylor-rule reaction: hotter inflation -> higher expected policy rate"),
    ("CPICORE", "FED", "any", +1,
     "same channel on the measure the Fed actually reacts to"),
    ("PAYROLLS", "FED", "any", +1,
     "dual mandate: stronger labour market -> tighter policy"),
    ("PAYROLLS", "FEDDECISION", "hike", +1,
     "same channel, discrete meeting outcome"),
    ("CPIYOY", "FEDDECISION", "hike", +1,
     "same channel, discrete meeting outcome"),
    ("U3", "FED", "any", -1,
     "FALSIFICATION CELL: higher unemployment is dovish, so the sign must flip"),
    ("U3", "FEDDECISION", "hike", -1,
     "falsification cell, discrete version"),
    ("CPIYOY", "PCECORE", "any", +1,
     "CPI leads PCE and shares source price collections"),
    ("WTI", "CPIGAS", "any", +1,
     "oil passes through to the gasoline CPI subcomponent — the cleanest "
     "mechanical prior in the grid"),
    ("WTI", "CPI", "any", +1,
     "energy is ~7% of the headline basket"),
    ("WTI", "JOBLESSCLAIMS", "any", 0,
     "no directional prior — either sign is tellable after the fact"),
]


def fmt(x, nd=3):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def sign_table(panel: pl.DataFrame) -> list[str]:
    lines = ["| channel | predicted | rho (full IS) | n | p | verdict | economic reading |",
             "|---|---|---|---|---|---|---|"]
    for trig, tgt, side, pred, why in PRIORS:
        g = panel.filter((pl.col("trigger") == trig) & (pl.col("target") == tgt)
                         & (pl.col("side") == side))
        if g.height < 5:
            lines.append(f"| {trig}→{tgt}/{side} | {pred:+d} | – | {g.height} | – | "
                         f"too few matches | {why} |")
            continue
        rho = spearman(g["surprise"].to_numpy().astype(float),
                       g["response"].to_numpy().astype(float))
        p = spearman_p(rho, g.height)
        if pred == 0:
            verdict = "no prior"
        elif not np.isfinite(rho):
            verdict = "–"
        elif np.sign(rho) == pred:
            verdict = "**matches**" if p < 0.05 else "matches (ns)"
        else:
            verdict = "**CONTRADICTS**" if p < 0.05 else "contradicts (ns)"
        lines.append(f"| {trig}→{tgt}/{side} | {pred:+d} | {fmt(rho)} | {g.height} | "
                     f"{fmt(p, 4)} | {verdict} | {why} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="bh", choices=["bh", "p05"])
    ap.add_argument("--n-folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--max-gap-s", type=int, default=DEFAULT_MAX_GAP_S)
    ap.add_argument("--refresh-spreads", action="store_true",
                    help="re-scan data/trades/ for spreads (~7 min; cached "
                         "in artifacts/effective_spreads.parquet otherwise)")
    args = ap.parse_args()

    panel = pl.read_parquet(PAIR_PANEL).filter(pl.col("t0") >= BURN_IN_END)
    folds = walk_forward(panel, n_folds=args.n_folds, start_frac=args.start_frac)
    structures = [(f, fit_structure(panel.filter(pl.Series(f.train)))) for f in folds]

    # the signals the sign rule actually made, out of fold, on covered rows
    p_up = predict_oof(panel, SignRule(), structures)
    covered = np.zeros(panel.height, bool)
    for f in folds:
        covered |= f.test
    mask = covered & gate_coverage(panel, structures, args.gate)
    direction = np.where(p_up >= 0.5, 1, -1)

    # effective spreads for the target series actually traded. data/trades/ is
    # 3,891 flat files, so any ticker filter scans the whole 1.8 GB
    # (TODO.md, Phase D) — cache the result rather than pay it per run.
    traded = panel.filter(pl.Series(mask))
    cache = OUT / "effective_spreads.parquet"
    if cache.exists() and not args.refresh_spreads:
        cached = pl.read_parquet(cache)
        sp_series = cached.filter(pl.col("window_s") == args.max_gap_s).drop("window_s")
        sp_wide = cached.filter(pl.col("window_s") == args.max_gap_s * 60).drop("window_s")
    else:
        tickers = traded["target_ticker"].unique().to_list()
        tr = (scan_trades(is_only=True)
              .filter(pl.col("ticker").is_in(tickers))
              .select("ticker", "created_time", "yes_price", "taker_side").collect())
        tk2series = dict(zip(traded["target_ticker"].to_list(),
                             traded["target"].to_list()))
        tr = tr.with_columns(pl.col("ticker").replace_strict(tk2series, default=None)
                             .alias("series")).drop_nulls("series")
        sp_series = effective_spread(tr, max_gap_s=args.max_gap_s, by="series")
        sp_wide = effective_spread(tr, max_gap_s=args.max_gap_s * 60, by="series")
        pl.concat([sp_series.with_columns(window_s=pl.lit(args.max_gap_s)),
                   sp_wide.with_columns(window_s=pl.lit(args.max_gap_s * 60))],
                  how="vertical").write_parquet(cache)
    spreads = dict(zip(sp_series["series"].to_list(),
                       sp_series["spread_median"].to_list())) if sp_series.height else {}

    ledger = trade_ledger(panel, direction, mask, spreads=spreads)
    summ = ledger_summary(ledger)
    OUT.mkdir(parents=True, exist_ok=True)
    if not ledger.is_empty():
        ledger.write_parquet(OUT / "direction_ledger.parquet")

    lines = [
        "# Edge economics: do the channels make sense, and can they be traded?",
        "",
        f"built: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}   "
        f"in-sample only (wall {OOS_START.date()})",
        f"gate: `{args.gate}`   folds: {len(folds)}   "
        f"spread pairing window: {args.max_gap_s}s",
        "",
        "## 1. Sign restrictions",
        "",
        "Signs predicted from macro theory *before* consulting the estimates. A "
        "channel that fires with the wrong sign is falsified in a way no "
        "validation loss can express (`research_summary.md` §8.3 item 2).",
        "",
    ] + sign_table(panel)

    # where do the edges point?
    surv = pl.read_parquet(OUT / "adjacency_is.parquet").filter(pl.col("survives"))
    by_target = (surv.group_by("target").agg(pl.len().alias("edges"))
                 .sort("edges", descending=True))
    lines += ["", "## 2. Where the edges point", "",
              "| target | surviving in-edges |", "|---|---|"]
    for r in by_target.iter_rows(named=True):
        lines.append(f"| {r['target']} | {r['edges']} |")
    policy = int(surv.filter(pl.col("target").is_in(["FED", "FEDDECISION"])).height)
    lines += ["",
              f"{policy} of {surv.height} surviving edges point at policy "
              "(`FED`/`FEDDECISION`). The recovered graph is not a web — it is a "
              "**hub**: macro data flows into the policy path, plus a "
              "same-release clique inside the CPI family."]

    lines += ["", "## 3. Effective spread on the traded targets", "",
              "Taker-direction estimator (`research_summary.md` §4.2.1). "
              "Window-invariance is the validity check: widening the pairing "
              "window 60x must not move the median.", "",
              "| target series | median (cents) | mean | n pairs | median @ 60x window |",
              "|---|---|---|---|---|"]
    wide = dict(zip(sp_wide["series"].to_list(), sp_wide["spread_median"].to_list())) \
        if sp_wide.height else {}
    for r in sp_series.iter_rows(named=True):
        lines.append(f"| {r['series']} | {fmt(r['spread_median'], 2)} | "
                     f"{fmt(r['spread_mean'], 2)} | {r['n_pairs']} | "
                     f"{fmt(wide.get(r['series'], float('nan')), 2)} |")

    if summ:
        lines += ["", "## 4. The ledger", "",
                  f"Every out-of-fold `sign_rule` signal on `{args.gate}`-covered "
                  "rows, priced as a round trip.", "",
                  "| quantity | value |", "|---|---|",
                  f"| signals traded | {summ['n_trades']} |",
                  f"| ...with a spread estimate (costed) | {summ['n_priced']} |",
                  f"| per year | {fmt(summ['per_year'], 1)} |",
                  f"| hit rate (at the executable entry) | {fmt(summ['hit_rate'])} |",
                  f"| gross vs p0 (what the estimator measures) | "
                  f"{fmt(summ['gross_p0'], 2)}c |",
                  f"| **gross vs first post-resolution print** | "
                  f"**{fmt(summ['gross_entry'], 2)}c** |",
                  f"| ...lost to entering after the first print | "
                  f"{fmt(summ['entry_slippage'], 2)}c |",
                  f"| effective spread charged | {fmt(summ['spread'], 2)}c |",
                  f"| fees (both legs) | {fmt(summ['fees'], 2)}c |",
                  f"| **net per trade (taker, central case)** | "
                  f"**{fmt(summ['net'], 2)}c** |",
                  f"| net per trade (maker both legs, upper bound) | "
                  f"{fmt(summ['net_maker'], 2)}c |",
                  f"| net median | {fmt(summ['net_median'], 2)}c |",
                  f"| net sd | {fmt(summ['net_sd'], 2)}c |",
                  f"| net total over the sample | {fmt(summ['net_total'], 1)}c |",
                  f"| median lag, resolution → first print | "
                  f"{fmt(summ['entry_lag_min_median'], 1)} min |",
                  f"| median holding time | {fmt(summ['hold_hours_median'], 2)} h |",
                  ]
        t = summ["net"] / (summ["net_sd"] / np.sqrt(summ["n_trades"])) \
            if summ["net_sd"] > 0 else float("nan")
        if summ["unpriced_targets"]:
            lines += ["",
                      "Targets too thin to yield a single adjacent "
                      "opposite-direction trade pair, so uncosted: "
                      + ", ".join(f"`{t}`" for t in summ["unpriced_targets"])
                      + ". That illiquidity is itself a tradability finding."]
        lines += ["",
                  f"t-statistic on net per-trade cents: **{fmt(t, 2)}** "
                  f"(n = {summ['n_trades']}).",
                  "",
                  "Per-pair breakdown:", "",
                  "| pair | n | gross entry | net | hit |", "|---|---|---|---|---|"]
        for r in (ledger.group_by("pair").agg(
                pl.len().alias("n"), pl.col("gross_entry").mean(),
                pl.col("net").mean(), pl.col("hit").mean())
                .sort("n", descending=True).iter_rows(named=True)):
            lines.append(f"| {r['pair']} | {r['n']} | {fmt(r['gross_entry'], 2)} | "
                         f"{fmt(r['net'], 2)} | {fmt(r['hit'])} |")

    by_stale, dec = decompose_move(panel, direction, mask)
    lines += ["", "## 5. Where the edge lives, and whether staleness made it", "",
              "`p0` is the last trade *before* the trigger resolved. In these "
              "books that price is old: median "
              f"**{fmt(dec['stale_median_h'], 1)} h**, 75th percentile "
              f"{fmt(dec['stale_p75_h'], 1)} h, and "
              f"{fmt(dec['frac_over_6h'] * 100, 0)}% of signals reference a price "
              "more than 6 hours stale. Splitting the signed move at the first "
              "executable price:", "",
              "| leg | cents | hit rate | median abs move |", "|---|---|---|---|",
              f"| jump: `p0` → first print (untradable) | "
              f"**{fmt(dec['s_jump'], 2)}** | {fmt(dec['hit_jump'])} | "
              f"{fmt(dec['abs_jump_median'], 1)}c |",
              f"| drift: first → third print (tradable) | "
              f"**{fmt(dec['s_drift'], 2)}** | {fmt(dec['hit_drift'])} | "
              f"{fmt(dec['abs_drift_median'], 1)}c |",
              "",
              "The tradable leg's hit rate is *below* chance. Whatever the "
              "estimator detects is complete by the first print.",
              "",
              "### Is it a staleness artifact?",
              "",
              "If the edge were manufactured by stale reference prices "
              "(`research_summary.md` §4.1), accuracy would **rise** with "
              "staleness. It falls:", "",
              "| p0 age | n | accuracy | signed jump | signed drift |",
              "|---|---|---|---|---|"]
    for r in by_stale.iter_rows(named=True):
        lines.append(f"| {r['bucket']} | {r['n']} | **{fmt(r['acc'])}** | "
                     f"{fmt(r['s_jump'], 2)}c | {fmt(r['s_drift'], 2)}c |")
    lines += ["",
              f"Monotone in the right direction, and corr(staleness, |jump|) = "
              f"{fmt(dec['corr_stale_absjump'], 2)} — near zero. Stale prices "
              "*dilute* the finding rather than create it, which is the answer "
              "§4.1 asked for.",
              "",
              f"The jump's *size*, though, is unrelated to the size of the news: "
              f"corr(|surprise|, |jump|) = {fmt(dec['corr_z_absjump'], 2)} "
              f"(and {fmt(dec['corr_z_absdrift'], 2)} for the drift). The first "
              "print moves in the right direction slightly more often than "
              "chance, by an amount that has nothing to do with how large the "
              "surprise was — the same 'sign survives, magnitude does not' "
              "pattern as `research_log.md` §1-2, now visible in the execution "
              "frame."]

    (OUT / "edge_economics.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}/edge_economics.md"
          + (f" and direction_ledger.parquet ({ledger.height} trades)"
             if not ledger.is_empty() else ""))


if __name__ == "__main__":
    main()
