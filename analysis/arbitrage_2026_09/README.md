# Arbitrage-shaped tests: cross-market structure without forecasting (2026-09-15)

Four studies that do **not** depend on predicting anything. The reason they are
worth doing is stated in `reports/arbitrage_findings.md` §0: every other result
in this repo is bottlenecked on `n` (median 9 target events per ordered pair),
and a coherence violation is a per-observation fact rather than a statistical
estimate.

    venv/bin/python analysis/arbitrage_2026_09/identity_cpi.py   # MoM vs YoY, through recovered moments
    venv/bin/python analysis/arbitrage_2026_09/identity_arb.py   # MoM vs YoY, strike-matched (needs identity_cpi first)
    venv/bin/python analysis/arbitrage_2026_09/coherence.py      # monotonicity + bucket sum-to-one
    venv/bin/python analysis/arbitrage_2026_09/vol_term.py       # implied-uncertainty term structure
    venv/bin/python analysis/arbitrage_2026_09/dispersion.py     # headline vs components

`identity_arb.py` reads the parquet `identity_cpi.py` writes; the rest are
independent. Captured output is in `out/`.

| script | question | answer |
|---|---|---|
| `identity_cpi.py` | Do the MoM and YoY ladders imply the same distribution? | Same location (wedge 0.046pp, at the noise floor); YoY ~21% wider. |
| `identity_arb.py` | Is that wedge arbitrage? | **No.** Break rate 12.8% from independent BLS rounding; net −1.07c. The 21% width gap is *less* than the 34% the rounding noise justifies. |
| `coherence.py` | Are ladders internally arbitrage-free? | ~2% of adjacent pairs violate monotonicity, flat across synchronicity tiers. Median 2.0c vs 2.3c cost; 34 genuinely tradable, 16 in FED. |
| `vol_term.py` | Does implied uncertainty carry structure? | −14% decay into the release; no systematic variance premium; cross-event effect dies once controlled for that decay. |
| `dispersion.py` | Is headline variance consistent with components? | 6% of days headline is arithmetically too narrow for its own core ladder. Weakly identified — hand-entered weights. |

## Two corrections this directory makes

* **The "124c bucket overround"** in `settlement_distribution_findings.md` is
  withdrawn. Median ladder mass is 111c at day granularity, **70c within 60
  minutes**, 74c within 5. Loose windows are stale, tight windows incomplete;
  trade prints cannot measure overround, only quotes can.
* **The "YoY ladder is 21% too wide"** reading from `identity_cpi.py` §2 is
  wrong, and `identity_arb.py` §5 shows why: rounding noise predicts 34%.

## Data note

`dispersion.py` hard-codes BLS relative importances (core ≈ 0.79, gasoline ≈
0.034) because `data/` carries no weights file — `research_summary.md` §3 asset
2 lists fetching them as outstanding. Every dispersion result is reported across
a weight sensitivity band for that reason. Replace `W` before citing.

Everything is in-sample only.
