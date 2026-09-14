# Settlement distribution — exploration (2026-09-14)

Working directory for `reports/settlement_distribution_plan.md`. Item 1 (the gate) is **run**; the writeup is
`reports/settlement_distribution_findings.md`.

The programme changes the target of prediction: stop forecasting the price path
(`m_{t+δ} − m_t`, which surprise was built for and which `research_log.md` §13
and `relations_findings.md` Addendum 2 have closed), and forecast the settling
statistic instead — a predictive distribution `p̂ = P(X_T > K) = Φ(−z)`,
`z = (K − μ̂_T)/σ̂_T`.

Run order, per plan §7. Item 1 was the gate; it has now been run and the
answer reorders everything after it — see `reports/settlement_distribution_findings.md`.

    venv/bin/python analysis/settlement_dist_2026_09/wing_calibration.py   # item 1, the gate
    venv/bin/python analysis/settlement_dist_2026_09/wing_overround.py     # is the bucket edge just overround?
    venv/bin/python analysis/settlement_dist_2026_09/wing_robustness.py    # four attempts to kill it

`wing_overround.py` reads the parquet `wing_calibration.py` writes, and
`wing_robustness.py` reads the one `wing_overround.py` writes, so run them in
order.

| script | question | answer |
|---|---|---|
| `wing_calibration.py` | Do legs priced 3-20c / 80-97c settle at their price? No reconstruction, year-split, clustered on `event_ticker`. | Threshold ladders yes (fair, -0.13c). Bucket ladders no (-6.6pp, +4.96c). |
| `wing_overround.py` | Is the bucket edge just the ladders summing to 124c instead of 100c? | No — survives mass-normalisation and the coherent-ladder subset. |
| `wing_robustness.py` | Missing winner, event concentration, regime, single series. | Survives all four. Decaying by year, though. |

Captured output is in `out/`.

## Notes carried in from the relations study

- `settlement_trade.py` in `analysis/relations_2026_09/` is the right shape to
  adapt for item 1 — real traded prices, real outcomes, hold to settlement, no
  pdf recovery anywhere.
- `implied.py:527 build_daily_implied_means` already emits `implied_std` per
  event per day, so the **market's** `σ` convergence schedule is a column.
- But `implied_std` is biased **down** until the ladder-coverage defect is
  fixed (`relations_findings.md` item 1: coverage 0.64-0.83, missing strikes
  disproportionately far-from-the-money). Anything stated in `σ` units is
  provisional until then. That fix is plan §7 item 2.

Everything is in-sample only; every script must route the wall through
`stg.splits.assert_no_oos`.
