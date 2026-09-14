# Settlement distribution — exploration (2026-09-14)

Working directory for `reports/settlement_distribution_plan.md`. **Empty so
far** — the design doc is written, no script has been run.

The programme changes the target of prediction: stop forecasting the price path
(`m_{t+δ} − m_t`, which surprise was built for and which `research_log.md` §13
and `relations_findings.md` Addendum 2 have closed), and forecast the settling
statistic instead — a predictive distribution `p̂ = P(X_T > K) = Φ(−z)`,
`z = (K − μ̂_T)/σ̂_T`.

Run order, per plan §7. Item 1 is the gate: if the wings are flat, nothing after
it is worth building.

    venv/bin/python analysis/settlement_dist_2026_09/wing_calibration.py   # item 1  (not written)

| script | item | question |
|---|---|---|
| `wing_calibration.py` | 1 | Do legs priced 3-20c / 80-97c settle at their price? No reconstruction, year-split, clustered on `target_event`. |

Captured output goes in `out/`.

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
