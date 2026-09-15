# Reconstruction-free ladder statistics (2026-09-15)

Writeup: `reports/quantile_findings.md`. Module under test:
`stg_infra/stg/events/quantile.py`, with `stg_infra/tests/test_quantile.py`.

    venv/bin/python -m pytest stg_infra/tests/test_quantile.py -q
    venv/bin/python analysis/quantile_2026_09/build_ladder_panel.py   # writes ladder_panel.parquet
    venv/bin/python analysis/quantile_2026_09/pit_direct.py           # item 1, re-tested
    venv/bin/python analysis/quantile_2026_09/response_vector.py      # A -> B, per moment

`build_ladder_panel.py` must run first; the other two read its parquet.

| script | question | answer |
|---|---|---|
| `build_ladder_panel.py` | Do quantile and integrated moments agree? | Location yes; width disagrees by series, `sigma_iqr/implied_std` running 0.85 (CPIYOY) to 1.17 (ADP). `q50` bracketed on only 43% of event-days. |
| `pit_direct.py` | Does item 1's calibration result survive without `recover_pdf`? | **No.** Mean PIT falls from 0.65-0.91 to 0.42-0.56; no series but FED rejects uniformity. Item 1's headline is withdrawn. |
| `response_vector.py` | Which moments of B move when A resolves? | Location yes (+0.0428 IQR, CI [+0.0121, +0.0741]); width no; skew no. |

## Why quantiles

`recover_pdf` integrates, so every moment inherits an assumption about strikes
that never traded — and 20-35% of listed strikes are absent, disproportionately
far-from-the-money. The quantile statistics are interior to the traded ladder by
construction: a unit test confirms that dropping both 5% wing legs leaves `iqr`
and `q50` unchanged. Where a ladder does not bracket a level the statistic is
`None` rather than an extrapolation — an explicit gate instead of a silent bias.

## Caveat that limits §1

19.9% of outcomes landed outside the traded strike range, and those are the
extreme ones by construction. So the PIT *location* result is clean, but width
and tail calibration are **not identified** from traded ladders alone. That
needs quotes, and the orderbook archive is sports-only from late 2025.

Everything is in-sample only.
