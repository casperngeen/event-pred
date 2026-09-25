# Relations study — items 1-4, exploration (2026-09-14)

Four independent explorations of `reports/relations_study_plan.md` §4's ranked
list. Each runs standalone against the current panels and answers one question;
none depends on another's conclusion being adopted. Run order does not matter,
except that everything needs a panel carrying the PIT columns.

    venv/bin/python scripts/build_panels.py                              # adds pit/s_pit/surprisal
    venv/bin/python analysis/relations_2026_09/pit_calibration.py        # item 1
    venv/bin/python analysis/relations_2026_09/stage1_variants.py        # item 1b  (~40 min)
    venv/bin/python analysis/relations_2026_09/magnitude_xval.py         # item 2
    venv/bin/python analysis/relations_2026_09/family_collapse.py        # item 3
    venv/bin/python analysis/relations_2026_09/channel_pooling.py        # item 4

Each script writes its captured run to `out/`, which is untracked — rerun the
script to regenerate it. The writeup is `reports/relations_findings.md`, which
carries each script's summary tables inline.

| script | item | question |
|---|---|---|
| `pit_calibration.py` | 1 | Is the Kalshi macro ladder calibrated? |
| `stage1_variants.py` | 1b | Do the 4 Stage-1 edges survive a change of surprise measure? |
| `magnitude_xval.py` | 2 | Is magnitude unrecoverable, or was the estimator bad? |
| `family_collapse.py` | 3 | Collapse the CPI family; does `z_PAYROLLS - z_U3` beat its parts? |
| `channel_pooling.py` | 4 | Does the block model buy power over 141 per-pair parameters? |

Everything is in-sample only; every script routes the wall through
`stg.splits.assert_no_oos`.

## Production changes these depend on

- `stg/events/implied.py` — `pmf_bin_index`, `pit`, `surprisal`.
- `stg/panel/surprise.py` — `pit`, `s_pit`, `surprisal`, `implied_median`,
  `surprise_median` added to the panel schema and to both contract paths.
- `scripts/build_panels.py` — also writes `surprise_panel_ungated.parquet`,
  which the calibration study needs (the gates make the PIT near-uniform by
  construction).
- `stg_infra/tests/test_implied.py` — 6 cases covering the above.
