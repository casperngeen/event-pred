# `analysis/` — one directory per study

Each subdirectory is a self-contained study: standalone scripts, a `README.md`
indexing them, and a writeup in `reports/`. Scripts are run from the repo root
with `venv/bin/python`, never imported.

| directory | scripts | writeup |
|---|---|---|
| `exploratory_2026_08` | 27 | `reports/research_log.md` — **frozen, see below** |
| `horizon_2026_09` | 5 | `reports/research_log.md` §13 |
| `data_quality_2026_09` | 1 | `reports/research_log.md` §12 |
| `agcrn_diagnostics_2026_09` | 1 | `reports/agcrn_postmortem.md` |
| `relations_2026_09` | 9 | `reports/relations_findings.md` |
| `quantile_2026_09` | 3 | `reports/quantile_findings.md` |
| `arbitrage_2026_09` | 5 | `reports/arbitrage_findings.md` |
| `settlement_dist_2026_09` | 4 | `reports/settlement_distribution_findings.md` |
| `leadlag_2026_09` | 18 | `reports/leadlag_findings.md`, `reports/strategy_spec.md`, `reports/liquidity_findings.md` |

## Output convention

Scripts print to stdout and write intermediate parquet. Both go to `<dir>/out/`,
which is **untracked**:

    venv/bin/python analysis/<dir>/<script>.py > analysis/<dir>/out/<script>.txt

`out/` is a working directory, not a deliverable. Several scripts read the
parquet another one leaves there — `identity_arb.py` reads `identity_cpi.py`'s,
`confirmation_stability.py` reads `maker_fill.py`'s, `longshot_stability.py`
reads across directories — so each `README.md` states its own run order. A
fresh clone has no `out/`; run the producers first.

## Why the captures are not committed

The conclusions live in each directory's `README.md` answer table and in
`reports/`, which carry the supporting tables inline. The captures add nothing
a reader can cite, and they are actively misleading as an audit trail:

- **No provenance.** A capture records the output, not the date, the commit or
  the command. Nothing ties one to the version of the script that produced it.
- **They go stale silently.** No test fails when a script changes and its
  capture does not.
- **Some are lossy.** Polars elides columns past a width, so captures under the
  `2026_09` studies print `shape: (6, 15)`, then eight columns and a `…`.

So a stale capture is worse than no capture: it reads like evidence. Rerun the
script instead — that is the only thing that produces a citable number.

If a captured run ever does need to be preserved against a specific commit, the
fix is a provenance header (date, `git rev-parse HEAD`, argv) printed by the
script itself, not a committed `.txt`.

## The one exception: `exploratory_2026_08/frozen_runs/`

That study is frozen and **cannot be rerun** — its scripts `sys.path.insert` a
directory outside this repo that no longer exists, and `fetch_kalshi_data.py` is
gone. "Rerun the script" is not available, so the captures are the only record
of the August 2026 measurement work and are committed under `frozen_runs/`
rather than `out/`. The name is the reason: they are kept because they are
irreproducible, not because captures are worth committing in general.
