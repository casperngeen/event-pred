# Memory constraints before running any script

The laptop has 16 GiB RAM. A script that concatenated about 9,900 trade parquet files exhausted it on 2026-10-30. Check memory before running a script, not after.

- Before running anything over `data/` (sports trades ~1.8 GB on disk, `data/sports` 43 GB, `data/sports_live` 35 GB), estimate peak RAM. Parquet expands several times when loaded, and string columns (`trade_id`, `ticker`) become Python objects.
- Never `pd.concat` every file in a directory. Stream one file at a time, read only the needed columns, and keep a small running aggregate (Counter, histogram, running sums).
- Do not run `drop_duplicates`, `merge` or `groupby` on a full multi-file table. Aggregate per file first, or dedupe on a compact key.
- The venv has `fastparquet` and no `pyarrow` or `duckdb`. Use `pd.read_parquet(f, columns=[...])`.
- Run any job that might exceed about 2 GiB in the background under `/usr/bin/time -l` and report its "maximum resident set size". `ulimit -v` does not work on macOS.
- Test on a handful of files first and extrapolate peak memory to the full set.
- Live hosts (sandbox 8 GiB, VM 7 GiB) must never run heavy jobs. Replays run on the Mac (see the live-host memory note).
- If the estimate is near or above 4 GiB, say so and ask before starting.

# Reporting sizing (final)

Every live run, future simulation and thesis result is reported at the final sizing: contracts = 20 x weight, rounded to the nearest 0.01 (`event_execution.SIZE_X`, `live_trade.CONTRACTS_K`). Do not report 1x or 5x numbers as results; label them as comparison arms if they appear. Justification used in the thesis: 20 contracts at full weight is a little below the historical median trade size (all sports trades, median 35; spreads/totals 25-48 by league), so orders are no larger than a typical trade.
