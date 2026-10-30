# event-pred

NUS final-year project: do Kalshi sports prediction markets (MLB, NBA, NFL) react to play-by-play events with a delay that a graph model can trade, and what is left after executable prices and fees?

## Layout

| Path | What it holds |
|---|---|
| `stg_infra/stg/` | The `stg` package: `stg.sports` (as-of data layer) and `stg.models.ingame` (GRU graph model). Tests in `stg_infra/tests/`. |
| `scripts/` | Data pulls, sports layer build, live paper-trading runner (`live_*.py`), exchange simulator (`event_execution.py`). |
| `analysis/graph_2026_10/` | The in-game studies: ablations, strategy study, event replay. |
| `cluster/` | Slurm scripts for training on the SoC GPU cluster (`cluster/README.md`). |
| `deploy/` | Daily live-run deployment (Lumid sandbox, backup VM). |
| `reports/` | Journals and results; `ingame_journal.md`, `holdout_spec.md`, `ablation_results.md`, `forward_test_log.md`. |
| `thesis/` | LaTeX source (`main.tex`). |

## Setup

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -e stg_infra
```

Parquet is read with `fastparquet` (no `pyarrow`). Kalshi access uses `KALSHI_KEY_ID` and `kalshi_api_key.txt`.

## Workflow

```bash
venv/bin/python scripts/fetch_sports.py list     # Kalshi markets, candles, trades (stages in its docstring)
venv/bin/python scripts/fetch_pbp.py             # play-by-play
venv/bin/python scripts/build_sports_layer.py    # as-of layer
# train the in-game model on the cluster: see cluster/README.md
venv/bin/python scripts/live_trade.py ...        # paper trade a day's games
venv/bin/python scripts/live_replay.py ...       # replay a past game
```

Arguments are in each script's docstring.

## In-game model and live run

- `live_model.py` rebuilds the model inputs from raw feeds every 5 s and averages five seeds' forecasts.
- `live_trade.py` forecasts, applies the frozen rule (`reports/holdout_spec.md`), keeps a paper book and publishes decisions.
- `live_books_ws.py` records the order book; `live_exec.py`, `event_execution.py` and `live_queue.py` simulate execution (exchange-clock ordering, queue position, fee ledger).
- `live_parity.py`, `live_rerun.py`, `live_rerun_nba.py` and `live_replay.py` check the live code against the training panel and re-run recorded games.

All results are reported at the final sizing: contracts = 20 × weight, rounded to the nearest 0.01. Gross P&L at executable prices comes first, then fees.

Heavy jobs over `data/` must stream files and stay within the laptop's 16 GiB; see `CLAUDE.md`.
