# Data archive (Google Drive)

`data/` and `analysis/graph_2026_10/out/` are not in git. The recordings below cannot be re-fetched from Kalshi or ESPN, and the thesis tables were built from them, so they are archived to the Drive folder
https://drive.google.com/drive/folders/1Mmg8yVk7T9r7x4LU_-OYPAYF8aAbVlx6
(archive made 2026-11-09 on the Mac; zstd -15, tar; checksums in `SHA256SUMS` in the same folder).

| File | Size | Contents (paths inside the tar are relative to the repo root) |
|---|---|---|
| `sports_live.tar.zst` | 15.8 GB | `data/sports_live/`: live recordings from the sandbox and backup VM (`vm`, `sandbox`, `sandbox_nfl`, `sandbox_archive`), the pinned replay inputs (`event_replay_inputs`, `replay_inputs`, `replay_hist`), and the 24-28 Oct book extracts (`books_10-24` to `books_10-28`, `book_l1`) |
| `thesis_results.tar.zst` | 2.0 GB | `analysis/graph_2026_10/out/` replay results behind the thesis tables: `b55_main_20x`, `b57_*`, `b59_control`, `b60_baselines`, `b69_*`, `b70_*`, `b73_live_vm`, `thesis_sizing20x`, `thesis_unfiltered_signals` |
| `oos_sealed.tar.zst` | 0.16 GB | `data/_OOS_DO_NOT_USE_*` (sealed out-of-sample data) |
| `SHA256SUMS` | - | `shasum -a 256` of the three archives |

Restore (from the repo root; needs `zstd`):

```bash
shasum -a 256 -c SHA256SUMS
zstd -dc sports_live.tar.zst | tar -xf -                              # -> data/sports_live
zstd -dc oos_sealed.tar.zst | tar -xf -                               # -> data/_OOS_DO_NOT_USE_*
zstd -dc thesis_results.tar.zst | tar -xf - -C analysis/graph_2026_10/out
```

Not archived (re-fetchable or too large): `data/sports` (43 GB, Kalshi pulls via `scripts/fetch_sports.py`; includes the sealed `_HOLDOUT_DO_NOT_USE`, which must be kept from the original copy), `data/trades`, `data/markets`, `results/ingame` (model forecasts; recopy from the cluster).

## Deduplication (2026-11-09)

Games recorded on both hosts exist in several trees with different arrival stamps. One copy per game was kept: the copy named by the `root` in the `out/*/games/*/*/result.json` files, otherwise the largest. 191 other copies (21.4 GB) were deleted, so the other host's version of those games is gone. Replays must use the pinned roots (`b74_pin_inputs.py`).

## Status

The archive was created on 2026-11-09. On 2026-11-12 the user confirmed that all four files (three archives plus `SHA256SUMS`) were uploaded to the folder above. Drive contents could not be independently accessed in this session. A local copy of the checksum manifest is retained at `reports/data_archive_SHA256SUMS.txt`. The local compressed copies were located at `/Users/caspe2/fyp_archive`; all three passed checksum verification before the four files were deleted. Uncompressed live recordings, sealed out-of-sample data, research results, models and the original sports holdout remain local.

## Live results (VM, 30 Oct - 8 Nov)

The VM runner was still at 1x sizing (`CONTRACTS_K = None` in its `live_trade.py`; the repo has 20). 101 games: about -1.7 c/contract in total at 1x (NBA spread -0.97, NBA total -1.02, NFL spread +0.07, NFL total +0.33 dollars; 93 contracts). These are not final-sizing results. Pull with `./live_pull.sh`; a real 20x figure needs a replay of these games, which needs their books and trades pulled from the VM (about 30 GB before dedup).

## Local cleanup (2026-11-12)

Removed the working sports API caches `data/sports/{trades,candles,candles_futures_1m}` and their derived `data/sports/derived/quotes` cache, plus disposable Python/Jupyter caches and `.DS_Store` files outside the virtual environment. Working market metadata, play-by-play, training panels, model results and legacy market/trade archives were retained. The original `data/sports/_HOLDOUT_DO_NOT_USE` was preserved.

Working-market tickers were removed from the three `data/sports/state/*_done.txt` completion lists so the fetcher can download the deleted data again. Other completion entries were retained, including holdout entries. To restore the working caches using the retained market metadata:

```bash
venv/bin/python scripts/fetch_sports.py candles
venv/bin/python scripts/fetch_sports.py trades
venv/bin/python scripts/fetch_sports.py futures1m
venv/bin/python scripts/build_sports_layer.py
```

These calls can take substantial time and disk space. Kalshi documents markets, candles and trades across its live and historical APIs at https://docs.kalshi.com/getting_started/historical_data. Fresh API responses may differ from the original snapshots. No remote data was modified.
