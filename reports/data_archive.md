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

The archive was created on 2026-11-09. Upload to the folder above is done by hand (Drive for desktop or the web page) and is not yet confirmed here. Originals stay on the Mac until it is.

## Live results (VM, 30 Oct - 8 Nov)

The VM runner was still at 1x sizing (`CONTRACTS_K = None` in its `live_trade.py`; the repo has 20). 101 games: about -1.7 c/contract in total at 1x (NBA spread -0.97, NBA total -1.02, NFL spread +0.07, NFL total +0.33 dollars; 93 contracts). These are not final-sizing results. Pull with `./live_pull.sh`; a real 20x figure needs a replay of these games, which needs their books and trades pulled from the VM (about 30 GB before dedup).
