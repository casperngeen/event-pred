# Sports arm, MLB 2025 pilot

Game results as timestamped surprises, and team futures as targets, in a
graph whose structure is known: self +, division rival −. Separate from the
macro study, on branch `casper/sports-graph`. It reads only the in-sample
archive (`data/markets/`, `data/trades/`) and writes only to `out/`
(git-ignored). Design: `reports/sports_study_plan.md`. Writeup:
`reports/sports_study.md`.

    venv/bin/python -W ignore analysis/sports_2026_09/build_mlb.py > analysis/sports_2026_09/out/build_mlb.txt   # ~10 s
    venv/bin/python -W ignore analysis/sports_2026_09/diffusion.py > analysis/sports_2026_09/out/diffusion.txt   # ~5 s
    venv/bin/python -W ignore analysis/sports_2026_09/graphs.py > analysis/sports_2026_09/out/graphs.txt         # ~3 min
    venv/bin/python -W ignore analysis/sports_2026_09/predict.py > analysis/sports_2026_09/out/predict.txt       # ~1 min
    venv/bin/python -W ignore analysis/sports_2026_09/signfix.py > analysis/sports_2026_09/out/signfix.txt       # ~1 min

As-of rerun (Phase 0 of `reports/sports_research_plan.md`): `build_mlb_asof.py`
rebuilds the four tables through `stg.sports` into `out/asof/`; the other
scripts read them with `SPORTS_PILOT_OUT`, and `SPORTS_PILOT_ASOF=1` makes
`graphs.py` / `signfix.py` fit fill values and the label scale on each fold's
days only. `out/fixonly/` is the original tables with just that switch.

    venv/bin/python scripts/build_sports_layer.py                                   # the data layer, once
    venv/bin/python -W ignore analysis/sports_2026_09/build_mlb_asof.py > analysis/sports_2026_09/out/asof/build_mlb.txt
    SPORTS_PILOT_OUT=analysis/sports_2026_09/out/asof SPORTS_PILOT_ASOF=1 venv/bin/python -W ignore analysis/sports_2026_09/graphs.py > analysis/sports_2026_09/out/asof/graphs.txt
    # likewise diffusion.py, predict.py, signfix.py

| script | what it does |
|---|---|
| `build_mlb_asof.py` | The same tables through the as-of layer: actual first pitch, p_pre from trades known before it, results from the final score. |
| `build_mlb.py` | Builds the game table: detected first pitch, pre-game win probability from both team markets, surprise z = 1[won] − p_pre. Also every futures trade (championship, pennant, playoffs, division) with canonical team codes, team → league/division read from the tickers, and a slate-day × team panel of futures prices. |
| `diffusion.py` | How fast a result reaches the futures: the price response to z during the game, at the first post-game print, and at +1/3/12/24 h, for the team's own contracts and its division rivals'. |
| `graphs.py` | Scores graphs against the known one (AUROC, rival-only AUROC, balanced sign accuracy): entity match and title embedding and title co-occurrence (text), price correlation, and graphs learned from surprise → response (pairwise, ridge, lasso, diagonal + rank-6, AGCRN adaptive/signed, and with the truth frozen in), fitted on growing parts of the season. Importable. |
| `predict.py` | Walk-forward: each graph trades the post-game drift of division contracts on its 150 strongest edges. Gross P&L against the mechanical graph. |
| `signfix.py` | Why the standard AGCRN gets no rival (negative) edge signs, and the fix: the schedule graph from titles as a frozen, unsigned second support, so the conv weights learn the rival sign (0.00 → 0.89 of rival signs). Also tests a signed adjacency restricted to schedule edges (no gain). |
