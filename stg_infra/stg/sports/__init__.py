"""The sports arm's as-of data layer (reports/sports_research_plan.md §4, Phase 0).

    loaders     holdout-guarded readers; every row stamped with ``known_at``
    contracts   what each market is: league, kind, team, strike, game, season
    outcomes    one outcome per event: complements merged, ladders flagged, labels
    prices      per-market bid/ask series (trades now, candles when pulled), mids
    asof        ``SportsData`` / ``SportsData.as_of(t)``, and bulk ``quotes_at``
    snapshots   pre-game snapshots (24 h / 1 h / 0 h before the actual start)
    features    as-of feature groups (§9)
    edges       the edge table (§8) and team x kind graphs
    checks      truncation test, known_at assertion, final-column rule

Build the derived layer first: ``venv/bin/python scripts/build_sports_layer.py``.
"""
