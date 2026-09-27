# Spillover beyond releases, 2026-09

This study adds more shocks to the macro graph than the 154 Kalshi release
instants: unscheduled price jumps, signed order-flow bursts, and 34 scheduled
releases that have a consensus but no Kalshi market. It then asks whether the
extra shocks make the graph learnable. Writeup: `reports/spillover_findings.md`.
In-sample only.

    venv/bin/python -W ignore analysis/spillover_2026_09/jumps.py > analysis/spillover_2026_09/out/jumps.txt                 # ~2 min
    venv/bin/python -W ignore analysis/spillover_2026_09/predictive.py > analysis/spillover_2026_09/out/predictive.txt       # ~40 s, reads jumps
    venv/bin/python -W ignore analysis/spillover_2026_09/dose.py > analysis/spillover_2026_09/out/dose.txt                   # ~5 s, reads jumps
    venv/bin/python -W ignore analysis/spillover_2026_09/flow.py > analysis/spillover_2026_09/out/flow.txt                   # ~1 min
    venv/bin/python -W ignore analysis/spillover_2026_09/releases.py > analysis/spillover_2026_09/out/releases.txt           # ~40 s
    venv/bin/python -W ignore analysis/spillover_2026_09/calendar_learn.py > analysis/spillover_2026_09/out/calendar_learn.txt   # ~15 s, reads releases
    venv/bin/python -W ignore analysis/spillover_2026_09/augmented.py > analysis/spillover_2026_09/out/augmented.txt         # ~15 min, reads releases
    venv/bin/python -W ignore analysis/spillover_2026_09/channel.py > analysis/spillover_2026_09/out/channel.txt             # ~2 min, imports augmented

`releases.py` reads the economic-calendar cache
`data/external/econ_calendar_us_2021q4_2025.parquet` (lum.id findata, 2021-10
→ 2025-12). It was fetched by `analysis/relations_2026_09/consensus_surprise.py`.
`augmented.py` also needs the event-time panel
(`analysis/event_time_2026_09/build_panel.py`) and imports its `bayes.py`.

| file | what it does |
|---|---|
| `_tape.py` | Not a script. Loads the threshold legs of the 17 series with a VWAP price and signed order flow per print, plus the macro release/settlement times. Also holds the shared response helpers: lead contract (ties broken by ticker, deterministically), as-of price, post-shock response, placebo time. |
| `jumps.py` | Persistent 15-minute jumps of ≥ 5/10/15/20¢ away from releases. The theory-signed response of every other series vs a placebo time, pooled, by channel and by pair (BH). |
| `predictive.py` | Walk-forward test on the jumps: the direction of the target's move, learned per pair or per channel from earlier jumps, against the theory direction and flipped-sign controls. |
| `dose.py` | Uses every ≥ 5¢ jump weighted by its size: does the placebo-adjusted response grow with the jump? |
| `flow.py` | Signed order-flow bursts (top 5% and 1% of a series' hourly net flow) as shocks. Responses of the series itself and of the others, vs placebo. |
| `calendar_triggers.py` | Not a script. The 34 calendar releases used as extra triggers, each with its family and hawkish sign. |
| `releases.py` | Those releases' standardised consensus surprises, kept only when more than 1 h from any Kalshi release. The theory-signed Kalshi responses at +1/4/24 h against a sign-flip null, by family × target type and by trigger. |
| `calendar_learn.py` | Walk-forward: can the direction of each release → market edge be learned without theory? Plus the full-sample learned signs against theory. |
| `augmented.py` | Kalshi + calendar instants in one design (776 edges). Theory rule, one slope, free edge graph, rank-4 low-rank graph and per-edge soft-sign Bayes, each trained with and without the calendar data and scored on the same test cells. Importable. |
| `channel.py` | Learns at channel level instead: 15 free-sign channel coefficients, and 68 source-type × target coefficients with no edge signs, on `augmented.py`'s cells. Also the full-sample channel coefficients. |
