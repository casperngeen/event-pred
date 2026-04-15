from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventPairsConfig:
    # universe
    min_event_days: int = 10
    top_events: int = 300

    # selection
    lookback_sel: int = 20
    beta_lookback: int = 10
    z_window: int = 10

    corr_min: float = 0.20
    corr_max: float = 0.95
    half_life_min: float = 1.0
    half_life_max: float = 10.0

    top_pairs: int = 50

    # trading/backtest
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_hold: int = 20
    cost_per_pair_turn: float = 0.0
    