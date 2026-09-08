"""Stage 2: dormant-horizon direction prediction on the Stage-1 structure.

The simple-learning successor to the AGCRN study — same graph, same edges,
but the label is the *sign* of the target's next-print move rather than a
weeks-ahead magnitude, and the models are a transparent ladder rather than a
sequence network. See ``reports/direction_study.md``.
"""

from stg.direction.dataset import build_pair_panel, pair_counts
from stg.direction.folds import Fold, walk_forward
from stg.direction.structure import FoldStructure, fit_structure
from stg.direction.learners import (
    BaseRate, Learner, Logit, SignRule, design, ladder, neighbour_signal,
)
from stg.direction.tradability import (
    decompose_move, effective_spread, ledger_summary, trade_ledger,
)
from stg.direction.evaluate import (
    coverage_composition, edge_stability, gate_coverage, predict_oof, run_ladder,
)

__all__ = [
    "build_pair_panel", "pair_counts",
    "Fold", "walk_forward",
    "FoldStructure", "fit_structure",
    "Learner", "BaseRate", "SignRule", "Logit", "ladder", "design",
    "neighbour_signal",
    "effective_spread", "trade_ledger", "ledger_summary", "decompose_move",
    "run_ladder", "edge_stability", "coverage_composition", "predict_oof",
    "gate_coverage",
]
