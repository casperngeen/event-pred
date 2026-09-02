"""Stage 1: direct structure estimation — the validated adjacency."""

from stg.structure.estimator import estimate_adjacency, adjacency_matrix
from stg.structure.mediation import test_mediation, mediation_grid
from stg.structure.horizon import estimate_by_horizon
from stg.structure.stats import (
    spearman, spearman_p, permutation_p, partial_spearman,
    benjamini_hochberg, bh_critical, block_permutation_sign_p,
)

__all__ = [
    "estimate_adjacency", "adjacency_matrix",
    "test_mediation", "mediation_grid", "estimate_by_horizon",
    "spearman", "spearman_p", "permutation_p", "partial_spearman",
    "benjamini_hochberg", "bh_critical", "block_permutation_sign_p",
]
