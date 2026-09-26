"""
model/forecast_head.py

THE HEAD THAT CAN ACTUALLY EXPRESS A DISLOCATION.

The existing heads cannot, and that is not a tuning problem:

  MeceOutputHead applies a grouped softmax, so its predicted leg prices
  sum to exactly $1.00 by construction. The quantity the project exists
  to predict -- a basket summing to $1.06 because one leg repriced on
  news and its siblings have not caught up -- is the one output that head
  is incapable of producing. Worse, it makes the model's basket-level
  deviation IDENTICALLY equal to the naive rule's:
      sum(predicted) == 1  =>  sum(observed - predicted) == sum(observed) - 1

  LadderOutputHead guarantees fair_a >= fair_b, so within a single
  forward pass its implied gap is <= 0 and it can never represent a
  monotonicity violation either.

The fix is to stop putting the mechanism in the HEAD and leave it in the
GRAPH, which is where a spatio-temporal model is supposed to carry it.
MECE membership and ladder adjacency are already edges; the attention
layers learn how a price shock on one node should propagate to its
structural neighbours. The head's only job is then to say what each node
will be worth, and the dislocation is a DERIVED quantity:

    basket dislocation(t+h) = sum_legs predicted_price(t+h) - 1
    ladder gap(t+h)         = predicted_B(t+h) - predicted_A(t+h)

Neither is constrained, so both can be non-zero, which is the whole point.

RESIDUAL PARAMETERISATION, AND WHY IT IS NOT COSMETIC.

    predicted_price(t+h) = observed_price(t) + delta(t, h)

Measured on this dataset: 90% of consecutive snapshot observations show
NO price change at all, and the leg's own last price beats the trained
reconstruction model by 4.4x on test (7.9x on val). A model that predicts
the price LEVEL spends its capacity relearning "the price is roughly what
it was", and the best solution to that objective is to echo the last
print -- which is exactly the staleness the project is trying to see
through.

Predicting the residual changes what is being learned. The final layer is
ZERO-INITIALISED, so the model starts life as the persistence baseline
exactly, and every subsequent change is a learned correction to it. Three
consequences:

  1. it cannot score worse than persistence by construction (delta = 0);
  2. any measured gain is unambiguously attributable to the graph;
  3. the learned quantity IS the mispricing -- how far a leg will move
     away from its last print -- rather than the print itself.

No clamping during training: prices are probabilities in [0, 1], but
clamping there kills gradients at the boundary exactly where large
dislocations live. Clamp at inference instead (`clamp=True`).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualForecastHead(nn.Module):
    """Per-node price forecast at several horizons, as a residual on the
    node's currently observed price. Unconstrained by design."""

    def __init__(self, embed_dim: int, horizons=(1, 2, 3, 6, 12),
                 hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        if not horizons:
            raise ValueError("horizons must be non-empty")
        self.horizons = tuple(int(h) for h in horizons)
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive, got {self.horizons}")
        hidden = hidden or max(embed_dim, 2 * embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, len(self.horizons)),
        )
        # START AS PERSISTENCE. Zeroing the final layer makes delta == 0 at
        # initialisation, so epoch 0 reproduces "the price stays where it
        # is" exactly -- the baseline that beats the current model 4.4x.
        # Training can then only move away from it by reducing loss.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, h_final: torch.Tensor, base_price: torch.Tensor,
                clamp: bool = False) -> torch.Tensor:
        """h_final (T, N, D), base_price (T, N) on the 0-1 scale.
        Returns (T, N, H) predicted prices, one per horizon."""
        if h_final.shape[:2] != base_price.shape:
            raise ValueError(
                f"shape mismatch: h_final {tuple(h_final.shape)} vs "
                f"base_price {tuple(base_price.shape)} -- base_price must be "
                f"(T, N) on the same chunk.")
        delta = self.net(h_final)                       # (T, N, H)
        pred = base_price.unsqueeze(-1) + delta
        return pred.clamp(0.0, 1.0) if clamp else pred

    def deltas(self, h_final: torch.Tensor) -> torch.Tensor:
        """The learned correction alone, (T, N, H). This is the model's
        actual claim -- 'this leg will move by X' -- and the quantity to
        inspect when asking whether it has learned anything beyond
        persistence, for which every delta would be 0."""
        return self.net(h_final)