"""
model/training_objective.py

Masked-leg reconstruction training objective for the MECE mechanism.

WHY MASKING MUST HAPPEN AT THE INPUT LEVEL, NOT JUST BY HIDING THE LABEL:
predicting a masked leg's price from other legs' full h_final embeddings
collapses toward trivial arithmetic (predicted = 1 - sum(others)) if
those other legs' embeddings still contain the masked leg's TRUE
contemporaneous price -- which they would, since spatial attention mixes
every basket member's CONTEMPORANEOUS feature together at every
snapshot via the shared hub. Computing the whole stack normally and only
looking at the output for one node afterward does NOT remove this
shortcut, because the other nodes' embeddings were already contaminated
during spatial attention, before the output head ever runs.

THE FIX: replace the target leg's INPUT feature vector at the target
snapshot with a learned [MASK] token BEFORE running the stack, so no
node's embedding -- including the masked leg's own contemporaneous
neighbours, reached via spatial attention -- ever receives its true
value at that snapshot. The node's PRESENCE in the graph (and mask=True)
is left unchanged, so it still participates as a valid attention
neighbour; only its FEATURE CONTENT is hidden. This exactly mirrors
BERT's [MASK]-token convention (a hidden token, not a removed sequence
position), not a weaker "hide the output label only" scheme.

The [MASK] token only replaces the RAW per-type content slots
(_TICKER_RAW_FEATURES = indices 0-8), never the type-indicator slot
(index 9, from COMBINED_N_FEATURES in stg/nodes/kalshi.py) -- the model
is still allowed to know "this position is a ticker leg", it just isn't
allowed to know THIS ticker's actual observed values.

FRESH TARGETS ONLY: for individual ticker-leg nodes specifically,
mask[t, n] == True already IS freshness (KalshiTickerNodes has no
carry-forward, unlike KalshiMeceBasketNodes) -- so any currently-True
position is already a valid, non-stale reconstruction target with no
further filtering needed. Only positions with mask[t, n] == True are
ever selected for masking, for this reason.

LADDER MECHANISM COVERAGE: masking is defined once, mechanism-agnostic
(select_and_mask only cares "is this a fresh ticker leg", not which
mechanism it belongs to) -- a single masked position can therefore score
against BOTH the MECE head and the ladder head in the same training
step, if that leg happens to participate in both. This was originally
built MECE-only (compute_loss, kept below under its original name for
backward compatibility) despite the ladder channel getting the larger
share of engineering effort (bidirectional message-passing fix,
LadderOutputHead) -- compute_ladder_loss below closes that gap using the
exact same masked positions and the exact same true_prices tensor, so
adding it required no change to select_and_mask at all.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MaskedLegReconstructionObjective(nn.Module):
    """Orchestrates: pick real (fresh) leg observations to hide, replace
    their input with a learned [MASK] token, run the full stack, and
    compare the MECE head's prediction at those positions against the
    TRUE (pre-masking) observed price.
    """

    def __init__(self, raw_feature_dim: int, mask_ratio: float = 0.15):
        super().__init__()
        self.raw_feature_dim = raw_feature_dim  # _TICKER_RAW_FEATURES, e.g. 9 -- NOT the type-indicator slot
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(raw_feature_dim) * 0.02)

    def select_and_mask(
        self,
        features: torch.Tensor,   # (T, N, F) -- F = COMBINED_N_FEATURES
        mask: torch.Tensor,       # (T, N) bool
        node_types: torch.Tensor,  # (N,) or (T, N) -- 0.0 = ticker, 1.0 = basket; only tickers get masked
        generator: Optional[torch.Generator] = None,
    ):
        """Returns (masked_features, target_mask, true_values):
            masked_features: (T, N, F), a COPY of features with selected
                positions' raw content slots replaced by the [MASK] token
            target_mask: (T, N) bool -- True at exactly the positions chosen
                to be masked/predicted this step
            true_values: the ORIGINAL raw content (0:raw_feature_dim) at
                the target positions, i.e. what the loss compares against
        """
        T, N, F = features.shape
        is_ticker = (node_types == 0.0)
        if is_ticker.dim() == 1:
            is_ticker = is_ticker.unsqueeze(0).expand(T, N)

        eligible = mask & is_ticker  # only real, fresh ticker observations can be masking targets
        rand = torch.rand(T, N, generator=generator, device=features.device)
        target_mask = eligible & (rand < self.mask_ratio)

        masked_features = features.clone()
        true_values = features[target_mask][:, : self.raw_feature_dim].clone()
        masked_features[target_mask, : self.raw_feature_dim] = self.mask_token

        return masked_features, target_mask, true_values

    def compute_mece_loss(
        self,
        mece_head_output: dict,   # output of MeceOutputHead.forward(): {t: {'leg_idx', 'hub_idx', 'fair_price'} or None}
        target_mask: torch.Tensor,  # (T, N) bool
        true_prices: torch.Tensor,  # (T, N) float -- the true yes_price (0-1 scale) per node, e.g. features[...,0]/100
    ) -> torch.Tensor:
        """Gathers the MECE head's predicted fair_price specifically at the
        masked target positions and computes MSE against the true price.
        Positions that were masked but don't appear in the head's output
        for that snapshot (e.g. the rest of their basket wasn't otherwise
        active) contribute nothing -- there's no prediction to compare."""
        losses = []
        for t, res in mece_head_output.items():
            if res is None:
                continue
            targets_t = target_mask[t].nonzero(as_tuple=True)[0]
            if targets_t.numel() == 0:
                continue
            leg_idx = res["leg_idx"]
            for leg in targets_t.tolist():
                hit = (leg_idx == leg).nonzero(as_tuple=True)[0]
                if hit.numel() == 0:
                    continue  # this masked leg's basket wasn't in the head's output this snapshot
                pred = res["fair_price"][hit[0]]
                true = true_prices[t, leg]
                losses.append((pred - true) ** 2)
        if not losses:
            return torch.tensor(0.0, device=target_mask.device, requires_grad=True)
        return torch.stack(losses).mean()

    # Kept as an alias -- this was the objective's original (MECE-only)
    # name, and verify_combined_graph.py's masked-leg reconstruction check
    # already calls it under this name.
    compute_loss = compute_mece_loss

    def compute_ladder_loss(
        self,
        ladder_head_output: dict,   # output of LadderOutputHead.forward(): {t: {'leg_a_idx','leg_b_idx','fair_a','fair_b'} or None}
        target_mask: torch.Tensor,  # (T, N) bool -- SAME masked positions select_and_mask produced
        true_prices: torch.Tensor,  # (T, N) float -- SAME true-price tensor compute_mece_loss uses
    ) -> torch.Tensor:
        """Ladder-mechanism counterpart to compute_mece_loss, using the
        exact same masked positions (a leg doesn't need a second, separate
        masking pass just because it's being scored against a different
        head). A masked leg can appear as leg_a in some validated pairs and
        leg_b in others simultaneously (e.g. a middle rung of a ladder) --
        every occurrence on either side contributes its own squared-error
        term against that leg's single true price, exactly mirroring how
        compute_mece_loss lets one masked leg score against every basket
        it's a member of. A masked leg that appears in neither head's
        output this snapshot (not a validated ladder leg at all, or its
        partner wasn't active) simply contributes nothing here, same as
        the MECE side."""
        losses = []
        for t, res in ladder_head_output.items():
            if res is None:
                continue
            targets_t = target_mask[t].nonzero(as_tuple=True)[0]
            if targets_t.numel() == 0:
                continue
            leg_a_idx, leg_b_idx = res["leg_a_idx"], res["leg_b_idx"]
            for leg in targets_t.tolist():
                for hit in (leg_a_idx == leg).nonzero(as_tuple=True)[0].tolist():
                    losses.append((res["fair_a"][hit] - true_prices[t, leg]) ** 2)
                for hit in (leg_b_idx == leg).nonzero(as_tuple=True)[0].tolist():
                    losses.append((res["fair_b"][hit] - true_prices[t, leg]) ** 2)
        if not losses:
            return torch.tensor(0.0, device=target_mask.device, requires_grad=True)
        return torch.stack(losses).mean()
