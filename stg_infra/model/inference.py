"""
model/inference.py

Turns a trained STGATBackbone's predictions into an actual per-leg
DEVIATION SIGNAL -- "how far is this leg's observed market price from
the model's own context-only fair-value estimate for it" -- which is the
actual point of the whole project, not just a reconstruction loss to
train against.

WHY THIS REUSES THE MASKING MACHINERY, NOT A PLAIN FORWARD PASS: running
the stack on a leg's TRUE, unmasked features and reading the output
head's prediction for that same leg is close to a truism, not a signal.
TypeSpecificProjection + both attention layers' RESIDUAL connections mean
a leg's own embedding still substantially encodes its own true feature
vector, so the head's "prediction" would mostly echo the observed price
straight back -- the exact shortcut the masked-leg reconstruction
OBJECTIVE was built to prevent during training (training_objective.py's
docstring). At inference the target isn't different in kind: a fair-value
estimate that's actually INDEPENDENT of a leg's own current price (so a
gap between the two means something) requires withholding that leg's
input the same way, via the SAME learned [MASK] token training used --
NOT a fresh/random token, which would evaluate a differently-masked model
than the one actually trained. masked_forward() below is therefore a
deterministic (not randomly-sampled) call into the same masking path
select_and_mask() uses for training.

WHAT THIS FILE DELIBERATELY DOES NOT DO: decide whether a resulting
deviation is big enough to trade, size a position, define a holding
period, or compute PnL. That's a materially different question from "can
the model produce an honest, non-leaking fair-value estimate" (which is
what's verified here) -- see verify_training_loop.py-style checks in
verify_inference.py for exactly what is and isn't claimed at this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from model.train import STGATBackbone, true_prices


def masked_forward(
    model: STGATBackbone,
    features: torch.Tensor,               # (T, N, F)
    mask: torch.Tensor,                    # (T, N) bool
    adjacency_by_type: Dict[str, list],    # edge_type -> length-T list of SparseSnapshotEdgesT
    target_mask: torch.Tensor,             # (T, N) bool -- EXACTLY which positions to withhold
    mask_token: torch.Tensor,              # the TRAINED objective's mask_token, not a fresh one
    raw_feature_dim: int,
) -> torch.Tensor:
    """Replaces exactly the positions in ``target_mask`` with
    ``mask_token`` and runs the backbone. Deterministic: unlike
    select_and_mask (random 15% each training step), the caller decides
    exactly which legs it wants a fair-value read on -- for signal
    generation that's "every real, currently-active MECE/ladder leg", not
    a sample."""
    masked_features = features.clone()
    masked_features[target_mask, :raw_feature_dim] = mask_token
    return model(masked_features, mask, adjacency_by_type)


def eligible_targets(features: torch.Tensor, mask: torch.Tensor,
                      adjacency_by_type: Dict[str, list]) -> torch.Tensor:
    """(T, N) bool: every currently-active ticker leg that is a member of
    at least one validated MECE basket or ladder pair this snapshot --
    i.e. every leg a fair-value signal could mean something for. A pure
    filler ticker with no mechanism membership has no head to read a
    prediction from at all, so it's excluded here rather than downstream.

    Sparse per-snapshot edge lists (see stg/bridge.py's rewrite), not a
    dense (T,N,N) sum -- membership is "does this global index appear as
    a row/col in this snapshot's edge_index", checked per snapshot in a
    plain loop (same convention every other per-snapshot consumer in
    this project already uses), never anything sized to N."""
    T, N = mask.shape
    is_ticker = features[..., -1] == 0.0
    mece_list = adjacency_by_type.get("mece_leg_to_basket")
    ladder_list = adjacency_by_type.get("ladder_monotonic")

    target_mask = torch.zeros(T, N, dtype=torch.bool, device=mask.device)
    for t in range(T):
        if mece_list is not None and mece_list[t].edge_index.numel() > 0:
            target_mask[t, mece_list[t].edge_index[0]] = True
        if ladder_list is not None and ladder_list[t].edge_index.numel() > 0:
            target_mask[t, ladder_list[t].edge_index[0]] = True
            target_mask[t, ladder_list[t].edge_index[1]] = True

    return mask & is_ticker & target_mask


@dataclass
class LegSignal:
    """One mechanism's fair-value read on one (snapshot, leg). A leg that
    belongs to both a MECE basket AND a ladder pair yields ONE LegSignal
    PER mechanism, not a single averaged number -- "does this leg agree
    with its basket" and "does this leg agree with its ladder neighbour"
    are different questions with different neighbourhoods behind them,
    collapsing them would throw away exactly the two-channel distinction
    spatial_attention.py was built to preserve."""

    t: int
    leg_idx: int          # index into the bundle's global node ordering (== node_ids[leg_idx] for the real ticker)
    mechanism: str        # "mece" | "ladder_a" | "ladder_b"
    predicted_fair_price: float   # 0-1 scale, model's context-only estimate
    observed_price: float         # 0-1 scale, the leg's actual last_yes_price this snapshot

    @property
    def deviation(self) -> float:
        """observed - predicted. Positive => market prices this leg
        HIGHER than the model's context-only estimate (a candidate
        "overpriced, model says sell" read); negative => the reverse.
        Sign convention deliberately matches mece_sum_to_one_check.py's
        deviation = sum_cents - 1.00 (positive = overpriced)."""
        return self.observed_price - self.predicted_fair_price


def generate_deviation_signals(
    model: STGATBackbone,
    features: torch.Tensor, mask: torch.Tensor,
    adjacency_by_type: Dict[str, list],
    mask_token: torch.Tensor, raw_feature_dim: int,
) -> List[LegSignal]:
    """One deterministic pass over a chunk: masks EVERY currently-active,
    mechanism-eligible ticker leg (see eligible_targets), then reads each
    head's prediction back out per masked leg, per mechanism it belongs
    to.

    Runs the WHOLE chunk in a single forward pass (all eligible legs
    masked simultaneously, not one leg at a time) -- masking one leg
    doesn't change what's eligible to mask for any OTHER leg, since
    eligibility is a graph-membership question fixed before masking, so
    there's no dependency forcing a per-leg loop; this is the same
    "compute once, read many" pattern the output heads already use
    across every basket/pair group in one call.
    """
    T, N, F = features.shape
    target_mask = eligible_targets(features, mask, adjacency_by_type)
    prices = true_prices(features)

    mece_adj = adjacency_by_type.get("mece_leg_to_basket")
    ladder_adj = adjacency_by_type.get("ladder_monotonic")

    model.eval()
    with torch.no_grad():
        h_final = masked_forward(model, features, mask, adjacency_by_type,
                                  target_mask, mask_token, raw_feature_dim)
        mece_out = model.mece_head(h_final, mece_adj) if mece_adj is not None else {}
        ladder_out = model.ladder_head(h_final, ladder_adj) if ladder_adj is not None else {}

    signals: List[LegSignal] = []
    for t in range(T):
        targets_t = target_mask[t].nonzero(as_tuple=True)[0].tolist()
        if not targets_t:
            continue
        mece_res = mece_out.get(t)
        ladder_res = ladder_out.get(t)
        for leg in targets_t:
            observed = prices[t, leg].item()
            if mece_res is not None:
                hit = (mece_res["leg_idx"] == leg).nonzero(as_tuple=True)[0]
                if hit.numel() > 0:
                    signals.append(LegSignal(t, leg, "mece", mece_res["fair_price"][hit[0]].item(), observed))
            if ladder_res is not None:
                for hit in (ladder_res["leg_a_idx"] == leg).nonzero(as_tuple=True)[0].tolist():
                    signals.append(LegSignal(t, leg, "ladder_a", ladder_res["fair_a"][hit].item(), observed))
                for hit in (ladder_res["leg_b_idx"] == leg).nonzero(as_tuple=True)[0].tolist():
                    signals.append(LegSignal(t, leg, "ladder_b", ladder_res["fair_b"][hit].item(), observed))
    return signals