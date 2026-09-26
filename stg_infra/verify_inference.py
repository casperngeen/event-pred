"""
verify_inference.py

Verifies model/inference.py the same way every other layer here was
verified: hand-built scenarios targeting the exact property that matters,
not a happy-path smoke test. Synthetic data again (see
verify_training_loop.py's environment note -- no real Kalshi data in this
sandbox).

This file does NOT claim the resulting signal is a profitable trading
strategy -- an untrained (random-weight) model's fair-value estimate is
not meaningfully predictive of anything, and there's no real data here to
train on regardless. What's actually being verified is narrower and
structural: does masked_forward()/generate_deviation_signals() genuinely
withhold each target leg's own value before reading a prediction for it
(the same no-leakage property training_objective.py's masking already
guarantees, re-checked at this new call site rather than assumed to carry
over), is the per-mechanism signal set complete and correctly attributed,
and is inference deterministic (no accidental randomness leaking into a
signal that's supposed to be reproducible for a given model + data).

Checks:
  1. No-leakage: corrupting the TRUE feature values of exactly the
     legs that get masked does not change a single predicted_fair_price
     -- while it DOES change observed_price, proving the corruption was
     real and the test isn't vacuously passing.
  2. Only mechanism-eligible legs ever appear in the signal set -- a
     filler ticker (no MECE/ladder membership) never shows up, since it
     has no head output to read at all.
  3. A leg belonging to BOTH a MECE basket and a ladder pair (the
     deliberately shared leg from verify_training_loop.py's synthetic
     bundle builder) yields signals under BOTH "mece" and one of
     "ladder_a"/"ladder_b" -- confirming per-mechanism attribution isn't
     collapsed or dropped.
  4. deviation == observed_price - predicted_fair_price for every signal.
  5. Determinism: two calls with identical inputs produce byte-identical
     output (unlike training's select_and_mask, nothing here should be
     randomly sampled).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime

import torch

from model.inference import eligible_targets, generate_deviation_signals, masked_forward
from model.train import STGATBackbone
from model.training_objective import MaskedLegReconstructionObjective
from verify_training_loop import build_synthetic_bundle, COMBINED_N_FEATURES

torch.manual_seed(0)

RAW_FEATURE_DIM = 9


def _fresh_model():
    return STGATBackbone(padded_feature_width=COMBINED_N_FEATURES, embed_dim=16, n_heads=4, max_len=300)


def check_no_leakage(bundle, model, mask_token):
    print("=== Check 1: masked_forward genuinely withholds each masked leg's own value ===")
    signals_before = generate_deviation_signals(
        model, bundle.features, bundle.mask, bundle.adjacency_by_type,
        mask_token, RAW_FEATURE_DIM,
    )
    assert signals_before, "no signals produced at all -- nothing to check"

    target_mask = eligible_targets(bundle.features, bundle.mask, bundle.adjacency_by_type)
    corrupted_features = bundle.features.clone()
    corrupted_features[target_mask, :RAW_FEATURE_DIM] = torch.rand_like(
        corrupted_features[target_mask, :RAW_FEATURE_DIM]
    ) * 1000.0 - 500.0  # wildly different from the original, comfortably outside the real 0-100 cents range

    signals_after = generate_deviation_signals(
        model, corrupted_features, bundle.mask, bundle.adjacency_by_type,
        mask_token, RAW_FEATURE_DIM,
    )

    by_key_before = {(s.t, s.leg_idx, s.mechanism): s for s in signals_before}
    by_key_after = {(s.t, s.leg_idx, s.mechanism): s for s in signals_after}
    assert set(by_key_before) == set(by_key_after), "corrupting inputs changed WHICH signals exist -- should only change values, if that"

    unchanged_pred = 0
    changed_obs = 0
    for key, before in by_key_before.items():
        after = by_key_after[key]
        assert before.predicted_fair_price == after.predicted_fair_price, (
            f"LEAK at {key}: predicted_fair_price changed from {before.predicted_fair_price} to "
            f"{after.predicted_fair_price} after corrupting that leg's OWN masked-out true value"
        )
        unchanged_pred += 1
        if before.observed_price != after.observed_price:
            changed_obs += 1
    print(f"  {unchanged_pred} signal(s) checked: predicted_fair_price unchanged for every single one")
    assert changed_obs > 0, "observed_price never changed either -- corruption didn't actually reach true_prices(), test is vacuous"
    print(f"  {changed_obs} signal(s) show a changed observed_price -- confirms the corruption was real, not a no-op")
    print("  PASS\n")


def check_eligibility_filtering(bundle, model, mask_token):
    print("=== Check 2: only mechanism-eligible legs ever appear in the signal set ===")
    signals = generate_deviation_signals(
        model, bundle.features, bundle.mask, bundle.adjacency_by_type,
        mask_token, RAW_FEATURE_DIM,
    )
    target_mask = eligible_targets(bundle.features, bundle.mask, bundle.adjacency_by_type)
    eligible_set = {(t, n) for t in range(bundle.features.shape[0])
                    for n in target_mask[t].nonzero(as_tuple=True)[0].tolist()}
    for s in signals:
        assert (s.t, s.leg_idx) in eligible_set, f"signal at {(s.t, s.leg_idx)} is not in the eligible set at all"
    print(f"  {len(signals)} signal(s) from {len(eligible_set)} eligible (t, leg) position(s) -- "
          f"every signal traces back to an eligible position")
    print("  PASS\n")


def check_shared_leg_dual_attribution(bundle, shared_leg_id, model, mask_token):
    print("=== Check 3: a leg in BOTH mechanisms gets BOTH a mece and a ladder signal ===")
    signals = generate_deviation_signals(
        model, bundle.features, bundle.mask, bundle.adjacency_by_type,
        mask_token, RAW_FEATURE_DIM,
    )
    shared = [s for s in signals if s.leg_idx == shared_leg_id]
    mechanisms = {s.mechanism for s in shared}
    assert "mece" in mechanisms, f"shared leg {shared_leg_id} never got a mece signal (mechanisms seen: {mechanisms})"
    assert ("ladder_a" in mechanisms) or ("ladder_b" in mechanisms), (
        f"shared leg {shared_leg_id} never got a ladder signal (mechanisms seen: {mechanisms})"
    )
    print(f"  shared leg {shared_leg_id}: {len(shared)} signal(s) across mechanisms {sorted(mechanisms)}")
    print("  PASS\n")


def check_deviation_arithmetic(bundle, model, mask_token):
    print("=== Check 4: deviation == observed_price - predicted_fair_price ===")
    signals = generate_deviation_signals(
        model, bundle.features, bundle.mask, bundle.adjacency_by_type,
        mask_token, RAW_FEATURE_DIM,
    )
    for s in signals:
        assert abs(s.deviation - (s.observed_price - s.predicted_fair_price)) < 1e-12
    print(f"  {len(signals)} signal(s) checked -- arithmetic holds exactly")
    print("  PASS\n")


def check_determinism(bundle, model, mask_token):
    print("=== Check 5: two calls on identical inputs produce identical output ===")
    s1 = generate_deviation_signals(model, bundle.features, bundle.mask, bundle.adjacency_by_type,
                                     mask_token, RAW_FEATURE_DIM)
    s2 = generate_deviation_signals(model, bundle.features, bundle.mask, bundle.adjacency_by_type,
                                     mask_token, RAW_FEATURE_DIM)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert (a.t, a.leg_idx, a.mechanism) == (b.t, b.leg_idx, b.mechanism)
        assert a.predicted_fair_price == b.predicted_fair_price
        assert a.observed_price == b.observed_price
    print(f"  {len(s1)} signal(s), byte-identical across two calls -- no hidden randomness")
    print("  PASS\n")


if __name__ == "__main__":
    bundle, shared_leg_id = build_synthetic_bundle(
        n_snapshots=200, start=datetime(2025, 5, 1), n_basket_hubs=4, legs_per_basket=4,
        n_ladder_pairs=8, n_filler_tickers=30, active_prob=0.5, seed=3,
    )
    model = _fresh_model()
    objective = MaskedLegReconstructionObjective(raw_feature_dim=RAW_FEATURE_DIM)
    mask_token = objective.mask_token.detach()

    check_no_leakage(bundle, model, mask_token)
    check_eligibility_filtering(bundle, model, mask_token)
    check_shared_leg_dual_attribution(bundle, shared_leg_id, model, mask_token)
    check_deviation_arithmetic(bundle, model, mask_token)
    check_determinism(bundle, model, mask_token)

    print("All checks passed -- masked_forward genuinely withholds each target leg's own value "
          "(no leakage at the inference call site either), signal attribution across mechanisms is "
          "correct and complete, and inference is deterministic. NOTE: this verifies the plumbing is "
          "honest, not that the signal is profitable -- that requires a model actually trained on real "
          "data, which this sandbox doesn't have.")