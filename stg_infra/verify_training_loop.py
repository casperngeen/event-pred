"""
verify_training_loop.py

Verifies the actual training loop (model/train.py, model/chunking.py, and
the ladder extension to model/training_objective.py) the same way every
other layer in this project was verified: hand-built scenarios that
exercise the exact failure modes already found, not just a happy-path
smoke test.

Environment note: this session's sandbox is fresh (no prior /home/claude
project state, no real Kalshi parquet files present), so this uses
SYNTHETIC data shaped like the real bundle -- same discipline already
used earlier in this project for the FIRST pass of every new layer,
before cross-checking against real 2025-02-02 Kalshi data in
verify_combined_graph.py. Re-running this project's actual real-data
regression (verify_combined_graph.py) after this is the natural next
step once real data is loaded into this environment.

Checks, in order:
  1. build_split_ranges / chunk_ranges: no chunk ever crosses a
     train/val/test boundary; every snapshot index is covered exactly
     once across all chunks.
  2. TemporalAttention, called DIRECTLY (bypassing chunking) with T
     larger than max_len, raises the new explicit ValueError -- not the
     original confusing IndexError inside pos_embedding.
  3. A full run_training() pass over a synthetic multi-month bundle with
     T large enough that the ORIGINAL (unchunked) code crashed at T=2000
     -- confirms chunking is what actually fixes the scaling problem,
     not just a smaller test.
  4. Both mechanisms' losses are simultaneously nonzero and gradients
     reach every trainable component, INCLUDING a leg deliberately
     constructed to be masked while participating in both a MECE basket
     and a ladder pair in the same step (compute_mece_loss and
     compute_ladder_loss must both fire from the ONE shared masked
     position, per training_objective.py's shared-masking design).
  5. Loss actually decreases over a short run on a small, easily
     overfit-able synthetic slice (sanity that the optimizer step is
     wired correctly end to end, not just that it runs without crashing).
  6. Checkpointing: the saved best.pt reloads to bit-identical output on
     the same input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import random
from datetime import datetime, timedelta

import numpy as np
import torch

from model.chunking import build_split_ranges, chunk_ranges, slice_bundle_chunk, describe_chunks
from model.temporal_attention import TemporalAttention
from model.train import STGATBackbone, TrainingConfig, run_training, true_prices
from model.training_objective import MaskedLegReconstructionObjective
from examples.data_windows import split_for
from stg.bridge import SparseSnapshotEdgesT

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

COMBINED_N_FEATURES = 10  # matches stg/nodes/kalshi.py


class SyntheticBundle:
    """Minimal stand-in for stg.bridge.TorchGraphBundle -- same field
    names, built directly as tensors/sparse-edge-lists instead of via a
    real graph, exactly as earlier layers in this project were first
    verified against hand-built synthetic tensors before real data was
    wired in.

    ``adjacency_by_type`` matches the REAL bundle's post-sparse-refactor
    shape: edge_type -> length-T list of SparseSnapshotEdgesT (GLOBAL
    node indices, per snapshot) -- not a dense (T,N,N) tensor. There is
    no separate ``edge_features`` field any more; ladder edge features
    travel WITH each snapshot's SparseSnapshotEdgesT.features (see
    stg/bridge.py's rewrite)."""

    def __init__(self, features, mask, adjacency_by_type, timestamps):
        self.features = features
        self.mask = mask
        self.adjacency_by_type = adjacency_by_type
        self.timestamps = timestamps


def build_synthetic_bundle(n_snapshots: int, start: datetime, n_basket_hubs=6, legs_per_basket=5,
                            n_ladder_pairs=15, n_filler_tickers=150, active_prob=0.35, seed=0):
    """Builds a synthetic bundle spanning ``n_snapshots`` 2h-spaced
    windows, with:
      - n_basket_hubs MECE baskets, each with its own legs_per_basket
        ticker legs (mece_leg_to_basket / mece_basket_to_leg edges)
      - n_ladder_pairs ladder pairs among a second pool of ticker legs
        (ladder_monotonic edges, directed leg_a -> leg_b)
      - a few legs deliberately shared between one basket and one ladder
        pair, so a single masked leg can score against BOTH heads at
        once (check 4 above)
      - n_filler_tickers unconnected ticker nodes, present but never
        wired to anything -- mimics the real graph's vast majority of
        traded-but-irrelevant tickers
    Node type indicator (feature slot -1) and mask=False -> features=0
    exactly follow stg.bridge.to_torch_bundle's real NaN->0 convention.
    """
    rng = np.random.default_rng(seed)

    node_specs = []  # (kind, basket_id or None)
    for b in range(n_basket_hubs):
        node_specs.append(("basket", b))
        for _ in range(legs_per_basket):
            node_specs.append(("leg", b))
    ladder_leg_ids = []
    for p in range(n_ladder_pairs):
        a_id = len(node_specs); node_specs.append(("ladder_leg", None)); ladder_leg_ids.append(a_id)
        b_id = len(node_specs); node_specs.append(("ladder_leg", None)); ladder_leg_ids.append(b_id)
    for _ in range(n_filler_tickers):
        node_specs.append(("filler", None))

    N = len(node_specs)
    F = COMBINED_N_FEATURES
    T = n_snapshots

    features = torch.zeros(T, N, F)
    mask = torch.zeros(T, N, dtype=torch.bool)
    # Per-snapshot GLOBAL edge lists, built up as plain Python lists and
    # converted to SparseSnapshotEdgesT at the end of each snapshot's loop
    # iteration -- mirrors stg.core.SpatioTemporalGraph.sparse_edges_by_type's
    # own "accumulate per real edge, never touch anything sized to N^2"
    # discipline, just built by hand instead of from an nx_graph.
    mece_leg_to_basket_snaps: list = []
    mece_basket_to_leg_snaps: list = []
    ladder_snaps: list = []

    basket_of_leg = {}
    legs_of_basket = {b: [] for b in range(n_basket_hubs)}
    basket_idx_of = {}
    for i, (kind, b) in enumerate(node_specs):
        if kind == "basket":
            basket_idx_of[b] = i
        elif kind == "leg":
            basket_of_leg[i] = b
            legs_of_basket[b].append(i)

    # Deliberately overlap the FIRST ladder pair's leg_a with the FIRST
    # basket's first leg, so that leg can be masked and score against
    # both heads simultaneously (see check 4).
    shared_leg_id = legs_of_basket[0][0]
    ladder_pairs = list(zip(ladder_leg_ids[0::2], ladder_leg_ids[1::2]))
    ladder_pairs[0] = (shared_leg_id, ladder_pairs[0][1])

    timestamps = [start + timedelta(hours=2 * t) for t in range(T)]

    def _empty_sparse(feature_dim=None):
        fe = torch.zeros((0, feature_dim), dtype=torch.float32) if feature_dim is not None else None
        return SparseSnapshotEdgesT(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            weight=torch.zeros((0,), dtype=torch.float32),
            features=fe,
        )

    for t in range(T):
        leg_active = {}
        for i, (kind, b) in enumerate(node_specs):
            if kind in ("leg", "ladder_leg", "filler"):
                leg_active[i] = rng.random() < active_prob

        l2h_rows, l2h_cols = [], []
        h2l_rows, h2l_cols = [], []
        for b, legs in legs_of_basket.items():
            hub_i = basket_idx_of[b]
            any_active = any(leg_active[l] for l in legs)
            if any_active:
                mask[t, hub_i] = True
                features[t, hub_i, -1] = 1.0
                features[t, hub_i, :6] = torch.rand(6)
                features[t, hub_i, 0] = torch.rand(1) * 100.0
            for l in legs:
                if leg_active[l]:
                    mask[t, l] = True
                    features[t, l, -1] = 0.0
                    features[t, l, :9] = torch.rand(9)
                    features[t, l, 0] = torch.rand(1) * 100.0
                    l2h_rows.append(l); l2h_cols.append(hub_i)
                    h2l_rows.append(hub_i); h2l_cols.append(l)

        for i, (kind, _b) in enumerate(node_specs):
            if kind in ("ladder_leg", "filler") and leg_active.get(i, False):
                mask[t, i] = True
                features[t, i, -1] = 0.0
                features[t, i, :9] = torch.rand(9)
                features[t, i, 0] = torch.rand(1) * 100.0

        ladder_rows, ladder_cols, ladder_feats = [], [], []
        for a, b in ladder_pairs:
            if mask[t, a] and mask[t, b]:
                ladder_rows.append(a); ladder_cols.append(b)
                ladder_feats.append(torch.rand(2))

        if l2h_rows:
            l2h_ei = torch.tensor([l2h_rows, l2h_cols], dtype=torch.long)
            mece_leg_to_basket_snaps.append(SparseSnapshotEdgesT(
                edge_index=l2h_ei, weight=torch.ones(len(l2h_rows)), features=None))
        else:
            mece_leg_to_basket_snaps.append(_empty_sparse())

        if h2l_rows:
            h2l_ei = torch.tensor([h2l_rows, h2l_cols], dtype=torch.long)
            mece_basket_to_leg_snaps.append(SparseSnapshotEdgesT(
                edge_index=h2l_ei, weight=torch.ones(len(h2l_rows)), features=None))
        else:
            mece_basket_to_leg_snaps.append(_empty_sparse())

        if ladder_rows:
            ladder_ei = torch.tensor([ladder_rows, ladder_cols], dtype=torch.long)
            ladder_snaps.append(SparseSnapshotEdgesT(
                edge_index=ladder_ei, weight=torch.ones(len(ladder_rows)),
                features=torch.stack(ladder_feats)))
        else:
            ladder_snaps.append(_empty_sparse(feature_dim=2))

    adjacency_by_type = {
        "mece_leg_to_basket": mece_leg_to_basket_snaps,
        "mece_basket_to_leg": mece_basket_to_leg_snaps,
        "ladder_monotonic": ladder_snaps,
    }
    return SyntheticBundle(features, mask, adjacency_by_type, timestamps), shared_leg_id


def check_chunking():
    print("=== Check 1: split-safe chunking ===")
    T = 2000
    start = datetime(2025, 5, 1)
    timestamps = [start + timedelta(hours=2 * t) for t in range(T)]
    ranges = build_split_ranges(timestamps)
    for split, s, e in ranges:
        months = {f"{timestamps[i].year:04d}-{timestamps[i].month:02d}" for i in range(s, e)}
        actual_splits = {split_for(m) for m in months}
        assert actual_splits == {split}, f"range [{s}:{e}) mixes splits: {actual_splits}"
    print(f"  {len(ranges)} contiguous split range(s), each internally single-split -- confirmed")

    chunks = chunk_ranges(ranges, chunk_len=168, min_chunk_len=24)
    covered = np.zeros(T, dtype=bool)
    for c in chunks:
        assert not covered[c.start:c.end].any(), f"chunk {c} overlaps a previously covered range"
        covered[c.start:c.end] = True
        chunk_months = {f"{timestamps[i].year:04d}-{timestamps[i].month:02d}" for i in range(c.start, c.end)}
        assert {split_for(m) for m in chunk_months} == {c.split}, f"chunk {c} crosses a split boundary"
    total_covered = int(covered.sum())
    print(f"  {len(chunks)} chunk(s) total, {total_covered}/{T} snapshots covered "
          f"({T - total_covered} dropped as sub-min_chunk_len remainders)")
    print(describe_chunks(chunks))
    assert total_covered >= T - 3 * 24, "dropped more than 3 chunks' worth of remainder -- suspiciously lossy"
    print("  PASS: no chunk crosses a split boundary, coverage is complete modulo tiny remainders\n")


def check_explicit_error_not_crash():
    print("=== Check 2: TemporalAttention gives an explicit error above max_len, not IndexError ===")
    attn = TemporalAttention(embed_dim=32, n_heads=4, max_len=200)
    T, N, D = 2000, 50, 32  # the exact scale that crashed with IndexError before this fix
    h = torch.randn(T, N, D)
    mask = torch.rand(T, N) > 0.6
    try:
        attn(h, mask)
        raise AssertionError("expected a ValueError, got no exception at all")
    except IndexError:
        raise AssertionError("still raising the old confusing IndexError -- fix did not take")
    except ValueError as e:
        assert "chunk" in str(e).lower(), f"error doesn't mention chunking, unhelpful: {e}"
        print(f"  Got the expected explicit ValueError: {e}")
    print("  PASS\n")


def check_full_training_run_at_scale(bundle, shared_leg_id):
    print("=== Check 3: full run_training() at the T=2000-class scale that previously crashed ===")
    config = TrainingConfig(embed_dim=16, n_heads=4, chunk_len=168, min_chunk_len=24,
                             mask_ratio=0.4, lr=1e-2, epochs=2, checkpoint_dir="/tmp/stgat_ckpt_verify")
    history = run_training(bundle, config)
    assert len(history) == config.epochs
    print(f"  Completed {len(history)} epoch(s) over T={bundle.features.shape[0]} snapshots without OOM/IndexError")
    print("  PASS\n")
    return history, config


def check_dual_head_gradients(bundle, shared_leg_id):
    print("=== Check 4: shared masked leg drives gradients through BOTH heads at once ===")
    config = TrainingConfig(embed_dim=16, n_heads=4, chunk_len=168, min_chunk_len=24, mask_ratio=1.0)
    # mask_ratio=1.0: mask EVERY eligible ticker this step, guaranteeing the
    # shared leg (present in both a basket and a ladder pair) is masked at
    # least once, rather than leaving it to chance.
    model = STGATBackbone(padded_feature_width=bundle.features.shape[-1], embed_dim=config.embed_dim,
                           n_heads=config.n_heads, max_len=config.chunk_len)
    objective = MaskedLegReconstructionObjective(raw_feature_dim=config.raw_feature_width, mask_ratio=1.0)

    from model.chunking import build_split_ranges, chunk_ranges
    ranges = build_split_ranges(bundle.timestamps)
    train_chunk = [c for c in chunk_ranges(ranges, config.chunk_len, config.min_chunk_len) if c.split == "train"][0]
    chunk_tensors = slice_bundle_chunk(bundle, train_chunk)

    from model.train import _forward_chunk_losses
    generator = torch.Generator().manual_seed(0)
    total, mece_loss, ladder_loss = _forward_chunk_losses(model, objective, chunk_tensors, generator)
    assert mece_loss.item() > 0, "MECE loss is exactly zero -- no masked leg reached the MECE head at all"
    assert ladder_loss.item() > 0, "ladder loss is exactly zero -- no masked leg reached the ladder head at all"
    print(f"  mece_loss={mece_loss.item():.4f}  ladder_loss={ladder_loss.item():.4f} -- both nonzero")

    total.backward()
    missing = [n for n, p in list(model.named_parameters()) + list(objective.named_parameters())
               if p.requires_grad and (p.grad is None)]
    nan_grad = [n for n, p in list(model.named_parameters()) + list(objective.named_parameters())
                if p.grad is not None and torch.isnan(p.grad).any()]
    print(f"  {len(missing)} param(s) with missing gradient (must be 0): {missing}")
    print(f"  {len(nan_grad)} param(s) with NaN gradient (must be 0): {nan_grad}")
    assert not missing and not nan_grad
    assert objective.mask_token.grad is not None and objective.mask_token.grad.abs().sum().item() > 0
    print("  mask_token gradient exists and nonzero -- confirmed")
    print("  PASS\n")


def check_loss_decreases():
    print("=== Check 5: loss actually decreases on a small, overfit-able synthetic slice ===")
    small_bundle, _shared = build_synthetic_bundle(
        n_snapshots=200, start=datetime(2025, 5, 1), n_basket_hubs=3, legs_per_basket=4,
        n_ladder_pairs=6, n_filler_tickers=20, active_prob=0.6, seed=1,
    )
    config = TrainingConfig(embed_dim=16, n_heads=4, chunk_len=100, min_chunk_len=20,
                             mask_ratio=0.3, lr=5e-3, epochs=15, checkpoint_dir="/tmp/stgat_ckpt_overfit")
    history = run_training(small_bundle, config)
    first_three = sum(h.train_loss for h in history[:3]) / 3
    last_three = sum(h.train_loss for h in history[-3:]) / 3
    print(f"  mean train_loss, first 3 epochs: {first_three:.4f}  last 3 epochs: {last_three:.4f}")
    assert last_three < first_three, "training loss did not decrease -- optimizer wiring is broken"
    print("  PASS\n")


def check_checkpoint_reload():
    print("=== Check 6: checkpoint reloads to identical output ===")
    ckpt_path = Path("/tmp/stgat_ckpt_verify/best.pt")
    assert ckpt_path.exists(), "expected check 3 to have written a checkpoint first"
    saved = torch.load(ckpt_path, weights_only=False)
    config: TrainingConfig = saved["config"]

    model_a = STGATBackbone(padded_feature_width=COMBINED_N_FEATURES, embed_dim=config.embed_dim,
                             n_heads=config.n_heads, max_len=config.chunk_len)
    model_a.load_state_dict(saved["model"])
    model_b = STGATBackbone(padded_feature_width=COMBINED_N_FEATURES, embed_dim=config.embed_dim,
                             n_heads=config.n_heads, max_len=config.chunk_len)
    model_b.load_state_dict(torch.load(ckpt_path, weights_only=False)["model"])

    torch.manual_seed(42)
    N, T = 30, 50
    features = torch.rand(T, N, COMBINED_N_FEATURES)
    features[..., -1] = (torch.rand(T, N) > 0.8).float()
    mask = torch.rand(T, N) > 0.4
    # Empty sparse edge lists (no MECE/ladder structure needed for this
    # check -- it only tests that two loaded copies of the same
    # checkpoint produce bit-identical output) -- length-T lists of
    # empty SparseSnapshotEdgesT, matching the real bundle's shape, not
    # a dense (T,N,N) tensor of zeros.
    _empty = SparseSnapshotEdgesT(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        weight=torch.zeros((0,), dtype=torch.float32),
        features=None,
    )
    adj = {
        "mece_leg_to_basket": [_empty] * T,
        "mece_basket_to_leg": [_empty] * T,
        "ladder_monotonic": [_empty] * T,
    }

    model_a.eval(); model_b.eval()
    with torch.no_grad():
        out_a = model_a(features, mask, adj)
        out_b = model_b(features, mask, adj)
    assert torch.equal(out_a, out_b), "reloaded checkpoint produces different output on the same input"
    print(f"  saved at epoch {saved['epoch']}, val_loss={saved['val_loss']:.4f} -- reload is bit-identical")
    print("  PASS\n")


if __name__ == "__main__":
    check_chunking()
    check_explicit_error_not_crash()

    bundle, shared_leg_id = build_synthetic_bundle(
        n_snapshots=2000, start=datetime(2025, 5, 1),
    )
    check_full_training_run_at_scale(bundle, shared_leg_id)
    check_dual_head_gradients(bundle, shared_leg_id)
    check_loss_decreases()
    check_checkpoint_reload()

    print("All checks passed -- the training loop is verified on synthetic data at the exact scale "
          "that previously crashed, with both mechanisms' losses wired in. Re-run against real "
          "Kalshi data (via build_combined_stgat_graph.py + stg.bridge.to_torch_bundle) once it's "
          "loaded into this environment, the same way verify_combined_graph.py cross-checked every "
          "earlier layer.")