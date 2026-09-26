"""
model/train.py

The actual training loop: wires the projection -> spatial attention ->
temporal attention backbone together with both output heads and the
masked-leg reconstruction objective (now covering MECE AND ladder, see
training_objective.py), and runs it over TIME-CHUNKED, split-safe windows
of a TorchGraphBundle (see model/chunking.py for why chunking exists and
how it keeps train/val/test strictly separate).

WHAT THIS FILE DOES vs. WHAT IT DOESN'T:
- Does: chunk a bundle, build the model + optimizer, run a real
  train/val epoch loop with loss aggregation across both mechanisms,
  and checkpoint the best-val-loss model state.
- Doesn't: load raw Kalshi data or build the graph itself -- that's
  build_combined_stgat_graph.py's job. run_training() takes an
  already-built TorchGraphBundle (stg.bridge.to_torch_bundle output) so
  this file has no Polars/data-loading dependency, matching the layering
  the rest of this project already uses (bridge.py keeps torch a soft
  dependency of the graph layer; this file keeps the raw-data pipeline a
  soft dependency of the model layer).

TRUE-PRICE CONVENTION: KalshiTickerNodes stores last_yes_price at feature
slot 0 in Kalshi's native cents scale (0-100), not the 0-1 probability
scale the output heads predict on (MeceOutputHead's softmax and
LadderOutputHead's fair_a/fair_b are both trained to land in [0, 1] via
the reconstruction loss). true_prices() below is the one place that
conversion happens, so every loss computation in this file uses the same
0-1-scaled target.
"""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from model.chunking import TemporalChunk, build_split_ranges, chunk_ranges, describe_chunks, slice_bundle_chunk
from model.data_validation import ValidationStats
from model.output_heads import LadderOutputHead, MeceOutputHead
from model.spatial_attention import TwoChannelSpatialAttention, TypeSpecificProjection
from model.temporal_attention import TemporalAttention
from model.training_objective import MaskedLegReconstructionObjective


def _rss_mb() -> float:
    """Current process peak RSS in MB (Linux: ru_maxrss is in KB). Cheap,
    stdlib-only (no psutil dependency) way to see whether memory is
    genuinely climbing across chunks/epochs -- the signature of the three
    dense-at-global-N bugs already found in this project -- versus a run
    that's simply CPU-bound and slow, which is a different problem with a
    different fix. Diagnostic only, added while localizing the hang that
    reproduces immediately after "Chunked bundle into: ..." with no
    further output."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def true_prices(features: torch.Tensor) -> torch.Tensor:
    """(T, N, F) -> (T, N), last_yes_price (slot 0) rescaled from Kalshi's
    native 0-100 cents convention to the 0-1 probability scale every
    output head is trained to predict on. See module docstring."""
    return features[..., 0] / 100.0


class STGATBackbone(nn.Module):
    """projection -> two-channel spatial attention -> temporal attention.
    Both output heads sit on top of this shared backbone, not inside it,
    since MeceOutputHead / LadderOutputHead are cheap per-snapshot linear
    ops applied to whatever h_final the backbone produces -- exactly the
    "one shared temporal backbone, two mechanism-specific heads" split
    this project's architecture discussion settled on."""

    def __init__(self, padded_feature_width: int, embed_dim: int = 32, n_heads: int = 4,
                 ladder_edge_feature_dim: int = 2, max_len: int = 512):
        """``padded_feature_width`` is the FULL per-node feature width
        including the type-indicator slot (COMBINED_N_FEATURES = 10 in
        stg/nodes/kalshi.py) -- NOT the same thing as
        TrainingConfig.raw_feature_width / the objective's
        raw_feature_dim (9), which is the type-indicator-EXCLUDED width
        select_and_mask replaces with the [MASK] token. Two different
        widths, two different names, deliberately not reused for each
        other."""
        super().__init__()
        # FEATURE STANDARDIZATION BUFFERS. Registered (not plain
        # attributes) so they travel with state_dict into the checkpoint
        # -- a model reloaded for inference MUST standardize its inputs
        # exactly the way training did, and silently failing to would
        # produce a model that looks fine and predicts nonsense.
        #
        # Identity by default (mean 0, std 1) so an un-fitted model
        # behaves exactly as before; run_training() fills these from the
        # TRAIN SPLIT ONLY before the first epoch.
        self.register_buffer("feat_mean", torch.zeros(padded_feature_width))
        self.register_buffer("feat_std", torch.ones(padded_feature_width))
        self.projection = TypeSpecificProjection(padded_feature_width, embed_dim)
        self.spatial = TwoChannelSpatialAttention(embed_dim, n_heads=n_heads,
                                                   ladder_edge_feature_dim=ladder_edge_feature_dim)
        self.temporal = TemporalAttention(embed_dim, n_heads=n_heads, max_len=max_len)
        self.mece_head = MeceOutputHead(embed_dim)
        self.ladder_head = LadderOutputHead(embed_dim)

    def standardize(self, features: torch.Tensor) -> torch.Tensor:
        """(x - mean) / std per feature slot, EXCEPT the type-indicator
        slot, which is left untouched.

        WHY THE TYPE SLOT IS EXCLUDED: TypeSpecificProjection routes on
        ``x[..., -1]`` being exactly 1.0 (MECE basket hub) or 0.0 (ticker
        leg), blending the two projections by that value. Standardizing
        it would turn those into arbitrary reals and silently mix the two
        projections for every node -- the exact "one set of weights doing
        double duty for two unrelated things" failure that having
        separate per-type projections exists to prevent.

        WHY THIS IS NEEDED AT ALL: see stg/nodes/kalshi.py's feature
        layout. Slot 8 is time_to_close in SECONDS (a market 30 days out
        is 2,592,000) while slots 5 and 6 are ratios in [-1, 1], and slot
        0 is a price in cents (0-100). Fed raw into nn.Linear with
        default init, the seconds-scale feature dominates the projection
        output by ~6 orders of magnitude: the embedding encodes little
        beyond time-to-close, downstream attention softmaxes saturate,
        and gradients for every other feature are swamped. That produces
        a loss which improves slightly for a few epochs and then
        flatlines well above a useful level, with train and val stuck
        close together -- which is what a real 5-month run produced
        (train ~0.433-0.441, val ~0.412-0.425, both flat, RMSE ~34 cents
        on a 0-100c contract).
        """
        mean, std = self.feat_mean.clone(), self.feat_std.clone()
        mean[-1], std[-1] = 0.0, 1.0        # leave the type indicator alone
        return (features - mean) / std

    def fit_standardization(self, mean: torch.Tensor, std: torch.Tensor):
        """Installs per-slot statistics. ``std`` is floored away from zero
        by the caller; a constant feature must not become a division by
        zero (it becomes all-zeros after centering, which is correct --
        a feature that never varies carries no information)."""
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def forward(self, features: torch.Tensor, mask: torch.Tensor,
                adjacency_by_type: Dict[str, list]) -> torch.Tensor:
        """features/mask/adjacency_by_type are all for ONE chunk already
        (T <= max_len) -- the caller (run_training below) is responsible
        for never handing this more than chunk_len snapshots at once.
        ``adjacency_by_type`` values are length-T lists of per-snapshot
        sparse edge structs (see stg/bridge.py's rewrite), not dense
        (T,N,N) tensors -- there's no separate ``edge_features`` argument
        any more since edge features now travel WITH each sparse
        snapshot entry. Returns h_final, (T, N, D)."""
        T, N, _ = features.shape
        t0 = time.time()
        features = self.standardize(features)
        h0 = self.projection(features)
        t1 = time.time()
        print(f"    [backbone] projection done  T={T} N={N}  {t1 - t0:.2f}s  rss={_rss_mb():.0f}MB", flush=True)
        h1 = self.spatial(h0, mask, adjacency_by_type)
        t2 = time.time()
        print(f"    [backbone] spatial done     {t2 - t1:.2f}s  rss={_rss_mb():.0f}MB", flush=True)
        h2 = self.temporal(h1, mask)
        t3 = time.time()
        print(f"    [backbone] temporal done    {t3 - t2:.2f}s  rss={_rss_mb():.0f}MB", flush=True)
        return h2


@dataclass
class TrainingConfig:
    embed_dim: int = 32
    n_heads: int = 4
    chunk_len: int = 168          # ~2 weeks at 2h resolution -- see chunking.py's trade-off discussion
    min_chunk_len: int = 24       # drop a trailing remainder shorter than this (unless it's a run's only chunk)
    mask_ratio: float = 0.15
    lr: float = 1e-3
    epochs: int = 20
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    raw_feature_width: int = 9    # _TICKER_RAW_FEATURES in stg/nodes/kalshi.py -- what select_and_mask replaces
    patience: int = 5             # stop after this many epochs with no val improvement; 0 disables


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    train_mece_loss: float
    train_ladder_loss: float
    val_mece_loss: float
    val_ladder_loss: float
    seconds: float


def _compute_train_feature_stats(get_chunk, train_chunks, feature_width: int):
    """Per-feature mean/std over OBSERVED positions in the TRAIN chunks only.

    TRAIN-ONLY IS NOT OPTIONAL. Fitting these statistics on val or test
    snapshots would leak information about held-out months into training
    -- a subtle form of the exact lookahead leakage data_windows.py's
    shared split and chunking.py's split-safe chunking were both built to
    prevent. It leaks quietly, too: the model still trains, the loss
    still falls, and the held-out score is simply optimistic by an
    unknown amount. So the statistics come from train_chunks, and val and
    test are standardized with those same numbers at inference time.

    Masked positions only: the dense tensors are mostly zero padding for
    nodes that aren't active in a snapshot, and including those zeros
    would drag every mean toward 0 and shrink every std by the padding
    ratio (~96%), producing statistics that describe the padding rather
    than the data.

    Uses a streaming sum / sum-of-squares pass so it never holds more
    than one chunk's observations at a time -- the same constraint
    everything else in this pipeline now respects.
    """
    count = 0
    total = torch.zeros(feature_width, dtype=torch.float64)
    total_sq = torch.zeros(feature_width, dtype=torch.float64)

    for chunk in train_chunks:
        ct = get_chunk(chunk)
        observed = ct["features"][ct["mask"]].to(torch.float64)  # (n_observed, F)
        if observed.numel() == 0:
            continue
        count += observed.shape[0]
        total += observed.sum(dim=0)
        total_sq += (observed ** 2).sum(dim=0)
        del ct, observed

    if count == 0:
        raise ValueError("No observed positions in any training chunk -- cannot fit feature statistics.")

    mean = total / count
    var = (total_sq / count) - mean ** 2
    std = var.clamp_min(0).sqrt()
    # A constant feature (std 0) would divide by zero; flooring to 1.0
    # makes it centre to exactly zero instead, which is the right answer
    # for a feature that carries no variation.
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean.to(torch.float32), std.to(torch.float32), count


def _forward_chunk_losses(model: STGATBackbone, objective: MaskedLegReconstructionObjective,
                           chunk_tensors: dict, generator: torch.Generator, label: str = ""):
    """One chunk's worth of masking + forward + both mechanisms' losses.
    Shared by the train step (which backprops the result) and the val
    step (which wraps this in torch.no_grad() instead) -- there's no
    train/val branching inside this function itself, so the two can never
    silently drift apart in what they compute.

    ``label`` is diagnostic-only (see run_training's per-chunk prints):
    threaded through purely so the [backbone] progress lines printed
    inside STGATBackbone.forward/TwoChannelSpatialAttention.forward can be
    told apart from other concurrent chunk-losses calls, while localizing
    the hang that reproduces right after chunking with no further output."""
    features, mask = chunk_tensors["features"], chunk_tensors["mask"]
    node_types = features[..., -1]  # last slot IS the type indicator -- see stg/nodes/kalshi.py

    t0 = time.time()
    masked_features, target_mask, _true_raw = objective.select_and_mask(
        features, mask, node_types, generator=generator,
    )
    prices = true_prices(features)
    print(f"    [{label}] select_and_mask done  {time.time() - t0:.2f}s  "
          f"n_masked={int(target_mask.sum())}  rss={_rss_mb():.0f}MB", flush=True)

    h_final = model(masked_features, mask, chunk_tensors["adjacency_by_type"])

    mece_adj = chunk_tensors["adjacency_by_type"].get("mece_leg_to_basket")
    ladder_adj = chunk_tensors["adjacency_by_type"].get("ladder_monotonic")

    zero = torch.tensor(0.0, device=features.device)
    t1 = time.time()
    mece_loss = zero
    if mece_adj is not None:
        mece_out = model.mece_head(h_final, mece_adj)
        mece_loss = objective.compute_mece_loss(mece_out, target_mask, prices)
    print(f"    [{label}] mece head+loss done   {time.time() - t1:.2f}s  rss={_rss_mb():.0f}MB", flush=True)

    t2 = time.time()
    ladder_loss = zero
    if ladder_adj is not None:
        ladder_out = model.ladder_head(h_final, ladder_adj)
        ladder_loss = objective.compute_ladder_loss(ladder_out, target_mask, prices)
    print(f"    [{label}] ladder head+loss done {time.time() - t2:.2f}s  rss={_rss_mb():.0f}MB", flush=True)

    total = mece_loss + ladder_loss
    return total, mece_loss, ladder_loss


def run_training(bundle, config: TrainingConfig) -> List[EpochResult]:
    """Full train/val loop over one TorchGraphBundle, chunked and
    split-safe (see model/chunking.py). Checkpoints the best-val-loss
    model to ``{config.checkpoint_dir}/best.pt``.

    ``bundle`` is a stg.bridge.TorchGraphBundle (or anything exposing the
    same .features/.mask/.adjacency_by_type/.timestamps fields) covering
    the FULL training range in one object -- chunking
    and train/val/test separation happen inside this function, not before
    it, so callers don't need their own copy of the split logic.
    """
    torch.manual_seed(config.seed)

    # ``bundle`` may be either a single in-memory TorchGraphBundle or a
    # MonthlyBundleStore (model/month_store.py), which keeps one month
    # resident at a time so a 12-14 month range costs the same RAM as a
    # single month. They are told apart by capability, not by isinstance,
    # so neither module has to import the other: a store materializes its
    # own chunks, a bundle is sliced.
    is_store = hasattr(bundle, "materialize_chunk")
    get_chunk = bundle.materialize_chunk if is_store else (lambda c: slice_bundle_chunk(bundle, c))
    feature_width = bundle.feature_width if is_store else int(bundle.features.shape[-1])
    device = torch.device("cpu") if is_store else bundle.features.device
    generator = torch.Generator(device=device).manual_seed(config.seed)

    if is_store:
        print(bundle.summary(), flush=True)

    ranges = build_split_ranges(bundle.timestamps)
    chunks = chunk_ranges(ranges, chunk_len=config.chunk_len, min_chunk_len=config.min_chunk_len)
    print("Chunked bundle into:\n" + describe_chunks(chunks))
    train_chunks = [c for c in chunks if c.split == "train"]
    val_chunks = [c for c in chunks if c.split == "val"]
    if not train_chunks:
        raise ValueError("No training chunks produced -- check bundle.timestamps against data_windows.py's split boundaries.")

    if not val_chunks:
        # Said LOUDLY and UP FRONT, not discovered afterwards: a run with
        # no val split still trains and still prints a val_loss, but that
        # number is 0.0000 because it is the mean of an empty list, NOT
        # because the model is perfect. Nothing can be selected, early-
        # stopped, or checked for overfitting in this mode.
        print(
            "\n*** WARNING: this run has NO VALIDATION CHUNKS. ***\n"
            "  Every snapshot in this bundle falls in the 'train' split per data_windows.py\n"
            f"  (TRAIN_END/VAL_END boundaries). The val_loss printed each epoch will be 0.0000\n"
            "  -- that is an empty average, not a score. Consequences for this run:\n"
            "    - no model selection / early stopping (best-by-val is undefined)\n"
            "    - no overfitting signal at all\n"
            "    - no held-out data, so nothing here can be reported as a result\n"
            "  To get a real train/val/test split, include months spanning the boundaries,\n"
            "  e.g. --months 2025-05 2025-06 2025-07 2025-08 2025-09.\n"
            "  Training will continue and the FINAL epoch will be checkpointed so the run\n"
            "  isn't wasted, but treat this as a plumbing run, not an experiment.\n",
            flush=True,
        )

    inferred_raw_width = feature_width - 1  # exclude the type-indicator slot -- see select_and_mask
    if inferred_raw_width != config.raw_feature_width:
        raise ValueError(
            f"config.raw_feature_width={config.raw_feature_width} doesn't match "
            f"the bundle's actual feature width ({feature_width} total, "
            f"{inferred_raw_width} raw slots after the type indicator) -- update "
            f"TrainingConfig.raw_feature_width (this must track _TICKER_RAW_FEATURES "
            f"in stg/nodes/kalshi.py, not be silently out of sync with it)."
        )
    model = STGATBackbone(
        padded_feature_width=feature_width, embed_dim=config.embed_dim, n_heads=config.n_heads,
        max_len=config.chunk_len,
    )
    # Fit input standardization on the TRAIN split before any training
    # happens (see _compute_train_feature_stats for why train-only).
    # Report data-quality filtering across the whole run (see
    # model/data_validation.py). Printed once, up front, because a run
    # that silently discards part of its dataset is a run whose numbers
    # can't be interpreted later.
    _vstats = ValidationStats()
    for _c in chunks:
        _ct = get_chunk(_c)
        _s = _ct.get("validation_stats")
        if _s is not None:
            _vstats.merge(_s)
        del _ct
    print("Data validity filtering (all splits):", flush=True)
    print(_vstats.summary(), flush=True)
    print(flush=True)

    print("Fitting feature standardization on train chunks ...", flush=True)
    _t_stats = time.time()
    feat_mean, feat_std, n_obs = _compute_train_feature_stats(get_chunk, train_chunks, feature_width)
    model.fit_standardization(feat_mean, feat_std)
    print(f"  fitted on {n_obs:,} observed positions in {time.time() - _t_stats:.1f}s", flush=True)
    print(f"  per-slot std BEFORE standardization: "
          f"{[round(float(s), 3) for s in feat_std]}", flush=True)
    _ratio = float(feat_std.max() / feat_std.clamp_min(1e-12).min())
    print(f"  widest/narrowest feature std ratio: {_ratio:,.0f}x "
          f"(this is what standardization removes)\n", flush=True)

    objective = MaskedLegReconstructionObjective(
        raw_feature_dim=config.raw_feature_width, mask_ratio=config.mask_ratio,
    )
    params = list(model.parameters()) + list(objective.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_since_improvement = 0
    history: List[EpochResult] = []

    for epoch in range(config.epochs):
        t0 = time.time()
        # Shuffle CHUNK order only -- never shuffle within a chunk, it's a
        # time series. With a MonthlyBundleStore the shuffle is grouped by
        # month (month order shuffled, chunks shuffled within each month)
        # so each month's file is loaded once per epoch instead of once
        # per chunk; see MonthlyBundleStore.month_of.
        rng = random.Random(config.seed + epoch)
        if is_store:
            by_month: Dict[str, List[int]] = {}
            for i, c in enumerate(train_chunks):
                by_month.setdefault(bundle.month_of(c), []).append(i)
            month_order = list(by_month)
            rng.shuffle(month_order)
            order = []
            for m in month_order:
                idxs = by_month[m]
                rng.shuffle(idxs)
                order.extend(idxs)
        else:
            order = list(range(len(train_chunks)))
            rng.shuffle(order)

        model.train()
        train_losses, train_mece, train_ladder = [], [], []
        for pos, idx in enumerate(order):
            chunk = train_chunks[idx]
            label = f"epoch{epoch} train {pos + 1}/{len(order)}"
            print(f"  [{label}] starting  chunk={chunk}  rss={_rss_mb():.0f}MB", flush=True)
            _t_chunk = time.time()
            chunk_tensors = get_chunk(chunk)
            total, mece_l, ladder_l = _forward_chunk_losses(model, objective, chunk_tensors, generator, label=label)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            train_losses.append(total.item())
            train_mece.append(mece_l.item())
            train_ladder.append(ladder_l.item())
            print(f"  [{label}] done  total={time.time() - _t_chunk:.2f}s  rss={_rss_mb():.0f}MB", flush=True)

        model.eval()
        val_losses, val_mece, val_ladder = [], [], []
        with torch.no_grad():
            for pos, c in enumerate(val_chunks):
                label = f"epoch{epoch} val {pos + 1}/{len(val_chunks)}"
                print(f"  [{label}] starting  chunk={c}  rss={_rss_mb():.0f}MB", flush=True)
                _t_chunk = time.time()
                chunk_tensors = get_chunk(c)
                total, mece_l, ladder_l = _forward_chunk_losses(model, objective, chunk_tensors, generator, label=label)
                val_losses.append(total.item())
                val_mece.append(mece_l.item())
                val_ladder.append(ladder_l.item())
                print(f"  [{label}] done  total={time.time() - _t_chunk:.2f}s  rss={_rss_mb():.0f}MB", flush=True)

        def _mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        result = EpochResult(
            epoch=epoch,
            train_loss=_mean(train_losses), val_loss=_mean(val_losses),
            train_mece_loss=_mean(train_mece), train_ladder_loss=_mean(train_ladder),
            val_mece_loss=_mean(val_mece), val_ladder_loss=_mean(val_ladder),
            seconds=time.time() - t0,
        )
        history.append(result)

        # Persist the curve AFTER EVERY EPOCH, not at the end. Two
        # reasons: a run that dies partway (this pipeline has form) still
        # leaves usable curve data, and the train/val curves are the
        # figure that answers "is it learning, converged, or overfitting"
        # -- a question that cannot be answered from the final loss value
        # alone, and which no amount of re-reading a terminal scrollback
        # reconstructs reliably.
        hist_path = ckpt_dir / "history.csv"
        with open(hist_path, "w") as fh:
            fh.write("epoch,train_loss,train_mece,train_ladder,val_loss,val_mece,val_ladder,seconds\n")
            for r in history:
                fh.write(f"{r.epoch},{r.train_loss:.6f},{r.train_mece_loss:.6f},"
                         f"{r.train_ladder_loss:.6f},{r.val_loss:.6f},{r.val_mece_loss:.6f},"
                         f"{r.val_ladder_loss:.6f},{r.seconds:.2f}\n")

        print(f"epoch {epoch:3d}  train_loss={result.train_loss:.4f} "
              f"(mece={result.train_mece_loss:.4f} ladder={result.train_ladder_loss:.4f})  "
              f"val_loss={result.val_loss:.4f} "
              f"(mece={result.val_mece_loss:.4f} ladder={result.val_ladder_loss:.4f})  "
              f"[{result.seconds:.1f}s]")

        # CHECKPOINT POLICY. The original condition here was
        # `if val_chunks and result.val_loss < best_val`, which meant a
        # run with no val split saved NOTHING while the caller still
        # printed "Done. Best checkpoint at ...". A real 40-minute run
        # completed, reported success, and left no model on disk. Saving
        # is now unconditional; only the FILENAME and the selection rule
        # depend on whether a val split exists, so a checkpoint can never
        # be silently skipped and can never be mistaken for a
        # val-selected one when it isn't.
        if val_chunks:
            if result.val_loss < best_val:
                best_val = result.val_loss
                epochs_since_improvement = 0
                torch.save(
                    {"model": model.state_dict(), "objective": objective.state_dict(),
                     "epoch": epoch, "val_loss": best_val, "selected_by": "val_loss",
                     "config": config},
                    ckpt_dir / "best.pt",
                )
                print(f"  -> new best val_loss {best_val:.4f}, checkpointed to {ckpt_dir / 'best.pt'}",
                      flush=True)
            else:
                # EARLY STOPPING. Added after a real 5-month run whose val
                # loss bottomed at epoch 14 (0.3741) and then climbed
                # steadily to 0.4236 by epoch 19 -- five epochs of pure
                # overfitting, costing ~4 minutes and moving the model
                # further from the one that actually got checkpointed.
                # best.pt is unaffected either way (it only ever advances
                # on improvement), so this purely stops wasting time.
                epochs_since_improvement += 1
                if config.patience and epochs_since_improvement >= config.patience:
                    print(f"\nEarly stopping: {epochs_since_improvement} epoch(s) since val_loss "
                          f"improved (best {best_val:.4f}). Stopping at epoch {epoch}; best.pt "
                          f"holds the best-val model.\n", flush=True)
                    break
        else:
            # No val split: "best" is undefined, so save the latest epoch
            # under a name that says exactly that rather than pretending
            # it was selected on held-out data.
            torch.save(
                {"model": model.state_dict(), "objective": objective.state_dict(),
                 "epoch": epoch, "val_loss": None, "selected_by": "last_epoch_no_val_split",
                 "config": config},
                ckpt_dir / "last.pt",
            )

    return history


if __name__ == "__main__":
    # Illustrative entry point, mirroring build_combined_stgat_graph.py's
    # own __main__ block -- this project's real data-loading wiring
    # (GraphBuilder / FixedWindowTemporal / the parquet loaders) isn't
    # importable from a bare model-layer checkout, so actually running
    # this end to end happens from a script that first builds the graph,
    # e.g.:
    #
    #   from build_combined_stgat_graph import build_combined_graph
    #   from stg.bridge import to_torch_bundle
    #   from model.train import TrainingConfig, run_training
    #
    #   stg = build_combined_graph(trades_df, markets_df, mece_leg_prices_df, ladder_pairs_df)
    #   bundle = to_torch_bundle(stg)
    #   history = run_training(bundle, TrainingConfig())
    pass