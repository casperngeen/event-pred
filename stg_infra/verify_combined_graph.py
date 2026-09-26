"""
verify_combined_graph.py

Sanity-checks build_combined_stgat_graph.py against REAL data for a single
day, before trusting it at full scale. Everything up to this point was only
tested with synthetic data (no real trades/markets were available in the
sandbox this was built in) -- this is the step that actually confirms it
works.

Targets 2025-02-02 specifically: this is the exact date behind the
KXHIGHLAX-25FEB03 example discussed at length earlier in this project
(6-leg weather bracket, dev=+$1.02) -- a good anchor because we know
concretely what a correct result should look like. It also falls inside
both LADDER_MONTHS and MECE_MONTHS, so one run checks both mechanisms.

Run from stg_infra/ (or adjust the sys.path/imports below to match wherever
you placed build_combined_stgat_graph.py).
"""

import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

# This file lives at <repo_root>/stg_infra/verify_combined_graph.py.
# Anchoring every path to that known, fixed location -- not to whatever the
# current working directory happens to be -- means this script works
# correctly no matter which directory you're standing in when you run it.
_THIS_DIR = Path(__file__).resolve().parent      # .../event-pred/stg_infra
_REPO_ROOT = _THIS_DIR.parent                     # .../event-pred

sys.path.insert(0, str(_THIS_DIR))
from build_combined_stgat_graph import build_combined_graph
from stg.bridge import to_torch_bundle, SparseSnapshotEdgesT
from model.spatial_attention import TypeSpecificProjection, TwoChannelSpatialAttention
from model.temporal_attention import TemporalAttention
from model.output_heads import MeceOutputHead, LadderOutputHead
from model.training_objective import MaskedLegReconstructionObjective

TARGET_DATE = date(2025, 2, 2)
KNOWN_MECE_EVENT = "KXHIGHLAX-25FEB03"

MARKETS_PATH = str(_REPO_ROOT / "data/markets/markets_kalshi_even/markets_2025-02.parquet")  # Feb = even month
TRADES_PATH = str(_REPO_ROOT / "data/trades/trades_kalshi_even/trades_2025-02.parquet")

MECE_LEG_PRICES_PATH = str(_REPO_ROOT / "mece_sum_to_one_leg_prices.parquet")
LADDER_PAIRS_PATH = str(_REPO_ROOT / "pairwise_monotonicity_taker_side_results_corrected.parquet")


def load_single_day():
    print(f"Loading markets/trades for {TARGET_DATE}...")
    markets = (
        pl.scan_parquet(MARKETS_PATH)
        .filter(pl.col("created_time").dt.date() == TARGET_DATE)
        .collect()
    )
    trades = (
        pl.scan_parquet(TRADES_PATH)
        .filter(pl.col("created_time").dt.date() == TARGET_DATE)
        .collect()
    )
    print(f"  {markets.height} market rows, {trades.height} trade rows for this day\n")
    return markets, trades


def verify():
    markets, trades = load_single_day()
    mece_leg_prices = pl.read_parquet(MECE_LEG_PRICES_PATH)
    ladder_pairs = pl.read_parquet(LADDER_PAIRS_PATH)

    stg = build_combined_graph(trades, markets, mece_leg_prices, ladder_pairs)

    print("\n=== Overall summary ===")
    print(stg.summary())

    node_types, edge_types = {}, {}
    for snap in stg:
        for nid in snap.node_ids:
            n = snap.get_node(nid)
            t = n.metadata.get("node_type", "unknown") if n else "unknown"
            node_types[t] = node_types.get(t, 0) + 1
        for u, v in snap.edges:
            et = snap.nx_graph.edges[u, v].get("metadata", {}).get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

    print("\n=== Node counts by type (summed across all snapshots) ===")
    for t, n in sorted(node_types.items()):
        print(f"  {t:20s} {n}")

    print("\n=== Edge counts by type (summed across all snapshots) ===")
    for t, n in sorted(edge_types.items()):
        print(f"  {t:25s} {n}")

    print(f"\n=== Specific check: does {KNOWN_MECE_EVENT} appear as a basket hub? ===")
    found = False
    for snap in stg:
        hub_id = f"MECE:{KNOWN_MECE_EVENT}"
        node = snap.get_node(hub_id)
        if node is not None:
            found = True
            print(f"  {snap.timestamp}: sum_cents={node.features[0]:.1f}  "
                  f"deviation={node.features[1]:+.2f}  "
                  f"legs_known={node.features[2]:.0f}/{node.features[3]:.0f}  "
                  f"coverage={node.features[4]:.2f}  freshness={node.features[5]:.2f}")
    if not found:
        print(f"  NOT FOUND -- expected this hub to appear at least once given the known example. "
              f"Check that mece_sum_to_one_leg_prices.parquet actually contains {KNOWN_MECE_EVENT}, "
              f"and that its legs are present in this day's trades file.")
    else:
        print(f"  Found. Compare the deviation values above against the known dev=+1.02 result "
              f"from earlier analysis -- they won't match exactly (that was a full-day aggregate, "
              f"this is per-snapshot), but should be in a sane, comparable range.")

    print("\n=== Torch bridge check ===")
    bundle = to_torch_bundle(stg)
    print(f"features:          {tuple(bundle.features.shape)}  dtype={bundle.features.dtype}")
    print(f"mask:              {tuple(bundle.mask.shape)}  dtype={bundle.mask.dtype}")
    # adjacency is now SPARSE per snapshot (see stg/bridge.py's rewrite) --
    # there's no single dense shape to print any more, so report total real
    # edges per type instead, which is the quantity that actually matters
    # now (memory is O(this), not O(T*N^2)).
    for et, snaps in bundle.adjacency_by_type.items():
        total_edges = sum(s.edge_index.shape[1] for s in snaps)
        has_features = any(s.features is not None for s in snaps)
        print(f"adjacency[{et:20s}]: {len(snaps)} snapshot(s), {total_edges} real edge(s) total"
              f"{'  (carries features)' if has_features else '  (no features)'}")
    print(f"total nodes across time (N): {len(bundle.node_ids)}")
    print(f"total snapshots (T):         {len(bundle.timestamps)}")

    import torch
    n_nan = torch.isnan(bundle.features).sum().item()
    print(f"\nNaN check: {n_nan} NaN values in features tensor (must be 0 -- NaN should always "
          f"be replaced by 0 with mask=False marking it, never leak through as NaN itself)")
    assert n_nan == 0, "BUG: NaN leaked into the final tensor -- check to_torch_bundle's nan_to_num step"

    coverage_pct = bundle.mask.float().mean().item()
    print(f"Overall mask coverage: {coverage_pct:.1%} of (snapshot, node) slots are real observations "
          f"(the rest are padding for nodes absent at that time -- expected to be well under 100% "
          f"given how sparse trading is across {len(bundle.node_ids)} distinct nodes over the day)")

    # Cross-check: the known hub's features, read through the FULL torch bridge, should exactly
    # match what was already printed directly from the SpatioTemporalGraph above -- confirming the
    # bridge conversion (padding, dtype cast, NaN handling) didn't silently corrupt anything.
    hub_id = f"MECE:{KNOWN_MECE_EVENT}"
    if hub_id in bundle.node_ids:
        hub_idx = bundle.node_ids.index(hub_id)
        print(f"\nCross-check via torch bundle -- {KNOWN_MECE_EVENT} hub features per snapshot "
              f"(should match the values printed above exactly):")
        for t, ts in enumerate(bundle.timestamps):
            if bundle.mask[t, hub_idx]:
                f = bundle.features[t, hub_idx]
                print(f"  {ts}: sum_cents={f[0]:.1f}  deviation={f[1]:+.2f}  "
                      f"legs_known={f[2]:.0f}/{f[3]:.0f}  coverage={f[4]:.2f}  freshness={f[5]:.2f}")

    print("\n=== Spatial attention integration check (real graph, not synthetic) ===")
    embed_dim = 32
    n_heads = 4
    # ladder edge feature width now lives on each snapshot's own
    # SparseSnapshotEdgesT.features -- scan for the first snapshot that
    # actually carries any, rather than reading a single top-level
    # bundle.edge_features tensor (which no longer exists).
    ladder_fe_dim = 2
    for s in bundle.adjacency_by_type.get("ladder_monotonic", []):
        if s.features is not None:
            ladder_fe_dim = s.features.shape[-1]
            break

    torch.manual_seed(0)
    proj = TypeSpecificProjection(bundle.features.shape[-1], embed_dim)
    attn = TwoChannelSpatialAttention(embed_dim, n_heads=n_heads, ladder_edge_feature_dim=ladder_fe_dim)

    t0 = time.perf_counter()
    h = proj(bundle.features)
    out = attn(h, bundle.mask, bundle.adjacency_by_type)
    elapsed = time.perf_counter() - t0

    print(f"Input embedding shape:  {tuple(h.shape)}")
    print(f"Output embedding shape: {tuple(out.shape)}")
    print(f"Forward pass took {elapsed:.2f}s for N={len(bundle.node_ids)}, T={len(bundle.timestamps)}")
    assert out.shape == h.shape

    n_nan_out = torch.isnan(out).sum().item()
    print(f"NaN check on attention output: {n_nan_out} (must be 0)")
    assert n_nan_out == 0, "BUG: NaN in spatial attention output on real data"

    if hub_id in bundle.node_ids:
        full_cov_t = None
        for t in range(len(bundle.timestamps)):
            if bundle.mask[t, hub_idx] and bundle.features[t, hub_idx, 4] >= 0.999:  # coverage == 1.0
                full_cov_t = t
                break
        if full_cov_t is not None:
            changed = not torch.allclose(out[full_cov_t, hub_idx], h[full_cov_t, hub_idx], atol=1e-6)
            print(f"\n{KNOWN_MECE_EVENT} hub at full coverage (t={full_cov_t}): "
                  f"output {'DIFFERS from' if changed else 'MATCHES (unexpected!)'} input embedding "
                  f"-- expected to differ, since it has real leg neighbours to attend over")

    # Channel-independence check on the REAL edge structure (not random synthetic
    # density) -- find one real node with ladder edges but no MECE edges, and vice
    # versa, then confirm disabling the OTHER channel doesn't touch its output.
    #
    # Rewritten for the sparse refactor: "zeroing out" a channel now means
    # substituting a length-T list of EMPTY SparseSnapshotEdgesT for that
    # edge type (there's no dense tensor to torch.zeros_like any more), and
    # "degree" is computed by scattering each snapshot's real edge_index
    # entries into a (T, N) count tensor rather than summing a dense (N,N)
    # matrix's rows/cols.
    N_nodes = len(bundle.node_ids)
    T_snaps = len(bundle.timestamps)

    def _empty_sparse_list(length, feature_dim=None):
        fe = torch.zeros((0, feature_dim), dtype=torch.float32) if feature_dim is not None else None
        return [SparseSnapshotEdgesT(
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            weight=torch.zeros((0,), dtype=torch.float32),
            features=fe,
        ) for _ in range(length)]

    def _sparse_degree(adj_list):
        degree = torch.zeros(T_snaps, N_nodes)
        if adj_list is None:
            return degree
        for t in range(T_snaps):
            edges_t = adj_list[t]
            if edges_t.edge_index.numel() > 0:
                idx = edges_t.edge_index.reshape(-1)
                degree[t].scatter_add_(0, idx, torch.ones(idx.numel()))
        return degree

    mece_adj = bundle.adjacency_by_type.get("mece_leg_to_basket")
    mece_adj_rev = bundle.adjacency_by_type.get("mece_basket_to_leg")
    ladder_adj = bundle.adjacency_by_type.get("ladder_monotonic")

    zero_ladder = dict(bundle.adjacency_by_type)
    if ladder_adj is not None:
        zero_ladder["ladder_monotonic"] = _empty_sparse_list(T_snaps, feature_dim=ladder_fe_dim)

    zero_mece = dict(bundle.adjacency_by_type)
    if mece_adj is not None:
        zero_mece["mece_leg_to_basket"] = _empty_sparse_list(T_snaps)
        zero_mece["mece_basket_to_leg"] = _empty_sparse_list(T_snaps)

    out_no_ladder = attn(h, bundle.mask, zero_ladder)
    out_no_mece = attn(h, bundle.mask, zero_mece)

    found_ladder_only = False
    if ladder_adj is not None and mece_adj is not None:
        ladder_degree = _sparse_degree(ladder_adj)  # (T, N)
        mece_degree = _sparse_degree(mece_adj) + _sparse_degree(mece_adj_rev)
        for t in range(T_snaps):
            candidates = ((ladder_degree[t] > 0) & (mece_degree[t] == 0)).nonzero(as_tuple=True)[0]
            if candidates.numel() > 0:
                n_idx = candidates[0].item()
                unaffected_by_mece = torch.allclose(out[t, n_idx], out_no_mece[t, n_idx], atol=1e-5)
                affected_by_ladder = not torch.allclose(out[t, n_idx], out_no_ladder[t, n_idx], atol=1e-5)
                print(f"\nReal ladder-only node check ({bundle.node_ids[n_idx]}, t={t}): "
                      f"unaffected by disabling MECE = {unaffected_by_mece} (expect True), "
                      f"affected by disabling ladder = {affected_by_ladder} (expect True)")
                found_ladder_only = True
                break
    if not found_ladder_only:
        print("\n(No node found with ladder edges but zero MECE edges in this window -- "
              "channel-independence check skipped for this run, not necessarily a problem, "
              "just means this particular day's active nodes happened to overlap.)")

    print("\nDone -- if everything above looks right, the spatial attention layer is verified "
          "on real data and safe to build the temporal layer on top of.")

    print("\n=== Temporal attention integration check (real graph) ===")
    temporal = TemporalAttention(embed_dim, n_heads=n_heads, max_len=len(bundle.timestamps) + 1)

    t0 = time.perf_counter()
    h_final = temporal(out, bundle.mask)
    elapsed = time.perf_counter() - t0
    print(f"Output shape: {tuple(h_final.shape)}  (took {elapsed:.2f}s)")
    assert h_final.shape == out.shape

    n_nan_temporal = torch.isnan(h_final).sum().item()
    print(f"NaN check: {n_nan_temporal} (must be 0)")
    assert n_nan_temporal == 0

    # The critical check, on the REAL graph: perturbing a LATER real snapshot
    # must not change an EARLIER one's output. Uses whichever two real
    # timestamps are actually present, not an assumption about indices.
    if len(bundle.timestamps) >= 2:
        early_t, late_t = 0, len(bundle.timestamps) - 1
        out_perturbed = out.clone()
        out_perturbed[late_t] += 100.0
        h_perturbed = temporal(out_perturbed, bundle.mask)
        leaked = not torch.allclose(h_final[early_t], h_perturbed[early_t], atol=1e-5)
        print(f"\nCausality check on real data: perturbing {bundle.timestamps[late_t]} changed "
              f"{bundle.timestamps[early_t]}'s output? {leaked} -- MUST be False")
        assert not leaked, "CAUSALITY VIOLATION on real data -- future snapshot leaked into an earlier one"

    # End-to-end gradient check through the full real pipeline.
    loss = (h_final[bundle.mask] ** 2).mean()
    loss.backward()
    all_params = list(proj.parameters()) + list(attn.parameters()) + list(temporal.parameters())
    n_missing = sum(1 for p in all_params if p.grad is None)
    n_nan_grad = sum(1 for p in all_params if p.grad is not None and torch.isnan(p.grad).any())
    print(f"\nEnd-to-end gradient check: {n_missing} params with missing gradient, "
          f"{n_nan_grad} with NaN gradient (both must be 0)")
    assert n_missing == 0 and n_nan_grad == 0

    print("\nFull pipeline (projection -> spatial -> temporal) verified on real data, "
          "forward and backward, with causality confirmed.")

    print("\n=== Output heads integration check (real graph) ===")
    mece_head = MeceOutputHead(embed_dim)
    ladder_head = LadderOutputHead(embed_dim)

    mece_leg_to_basket = bundle.adjacency_by_type.get("mece_leg_to_basket")
    ladder_adj_real = bundle.adjacency_by_type.get("ladder_monotonic")

    mece_out = mece_head(h_final, mece_leg_to_basket) if mece_leg_to_basket is not None else {}
    ladder_out = ladder_head(h_final, ladder_adj_real) if ladder_adj_real is not None else {}

    n_baskets_checked, n_pairs_checked = 0, 0
    for t, res in mece_out.items():
        if res is None:
            continue
        for hub in res["hub_idx"].unique():
            group_sum = res["fair_price"][res["hub_idx"] == hub].sum().item()
            assert abs(group_sum - 1.0) < 1e-4, (
                f"MECE sum-to-1 violated on real data at t={t}, hub={bundle.node_ids[hub]}: sum={group_sum}"
            )
            n_baskets_checked += 1
    print(f"MECE sum-to-1 constraint verified across {n_baskets_checked} real (snapshot, basket) groups "
          f"-- every single one summed to 1.0 exactly")

    for t, res in ladder_out.items():
        if res is None:
            continue
        violated = (res["fair_a"] < res["fair_b"] - 1e-5).sum().item()
        assert violated == 0, f"Ladder monotonicity violated on real data at t={t}: {violated} pairs"
        n_pairs_checked += res["fair_a"].numel()
    print(f"Ladder monotonicity constraint verified across {n_pairs_checked} real (snapshot, pair) "
          f"instances -- fair_a >= fair_b held for every single one")

    # Specific check: does the known KXHIGHLAX basket actually show up in the head's output?
    if hub_id in bundle.node_ids:
        hub_g = bundle.node_ids.index(hub_id)
        for t, res in mece_out.items():
            if res is not None and (res["hub_idx"] == hub_g).any():
                legs_here = res["leg_idx"][res["hub_idx"] == hub_g]
                prices_here = res["fair_price"][res["hub_idx"] == hub_g]
                print(f"\n{KNOWN_MECE_EVENT} predicted fair prices at {bundle.timestamps[t]}:")
                for leg, price in zip(legs_here.tolist(), prices_here.tolist()):
                    print(f"    {bundle.node_ids[leg]:35s}  {price:.3f}")

    # Fresh forward pass for this gradient check specifically -- h_final above
    # was already backwarded through in the temporal attention section, and
    # its computation graph has been freed; a real training loop only ever
    # does one forward + one backward per step anyway, so this mirrors that
    # more accurately than trying to chain a second backward onto the same
    # freed graph.
    h_grad_check = proj(bundle.features)
    h_grad_check = attn(h_grad_check, bundle.mask, bundle.adjacency_by_type)
    h_grad_check = temporal(h_grad_check, bundle.mask)
    mece_out_gc = mece_head(h_grad_check, mece_leg_to_basket) if mece_leg_to_basket is not None else {}
    ladder_out_gc = ladder_head(h_grad_check, ladder_adj_real) if ladder_adj_real is not None else {}

    loss = torch.tensor(0.0)
    for t, res in mece_out_gc.items():
        if res is not None:
            loss = loss + (res["fair_price"] ** 2).sum()
    for t, res in ladder_out_gc.items():
        if res is not None:
            loss = loss + (res["fair_a"] ** 2).sum() + (res["fair_b"] ** 2).sum()

    for p in (list(proj.parameters()) + list(attn.parameters()) + list(temporal.parameters())
              + list(mece_head.parameters()) + list(ladder_head.parameters())):
        p.grad = None  # fresh gradients for this check, not accumulated from earlier sections
    loss.backward()

    all_params = (
        list(proj.parameters()) + list(attn.parameters()) + list(temporal.parameters())
        + list(mece_head.parameters()) + list(ladder_head.parameters())
    )
    n_missing = sum(1 for p in all_params if p.grad is None)
    n_nan_grad = sum(1 for p in all_params if p.grad is not None and torch.isnan(p.grad).any())
    print(f"\nFull-stack gradient check (all 5 components): {n_missing} missing, {n_nan_grad} NaN (both must be 0)")
    assert n_missing == 0 and n_nan_grad == 0

    print("\nFull pipeline, including constraint-respecting output heads, verified end to end "
          "on real data -- ready for the training objective itself.")

    print("\n=== Masked-leg reconstruction objective check (real graph) ===")
    raw_feature_dim = bundle.features.shape[-1] - 1  # everything except the type-indicator slot
    objective = MaskedLegReconstructionObjective(raw_feature_dim, mask_ratio=0.15)
    node_types_real = bundle.features[..., -1]

    def run_stack(feat):
        hh = proj(feat)
        hh = attn(hh, bundle.mask, bundle.adjacency_by_type)
        hh = temporal(hh, bundle.mask)
        return mece_head(hh, mece_leg_to_basket) if mece_leg_to_basket is not None else {}

    gen = torch.Generator().manual_seed(7)
    masked_features, target_mask, _ = objective.select_and_mask(
        bundle.features, bundle.mask, node_types_real, generator=gen
    )
    n_masked = target_mask.sum().item()
    print(f"Masked {n_masked} real (snapshot, ticker) positions this run (mask_ratio=0.15)")

    out1 = run_stack(masked_features)
    true_prices_real = (bundle.features[..., 0] / 100.0).clamp(0, 1)  # yes_price cents -> [0,1]
    loss = objective.compute_loss(out1, target_mask, true_prices_real)
    print(f"Reconstruction loss: {loss.item():.4f}")

    # The critical check, on REAL data: dramatically perturb the TRUE
    # (pre-masking) value of every masked leg, re-mask the SAME positions,
    # and confirm every prediction is byte-for-byte unchanged. If it isn't,
    # the true value leaked through somewhere despite masking.
    if n_masked > 0:
        features_perturbed = bundle.features.clone()
        pos = target_mask.nonzero(as_tuple=False)
        for t, n in pos.tolist():
            features_perturbed[t, n, :raw_feature_dim] = 999.0
        gen2 = torch.Generator().manual_seed(7)
        masked_features2, target_mask2, _ = objective.select_and_mask(
            features_perturbed, bundle.mask, node_types_real, generator=gen2
        )
        assert torch.equal(target_mask, target_mask2), "mask selection was not reproducible with the same seed"
        out2 = run_stack(masked_features2)

        leaked = False
        for t, res in out1.items():
            if res is None or out2.get(t) is None:
                continue
            if not torch.allclose(res["fair_price"], out2[t]["fair_price"], atol=1e-6):
                leaked = True
        print(f"\nNo-leakage check on real data: perturbing masked legs' true values changed any "
              f"prediction? {leaked} -- MUST be False")
        assert not leaked, "LEAKAGE on real data: a masked leg's true value reached the model"

    if n_masked > 0:
        loss.backward()
        grad_ok = objective.mask_token.grad is not None and (objective.mask_token.grad != 0).any().item()
        print(f"\nmask_token gradient exists and nonzero? {grad_ok}")
        assert grad_ok
    else:
        print("\n(0 positions were masked by chance this run -- nothing to compute a gradient over. "
              "Not a bug: with mask_ratio=0.15, this can happen on a small eligible pool by pure chance. "
              "Should not occur at the real 788-node scale, where hundreds of positions are eligible.)")

    print("\nFull training objective verified on real data: masking prevents leakage, loss computes, "
          "gradients reach the mask token. The STGAT backbone is complete and ready to actually train.")

    print("\n=== Ladder-mechanism masked reconstruction check (real graph) ===")
    # The original check above only ever exercised compute_mece_loss (this
    # objective was MECE-only when this script was first written) -- this
    # re-runs the SAME masked positions through compute_ladder_loss, closing
    # the same real-data coverage gap the ladder loss itself closed
    # synthetically in verify_training_loop.py.
    #
    # DELIBERATELY recomputes masked_features from scratch here (same seed
    # -> same target_mask, verified below) rather than reusing the MECE
    # section's masked_features tensor: that tensor's OWN creation op
    # (select_and_mask's mask_token assignment) is an ancestor of the graph
    # `loss.backward()` already consumed a few lines above, and PyTorch
    # frees an autograd graph's saved buffers after backward() unless
    # retain_graph=True -- reusing that same tensor here would try to
    # backward through that already-freed ancestor a second time
    # (`RuntimeError: Trying to backward through the graph a second time`).
    # Same fix already applied once before in this file, for the same
    # reason -- see the "Fresh forward pass for this gradient check
    # specifically" comment further up.
    def run_ladder_stack(feat):
        hh = proj(feat)
        hh = attn(hh, bundle.mask, bundle.adjacency_by_type)
        hh = temporal(hh, bundle.mask)
        return ladder_head(hh, ladder_adj_real) if ladder_adj_real is not None else {}

    gen_ladder = torch.Generator().manual_seed(7)
    masked_features_ladder, target_mask_ladder, _ = objective.select_and_mask(
        bundle.features, bundle.mask, node_types_real, generator=gen_ladder
    )
    assert torch.equal(target_mask, target_mask_ladder), "mask selection drifted for the ladder check"

    ladder_out1 = run_ladder_stack(masked_features_ladder)
    ladder_loss = objective.compute_ladder_loss(ladder_out1, target_mask, true_prices_real)
    print(f"Ladder reconstruction loss: {ladder_loss.item():.4f}")

    if n_masked > 0 and ladder_adj_real is not None:
        gen_ladder2 = torch.Generator().manual_seed(7)
        masked_features2_ladder, target_mask2_ladder, _ = objective.select_and_mask(
            features_perturbed, bundle.mask, node_types_real, generator=gen_ladder2
        )
        assert torch.equal(target_mask, target_mask2_ladder), "mask selection drifted for the ladder no-leakage check"
        ladder_out2 = run_ladder_stack(masked_features2_ladder)
        ladder_leaked = False
        for t, res in ladder_out1.items():
            res2 = ladder_out2.get(t)
            if res is None or res2 is None:
                continue
            if not torch.allclose(res["fair_a"], res2["fair_a"], atol=1e-6) or \
               not torch.allclose(res["fair_b"], res2["fair_b"], atol=1e-6):
                ladder_leaked = True
        print(f"No-leakage check (ladder head) on real data: perturbing masked legs' true values "
              f"changed any prediction? {ladder_leaked} -- MUST be False")
        assert not ladder_leaked, "LEAKAGE on real data: a masked leg's true value reached the ladder head"

        if ladder_loss.item() > 0:
            for p in objective.parameters():
                p.grad = None
            ladder_loss.backward()
            grad_ok = objective.mask_token.grad is not None and (objective.mask_token.grad != 0).any().item()
            print(f"mask_token gradient (via ladder loss) exists and nonzero? {grad_ok}")
            assert grad_ok
    print("\nLadder mechanism's masked-reconstruction path verified on real data -- both mechanisms "
          "now covered by the training objective, not just MECE.")

    print("\n=== Training loop / chunking integration check (real graph) ===")
    # This one-day window is only 10 snapshots, so chunking has nothing
    # meaningful to cut here (chunk_len will exceed the whole window) --
    # this section exists to confirm the ACTUAL model.train wiring (not a
    # hand-assembled proj/attn/temporal/head stack, as every check above
    # this point used) runs correctly against a real bundle end to end,
    # the same way it was already confirmed against synthetic data at
    # T=2000 in verify_training_loop.py. Run this against a real
    # multi-month bundle (run_real_training.py) to actually exercise
    # multiple chunks and the train/val/test split.
    from model.chunking import build_split_ranges, chunk_ranges, describe_chunks
    from model.train import STGATBackbone, TrainingConfig, _forward_chunk_losses

    ranges = build_split_ranges(bundle.timestamps)
    chunks = chunk_ranges(ranges, chunk_len=max(len(bundle.timestamps), 24), min_chunk_len=1)
    print(describe_chunks(chunks))

    real_model = STGATBackbone(padded_feature_width=bundle.features.shape[-1], embed_dim=embed_dim,
                                n_heads=n_heads, max_len=max(len(bundle.timestamps), 24))
    real_objective = MaskedLegReconstructionObjective(raw_feature_dim=raw_feature_dim, mask_ratio=0.15)
    real_optimizer = torch.optim.Adam(
        list(real_model.parameters()) + list(real_objective.parameters()), lr=1e-3
    )
    gen3 = torch.Generator().manual_seed(0)
    chunk_tensors = {
        "features": bundle.features, "mask": bundle.mask,
        "adjacency_by_type": bundle.adjacency_by_type,
    }
    total, mece_l, ladder_l = _forward_chunk_losses(real_model, real_objective, chunk_tensors, gen3)
    real_optimizer.zero_grad()
    total.backward()
    real_optimizer.step()
    print(f"One real optimizer step completed: total={total.item():.4f} "
          f"(mece={mece_l.item():.4f}, ladder={ladder_l.item():.4f})")
    print("\nTraining-loop wiring (model.train.STGATBackbone + chunking) verified against a real "
          "bundle -- run run_real_training.py against a real multi-month range for an actual "
          "training run with genuine chunking and a real train/val/test split.")


if __name__ == "__main__":
    verify()