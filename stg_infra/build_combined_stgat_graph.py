"""
build_combined_stgat_graph.py

Example wiring both mechanisms into ONE GraphBuilder, producing a single
combined graph where MECE baskets (hyperedges via a shared hub) and ladder
chains (directed edges) coexist -- matching the STGAT design decision that
both mechanisms share one temporal attention backbone, even though they use
different edge topologies.

Node types present in the resulting graph, distinguishable via
node.metadata["node_type"]:
    "ticker"        -- an individual market leg (from KalshiTickerNodes)
    "mece_basket"   -- a synthetic hub, one per validated MECE basket

Edge types present, distinguishable via edge.metadata["edge_type"]:
    "mece_leg_to_basket" / "mece_basket_to_leg"  -- bidirectional hyperedge spokes
    "ladder_monotonic"                            -- one-directional, leg_a -> leg_b

This is illustrative wiring, not a runnable script on its own -- it assumes
trades_df/markets_df/leg_prices_df/pairs_df are already loaded from your
actual data (see data_windows.py and the pipeline scripts already built for
loading the right months per mechanism).
"""

import polars as pl

from stg.builders.builder import GraphBuilder
from stg.temporal.strategies import FixedWindowTemporal  # adjust if using a different temporal strategy
from stg.nodes.kalshi import KalshiTickerNodes, KalshiMeceBasketNodes
from stg.edges.kalshi import KalshiMeceHyperedges, KalshiLadderChainEdges


def build_combined_graph(
    trades_df: pl.DataFrame,
    markets_df: pl.DataFrame,
    mece_leg_prices_df: pl.DataFrame,
    ladder_pairs_df: pl.DataFrame,
):
    """
    Parameters
    ----------
    trades_df : the raw trades table for whichever MASTER_MONTHS window
        covers both mechanisms (see data_windows.py) -- must include every
        ticker that's either a ladder leg or a MECE basket leg.
    markets_df : the raw markets table, same window -- used by
        KalshiTickerNodes for metadata/close_time.
    mece_leg_prices_df : mece_sum_to_one_leg_prices.parquet (or equivalent)
        -- defines validated MECE basket membership.
    ladder_pairs_df : pairwise_monotonicity_taker_side_results_corrected.parquet
        (or equivalent) -- defines validated ladder pairs. leg_a/leg_b
        required; historical violation columns optional but recommended
        as edge features.
    """
    # Every ticker either mechanism actually cares about -- passed to
    # KalshiTickerNodes so ONLY these specific tickers carry forward across
    # quiet windows (not every ticker that ever trades, which would explode
    # node counts for tickers no mechanism here uses). See KalshiTickerNodes'
    # docstring for why this carry-forward was added: without it, a MECE
    # basket's softmax group at a low-freshness snapshot could degenerate to
    # a single member, since a leg with no fresh trade was never even a node
    # to connect to its hub.
    tracked_tickers = set(mece_leg_prices_df["ticker"].unique().to_list())
    tracked_tickers |= set(ladder_pairs_df["leg_a"].unique().to_list())
    tracked_tickers |= set(ladder_pairs_df["leg_b"].unique().to_list())

    builder = (
        GraphBuilder()
        .with_temporal(FixedWindowTemporal(every="2h"))  # match whatever window size the rest of the pipeline uses
        .with_nodes(KalshiTickerNodes(markets_df=markets_df, tracked_tickers=tracked_tickers))
        .with_nodes(KalshiMeceBasketNodes(mece_leg_prices_df))
        .with_edges(KalshiMeceHyperedges(mece_leg_prices_df))
        .with_edges(KalshiLadderChainEdges(
            ladder_pairs_df,
            feature_cols=["same_side_yes_violation", "same_side_yes_gap"],
        ))
        .with_auxiliary_time_col("markets", "created_time")
    )

    stg = builder.build(trades_df, auxiliary={"markets": markets_df})

    print(stg.summary())
    return stg


if __name__ == "__main__":
    # Wire up your actual data loading here, e.g.:
    #
    # trades_df = load_trades_for(MASTER_MONTHS)
    # markets_df = load_markets_for(MASTER_MONTHS)
    # mece_leg_prices_df = pl.read_parquet("mece_sum_to_one_leg_prices.parquet")
    # ladder_pairs_df = pl.read_parquet("pairwise_monotonicity_taker_side_results_corrected.parquet")
    # stg = build_combined_graph(trades_df, markets_df, mece_leg_prices_df, ladder_pairs_df)
    #
    # feature_tensor, mask = stg.feature_tensor_padded()   # (T, N, F), (T, N)
    # adjacency_tensor = stg.adjacency_tensor()             # (T, N, N)
    pass
