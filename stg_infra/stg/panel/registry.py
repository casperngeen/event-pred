"""The macro-series universe: canonicalisation, universe filter, same-release
groups, and trigger/target role hints.

This is the single place that decides *what a node is*. Everything downstream
(surprise panel, node panel, structure estimator, graph strategies) asks this
module rather than re-deriving series names from ticker prefixes.

Design notes
------------
* **Alias merge before the ``min_events`` filter.** ``PROLLS`` and ``PAYROLLS``
  are the same BLS release under two ticker conventions; merging them first
  raises events-per-node, which matters for the capacity accounting in the
  AGCRN study. Same for ``JOBLESS`` (old single-threshold binaries) and
  ``KXJOBLESSCLAIMS`` (new multi-strike ladders).
* **``rules_primary`` is not in the archive** (checked: the markets parquet has
  no such column), so canonicalisation is a hand-curated table informed by
  ``title`` / ``yes_sub_title``, not an automated rule.
* **Role hints are hints.** Whether a series can actually act as a *trigger*
  is decided empirically by :mod:`stg.panel.surprise` (does it yield >= n
  events with a computable signed surprise). The ``role`` field here records
  the *expected* outcome and the reason, so "correctly a target-only series"
  stays distinguishable from "surprise builder has a bug".

* **Triggers and targets are separate universes.** They are not the same
  filter relaxed by a threshold — they have different *requirements*:

  - a **trigger** needs a recoverable implied distribution *and* a resolved
    outcome, so that ``surprise = actual - implied_mean`` means something;
  - a **target** needs only a price path around the trigger's resolution.

  A target is therefore strictly cheaper, which is why the target universe can
  include series that could never be triggers. Two fields express this:

  ``can_trigger``
      Structural. ``False`` for asset-price series, which are targets *by
      construction* — their resolution is a price snapshot, not an information
      event, so a "surprise" computed against them is the artefact
      ``edge_economics.md`` §2(b) diagnosed on WTI. These never enter the
      surprise panel.

  ``scheduled_release``
      The §2(b) research criterion itself: does resolution constitute a
      *scheduled information release*, rather than merely a resolution
      timestamp? Annotated honestly per series (so ``WTI`` is ``False``), but
      it only *gates* under ``trigger_universe(require_release=True)`` — which
      defaults off, leaving current results unchanged. Turning it on is the
      one-line form of the open "WTI in Universe A?" decision in
      ``reports/TODO.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import polars as pl

from stg.panel._io import load_markets

# --------------------------------------------------------------------------
# Canonical node -> {raw ticker-prefix aliases}. Applied before any filtering.
# --------------------------------------------------------------------------
ALIASES: dict[str, tuple[str, ...]] = {
    "PAYROLLS": ("PAYROLLS", "PROLLS"),
    "JOBLESSCLAIMS": ("JOBLESSCLAIMS", "JOBLESS"),
}
# reverse index: raw prefix -> canonical
_RAW_TO_CANON: dict[str, str] = {
    raw: canon for canon, raws in ALIASES.items() for raw in raws
}


def canonical(series_raw: str) -> str:
    return _RAW_TO_CANON.get(series_raw, series_raw)


# --------------------------------------------------------------------------
# The curated macro universe. Anything not listed here is not a node, no
# matter how many events it has (keeps sports/crypto/weather and one-off
# political contracts out without a fragile keyword filter).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SeriesSpec:
    canon: str
    series_type: str           # matches stg/events/* labels where one exists
    kind: str                  # "threshold" | "bucket" | "categorical"
    role: str                  # "trigger" | "target_only"   (a hint, not a gate)
    role_reason: str
    same_release: Optional[str] = None   # official-release group key
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # -- trigger eligibility (see module docstring) -------------------------
    can_trigger: bool = True        # structural; False => target-only always
    scheduled_release: bool = True  # §2(b) criterion; gates only on request


_SPECS: tuple[SeriesSpec, ...] = (
    # ---- BLS CPI report (all resolve from one print, same morning) --------
    SeriesSpec("CPI", "cpi_mom", "threshold", "trigger",
               "headline CPI MoM ladder, deep and regular", "bls_cpi"),
    SeriesSpec("CPICORE", "core_cpi_mom", "threshold", "trigger",
               "core CPI MoM ladder", "bls_cpi"),
    SeriesSpec("CPIYOY", "cpi_yoy", "threshold", "trigger",
               "headline CPI YoY ladder", "bls_cpi"),
    SeriesSpec("CPICOREYOY", "core_cpi_yoy", "threshold", "trigger",
               "core CPI YoY ladder", "bls_cpi"),
    SeriesSpec("CPIGAS", "cpi_gas", "threshold", "trigger",
               "CPI gasoline subcomponent", "bls_cpi"),
    SeriesSpec("CPISHELTER", "cpi_shelter", "threshold", "trigger",
               "CPI shelter subcomponent", "bls_cpi"),
    SeriesSpec("CPIFOOD", "cpi_food", "threshold", "target_only",
               "subcomponent ladder, often < 3 fresh legs pre-resolution",
               "bls_cpi"),
    SeriesSpec("CPIAPPAREL", "cpi_apparel", "threshold", "target_only",
               "thin subcomponent ladder", "bls_cpi"),
    SeriesSpec("CPIUSEDCAR", "cpi_usedcar", "threshold", "target_only",
               "thin subcomponent ladder", "bls_cpi"),
    # ---- BLS Employment Situation (payrolls + U3, one release) ------------
    SeriesSpec("PAYROLLS", "payrolls", "threshold", "trigger",
               "nonfarm payrolls; strike spacing ~25-100k",
               "bls_employment", ("PROLLS",)),
    SeriesSpec("U3", "unemployment", "threshold", "trigger",
               "unemployment rate; falsification cell (dovish sign flips)",
               "bls_employment"),
    # ---- ADP (private, precedes NFP) -------------------------------------
    SeriesSpec("ADP", "adp", "threshold", "trigger",
               "ADP private payrolls", "adp"),
    # ---- weekly initial jobless claims (its own BLS UI release) ----------
    SeriesSpec("JOBLESSCLAIMS", "jobless_claims", "threshold", "trigger",
               "new KXJOBLESSCLAIMS ladders 'At least X' are deep; old JOBLESS "
               "'Above X' binaries add events but no cross-section",
               "bls_claims", ("JOBLESS",)),
    # ---- BEA -----------------------------------------------------------
    SeriesSpec("GDP", "gdp", "threshold", "trigger",
               "quarterly GDP advance/second/third estimates", "bea_gdp"),
    SeriesSpec("PCECORE", "pce_core", "threshold", "target_only",
               "core PCE ladder, few events with a deep cross-section",
               "bea_pce"),
    # ---- ISM ---------------------------------------------------------
    SeriesSpec("ISMPMI", "ism_pmi", "threshold", "target_only",
               "only ~7 IS events", "ism"),
    # ---- Oil (front-month WTI settle) -----------------------------------
    # scheduled_release=False: the settle is a price snapshot, not a data
    # release. edge_economics.md §2(b) -- WTI->CPIGAS, the most mechanically
    # certain link in the grid, is flat (rho=+0.04, n=80). Left *enabled* as a
    # trigger (require_release defaults off) because "WTI in Universe A?" is
    # still an open decision in reports/TODO.md; flipping the gate is how that
    # decision gets made, not this annotation.
    SeriesSpec("WTI", "wti", "bucket", "trigger",
               "85% bucket ladder; weekly cadence, highest event count",
               "wti_settle", scheduled_release=False),
    SeriesSpec("WTIW", "wti_weekly", "bucket", "trigger",
               "weekly WTI variant", "wti_settle", scheduled_release=False),
    # ---- Fed --------------------------------------------------------
    SeriesSpec("FED", "fed_level", "threshold", "target_only",
               "numeric 'Above X%' rate-level ladder. A surprise is computable "
               "(~31 events) but tiny: median |surprise| ~1.5bps vs CPI's ~7bps, "
               "and ~10x below FED's own implied std -- the policy level is "
               "public and mostly anticipated. Good target, weak trigger; the "
               "estimator still tests it and FED-> edges are expected to be "
               "null.", "fomc"),
    SeriesSpec("FEDDECISION", "decision_bps", "categorical", "target_only",
               "ordered ladder over bps changes; modelled by side "
               "(P(hike)/P(cut)), not as a magnitude surprise", "fomc"),
    SeriesSpec("RATECUT", "ratecut", "categorical", "target_only",
               "binary 'Cuts' contract, no numeric axis", "fomc"),
    # ---- Asset-price targets (can_trigger=False) -------------------------
    # Targets by construction: a macro surprise is *news to* these markets, so
    # they carry the response but can never generate one. This is the
    # Kuttner (2001) / Gurkaynak-Sack-Swanson (2005) design -- scheduled macro
    # surprise -> asset-price response -- already cited in research_summary.md
    # §6.2/§7.3, so the channel is available a priori rather than fitted.
    #
    # Two properties the macro targets lack. (i) Liquidity: median volume per
    # ticker is 3,288 (INXD) and 6,999 (NASDAQ100D) against the macro
    # contracts' 1-10 trades/day band. (ii) Identification: an equity index
    # does not resolve off the CPI print, so the "both contracts resolve from
    # the same official release" confound that makes the CPI clique
    # uninterpretable (reports/TODO.md §Identification) simply does not arise.
    #
    # These close intraday the same day (20:00 / 21:00 UTC) while a BLS print
    # lands at 13:30 UTC, so the target contract is live at the trigger's
    # resolution and matches at a gap of ~0.3 days.
    #
    # Membership here is bounded by ``data/trades/``, not by what Kalshi
    # listed. The public trades API retains ~67 days (verified 2026-09-08:
    # earliest served 2026-07-02), so any series absent from the local archive
    # can never be reconstructed for the in-sample window. TNOTED / TNOTEW /
    # NASDAQ100D are exactly that case -- listed, with real aggregate
    # ``volume`` in the markets metadata, but zero trades locally and
    # unrecoverable. They are deliberately NOT registered: the 10-year yield
    # would otherwise be the cleanest policy-path target available.
    #
    # ``INX`` is also omitted: it runs on the *same* 464 event days as INXU
    # (parallel ticker families, not successive generations), so registering
    # both would double-count the same underlying. INXD's days are largely its
    # own, so it stands as a separate node rather than an alias.
    #
    # Caveat for multiple testing: INXU and INXD track one underlying, so they
    # are not independent targets even where their event days differ.
    SeriesSpec("INXU", "sp500_daily", "threshold", "target_only",
               "S&P 500 daily close, threshold ladder ('above X'). 646,547 "
               "trades over 2022-08 -> 2025-11 -- the deepest continuous "
               "asset-price target in the archive.",
               same_release=None, can_trigger=False, scheduled_release=False),
    SeriesSpec("INXD", "sp500_daily_bucket", "bucket", "target_only",
               "S&P 500 daily close, bucket ladder. 565,941 trades but stops "
               "2024-12-31; complements INXU on mostly disjoint event days.",
               same_release=None, can_trigger=False, scheduled_release=False),
    SeriesSpec("NASDAQ100U", "nasdaq100_daily", "threshold", "target_only",
               "Nasdaq-100 daily close, threshold ladder. 214,804 trades but "
               "only from 2024-10, so it spans the late folds alone.",
               same_release=None, can_trigger=False, scheduled_release=False),
)

SPECS: dict[str, SeriesSpec] = {s.canon: s for s in _SPECS}


def same_release_groups() -> dict[str, list[str]]:
    """group key -> [canonical series] that resolve from the same official print."""
    out: dict[str, list[str]] = {}
    for s in _SPECS:
        if s.same_release:
            out.setdefault(s.same_release, []).append(s.canon)
    return {k: v for k, v in out.items() if len(v) > 1}


def same_release_pairs() -> set[frozenset[str]]:
    """Unordered {A, B} pairs sharing an official release."""
    pairs: set[frozenset[str]] = set()
    for members in same_release_groups().values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add(frozenset((members[i], members[j])))
    return pairs


def is_same_release(a: str, b: str) -> bool:
    return frozenset((a, b)) in same_release_pairs()


# --------------------------------------------------------------------------
# Universe: apply aliases, count events, filter by min_events.
# --------------------------------------------------------------------------
def event_counts(markets: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """canonical series -> in-sample resolved-event count (post alias-merge).

    Columns: ``canon, n_events, n_contracts, in_universe``.
    """
    mk = markets if markets is not None else load_markets(is_only=True)
    mk = mk.with_columns(
        pl.col("series_raw")
        .map_elements(canonical, return_dtype=pl.Utf8)
        .alias("canon")
    )
    g = (
        mk.group_by("canon")
        .agg(
            pl.col("event_ticker").n_unique().alias("n_events"),
            pl.len().alias("n_contracts"),
        )
        .with_columns(pl.col("canon").is_in(list(SPECS)).alias("in_universe"))
        # ``canon`` breaks ties. Without it the order of equal-event-count
        # series depends on the order polars happens to emit groups in, so
        # ``universe()`` returned a different list on identical calls (CPIFOOD
        # / CPIUSEDCAR at 21 events, CPIGAS / CPISHELTER / GDP at 18). That
        # order sets the grid iteration order, and any seeded RNG consumed in
        # grid order inherits the nondeterminism -- the same failure mode
        # already fixed in panel/targets.py::representative_tickers.
        .sort(["n_events", "canon"], descending=[True, False])
    )
    return g


def _eligible(min_events: int, markets: Optional[pl.DataFrame],
              predicate) -> list[str]:
    """Curated series with >= min_events IS events, filtered by ``predicate``.

    N is a *consequence* of ``min_events`` and the alias table, reported by
    callers rather than hardcoded. Order is by descending event count.
    """
    counts = event_counts(markets)
    keep = (
        counts.filter(pl.col("in_universe") & (pl.col("n_events") >= min_events))
        ["canon"].to_list()
    )
    return [c for c in keep if predicate(SPECS[c])]


def target_universe(min_events: int = 5,
                    markets: Optional[pl.DataFrame] = None) -> list[str]:
    """Series usable as *targets*: every curated node clearing ``min_events``.

    A target needs only a price path around the trigger's resolution, so
    nothing is excluded on information-content grounds — including the
    asset-price series, which exist in the registry for exactly this role.
    """
    return _eligible(min_events, markets, lambda s: True)


def trigger_universe(min_events: int = 5,
                     markets: Optional[pl.DataFrame] = None,
                     *, require_release: bool = False) -> list[str]:
    """Series usable as *triggers*: those that can carry a signed surprise.

    ``can_trigger`` is structural and always applies, so asset-price series are
    never triggers. ``require_release`` additionally applies the
    ``edge_economics.md`` §2(b) criterion — resolution must be a *scheduled
    information release*, not merely a resolution timestamp — which drops
    WTI/WTIW. It defaults off so this filter is a no-op against the results
    reported in ``direction_study.md``; see the open decision in
    ``reports/TODO.md``.
    """
    return _eligible(
        min_events, markets,
        lambda s: s.can_trigger and (s.scheduled_release or not require_release),
    )


def universe(min_events: int = 5, markets: Optional[pl.DataFrame] = None) -> list[str]:
    """Deprecated alias for :func:`target_universe`.

    Kept so existing callers and saved scripts keep working. New code should
    name the role it means — the two universes are no longer the same list.
    """
    return target_universe(min_events, markets)


def ticker_prefixes(canon: str) -> tuple[str, ...]:
    """All raw ticker prefixes that map to this canonical node."""
    spec = SPECS.get(canon)
    base = (canon,)
    extra = spec.aliases if spec else ()
    merged = ALIASES.get(canon, ())
    return tuple(dict.fromkeys(base + extra + merged))


def series_filter_expr(canon: str, col: str = "series_raw") -> pl.Expr:
    """Polars predicate selecting rows belonging to a canonical node."""
    return pl.col(col).is_in(list(ticker_prefixes(canon)))
