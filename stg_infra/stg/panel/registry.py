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
    role: str                  # "trigger" | "target_only"
    role_reason: str
    same_release: Optional[str] = None   # official-release group key
    aliases: tuple[str, ...] = field(default_factory=tuple)


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
    SeriesSpec("WTI", "wti", "bucket", "trigger",
               "85% bucket ladder; weekly cadence, highest event count",
               "wti_settle"),
    SeriesSpec("WTIW", "wti_weekly", "bucket", "trigger",
               "weekly WTI variant", "wti_settle"),
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
        .sort("n_events", descending=True)
    )
    return g


def universe(min_events: int = 5, markets: Optional[pl.DataFrame] = None) -> list[str]:
    """Canonical node names in the curated macro list with >= min_events IS events.

    N is a *consequence* of ``min_events`` and the alias table, reported here
    rather than hardcoded.
    """
    counts = event_counts(markets)
    keep = (
        counts.filter(pl.col("in_universe") & (pl.col("n_events") >= min_events))
        ["canon"].to_list()
    )
    # deterministic order: by descending event count
    return keep


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
