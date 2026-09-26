"""
model/run_guards.py

CHEAP ASSERTIONS THAT MAKE THE KNOWN BUG CLASSES IMPOSSIBLE TO REPEAT.

Every guard here exists because the corresponding mistake was actually
made in this project and was caught by luck, by a reader noticing two
numbers that should have differed, or not at all until much later. The
purpose is not general defensive programming; it is to make the next long
run on more months fail LOUDLY at second zero instead of producing a
plausible-looking table that has to be disbelieved afterwards.

Each function raises SystemExit with an explanation rather than returning
a status, because a run that trips one of these should not continue.

THE FIVE FAILURES BEING GUARDED, and what caught them the first time:

  1. Split leakage across an added month. Not yet hit, and the one that
     would invalidate the whole thesis rather than one table. Adding
     months without moving TRAIN_END/VAL_END silently reclassifies them.
     -> check_split_integrity

  2. A month bundle built with a different edge builder. Not yet hit.
     Would degrade one channel on some months only, invisibly.
     -> check_month_schema

  3. An ablation flag that ablated nothing (a --drop-edges choices list
     containing "leg_to_hub"/"hub_to_leg", neither of which is a real
     key, while omitting the real "mece_basket_to_leg"). Caught only
     because the "no graph" run came back bit-for-bit identical to the
     previous one across all 18 epochs.
     -> check_drop_edges (and the BACKBONE WILL SEE line it prints)

  4. Training weights leaking into the REPORTED metric. The mover weight
     was applied to the reported persistence baseline as well as to the
     loss, tripling it (0.006012 = 3 x 0.002004) and making a true 1.070x
     read as 0.996x. Caught by hand, three runs later.
     -> check_ladder_persistence_decomposition, BaselineFingerprint

  5. A metric that is printed but never written to history.csv
     (ladder_moved), so the selected epoch's value was unrecoverable
     after the run ended.
     -> check_history_columns

Also here: verdict(), because the same one-sided confidence-interval bug
has now been written twice in two different evaluation scripts. There
should be exactly one place that turns an interval into a word.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_windows import split_for  # noqa: E402

SPLIT_ORDER = ("train", "val", "test")


def _month_str(ts) -> str:
    return f"{ts.year:04d}-{ts.month:02d}"


def _die(title: str, *lines: str) -> "None":
    body = "\n".join(f"  {ln}" for ln in lines)
    raise SystemExit(f"\nGUARD FAILED -- {title}\n{body}\n")


# ---------------------------------------------------------------------
# 1. Split integrity
# ---------------------------------------------------------------------

def check_split_integrity(timestamps: Sequence, chunks: Sequence, *,
                          verbose: bool = True) -> str:
    """Chunks are disjoint, homogeneous, and chronologically ordered by
    split.

    THREE SEPARATE THINGS, all of which have to hold:

      (a) DISJOINT AND IN RANGE. Two chunks sharing an index would train
          on a val snapshot without any label saying so.

      (b) HOMOGENEOUS. Every timestamp inside a chunk must map, through
          data_windows.split_for, to that chunk's own split label. This
          is guaranteed by construction in chunking.py -- which is
          exactly why it is worth checking, since the guarantee lives in
          a different file from the code that relies on it.

      (c) CHRONOLOGICAL. max(train) < min(val) < min(test). This is the
          one that breaks when MONTHS ARE ADDED. data_windows.py splits
          by string comparison against TRAIN_END / VAL_END, so appending
          2025-10 onward silently classifies all of it as test; appending
          months EARLIER than TRAIN_END silently adds training data
          before the val window, which is fine, while a mis-set boundary
          can interleave them, which is not.

    This is the guard whose failure would invalidate every result at once
    rather than one table, so it runs before anything is loaded.
    """
    n = len(timestamps)
    if not chunks:
        _die("no chunks", "build_split_ranges/chunk_ranges produced nothing.")

    # (a) disjoint, in range
    seen: List[Optional[int]] = [None] * n
    for ci, c in enumerate(chunks):
        if c.start < 0 or c.end > n or c.start >= c.end:
            _die("chunk out of range",
                 f"chunk {ci} = [{c.start}:{c.end}) against {n} snapshots.")
        for i in range(c.start, c.end):
            if seen[i] is not None:
                _die("overlapping chunks",
                     f"snapshot index {i} appears in chunk {seen[i]} and chunk {ci}.",
                     "Overlap means the same snapshot is used twice, and if the two",
                     "chunks carry different split labels it is training on held-out data.")
            seen[i] = ci

    # (b) homogeneous
    per_split: Dict[str, List[int]] = {}
    for ci, c in enumerate(chunks):
        for i in range(c.start, c.end):
            lbl = split_for(_month_str(timestamps[i]))
            if lbl != c.split:
                _die("chunk spans a split boundary",
                     f"chunk {ci} is labelled {c.split!r} but snapshot {i} "
                     f"({timestamps[i]}, month {_month_str(timestamps[i])}) is {lbl!r}.",
                     "chunking.py cuts on split runs first, so this means the timeline",
                     "and data_windows.split_for disagree -- check the months passed in.")
        per_split.setdefault(c.split, []).append(ci)

    # (c) chronological
    bounds: Dict[str, tuple] = {}
    for split in SPLIT_ORDER:
        idxs = [i for c in chunks if c.split == split for i in range(c.start, c.end)]
        if idxs:
            bounds[split] = (timestamps[min(idxs)], timestamps[max(idxs)], len(idxs))

    present = [s for s in SPLIT_ORDER if s in bounds]
    for a, b in zip(present, present[1:]):
        if not bounds[a][1] < bounds[b][0]:
            _die("splits are not chronologically ordered",
                 f"{a} ends {bounds[a][1]} but {b} starts {bounds[b][0]}.",
                 "Every result in this project assumes strictly forward-in-time",
                 "evaluation. Fix TRAIN_END / VAL_END in data_windows.py for the",
                 "month range being used -- adding months does NOT move them.")

    lines = ["SPLIT INTEGRITY: disjoint, homogeneous, chronological -- OK"]
    for split in present:
        lo, hi, k = bounds[split]
        lines.append(f"  {split:<5} {k:>6,} snapshots   {lo}  ->  {hi}")
    missing = [s for s in SPLIT_ORDER if s not in bounds]
    if missing:
        lines.append(f"  NOTE: no snapshots at all for {missing}. If a split you "
                     f"expect is missing, TRAIN_END/VAL_END do not match these months.")
    report = "\n".join(lines)
    if verbose:
        print(report, flush=True)
    return report


# ---------------------------------------------------------------------
# 2. Month schema
# ---------------------------------------------------------------------

def check_month_schema(store, *, verbose: bool = True) -> str:
    """Every month exposes the same adjacency key set and feature width.

    MonthlyBundleStore already rejects a feature-width mismatch. It did
    not check EDGE TYPES, and that is the more dangerous one: a month
    built before an edge builder existed simply has no entries under that
    key, so the spatial channel is silently absent for part of the range
    while the run reports one number averaged over both regimes.

    Needs ``store.adjacency_keys``, collected during the store's existing
    one-pass setup (see model/month_store.py). Absent, this says so
    rather than doing a second full pass over the cache.
    """
    keys_by_month = getattr(store, "adjacency_keys", None)
    if not keys_by_month:
        msg = ("MONTH SCHEMA: skipped -- this MonthlyBundleStore does not expose "
               "adjacency_keys. Update model/month_store.py to collect them.")
        if verbose:
            print(msg, flush=True)
        return msg

    sets = {m: tuple(sorted(k)) for m, k in keys_by_month.items()}
    distinct = sorted(set(sets.values()))
    if len(distinct) > 1:
        lines = ["months do not agree on which edge types exist:"]
        for variant in distinct:
            who = [m for m, v in sets.items() if v == variant]
            lines.append(f"  {who} -> {list(variant)}")
        lines += [
            "A month missing an edge type contributes no spatial structure of that",
            "kind, so the run averages a with-graph regime and a without-graph one",
            "and reports the mean as if it were a single condition.",
            "Rebuild the odd months' caches before training.",
        ]
        _die("month schema mismatch", *lines)

    report = (f"MONTH SCHEMA: {len(sets)} month(s) agree on edge types "
              f"{list(distinct[0])} -- OK")
    if verbose:
        print(report, flush=True)
    return report


# ---------------------------------------------------------------------
# 3. Ablation actually ablates
# ---------------------------------------------------------------------

def check_drop_edges(drop_edges: Iterable[str], present_keys: Iterable[str], *,
                     verbose: bool = True) -> str:
    """Names passed to --drop-edges exist, and says what is left.

    The original failure: the argparse ``choices`` list was written by
    hand and contained two keys that model/spatial_attention.py never
    reads. Dropping them removed nothing, the run completed normally, and
    the resulting "no graph" history.csv was bit-for-bit identical to the
    previous run's -- which is the only reason it was noticed.

    argparse validates against the choices list; this validates against
    the CACHE. Those are different things and only the second one is
    evidence.
    """
    keys = sorted(present_keys)
    drop = list(drop_edges or [])
    bad = [k for k in drop if k not in keys]
    if bad:
        _die("--drop-edges names are not in the graph",
             f"not present: {bad}",
             f"actual edge types in this cache: {keys}",
             "Dropping a non-existent key removes nothing. The run would complete",
             "and look like a finished ablation while the model was unchanged.")
    kept = [k for k in keys if k not in drop]
    report = (f"GRAPH EDGES present: {keys}\n"
              f"  dropped from backbone : {drop or 'none'}\n"
              f"  BACKBONE WILL SEE     : {kept or 'NOTHING -- no spatial channel at all'}")
    if verbose:
        print(report, flush=True)
    return report


# ---------------------------------------------------------------------
# 4. Hub feature-slot semantics
# ---------------------------------------------------------------------

def check_hub_slot(ct, slot_legs_total: int, *, max_snapshots: int = 12,
                   verbose: bool = True) -> str:
    """The basket hub's legs-total slot is an integer count, >= 2, and at
    least as large as the number of leg edges pointing into that hub.

    This slot is what diagnose_basket_coverage.py divides by to get a
    coverage fraction, and what any basket-sum reconstruction implicitly
    trusts. If a cache rebuild shifts the feature layout, the slot keeps
    returning a float and every downstream fraction is quietly wrong
    rather than absent -- the worst failure mode, because nothing errors.

    Sampled, not exhaustive: a layout error is systematic, so a dozen
    snapshots settle it.
    """
    import torch  # local: keeps this module importable without torch present

    features = ct["features"]
    adj = ct["adjacency_by_type"].get("mece_leg_to_basket")
    if adj is None:
        msg = "HUB SLOT: skipped -- no mece_leg_to_basket edges in the probe chunk."
        if verbose:
            print(msg, flush=True)
        return msg

    if features.shape[-1] <= slot_legs_total:
        _die("hub slot index out of range",
             f"SLOT_LEGS_TOTAL={slot_legs_total} but features have width "
             f"{features.shape[-1]}.")

    checked = 0
    totals: List[float] = []
    for t in range(min(len(adj), features.shape[0])):
        e = adj[t]
        if e.edge_index.numel() == 0:
            continue
        leg, hub = e.edge_index[0], e.edge_index[1]
        for h in torch.unique(hub).tolist():
            n_edges = int((hub == h).sum())
            total = float(features[t, h, slot_legs_total])
            if not math.isfinite(total):
                _die("hub legs-total is not finite",
                     f"snapshot {t}, hub node {h}: value {total!r}.")
            if abs(total - round(total)) > 1e-6:
                _die("hub legs-total is not an integer count",
                     f"snapshot {t}, hub node {h}: value {total!r}.",
                     f"Feature slot {slot_legs_total} is being read as the number of",
                     "legs in the basket. A non-integer means the feature layout has",
                     "changed (COMBINED_N_FEATURES / stg/nodes/kalshi.py) and every",
                     "coverage fraction computed from it is wrong but plausible.")
            if round(total) < 2:
                _die("hub legs-total below 2",
                     f"snapshot {t}, hub node {h}: {total!r}.",
                     "A MECE basket with fewer than two legs has no sum constraint.")
            if n_edges > round(total):
                _die("more leg edges than the hub claims legs",
                     f"snapshot {t}, hub node {h}: {n_edges} edges, legs_total={total!r}.",
                     "Either the slot is not legs-total, or basket membership and the",
                     "edge builder disagree.")
            totals.append(total)
        checked += 1
        if checked >= max_snapshots:
            break

    if not totals:
        msg = "HUB SLOT: skipped -- no hubs found in the sampled snapshots."
        if verbose:
            print(msg, flush=True)
        return msg

    report = (f"HUB SLOT {slot_legs_total}: integer leg counts, "
              f"min {min(totals):.0f} max {max(totals):.0f}, "
              f"{len(totals):,} hub-snapshots over {checked} snapshots -- OK")
    if verbose:
        print(report, flush=True)
    return report


# ---------------------------------------------------------------------
# 5. Baselines are model-independent, so they must never move
# ---------------------------------------------------------------------

def _mech_baselines(val_mech: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for k, v in (val_mech or {}).items():
        row = {"n": int(v.get("n", 0)), "persistence": float(v.get("persistence", float("nan")))}
        if "shrink" in v:
            row["shrink"] = float(v["shrink"])
            row["alpha"] = float(v["alpha"])
        out[k] = row
    return out


def _close(a: float, b: float, rel: float) -> bool:
    if not (math.isfinite(a) and math.isfinite(b)):
        return (a != a) and (b != b)      # both NaN counts as equal
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


class BaselineFingerprint:
    """Persistence and shrinkage baselines depend ONLY on the data. They
    must be bit-stable across epochs, and identical between two runs that
    differ only in the model.

    TWO DISTINCT CHECKS:

      WITHIN A RUN. The baselines are recomputed every epoch from the same
      val positions, so any epoch-to-epoch movement means the supervised
      population is drifting -- a masking or eligibility bug, since the
      model cannot affect a baseline.

      ACROSS RUNS (--baseline-ref). This is the one that catches training
      weights leaking into reported metrics. When the mover weight was
      applied to the reported baseline as well as to the loss, the ladder
      persistence figure tripled; the run still looked internally
      consistent, and the error only surfaced when someone divided two
      numbers from different runs by hand. Pointing an ablation at the
      reference run's fingerprint makes that a startup failure instead.

      It is also what makes a paired ablation PAIRED rather than lucky:
      the graph-ablation comparison is only a clean comparison because
      both runs happened to share persistence 0.008251 and shrink
      0.007719 at alpha 0.869975. That was verified by eye afterwards.
      Now it is asserted.
    """

    def __init__(self, path: Path, ref_path: Optional[Path] = None,
                 rel_tol: float = 1e-9, ref_rel_tol: float = 1e-6):
        self.path = Path(path)
        self.rel_tol = rel_tol
        self.ref_rel_tol = ref_rel_tol
        self.first: Optional[dict] = None
        self.ref: Optional[dict] = None
        if ref_path is not None:
            p = Path(ref_path)
            if not p.exists():
                _die("--baseline-ref not found", f"{p}",
                     "Point it at the baselines.json of the run this one is paired",
                     "with, or omit the flag.")
            self.ref = json.loads(p.read_text())

    def check(self, epoch: int, val_mech: Dict[str, dict],
              node_persistence: float, node_n: int, *, verbose: bool = True):
        cur = {"node": {"n": int(node_n), "persistence": float(node_persistence)},
               **_mech_baselines(val_mech)}

        if self.first is None:
            self.first = cur
            self.path.write_text(json.dumps(cur, indent=2, sort_keys=True))
            if verbose:
                print(f"  baselines fingerprinted -> {self.path}", flush=True)
            if self.ref is not None:
                self._compare(cur, self.ref, self.ref_rel_tol, "--baseline-ref run",
                              extra=[
                                  "Two runs scored on the same split MUST see identical",
                                  "baselines: they are computed from the data, not the",
                                  "model. A difference means the runs are not comparable,",
                                  "so any ratio between them is meaningless.",
                                  "Most likely causes: different --months, a different",
                                  "--supervise mode, a different --ladder-supervise",
                                  "population, or a training weight leaking into the",
                                  "REPORTED metric (this has happened).",
                              ])
                if verbose:
                    print("  baselines match --baseline-ref -- runs are paired", flush=True)
            return

        self._compare(cur, self.first, self.rel_tol, f"epoch 0 of this run (now epoch {epoch})",
                      extra=[
                          "Baselines cannot depend on the model, so a moving baseline",
                          "means the supervised POPULATION is changing between epochs.",
                          "Look at masking / eligibility, not at the optimiser.",
                      ])

    def _compare(self, cur: dict, other: dict, rel: float, what: str,
                 extra: Sequence[str] = ()):
        problems: List[str] = []
        for k in sorted(set(cur) | set(other)):
            a, b = cur.get(k), other.get(k)
            if a is None or b is None:
                problems.append(f"{k}: present in one, absent in the other "
                                f"({'this run' if a else what})")
                continue
            for field in sorted(set(a) | set(b)):
                x, y = a.get(field), b.get(field)
                if x is None or y is None:
                    problems.append(f"{k}.{field}: missing on one side")
                elif isinstance(x, int) and isinstance(y, int):
                    if x != y:
                        problems.append(f"{k}.{field}: {x:,} vs {y:,}")
                elif not _close(float(x), float(y), rel):
                    problems.append(f"{k}.{field}: {float(x):.9g} vs {float(y):.9g}")
        if problems:
            _die(f"baselines differ from {what}", *problems, "", *extra)


# ---------------------------------------------------------------------
# 6. Ladder persistence decomposition -- the weighting-leak detector
# ---------------------------------------------------------------------

def check_ladder_persistence_decomposition(val_mech: Dict[str, dict], *,
                                           lo: float = 0.85, hi: float = 1.000001,
                                           verbose: bool = True) -> Optional[str]:
    """An exact arithmetic identity that the weighting bug violated by 3x.

    A ladder pair where neither leg reprices has gap(t+h) == gap(t)
    exactly, so its PERSISTENCE squared error is exactly zero. Summed over
    the whole population, the frozen pairs therefore contribute nothing:

        persistence_all * n_all  ==  persistence_moved * n_moved

    The identity is not quite exact only because 'moved' is defined by
    MOVE_EPS rather than by exact equality, so pairs that moved by less
    than the threshold are counted as frozen while carrying a tiny
    non-zero error. That makes the ratio slightly BELOW 1, never above.

    A ratio ABOVE 1 means the pooled baseline is smaller than the moved
    one scaled by its share, which cannot happen from data -- it happens
    when a training weight is applied to the reported baseline. That is
    exactly what occurred (0.006012 = 3 x 0.002004), and it turned a true
    1.070x into a reported 0.996x for three runs before anyone noticed.

    Returns None when both populations are not present (e.g. --w-ladder 0
    without --monitor-unweighted).
    """
    a, m = (val_mech or {}).get("ladder"), (val_mech or {}).get("ladder_moved")
    if not a or not m or not a.get("n") or not m.get("n"):
        return None
    tot_all = float(a["persistence"]) * int(a["n"])
    tot_moved = float(m["persistence"]) * int(m["n"])
    if tot_all <= 0:
        return None
    ratio = tot_moved / tot_all

    if not (lo <= ratio <= hi):
        _die("ladder persistence decomposition violated",
             f"persistence_moved * n_moved / (persistence_all * n_all) = {ratio:.6f}",
             f"  all   : persistence {float(a['persistence']):.6f}  n {int(a['n']):,}",
             f"  moved : persistence {float(m['persistence']):.6f}  n {int(m['n']):,}",
             "",
             "Frozen pairs satisfy gap(t+h) == gap(t) exactly, so they contribute",
             "exactly zero persistence error and this ratio must sit just below 1.",
             "",
             "ABOVE 1 means a training weight has leaked into the REPORTED metric.",
             "That bug shipped once already: the mover weight tripled the reported",
             "persistence baseline and a true 1.070x was logged as 0.996x.",
             "Reporting must be unweighted; only the loss may be weighted.",
             "",
             "WELL BELOW 1 means pairs moving by less than MOVE_EPS carry a large",
             "share of the error -- reconsider the threshold rather than ignoring it.")

    report = f"ladder persistence decomposition {ratio:.6f} (frozen carry the rest) -- OK"
    if verbose:
        print(f"  {report}", flush=True)
    return report


# ---------------------------------------------------------------------
# 7. Every printed metric reaches the file
# ---------------------------------------------------------------------

def check_history_columns(val_mech: Dict[str, dict], columns: Sequence[str], *,
                          known: Sequence[str]) -> None:
    """A mechanism that is computed and printed but has no history.csv
    column is evidence that exists only for as long as the terminal
    scrollback does.

    ladder_moved was in exactly that state: it was the metric
    --ladder-supervise moved trained and early-stopped on, and the
    selected epoch-8 best could not be recovered after the run because
    the trajectory was never written anywhere.
    """
    unknown = sorted(set(val_mech or {}) - set(known))
    if unknown:
        _die("a mechanism has no history.csv column",
             f"computed and printed but not logged: {unknown}",
             f"columns currently written: {list(columns)}",
             "Add it to MECH_COLUMNS in train_forecast.py. A metric that is only",
             "ever printed cannot be plotted, cannot be compared across runs, and",
             "disappears when the terminal does -- which already cost one run's",
             "selected checkpoint.")


# ---------------------------------------------------------------------
# 8. Finite selection metric
# ---------------------------------------------------------------------

def require_finite_selection(name: str, value: float) -> float:
    """A NaN selection metric silently loses every comparison, so `best`
    is never updated, no checkpoint is written, and early stopping fires
    at the patience limit as if nothing had improved -- which looks
    exactly like a model that cannot learn.

    NaN here is normal for a mechanism with weight 0 and no
    --monitor-unweighted. The correct response is to fail immediately and
    say so, not to run for forty minutes and save nothing.
    """
    if value is None or not math.isfinite(float(value)):
        _die(f"selection metric {name} is {value!r}",
             "Nothing can be selected or early-stopped on a non-finite quantity:",
             "every comparison against it is False, so no checkpoint is ever saved",
             "and the run stops at the patience limit having kept nothing.",
             "",
             "Usual cause: --select-on names a mechanism whose weight is 0. Either",
             "give it a weight, add --monitor-unweighted to score it without",
             "training on it, or select on something this run actually computes.")
    return float(value)


# ---------------------------------------------------------------------
# 9. One place that turns an interval into a word
# ---------------------------------------------------------------------

def verdict(lo: float, hi: float) -> str:
    """Three outcomes, not two.

    Written here once because the two-branch version -- "lo > 0 means
    better, otherwise inconclusive" -- has now been written twice in two
    different evaluation scripts, and in both of them an interval of
    [-5.04, -2.58] printed as "INCLUDES 0". An interval lying entirely
    BELOW zero is not an absence of evidence; it is evidence of the
    opposite, and it is the single most important thing a PnL comparison
    can say.
    """
    if lo != lo or hi != hi:
        return "undefined"
    if lo > 0.0:
        return "beats"
    if hi < 0.0:
        return "worse"
    return "inconclusive"


def format_verdict(label: str, lo: float, hi: float, point: Optional[float] = None) -> str:
    v = verdict(lo, hi)
    word = {"beats": "BEATS  (CI entirely above 0)",
            "worse": "WORSE  (CI entirely below 0)",
            "inconclusive": "inconclusive (CI includes 0)",
            "undefined": "undefined (CI not computable)"}[v]
    pt = f"  point {point:+.4f}" if point is not None else ""
    return f"{label:<34} [{lo:+.4f}, {hi:+.4f}]{pt}   {word}"