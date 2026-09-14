#!/usr/bin/env python
"""Item 4 — channel pooling as a stochastic block model.
(relations_study_plan §3.4, research_log.md §11)

    venv/bin/python analysis/relations_2026_09/channel_pooling.py

Stage 1 spends one free parameter per ordered pair — 141 of them — to estimate
a graph that 4 edges survive.  The block-model alternative gives every node a
*type* (inflation, labour, growth, energy, policy), fixes one theory sign per
**channel** (type -> type), and pools every pair in the channel into a single
test.  Nothing is fitted: the sign is imposed by economics, not estimated, so
there is no in-sample/out-of-sample gap of the kind the walk-forward apparatus
exists to measure.

``research_log.md`` §11 measured the data->policy channel at 456 rows, 55.0%
aligned sign agreement against a 50.1% +/- 2.3% block-permutation null,
p = 0.021, on the raw ``surprise``.  This re-runs it

  * on the corrected panel (§14 changed the resolved values under it),
  * on ``s_pit`` as well as ``surprise`` — §1.4's argument is that pooling raw
    surprises across series is not even meaningful, because a 0.1pp CPI miss and
    a 50k payrolls miss are not the same unit, whereas averaging PIT positions
    is,
  * channel by channel rather than as one aggregate, so a null channel is
    visible instead of averaged away,
  * with same-release pairs excluded throughout, as §11 did.

The null must be the block permutation: shuffling within a trigger series
preserves the r ~= 0.9 dependence between CPI and CPICORE, which resolve from
one print.  The naive per-cell shuffle overstates significance ~10x
(research_log.md §3).

In-sample only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "stg_infra")

from stg.panel._io import load_markets, scan_trades
from stg.panel.registry import is_same_release, target_universe
from stg.panel.surprise import usable_triggers
from stg.panel.targets import response_panel, target_frames
from stg.splits import assert_no_oos
from stg.structure.stats import block_permutation_sign_p, spearman, permutation_p

PANELS = Path("artifacts/panels")
OUT = Path("analysis/relations_2026_09/out")

# Node types.  The block model's whole content is in this table plus SIGN.
TYPE = {
    "CPI": "inflation", "CPICORE": "inflation", "CPIYOY": "inflation",
    "CPICOREYOY": "inflation", "CPIGAS": "inflation", "CPIUSEDCAR": "inflation",
    "CPISHELTER": "inflation", "CPIFOOD": "inflation", "CPIAPPAREL": "inflation",
    "PCECORE": "inflation",
    "PAYROLLS": "labour", "U3": "labour", "JOBLESSCLAIMS": "labour",
    "ADP": "labour",
    "GDP": "growth", "ISMPMI": "growth", "RECESSION": "growth",
    "FED": "policy", "FEDDECISION": "policy",
    "WTI": "energy", "WTIW": "energy",
}

# Does a *higher* print from this series push the policy path up (hawkish, +1)
# or down (dovish, -1)?  Unemployment and jobless claims are the dovish cells;
# they are what makes this a falsifiable sign restriction rather than a
# relabelling.
HAWKISH = {
    "CPI": +1, "CPICORE": +1, "CPIYOY": +1, "CPICOREYOY": +1, "PCECORE": +1,
    "CPIGAS": +1, "CPIUSEDCAR": +1, "CPISHELTER": +1, "CPIFOOD": +1,
    "CPIAPPAREL": +1,
    "PAYROLLS": +1, "ADP": +1,
    "U3": -1, "JOBLESSCLAIMS": -1,
    "GDP": +1, "ISMPMI": +1,
    "WTI": +1, "WTIW": +1,
}

# Which way the *target* leg moves when the policy path rises.  FEDDECISION/cut
# is the falsification cell: it must come out with the opposite sign, and a
# model that scores it as a hit is scoring noise.
TARGET_SIGN = {("FEDDECISION", "hike"): +1, ("FEDDECISION", "cut"): -1,
               ("FED", "any"): +1}


def cells(panel: pl.DataFrame, measure: str, mk, tr) -> list[dict]:
    """One entry per (trigger, target, side) with the aligned surprise."""
    triggers = usable_triggers(panel, 10)
    out = []
    frame_cache: dict = {}
    for (target, side), tsign in TARGET_SIGN.items():
        key = (target, side)
        if key not in frame_cache:
            frame_cache[key] = target_frames(target, side, markets=mk, trades=tr)
        frames = frame_cache[key]
        if frames[0].height == 0:
            continue
        for trig in triggers:
            if trig == target or trig not in HAWKISH or trig not in TYPE:
                continue
            if is_same_release(trig, target):
                continue
            sp = (panel.filter(pl.col("series") == trig)
                  .with_columns(pl.col(measure).alias("surprise"))
                  .filter(pl.col("surprise").is_finite()))
            if sp.height == 0:
                continue
            rp = response_panel(sp, target, side, horizon="dormant",
                                markets=mk, trades=tr, frames=frames)
            if rp.height == 0:
                continue
            s = rp["surprise"].to_numpy().astype(float) * HAWKISH[trig] * tsign
            r = rp["response"].to_numpy().astype(float)
            out.append(dict(trigger=trig, target=target, side=side,
                            channel=f"{TYPE[trig]}->{TYPE[target]}",
                            aligned=s, response=r, n=len(s),
                            # carried so the pooled test can cluster on it:
                            # one FOMC meeting is hit by every trigger in the
                            # grid, so rows sharing a target event are not
                            # independent draws
                            target_event=rp["target_event"].to_list()))
    return out


def pooled_rho(cs: list[dict]) -> tuple[float, float]:
    """Spearman of aligned surprise on response, ranked **within** each cell.

    Ranking within the cell before pooling is what makes this legitimate: cents
    of response are not comparable across targets, and a raw pool would let the
    target with the widest price range dominate.
    """
    S, R = [], []
    for c in cs:
        if c["n"] < 3:
            continue
        S.append(_rank(c["aligned"]))
        R.append(_rank(c["response"]))
    if not S:
        return float("nan"), float("nan")
    s, r = np.concatenate(S), np.concatenate(R)
    rho = spearman(s, r)
    rng = np.random.default_rng(0)
    return float(rho), float(permutation_p(s, r, 2000, rng=rng))


def _rank(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return rankdata(x) / (len(x) + 1.0)


def cluster_sign_p(cs: list[dict], n_perm: int = 5000, seed: int = 0) -> dict:
    """Sign agreement against a null that clusters on ``target_event``.

    The block permutation above shuffles *within a trigger series*, which fixes
    the CPI/CPICORE dependence — but it leaves the other one untouched: one FOMC
    meeting is the target of every trigger in the grid, so a single meeting that
    happened to move the right way lends a hit to twenty rows at once.  §13.5
    found two headline configurations that did not survive clustering on
    ``target_event``, and the standing instruction in ``reports/INDEX.md`` is to
    cite clustered figures only.

    Implemented as a cluster-level sign flip (a wild bootstrap with Rademacher
    weights): every row sharing a target event has its response flipped
    together, so the within-cluster correlation is preserved exactly under the
    null while the hit rate is centred on 0.5.
    """
    rng = np.random.default_rng(seed)
    aligned = np.concatenate([c["aligned"] for c in cs])
    response = np.concatenate([c["response"] for c in cs])
    events = np.concatenate([np.asarray(c["target_event"]) for c in cs])
    keep = (aligned != 0) & (response != 0)
    aligned, response, events = aligned[keep], response[keep], events[keep]
    if aligned.size == 0:
        return {}
    uniq, idx = np.unique(events, return_inverse=True)
    obs = float((aligned * response > 0).mean())
    null = np.empty(n_perm)
    for k in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=uniq.size)[idx]
        null[k] = float((aligned * response * flip > 0).mean())
    return dict(clustered_agree=obs, n_clusters=int(uniq.size),
                cl_null=float(null.mean()), cl_sd=float(null.std()),
                p_clustered=float((1 + int((null >= obs).sum())) / (1 + n_perm)))


def report(cs: list[dict], label: str) -> dict:
    triples = [(c["trigger"], c["aligned"], c["response"]) for c in cs]
    if not triples:
        return {}
    res = block_permutation_sign_p(triples, n_perm=5000)
    rho, p_rho = pooled_rho(cs)
    out = dict(channel=label, pairs=len(cs), rows=res["n"],
               sign_agree=res["sign_agreement"], null=res["null_mean"],
               null_sd=res["null_sd"], p_sign=res["p"],
               pooled_rho=rho, p_rho=p_rho)
    out.update(cluster_sign_p(cs))
    return out


def main() -> None:
    panel = pl.read_parquet(PANELS / "surprise_panel.parquet")
    assert_no_oos(panel, time_col="close_time")
    mk = load_markets(is_only=True)
    tr = scan_trades(is_only=True)

    rows = []
    for measure in ("surprise", "s_pit"):
        cs = cells(panel, measure, mk, tr)
        print(f"\n######## measure = {measure} ########")
        print(f"cells {len(cs)}   rows {sum(c['n'] for c in cs)}")

        blocks = {}
        for c in cs:
            blocks.setdefault(c["channel"], []).append(c)

        res = []
        for name, group in sorted(blocks.items()):
            r = report(group, name)
            if r:
                res.append(r)
        # the aggregate §11 reported: everything that is not energy
        data_to_policy = [c for c in cs
                          if c["channel"] in ("inflation->policy", "labour->policy",
                                              "growth->policy")]
        if data_to_policy:
            res.append(report(data_to_policy, "ALL data->policy"))
        allc = report(cs, "ALL channels")
        if allc:
            res.append(allc)
        tab = pl.DataFrame(res).with_columns(pl.lit(measure).alias("measure"))
        rows.append(tab)
        with pl.Config(tbl_rows=30, float_precision=4, tbl_width_chars=200):
            print(tab)

        print("\n-- per-pair contribution within inflation->policy --")
        infl = [c for c in blocks.get("inflation->policy", [])]
        if infl:
            per = pl.DataFrame([
                dict(trigger=c["trigger"], target=c["target"], side=c["side"],
                     n=c["n"],
                     hits=float(((c["aligned"] * c["response"]) > 0).sum()),
                     rate=float(((c["aligned"] * c["response"]) > 0).sum()
                                / max(((c["aligned"] != 0) & (c["response"] != 0)).sum(), 1)))
                for c in infl])
            with pl.Config(tbl_rows=40, float_precision=3):
                print(per.sort("n", descending=True))
        sys.stdout.flush()

    OUT.mkdir(parents=True, exist_ok=True)
    pl.concat(rows).write_parquet(OUT / "channel_pooling.parquet")


if __name__ == "__main__":
    main()
