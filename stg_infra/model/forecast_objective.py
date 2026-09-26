"""
model/forecast_objective.py

HORIZON FORECASTING LOSS, WITH PERSISTENCE SCORED ALONGSIDE IT.

The training target is the price a node actually trades at h snapshots
later, NOT its price now. That single change is what turns the model from
a reconstructor into a forecaster, and it is the reason the previous
objective could not have produced the behaviour this project wants:
masked reconstruction asks "given the neighbours, what is this leg worth
NOW", and when the leg's printed price is a day-old stale quote, the
TRAINING LABEL IS THAT STALE QUOTE. The model was being rewarded for
reproducing exactly the lag it was supposed to detect.

NO LEAKAGE. h_final[t] is produced by causal temporal attention
(model/temporal_attention.py builds an unconditional lower-triangular
mask) over spatial features at times <= t, so predicting t+h from it
cannot see the answer. This is a property of the backbone, not of this
module -- if that causal mask ever became conditional, every number
produced here would be meaningless, which is why it is stated here too.

TICKER NODES ONLY. Slot 0 means `last_yes_price` for a ticker leg but
`sum_cents` for a MECE basket hub (see model/data_validation.py), so a
single pooled regression over both would be fitting two different
quantities to one head. Basket dislocation is DERIVED from predicted legs
afterwards, which is the point of the unconstrained head.

PERSISTENCE IS SCORED EVERY BATCH, NOT AT THE END. On this dataset 90% of
consecutive observations show no price change, and the leg's own last
price beat the previous model by 4.4x. A forecasting run that does not
print the persistence loss next to its own is a run that can spend twenty
epochs getting better at something a one-line rule already does better.
The ratio model/persistence is the only number in the log that matters:
below 1.0 is progress, at or above 1.0 is not, whatever the raw loss does.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# Half a cent. Below this a Kalshi price is unchanged -- the tick is 1c, so
# any real reprice clears this comfortably and float noise does not. Same
# value as diagnose_ladder_conditional.py, deliberately: the training
# objective and the diagnostic that motivated it must agree on what
# "moved" means, or the retrain is not testing the diagnostic's claim.
MOVE_EPS = 0.005


class HorizonForecastObjective(nn.Module):
    """Builds (t -> t+h) supervision from a chunk and scores it against
    the persistence baseline on identical positions."""

    def __init__(self, horizons=(1, 2, 3, 6, 12), huber_beta: float = 0.0):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        # Huber (smooth L1) optionally, because price jumps are heavy
        # tailed and a few large moves can otherwise dominate the
        # gradient. beta = 0 keeps plain MSE.
        self.huber_beta = float(huber_beta)

    def _elementwise(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = pred - target
        if self.huber_beta > 0:
            a = err.abs()
            return torch.where(a <= self.huber_beta,
                               0.5 * err ** 2 / self.huber_beta,
                               a - 0.5 * self.huber_beta)
        return err ** 2

    def compute_loss(self, pred: torch.Tensor, prices: torch.Tensor,
                     mask: torch.Tensor, is_ticker: torch.Tensor) -> Dict[str, object]:
        """pred (T, N, H) predicted prices; prices (T, N) observed on the
        0-1 scale; mask (T, N) observed-ness; is_ticker (T, N) bool.

        A position (t, n) supervises horizon h only when the node is a
        ticker AND observed at t AND observed at t+h -- otherwise there is
        no target, and inventing one (carrying forward, say) would teach
        the model that unquoted means unchanged, which is the staleness
        assumption this objective exists to remove.
        """
        T, N, H = pred.shape
        if len(self.horizons) != H:
            raise ValueError(f"head produced {H} horizons, objective expects "
                             f"{len(self.horizons)}")
        if prices.shape != (T, N) or mask.shape != (T, N):
            raise ValueError(f"prices/mask must be (T, N) = ({T}, {N})")

        device = pred.device
        total = torch.zeros((), device=device)
        n_total = 0
        per_h: Dict[int, Dict[str, float]] = {}
        strata: Dict[str, Dict[str, float]] = {}

        for i, h in enumerate(self.horizons):
            if T <= h:
                per_h[h] = {"n": 0, "model": float("nan"), "persistence": float("nan")}
                continue
            src = slice(0, T - h)
            dst = slice(h, T)
            valid = mask[src] & mask[dst] & is_ticker[src]
            n_valid = int(valid.sum())
            if n_valid == 0:
                per_h[h] = {"n": 0, "model": float("nan"), "persistence": float("nan")}
                continue

            target = prices[dst][valid]
            p_model = pred[src, :, i][valid]
            # PERSISTENCE ON THE SAME POSITIONS: predict the price at t.
            # Scored with the identical elementwise function so the two
            # numbers are directly comparable rather than merely adjacent.
            p_persist = prices[src][valid]

            loss_h = self._elementwise(p_model, target).mean()
            with torch.no_grad():
                base_h = self._elementwise(p_persist, target).mean()

            # STRATIFY BY WHETHER THE PRICE ACTUALLY MOVED.
            #
            # 53.9% of observed positions show no price change from the
            # previous snapshot -- and (see claude/staleness-correction.md)
            # those are REAL trades at an unchanged price, not stale prints,
            # so they cannot be filtered out as noise. But on them
            # persistence is exactly right and the model can only lose,
            # while on movers persistence is uninformative and the model has
            # room. A pooled ratio averages the two regimes and hides both:
            # a model that is genuinely good on movers looks like it is doing
            # nothing, because the majority class drowns it.
            with torch.no_grad():
                moved = (target - p_persist).abs() > 0.005
                for tag, sub in (("moved", moved), ("flat", ~moved)):
                    k = int(sub.sum())
                    if k == 0:
                        continue
                    m_ = float(self._elementwise(p_model[sub], target[sub]).mean())
                    b_ = float(self._elementwise(p_persist[sub], target[sub]).mean())
                    d = strata.setdefault(tag, {"n": 0, "model": 0.0, "persistence": 0.0})
                    d["n"] += k
                    d["model"] += m_ * k
                    d["persistence"] += b_ * k

            total = total + loss_h * n_valid
            n_total += n_valid
            per_h[h] = {"n": n_valid,
                        "model": float(loss_h.detach()),
                        "persistence": float(base_h)}

        if n_total == 0:
            # A chunk can legitimately supervise nothing (too short, or no
            # node observed at both ends). Returning a zero that still
            # carries grad_fn keeps the caller's backward() valid.
            zero = pred.sum() * 0.0
            return {"loss": zero, "n": 0, "per_horizon": per_h, "strata": {},
                    "persistence": float("nan"), "ratio": float("nan")}

        loss = total / n_total
        pers = sum(v["persistence"] * v["n"] for v in per_h.values() if v["n"]) / n_total
        for d in strata.values():
            if d["n"]:
                d["model"] /= d["n"]
                d["persistence"] /= d["n"]
                d["ratio"] = (d["model"] / d["persistence"]
                              if d["persistence"] > 0 else float("nan"))
        return {"loss": loss, "n": n_total, "per_horizon": per_h, "strata": strata,
                "persistence": pers,
                "ratio": float(loss.detach()) / pers if pers > 0 else float("nan")}


def _shrinkage_mse(sxx, sxy, syy, n):
    """Best-possible ONE-PARAMETER baseline, in closed form.

    Predict x(t+h) = alpha * x(t). The alpha minimising squared error is
    alpha* = sum(x*y) / sum(x*x), and the resulting MSE is

        [ sum(y^2) - (sum(x*y))^2 / sum(x^2) ] / n

    WHY THIS CONTROL IS NECESSARY. Dislocations and gaps mean-revert, so
    predicting a SHRUNKEN version of the current value beats predicting it
    unchanged. That is a single scalar and needs no graph, no attention and
    no training -- and it is the first thing gradient descent finds, since
    pulling every predicted leg slightly toward basket consistency is one
    global adjustment. A model that merely discovers shrinkage will beat
    the persistence baseline and look like a result.

    alpha is fitted HERE, on the split being scored, which makes this an
    ORACLE and therefore optimistic. That asymmetry is deliberate: if the
    model cannot beat the best scalar even when the scalar is handed the
    answer, the model is at best a scalar. If it does beat it, it is doing
    something a scalar cannot.
    """
    if n <= 0 or sxx <= 0:
        return float("nan"), float("nan")
    alpha = sxy / sxx
    mse = (syy - (sxy * sxy) / sxx) / n
    return max(mse, 0.0), alpha


class MechanismForecastObjective(HorizonForecastObjective):
    """Node forecasting PLUS direct supervision of the two derived
    quantities the project actually trades on.

    WHY THE DERIVED TERMS ARE NOT REDUNDANT. Per-node MSE treats every leg
    equally, but a ladder violation is a DIFFERENCE between two adjacent
    rungs and a MECE dislocation is a SUM over siblings. Errors that are
    correlated across a family -- the whole basket drifting together --
    cancel almost entirely in the difference and in the deviation from
    $1.00, yet they dominate the node-level loss. A model tuned only on
    node MSE therefore spends its capacity on the component that does not
    matter here. Supervising the derived quantity directly is the standard
    remedy, and it aligns the training objective with the evaluation.

        ladder:  gap(t+h)  = price_B(t+h) - price_A(t+h)
        mece:    disloc(t+h) = sum_legs price(t+h) - 1

    BOTH GET THEIR OWN PERSISTENCE BASELINE, scored on identical
    positions. This is not decoration: measured on this dataset, 96.8% of
    ladder violations survive two hours unchanged, so "the gap stays where
    it is" is an extremely strong predictor of the gap. A model that
    improves node-level loss while losing to gap-persistence has learned
    nothing about violations, and only a per-quantity ratio reveals that.

    Baskets are included only when EVERY leg is observed at t+h -- a
    partially observed sum is not a dislocation, it is a coverage
    artifact, and the project has already invalidated one result by
    treating those as interchangeable.
    """

    def __init__(self, horizons=(1, 2, 3, 6, 12), huber_beta: float = 0.0,
                 w_node: float = 1.0, w_ladder: float = 1.0, w_mece: float = 1.0,
                 monitor_unweighted: bool = False, ladder_supervise: str = "all",
                 moved_weight: float = 9.0):
        super().__init__(horizons=horizons, huber_beta=huber_beta)
        self.w_node, self.w_ladder, self.w_mece = float(w_node), float(w_ladder), float(w_mece)
        # Score mechanisms whose weight is 0 anyway, for monitoring only.
        self.monitor_unweighted = bool(monitor_unweighted)
        # "all" | "moved" -- which ladder population the LOSS is computed on.
        # Both are always reported; this only decides which one is trained.
        if ladder_supervise not in ("all", "moved", "weighted"):
            raise ValueError(f"ladder_supervise must be 'all', 'moved' or "
                             f"'weighted', got {ladder_supervise!r}")
        self.ladder_supervise = ladder_supervise
        self.moved_weight = float(moved_weight)

    def _ladder_terms(self, pred, prices, mask, ladder_adj, i, h, T,
                      restrict_moved=False, moved_weight=1.0):
        """(sum_sq_model, sum_sq_persist, count, shrink_sums) for one horizon.

        shrink_sums accumulates what the OPTIMAL SHRINKAGE baseline needs:
        predict x(t+h) = alpha * x(t) with the single alpha that minimises
        squared error. See _shrinkage_mse for why this control exists.

        ``restrict_moved`` keeps only pairs where at least one leg repriced
        between t and t+h. ~90% of ladder pairs do not (measured: 89.8% at
        h=1 on val), and for those the target IS the persistence prediction,
        exactly. Averaging over them scores the model almost entirely on
        cases whose correct answer is "unchanged", which is what produced a
        pooled alpha of 0.996 and hid a conditional alpha of 0.09-0.48 in
        the violation buckets. See claude/ladder-conclusion-overturned.md.

        READ THIS BEFORE REPORTING ANY NUMBER PRODUCED WITH IT. Whether a
        leg reprices between t and t+h is NOT KNOWN AT t. Selecting on it
        is therefore an ORACLE condition, in exactly the sense the
        shrinkage baseline is an oracle:

          - As a TRAINING signal it is legitimate. The model's inputs still
            stop at t; this only decides which examples its capacity is
            spent on, like hard-example mining.
          - As an EVALUATION it is NOT deployable. A ratio measured on
            moved-only pairs cannot be achieved live, because at t you
            cannot choose to trade only the pairs that will move.

        So a run may TRAIN on this subset, but the headline number must
        still be the all-pairs one. compute_loss reports both, always, and
        labels which is which.
        """
        sm_u = sp_u = sm_w = torch.zeros((), device=pred.device)
        n = 0
        wt = 0.0
        sxx = sxy = syy = 0.0
        for t in range(T - h):
            e = ladder_adj[t]
            if e.edge_index.numel() == 0:
                continue
            a, b = e.edge_index[0], e.edge_index[1]
            ok = mask[t, a] & mask[t, b] & mask[t + h, a] & mask[t + h, b]
            mv = (((prices[t + h, a] - prices[t, a]).abs() > MOVE_EPS) |
                  ((prices[t + h, b] - prices[t, b]).abs() > MOVE_EPS))
            if restrict_moved:
                ok = ok & mv
            if not bool(ok.any()):
                continue
            a, b = a[ok], b[ok]
            tgt = prices[t + h, b] - prices[t + h, a]
            pm = pred[t, b, i] - pred[t, a, i]
            pp = prices[t, b] - prices[t, a]          # gap persistence
            em, ep = self._elementwise(pm, tgt), self._elementwise(pp, tgt)
            # REPORT UNWEIGHTED, TRAIN WEIGHTED. These must be separated.
            # Weighting the reported metric as well made the 'ladder' row a
            # weighted ratio rather than the deployable all-pairs one, and
            # since frozen pairs contribute exactly 0 to persistence, a
            # mover weight of W simply multiplied the persistence baseline
            # by W (logged: 0.006012 = 3 x 0.002004). The ratio then looked
            # like 0.996x when the true all-pairs figure was 1.070x. Worse,
            # the shrinkage sums below are unweighted, so vs_shrink compared
            # a weighted model against an unweighted baseline and printed
            # 2.992x, which is not a quantity at all.
            sm_u = sm_u + em.sum()
            sp_u = sp_u + ep.sum()
            n += int(a.numel())
            if moved_weight != 1.0 and not restrict_moved:
                # Upweight movers WITHOUT discarding the frozen majority. A
                # model trained on movers alone never sees a pair that stays
                # put, so it learns that things always move and then predicts
                # movement everywhere -- measured: all-pairs ladder went from
                # 1.09x to 1.98x while the moved-only metric improved. The
                # weights are reported in the numerator and the denominator
                # alike, so the ratio against persistence stays honest.
                w_ = torch.where(mv[ok], torch.full_like(em, moved_weight),
                                 torch.ones_like(em))
                sm_w = sm_w + (em * w_).sum()
                wt += float(w_.sum())
            else:
                sm_w = sm_w + em.sum()
                wt += float(a.numel())
            with torch.no_grad():
                sxx += float((pp * pp).sum())
                sxy += float((pp * tgt).sum())
                syy += float((tgt * tgt).sum())
        return sm_u, sp_u, n, (sxx, sxy, syy), (sm_w, wt)

    def _mece_terms(self, pred, prices, mask, mece_adj, i, h, T):
        sm = sp = torch.zeros((), device=pred.device)
        n = 0
        sxx = sxy = syy = 0.0
        for t in range(T - h):
            e = mece_adj[t]
            if e.edge_index.numel() == 0:
                continue
            leg, hub = e.edge_index[0], e.edge_index[1]
            hubs, inv = torch.unique(hub, return_inverse=True)
            k = hubs.numel()
            legs_total = torch.zeros(k, device=pred.device).index_add_(
                0, inv, torch.ones_like(leg, dtype=pred.dtype))
            seen = (mask[t, leg] & mask[t + h, leg]).to(pred.dtype)
            legs_seen = torch.zeros(k, device=pred.device).index_add_(0, inv, seen)
            # EVERY leg observed at both ends, or the sum is a coverage artifact
            full = legs_seen == legs_total
            if not bool(full.any()):
                continue
            s_pred = torch.zeros(k, device=pred.device).index_add_(0, inv, pred[t, leg, i])
            s_tgt = torch.zeros(k, device=pred.device).index_add_(0, inv, prices[t + h, leg])
            s_now = torch.zeros(k, device=pred.device).index_add_(0, inv, prices[t, leg])
            sm = sm + self._elementwise(s_pred[full] - 1.0, s_tgt[full] - 1.0).sum()
            sp = sp + self._elementwise(s_now[full] - 1.0, s_tgt[full] - 1.0).sum()
            with torch.no_grad():
                x = s_now[full] - 1.0
                y = s_tgt[full] - 1.0
                sxx += float((x * x).sum())
                sxy += float((x * y).sum())
                syy += float((y * y).sum())
            n += int(full.sum())
        # MECE is never reweighted, so the "weighted" slot is the plain sum
        # and its weight total is the count -- keeps the unpacking uniform.
        return sm, sp, n, (sxx, sxy, syy), (sm, float(n))

    def compute_loss(self, pred, prices, mask, is_ticker,
                     ladder_adj=None, mece_adj=None):
        base = super().compute_loss(pred, prices, mask, is_ticker)
        T = pred.shape[0]
        out = dict(base)
        # KEEP THE NODE TERM SEPARATELY. out["loss"] becomes the weighted
        # SUM of three terms below, while out["persistence"] stays the
        # node-only baseline. Dividing one by the other compares three
        # losses against one baseline and inflates the ratio -- which is
        # exactly the bug that printed "2.789x" while every chunk reported
        # 1.000x. Callers wanting the honest headline use node_loss.
        out["node_loss"] = float(base["loss"].detach()) if base["n"] else float("nan")
        out["node_ratio"] = base["ratio"]
        total = self.w_node * base["loss"]
        parts = {}

        # The ladder appears TWICE and always: once over all pairs, once over
        # pairs where a leg repriced. Only one carries the training weight
        # (ladder_supervise decides which); the other is scored at weight 0
        # purely so both numbers are in every log. That is not redundancy --
        # 'moved' is an oracle condition (see _ladder_terms), so a run that
        # trains on it must still show the deployable all-pairs figure beside
        # it, or the flattering number quietly becomes the headline.
        _sup = getattr(self, "ladder_supervise", "all")
        _moved_trained = _sup == "moved"
        _mw = self.moved_weight if _sup == "weighted" else 1.0
        for name, adj, fn, w, kw in (
                ("ladder", ladder_adj, self._ladder_terms,
                 0.0 if _moved_trained else self.w_ladder,
                 {"moved_weight": _mw}),
                ("ladder_moved", ladder_adj, self._ladder_terms,
                 self.w_ladder if _moved_trained else 0.0, {"restrict_moved": True}),
                ("mece", mece_adj, self._mece_terms, self.w_mece, {})):
            if adj is None:
                continue
            # A ZERO WEIGHT SILENCES THE LOSS, NOT THE METRIC. Skipping the
            # whole block at w=0 -- the original behaviour -- meant a run
            # with --w-ladder 0 logged n=0 and nan for ladder, so there was
            # no way to see whether optimising MECE was quietly degrading
            # the other mechanism. Scoring without training on it costs a
            # forward-only pass over that mechanism's pairs, which is real
            # (the ladder side is ~426k pairs per snapshot), so for a
            # mechanism that is switched off entirely it stays opt-in via
            # --monitor-unweighted.
            #
            # The two ladder rows are the exception: when the ladder is in
            # play at all, BOTH are scored whichever one holds the weight,
            # because the whole point is to see them side by side.
            ladder_live = name.startswith("ladder") and self.w_ladder != 0.0
            if not ladder_live and w == 0.0 and not getattr(
                    self, "monitor_unweighted", False):
                continue
            sm = sp = sw = torch.zeros((), device=pred.device)
            n = 0
            wtot = 0.0
            SXX = SXY = SYY = 0.0
            for i, h in enumerate(self.horizons):
                if T <= h:
                    continue
                a, b, c, (xx, xy, yy), (aw, wn) = fn(
                    pred, prices, mask, adj, i, h, T, **kw)
                sm, sp, n = sm + a, sp + b, n + c
                sw, wtot = sw + aw, wtot + wn
                SXX += xx; SXY += xy; SYY += yy
            if n:
                lm = sm / n
                lp = float(sp.detach()) / n
                shrink, alpha = _shrinkage_mse(SXX, SXY, SYY, n)
                if w != 0.0:
                    # The LOSS uses the weighted mean; every reported number
                    # above and below uses the unweighted one.
                    total = total + w * (sw / wtot if wtot else lm)
                parts[name] = {"n": n, "model": float(lm.detach()),
                               "persistence": lp,
                               "ratio": float(lm.detach()) / lp if lp > 0 else float("nan"),
                               "shrink": shrink, "alpha": alpha,
                               "vs_shrink": (float(lm.detach()) / shrink
                                             if shrink > 0 else float("nan")),
                               # RAW SUMS, not the per-chunk MSE. alpha must be
                               # refitted on the POOLED sums across chunks --
                               # averaging per-chunk shrinkage MSEs fits a
                               # different alpha per chunk and reports a
                               # baseline no single scalar could achieve.
                               "sxx": SXX, "sxy": SXY, "syy": SYY}
        out["loss"] = total
        out["mechanisms"] = parts
        return out


def format_mechanisms(parts: Dict[str, Dict[str, float]]) -> str:
    """Per-mechanism model/persistence, which is where a violation-timing
    claim lives or dies -- the node-level ratio can improve while the gap
    ratio does not."""
    if not parts:
        return "(no mechanism supervision this chunk)"
    out = []
    for k, v in sorted(parts.items()):
        seg = (f"{k}: {v['model']:.5f} vs persist {v['persistence']:.5f}"
               f"={v['ratio']:.3f}x")
        sh = v.get("shrink")
        if sh is not None and sh == sh and v.get("alpha") is not None:   # present, not NaN
            seg += (f" vs shrink(a={v['alpha']:.3f}) {v['shrink']:.5f}"
                    f"={v['vs_shrink']:.3f}x")
        out.append(seg + f" (n={v['n']:,})")
    return "  |  ".join(out)


def format_strata(strata: Dict[str, Dict[str, float]]) -> str:
    """model vs persistence split by whether the price moved. On 'flat'
    persistence is exact and the model can only lose; 'moved' is where any
    real forecasting skill has to show up."""
    if not strata:
        return "(no strata)"
    parts = []
    for tag in ("moved", "flat"):
        v = strata.get(tag)
        if not v or not v["n"]:
            continue
        parts.append(f"{tag}: {v['model']:.5f}/{v['persistence']:.5f}"
                     f"={v.get('ratio', float('nan')):.3f}x (n={v['n']:,})")
    return "  ".join(parts) if parts else "(no strata)"


def format_horizons(per_h: Dict[int, Dict[str, float]]) -> str:
    """One compact line: model vs persistence per horizon, with the ratio.
    Printed every epoch so a run that is losing to a one-line rule is
    obvious immediately rather than after it finishes."""
    parts = []
    for h in sorted(per_h):
        v = per_h[h]
        if not v["n"]:
            continue
        r = v["model"] / v["persistence"] if v["persistence"] else float("nan")
        parts.append(f"h{h}: {v['model']:.5f}/{v['persistence']:.5f}={r:.3f}x")
    return "  ".join(parts) if parts else "(no supervised positions)"