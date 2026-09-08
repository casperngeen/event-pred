"""The complexity ladder for the dormant-horizon direction task.

``research_summary.md`` §9 Phase 2 item 5 asks for an ablation ladder whose
rungs 1-2 are "single-pair regression on Stage-1 structure" and "multi-trigger
regression aggregating over active Stage-1 neighbours". The AGCRN post-mortem
(``reports/agcrn_postmortem.md``) says to run it as a *direction classifier at
the dormant horizon* rather than a magnitude regression at the snapshot
horizon. These are those rungs.

Every rung answers one question by adding exactly one ingredient:

    R0 base_rate        is there a constant that beats a coin flip?
    R1 sign_rule        does sign(edge) x sign(surprise) alone predict?
    R1s sign_rule_bh    ... restricted to edges surviving BH *in that fold*
    R2 edge_logit       does a fitted weight on the edge-weighted surprise help?
    R3 no_structure     ablation: raw surprise + context, edge weight removed
    R4 feature_logit    edge-weighted surprise + context
    R5 neighbour_logit  + the aggregate signal from co-active in-neighbours

R3 is the ablation that matters: R4 minus R3 is what the estimated *structure*
buys over knowing the surprise alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from stg.direction.structure import FoldStructure

CONTEXT = ("abs_z", "dtc", "p0c", "same_rel")


def neighbour_signal(panel: pl.DataFrame, st: FoldStructure,
                     survivors_only: bool = False) -> np.ndarray:
    """Aggregate edge-weighted surprise from *other* triggers already resolved
    into the same target event.

    Strictly causal: only rows with an earlier ``t0`` on the same
    ``target_event`` contribute, so this is information a decision-maker holds
    at the moment the current trigger resolves.
    """
    sig = st.signal(panel["pair"].to_list(), panel["z_surprise"].to_numpy(),
                    survivors_only)
    df = panel.select("target_event", "t0").with_columns(
        pl.Series("sig", sig), pl.Series("idx", np.arange(panel.height)))
    out = np.zeros(panel.height)
    for (_ev,), g in df.sort("t0").group_by("target_event", maintain_order=True):
        s = g["sig"].to_numpy()
        prior = np.concatenate([[0.0], np.cumsum(s)[:-1]])
        out[g["idx"].to_numpy()] = prior
    return out


def design(panel: pl.DataFrame, st: FoldStructure, cols: tuple[str, ...],
           survivors_only: bool = False) -> np.ndarray:
    """Build the named feature columns for one rung."""
    z = panel["z_surprise"].to_numpy().astype(float)
    pairs = panel["pair"].to_list()
    sig = st.signal(pairs, z, survivors_only)
    avail = {
        "signal": sig,
        "sign_signal": np.sign(sig),
        "z": np.clip(z, -5, 5),
        "abs_z": np.clip(np.abs(z), 0, 5),
        "dtc": panel["days_to_close"].to_numpy().astype(float) / 30.0,
        "p0c": (panel["p0"].to_numpy().astype(float) - 50.0) / 50.0,
        "same_rel": panel["same_release"].to_numpy().astype(float),
        "nbr": neighbour_signal(panel, st, survivors_only),
    }
    return np.column_stack([avail[c] for c in cols])


class Learner:
    name = "learner"

    def fit(self, train: pl.DataFrame, st: FoldStructure) -> "Learner":
        raise NotImplementedError

    def predict_proba(self, test: pl.DataFrame, st: FoldStructure) -> np.ndarray:
        """P(response > 0), one per row."""
        raise NotImplementedError


@dataclass
class BaseRate(Learner):
    name: str = "base_rate"
    p: float = 0.5

    def fit(self, train, st):
        self.p = float((train["y"].to_numpy() > 0).mean())
        return self

    def predict_proba(self, test, st):
        return np.full(test.height, self.p)


@dataclass
class SignRule(Learner):
    """sign(edge) x sign(surprise), with a single calibration parameter.

    The rule itself is fitted only through the *sign* of each train-fold edge;
    the one continuous parameter is the rule's own train hit rate, which turns
    its vote into a probability so log-loss and AUC stay comparable across
    rungs. Rows whose pair has no train-fold edge fall back to the base rate.
    """
    survivors_only: bool = False
    name: str = "sign_rule"
    hit: float = 0.5
    p_base: float = 0.5

    def fit(self, train, st):
        v = np.sign(design(train, st, ("signal",), self.survivors_only)[:, 0])
        y = train["y"].to_numpy().astype(float)
        m = v != 0
        self.hit = float((v[m] == y[m]).mean()) if m.any() else 0.5
        self.p_base = float((y > 0).mean())
        return self

    def predict_proba(self, test, st):
        v = np.sign(design(test, st, ("signal",), self.survivors_only)[:, 0])
        p = np.where(v > 0, self.hit, 1.0 - self.hit)
        return np.where(v == 0, self.p_base, p)


@dataclass
class Logit(Learner):
    """Standardised logistic regression on a named feature set."""
    cols: tuple[str, ...] = ("signal",)
    name: str = "logit"
    survivors_only: bool = False
    C: float = 1.0

    def fit(self, train, st):
        X = design(train, st, self.cols, self.survivors_only)
        y = (train["y"].to_numpy() > 0).astype(int)
        self.mu_ = X.mean(0)
        self.sd_ = np.where(X.std(0) > 1e-9, X.std(0), 1.0)
        if len(np.unique(y)) < 2:
            self.model_ = None
            self.p_ = float(y.mean())
            return self
        self.model_ = LogisticRegression(C=self.C, max_iter=1000)
        self.model_.fit((X - self.mu_) / self.sd_, y)
        return self

    def predict_proba(self, test, st):
        if self.model_ is None:
            return np.full(test.height, self.p_)
        X = design(test, st, self.cols, self.survivors_only)
        return self.model_.predict_proba((X - self.mu_) / self.sd_)[:, 1]


def ladder() -> list[Learner]:
    return [
        BaseRate(),
        SignRule(name="sign_rule"),
        SignRule(name="sign_rule_bh", survivors_only=True),
        Logit(cols=("signal",), name="edge_logit"),
        Logit(cols=("z",) + CONTEXT, name="no_structure"),
        Logit(cols=("signal", "sign_signal") + CONTEXT, name="feature_logit"),
        Logit(cols=("signal", "sign_signal", "nbr") + CONTEXT, name="neighbour_logit"),
    ]
