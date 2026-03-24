"""Built-in feature transformation strategies."""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from stg_infra.stg.strategies.protocols import FeatureStrategy


class StandardScaleFeatures:
    def __init__(self) -> None:
        self._ns = StandardScaler(); self._es = StandardScaler()
    def transform_node_features(self, f: np.ndarray) -> np.ndarray:
        return self._ns.fit_transform(f) if f.shape[0] > 1 else f
    def transform_edge_features(self, f: np.ndarray) -> np.ndarray:
        return self._es.fit_transform(f) if f.shape[0] > 1 else f


class MinMaxScaleFeatures:
    def __init__(self) -> None:
        self._ns = MinMaxScaler(); self._es = MinMaxScaler()
    def transform_node_features(self, f: np.ndarray) -> np.ndarray:
        return self._ns.fit_transform(f) if f.shape[0] > 1 else f
    def transform_edge_features(self, f: np.ndarray) -> np.ndarray:
        return self._es.fit_transform(f) if f.shape[0] > 1 else f


class LogTransformFeatures:
    def transform_node_features(self, f: np.ndarray) -> np.ndarray:
        return np.sign(f) * np.log1p(np.abs(f))
    def transform_edge_features(self, f: np.ndarray) -> np.ndarray:
        return np.sign(f) * np.log1p(np.abs(f))


class ChainFeatures:
    def __init__(self, strategies: List[FeatureStrategy]) -> None:
        self.strategies = strategies
    def transform_node_features(self, f: np.ndarray) -> np.ndarray:
        for s in self.strategies: f = s.transform_node_features(f)
        return f
    def transform_edge_features(self, f: np.ndarray) -> np.ndarray:
        for s in self.strategies: f = s.transform_edge_features(f)
        return f
