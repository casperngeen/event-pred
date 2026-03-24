"""Control path construction for Graph Neural CDEs.

Converts a ``SpatioTemporalGraph`` into tensors suitable for neural CDE /
latent ODE solvers (e.g. ``torchcde``, ``diffrax``).

Typical workflow
----------------
::

    from stg_infra.stg.temporal.interpolation import CDEPathBuilder

    builder = CDEPathBuilder(stg, fill_strategy="linear", adjacency_mode="fixed")
    data = builder.build()
    # data["X"]         : (N, T, F+1)  — node trajectories + time channel
    # data["times"]     : (T,)          — normalised timestamps in [0, 1]
    # data["adjacency"] : (N, N) or (T, N, N)
    # data["node_ids"]  : list[Hashable]
    # data["mask"]      : (T, N) bool   — True where node was observed

With torchcde::

    coeffs, t = CDEPathBuilder.to_torchcde(data)
    # coeffs shape: (N, T-1, 4*(F+1)) — natural cubic spline coefficients
    # Feed into a torchcde.CubicSpline and your GN-CDE model.

With scipy::

    splines = CDEPathBuilder.to_scipy_splines(data)
    # splines[i](t) → feature vector for node i at time t
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Hashable, List

import numpy as np
import torch
import torchcde

from stg_infra.stg.core import SpatioTemporalGraph


class CDEPathBuilder:
    """Build continuous control paths from a :class:`SpatioTemporalGraph`.

    Parameters
    ----------
    stg:
        Source spatio-temporal graph.
    time_normalise:
        Rescale timestamps to ``[0, 1]``. Recommended for ODE solvers that
        are sensitive to the scale of *t*.
    append_time:
        Append the (normalised) timestamp as the last feature channel of
        every node. Neural CDEs require ``dX/dt`` to be well-defined; adding
        *t* as a channel guarantees this even when features are constant.
    fill_strategy:
        How to fill feature values at timesteps where a node was absent:

        ``"linear"``
            Linear interpolation between observed values; forward/backward
            extrapolation at the edges uses the nearest observed value.
        ``"forward"``
            Carry the last observed value forward (zero-fill before first
            observation).
        ``"zero"``
            Replace all missing values with 0.
    adjacency_mode:
        ``"fixed"``  — single (N, N) matrix taken from the **first** snapshot.
        ``"mean"``   — element-wise average over all snapshots → (N, N).
        ``"temporal"`` — full (T, N, N) tensor.
    """

    def __init__(
        self,
        stg: SpatioTemporalGraph,
        time_normalise: bool = True,
        append_time: bool = True,
        fill_strategy: str = "linear",
        adjacency_mode: str = "fixed",
    ) -> None:
        self.stg = stg
        self.time_normalise = time_normalise
        self.append_time = append_time
        self.fill_strategy = fill_strategy
        self.adjacency_mode = adjacency_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """Build the CDE-ready data dictionary.

        Returns
        -------
        dict with keys:

        ``X``
            ``np.ndarray`` of shape ``(N, T, F)`` or ``(N, T, F+1)`` if
            ``append_time=True``.  Node feature trajectories with missing
            values filled according to ``fill_strategy``.
        ``times``
            ``np.ndarray`` of shape ``(T,)``.  Normalised (or raw POSIX)
            timestamps depending on ``time_normalise``.
        ``raw_times``
            ``np.ndarray`` of shape ``(T,)``.  Raw POSIX timestamps.
        ``adjacency``
            ``np.ndarray`` of shape ``(N, N)`` or ``(T, N, N)`` depending
            on ``adjacency_mode``.
        ``node_ids``
            ``list[Hashable]`` — node ordering used for axis N above.
        ``mask``
            ``np.ndarray`` bool of shape ``(T, N)`` — ``True`` where the
            node was actually observed in that snapshot.
        """
        node_ids = self.stg.all_node_ids()
        N = len(node_ids)
        T = len(self.stg)

        if N == 0 or T == 0:
            raise ValueError("STG is empty — cannot build CDE path.")

        raw_times = self._extract_times()

        t_min = raw_times[0]
        t_range = raw_times[-1] - raw_times[0]
        if self.time_normalise and t_range > 0:
            times = (raw_times - t_min) / t_range
        else:
            times = raw_times.copy()

        # (T, N, F) with NaN where absent, plus boolean mask
        padded, mask = self.stg.feature_tensor_padded()

        filled = self._fill(padded, mask)  # (T, N, F)

        if self.append_time:
            # Broadcast (T,) → (T, N, 1) then concatenate
            time_channel = np.tile(times[:, None, None], (1, N, 1))  # (T, N, 1)
            filled = np.concatenate([filled, time_channel], axis=2)   # (T, N, F+1)

        # torchcde convention: (batch=N, time=T, channels=F+1)
        X = filled.transpose(1, 0, 2)  # (N, T, F+1)

        adjacency = self._build_adjacency(node_ids)

        return {
            "X": X,
            "times": times,
            "raw_times": raw_times,
            "adjacency": adjacency,
            "node_ids": node_ids,
            "mask": mask,  # (T, N)
        }

    # ------------------------------------------------------------------
    # Optional downstream helpers (soft dependencies)
    # ------------------------------------------------------------------

    @staticmethod
    def to_torchcde(data: Dict[str, Any]) -> Any:
        """Convert ``build()`` output to ``torchcde`` cubic spline coefficients.

        Requires ``torch`` and ``torchcde`` (``pip install torchcde torch``).

        Returns
        -------
        coeffs
            Natural cubic spline coefficients, shape ``(N, T-1, 4*(F+1))``.
            Pass to ``torchcde.CubicSpline(coeffs, t)`` inside your model.
        times_t
            ``torch.Tensor`` of shape ``(T,)`` — the normalised time grid.
        """
        X = torch.tensor(data["X"], dtype=torch.float32)          # (N, T, F+1)
        times_t = torch.tensor(data["times"], dtype=torch.float32)  # (T,)
        coeffs = torchcde.natural_cubic_spline_coeffs(X, t=times_t)
        return coeffs, times_t

    @staticmethod
    def to_scipy_splines(data: Dict[str, Any]) -> List[Any]:
        """Return a ``CubicSpline`` per node mapping *t* → feature vector.

        Requires ``scipy`` (``pip install scipy``).

        Each spline takes a scalar or array of times (in ``data["times"]``
        units) and returns an array of shape ``(..., F+1)``.
        """
        try:
            from scipy.interpolate import CubicSpline
        except ImportError as exc:
            raise ImportError("scipy is required: pip install scipy") from exc

        times = data["times"]        # (T,)
        X = data["X"]                # (N, T, F+1)
        return [CubicSpline(times, X[n]) for n in range(X.shape[0])]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_times(self) -> np.ndarray:
        times: List[float] = []
        for snap in self.stg:
            ts = snap.timestamp
            times.append(ts.timestamp() if isinstance(ts, datetime) else float(ts))
        return np.array(times, dtype=np.float64)

    def _fill(self, padded: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill NaN in ``(T, N, F)`` according to ``self.fill_strategy``."""
        filled = padded.copy()
        T, N, F = filled.shape

        if self.fill_strategy == "zero":
            np.nan_to_num(filled, copy=False, nan=0.0)

        elif self.fill_strategy == "forward":
            for n in range(N):
                last = np.zeros(F, dtype=np.float64)
                for t in range(T):
                    if mask[t, n]:
                        last = filled[t, n].copy()
                    else:
                        filled[t, n] = last

        elif self.fill_strategy == "linear":
            all_t = np.arange(T, dtype=np.float64)
            for n in range(N):
                obs_t = np.where(mask[:, n])[0]
                if len(obs_t) == 0:
                    filled[:, n, :] = 0.0
                    continue
                if len(obs_t) == 1:
                    filled[:, n, :] = filled[obs_t[0], n, :]
                    continue
                for f in range(F):
                    filled[:, n, f] = np.interp(
                        all_t, obs_t.astype(np.float64), filled[obs_t, n, f]
                    )

        else:
            raise ValueError(
                f"Unknown fill_strategy {self.fill_strategy!r}. "
                "Choose 'linear', 'forward', or 'zero'."
            )

        return filled

    def _build_adjacency(self, node_ids: List[Hashable]) -> np.ndarray:
        node_idx: Dict[Hashable, int] = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        if self.adjacency_mode == "fixed":
            A = np.zeros((N, N), dtype=np.float64)
            snap = self.stg[0]
            for u, v in snap.edges:
                if u in node_idx and v in node_idx:
                    e = snap.get_edge(u, v)
                    A[node_idx[u], node_idx[v]] = e.weight if e else 1.0
            return A

        elif self.adjacency_mode == "mean":
            A_sum = np.zeros((N, N), dtype=np.float64)
            for snap in self.stg:
                for u, v in snap.edges:
                    if u in node_idx and v in node_idx:
                        e = snap.get_edge(u, v)
                        A_sum[node_idx[u], node_idx[v]] += e.weight if e else 1.0
            return A_sum / max(len(self.stg), 1)

        elif self.adjacency_mode == "temporal":
            T = len(self.stg)
            A = np.zeros((T, N, N), dtype=np.float64)
            for t, snap in enumerate(self.stg):
                for u, v in snap.edges:
                    if u in node_idx and v in node_idx:
                        e = snap.get_edge(u, v)
                        A[t, node_idx[u], node_idx[v]] = e.weight if e else 1.0
            return A

        else:
            raise ValueError(
                f"Unknown adjacency_mode {self.adjacency_mode!r}. "
                "Choose 'fixed', 'mean', or 'temporal'."
            )
