"""Serialisation helpers for CDE-ready graph data.

Saves / loads the dict produced by :class:`~stg_infra.stg.temporal.interpolation.CDEPathBuilder`
to disk in either NumPy (``.npy``) or PyTorch (``.pt``) format.

NumPy layout (directory)::

    X.npy          — (N, T, F+1) node feature trajectories
    times.npy      — (T,)        normalised timestamps
    raw_times.npy  — (T,)        POSIX timestamps before normalisation
    adjacency.npy  — (N, N) or (T, N, N)
    mask.npy       — (T, N) bool — True where node was observed
    meta.json      — {"node_ids": [...], "shapes": {...}}

PyTorch layout (directory) — same names but ``.pt`` extension.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

_ARRAY_KEYS = ("X", "times", "raw_times", "adjacency", "mask")


class CDEExporter:
    """Save / load :class:`CDEPathBuilder` output."""

    # ------------------------------------------------------------------
    # NumPy format
    # ------------------------------------------------------------------

    @staticmethod
    def save(data: Dict[str, Any], directory: str) -> Dict[str, str]:
        """Save arrays as ``.npy`` files and metadata as ``meta.json``.

        Parameters
        ----------
        data:
            Dict returned by ``CDEPathBuilder.build()``.
        directory:
            Target directory (created if absent).

        Returns
        -------
        dict mapping key names to saved file paths.
        """
        os.makedirs(directory, exist_ok=True)
        paths: Dict[str, str] = {}

        for key in _ARRAY_KEYS:
            if key not in data:
                continue
            p = os.path.join(directory, f"{key}.npy")
            np.save(p, data[key])
            paths[key] = p
            logger.info("Saved %-12s → %s  shape=%s", key, p, data[key].shape)

        meta = {
            "node_ids": [str(n) for n in data.get("node_ids", [])],
            "shapes": {
                k: list(data[k].shape)
                for k in _ARRAY_KEYS
                if k in data
            },
        }
        p = os.path.join(directory, "meta.json")
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        paths["meta"] = p
        logger.info("Saved metadata  → %s", p)

        return paths

    @staticmethod
    def load(directory: str) -> Dict[str, Any]:
        """Load previously saved NumPy CDE data.

        Parameters
        ----------
        directory:
            Directory written by :meth:`save`.

        Returns
        -------
        dict with the same keys as ``CDEPathBuilder.build()``.
        """
        data: Dict[str, Any] = {}

        for key in _ARRAY_KEYS:
            p = os.path.join(directory, f"{key}.npy")
            if os.path.exists(p):
                data[key] = np.load(p)
            else:
                logger.debug("No file for key %r in %s", key, directory)

        meta_p = os.path.join(directory, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                meta = json.load(f)
            data["node_ids"] = meta.get("node_ids", [])

        return data

    # ------------------------------------------------------------------
    # PyTorch format
    # ------------------------------------------------------------------

    @staticmethod
    def save_pt(data: Dict[str, Any], directory: str) -> Dict[str, str]:
        """Save arrays as PyTorch ``.pt`` tensors.

        The ``mask`` array is stored as ``torch.bool``; all others as
        ``torch.float32``.

        Parameters
        ----------
        data:
            Dict returned by ``CDEPathBuilder.build()``.
        directory:
            Target directory (created if absent).

        Returns
        -------
        dict mapping key names to saved file paths.
        """

        os.makedirs(directory, exist_ok=True)
        paths: Dict[str, str] = {}

        for key in _ARRAY_KEYS:
            if key not in data:
                continue
            arr = data[key]
            dtype = torch.bool if key == "mask" else torch.float32
            t = torch.tensor(arr, dtype=dtype)
            p = os.path.join(directory, f"{key}.pt")
            torch.save(t, p)
            paths[key] = p
            logger.info("Saved %-12s → %s  shape=%s", key, p, tuple(t.shape))

        meta = {"node_ids": [str(n) for n in data.get("node_ids", [])]}
        p = os.path.join(directory, "meta.json")
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)
        paths["meta"] = p
        logger.info("Saved metadata  → %s", p)

        return paths

    @staticmethod
    def load_pt(directory: str) -> Dict[str, Any]:
        """Load previously saved PyTorch CDE data.

        Returns
        -------
        dict with keys matching ``CDEPathBuilder.build()``;
        values are ``torch.Tensor`` objects.
        """
        data: Dict[str, Any] = {}

        for key in _ARRAY_KEYS:
            p = os.path.join(directory, f"{key}.pt")
            if os.path.exists(p):
                data[key] = torch.load(p, weights_only=True)
            else:
                logger.debug("No file for key %r in %s", key, directory)

        meta_p = os.path.join(directory, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                meta = json.load(f)
            data["node_ids"] = meta.get("node_ids", [])

        return data
