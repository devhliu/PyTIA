"""Noise floor estimation and negative-value clamping utilities."""

from __future__ import annotations

import numpy as np


def clamp_negative_to_zero(data: np.ndarray) -> np.ndarray:
    """Replace negative values with zero, returning float32."""
    return np.maximum(data, 0.0).astype(np.float32)


def compute_noise_floor(A: np.ndarray, mode: str, absolute: float, rel_frac: float) -> np.ndarray:
    """Compute per-voxel noise floor.

    Parameters
    ----------
    A : ndarray, shape (N_vox, N_time)
        Activity values (Bq or Bq/ml) after clamping.
    mode : ``"absolute"`` or ``"relative"``
    absolute : float
        Absolute floor value (used when *mode* is ``"absolute"``).
    rel_frac : float
        Fraction of per-voxel max (used when *mode* is ``"relative"``).

    Returns
    -------
    ndarray, shape (N_vox,)
    """
    if mode == "absolute":
        return np.full((A.shape[0],), float(absolute), dtype=np.float32)
    if mode == "relative":
        finite_rows = np.any(np.isfinite(A), axis=1)
        mx = np.full((A.shape[0],), np.nan, dtype=np.float64)
        if np.any(finite_rows):
            mx[finite_rows] = np.nanmax(A[finite_rows], axis=1)
        return (mx * float(rel_frac)).astype(np.float32)
    raise ValueError(f"Unknown noise_floor mode: {mode!r}. Use 'absolute' or 'relative'.")


def valid_mask_from_floor(A: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Return boolean mask ``(N_vox, N_time)`` of values above the floor."""
    return A >= floor[:, None]
