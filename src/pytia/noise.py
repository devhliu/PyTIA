from __future__ import annotations

import numpy as np


def clamp_negative_to_zero(data4: np.ndarray) -> np.ndarray:
    return np.maximum(data4, 0.0, dtype=np.float32)


def compute_noise_floor(
    A: np.ndarray, mode: str, absolute: float, rel_frac: float
) -> np.ndarray:
    """
    A: (N_vox, N_time) in Bq/ml (or Bq) after clamp.
    Returns per-voxel floor (N_vox,)
    """
    if mode == "absolute":
        return np.full((A.shape[0],), float(absolute), dtype=np.float32)
    # relative
    mx = np.nanmax(A, axis=1)
    return (mx * float(rel_frac)).astype(np.float32)


def valid_mask_from_floor(A: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """
    Returns boolean valid mask (N_vox, N_time).
    """
    return A >= floor[:, None]