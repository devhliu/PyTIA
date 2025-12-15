from __future__ import annotations

import numpy as np

# class ids (voxel mode)
CLASS_FALLING = 1
CLASS_RISING = 2
CLASS_HUMP = 3
CLASS_AMBIG = 4


def classify_curves(A: np.ndarray, valid: np.ndarray, eps_rel: float = 0.02) -> np.ndarray:
    """
    Vectorized classification.
    A: (N_vox, N_time) non-negative
    valid: (N_vox, N_time)
    Returns class_id (N_vox,)
    """
    N, T = A.shape
    # Use -inf for invalid to find peak among valid points
    A_for_peak = np.where(valid, A, -np.inf)
    idx_max = np.argmax(A_for_peak, axis=1)

    # Find first/last valid indices
    has_valid = np.any(valid, axis=1)
    first_valid = np.argmax(valid, axis=1)
    last_valid = T - 1 - np.argmax(valid[:, ::-1], axis=1)

    # Gradient sign counts (ignoring invalid transitions by masking)
    dA = np.diff(A, axis=1)
    dv = valid[:, 1:] & valid[:, :-1]
    # scale eps by voxel max to avoid noisy sign flips
    mx = np.nanmax(np.where(valid, A, np.nan), axis=1)
    eps = np.maximum(mx * float(eps_rel), 1e-6)
    n_pos = np.sum((dA > eps[:, None]) & dv, axis=1)
    n_neg = np.sum((dA < -eps[:, None]) & dv, axis=1)

    falling = has_valid & (idx_max == first_valid) & (n_neg > 0)
    rising = has_valid & (idx_max == last_valid) & (n_pos > 0)
    hump = has_valid & (~falling) & (~rising) & (idx_max > first_valid) & (idx_max < last_valid)

    cls = np.full((N,), CLASS_AMBIG, dtype=np.uint8)
    cls[falling] = CLASS_FALLING
    cls[rising] = CLASS_RISING
    cls[hump] = CLASS_HUMP
    return cls