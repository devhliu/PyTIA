from __future__ import annotations

import numpy as np


def hybrid_piecewise_hat_at_samples(A: np.ndarray, valid: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear Ahat at sampled times using nearest valid points left/right.

    - Vectorized over voxels.
    - Uses small loops over time dimension only (T), not voxel loops.
    - Interpolates in *time* (not index), so non-uniform sampling is handled.

    A: (N,T)
    valid: (N,T)
    times: (T,)
    Returns Ahat: (N,T) with NaNs where cannot predict.
    """
    N, T = A.shape
    t = times.astype(np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    Ahat[valid] = A[valid].astype(np.float32)

    left = np.full((N, T), -1, dtype=np.int32)
    last = np.full((N,), -1, dtype=np.int32)
    for j in range(T):
        last = np.where(valid[:, j], j, last)
        left[:, j] = last

    right = np.full((N, T), -1, dtype=np.int32)
    nxt = np.full((N,), -1, dtype=np.int32)
    for j in range(T - 1, -1, -1):
        nxt = np.where(valid[:, j], j, nxt)
        right[:, j] = nxt

    need = (~valid) & (left >= 0) & (right >= 0)
    if not np.any(need):
        return Ahat

    # where left == right, we can just copy that value
    same = need & (left == right)
    if np.any(same):
        li = left[same]
        ii, jj = np.nonzero(same)
        Ahat[same] = A[ii, li].astype(np.float32)

    interp = need & (left != right)
    if not np.any(interp):
        return Ahat

    ii, jj = np.nonzero(interp)
    li = left[interp]
    ri = right[interp]

    tL = t[li]
    tR = t[ri]
    tJ = t[jj]

    aL = A[ii, li]
    aR = A[ii, ri]

    denom = (tR - tL)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    w = (tJ - tL) / denom

    Ahat[interp] = ((1.0 - w) * aL + w * aR).astype(np.float32)
    return Ahat