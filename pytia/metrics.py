"""Goodness-of-fit metrics for curve fitting."""

from __future__ import annotations

import numpy as np


def r2_score(y: np.ndarray, yhat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Vectorized coefficient of determination (R²) per voxel.

    Parameters
    ----------
    y, yhat : ndarray, shape (N_vox, T)
    valid : bool ndarray, shape (N_vox, T)

    Returns
    -------
    ndarray, shape (N_vox,), dtype float32
    """
    yv = np.where(valid, y, np.nan)
    yhv = np.where(valid, yhat, np.nan)
    # SS_res
    ss_res = np.nansum((yv - yhv) ** 2, axis=1)
    # SS_tot
    mu = np.nanmean(yv, axis=1)
    ss_tot = np.nansum((yv - mu[:, None]) ** 2, axis=1)
    r2 = np.full((y.shape[0],), np.nan, dtype=np.float32)
    finite = np.isfinite(ss_res) & np.isfinite(ss_tot)

    regular = finite & (ss_tot > 0)
    r2[regular] = (1.0 - ss_res[regular] / ss_tot[regular]).astype(np.float32)

    zero_var = finite & np.isclose(ss_tot, 0.0)
    if np.any(zero_var):
        perfect = zero_var & np.isclose(ss_res, 0.0)
        imperfect = zero_var & ~perfect
        r2[perfect] = 1.0
        r2[imperfect] = 0.0
    return r2
