from __future__ import annotations

import numpy as np


def r2_score(y: np.ndarray, yhat: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Vectorized R^2 per voxel. y,yhat shape (N_vox,T), valid shape (N_vox,T)
    Returns (N_vox,) float
    """
    yv = np.where(valid, y, np.nan)
    yhv = np.where(valid, yhat, np.nan)
    # SS_res
    ss_res = np.nansum((yv - yhv) ** 2, axis=1)
    # SS_tot
    mu = np.nanmean(yv, axis=1)
    ss_tot = np.nansum((yv - mu[:, None]) ** 2, axis=1)
    r2 = np.full((y.shape[0],), np.nan, dtype=np.float32)
    ok = np.isfinite(ss_res) & np.isfinite(ss_tot) & (ss_tot > 0)
    r2[ok] = (1.0 - ss_res[ok] / ss_tot[ok]).astype(np.float32)
    return r2