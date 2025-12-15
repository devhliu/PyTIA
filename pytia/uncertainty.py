from __future__ import annotations

import numpy as np


def residual_bootstrap(
    A: np.ndarray,
    Ahat: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Residual bootstrap: A* = clip(Ahat + r*), resampling residuals along time per voxel.
    Returns Astar (N,T)
    """
    r = np.where(valid, A - Ahat, 0.0).astype(np.float32)
    N, T = A.shape

    # For each voxel, sample indices 0..T-1; but only meaningful residuals where valid
    idx = rng.integers(0, T, size=(N, T), dtype=np.int32)
    rstar = np.take_along_axis(r, idx, axis=1)
    Astar = Ahat + rstar
    return np.maximum(Astar, 0.0)