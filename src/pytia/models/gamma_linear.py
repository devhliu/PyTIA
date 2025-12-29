from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_func

from ..metrics import r2_score


def fit_gamma_linear_wls(
    A: np.ndarray, times: np.ndarray, valid: np.ndarray, lambda_phys: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearized gamma fit on log(A):
      ln A = lnK + alpha ln t - beta t
    Vectorized WLS using per-voxel normal equations.

    Returns:
      params: (N,3) [lnK, alpha, beta]
      tpeak: (N,)
      yhat: (N,T) predicted A
      r2: (N,)
    """
    N, T = A.shape
    t = times.astype(np.float64)
    if np.any(t <= 0):
        raise ValueError("Gamma linear fit requires times > 0.")
    x0 = np.ones((T,), dtype=np.float64)
    x1 = np.log(t)
    x2 = -t

    # Need A>0 at valid points for log; redefine valid2
    valid2 = valid & (A > 0)
    w = valid2.astype(np.float64)

    y = np.where(valid2, np.log(A), 0.0).astype(np.float64)

    # Precompute sums for each voxel: Σ w * f
    Sw = np.sum(w, axis=1)
    Sx1 = np.sum(w * x1[None, :], axis=1)
    Sx2 = np.sum(w * x2[None, :], axis=1)
    Sx1x1 = np.sum(w * (x1[None, :] ** 2), axis=1)
    Sx1x2 = np.sum(w * (x1[None, :] * x2[None, :]), axis=1)
    Sx2x2 = np.sum(w * (x2[None, :] ** 2), axis=1)

    Sy = np.sum(w * y, axis=1)
    Sx1y = np.sum(w * (x1[None, :] * y), axis=1)
    Sx2y = np.sum(w * (x2[None, :] * y), axis=1)

    # Assemble G and b for each voxel:
    # [ Sw     Sx1    Sx2 ] [lnK ] = [ Sy  ]
    # [ Sx1   Sx1x1  Sx1x2] [alpha]   [Sx1y]
    # [ Sx2   Sx1x2  Sx2x2] [beta ]   [Sx2y]
    G = np.zeros((N, 3, 3), dtype=np.float64)
    b = np.zeros((N, 3), dtype=np.float64)

    G[:, 0, 0] = Sw
    G[:, 0, 1] = Sx1
    G[:, 0, 2] = Sx2
    G[:, 1, 0] = Sx1
    G[:, 1, 1] = Sx1x1
    G[:, 1, 2] = Sx1x2
    G[:, 2, 0] = Sx2
    G[:, 2, 1] = Sx1x2
    G[:, 2, 2] = Sx2x2

    b[:, 0] = Sy
    b[:, 1] = Sx1y
    b[:, 2] = Sx2y

    params = np.full((N, 3), np.nan, dtype=np.float64)

    # Solve batched 3x3; skip ill-conditioned
    # Condition proxy: det != 0 and enough points
    enough = Sw >= 3
    det = np.linalg.det(G[enough])
    ok = np.zeros((N,), dtype=bool)
    ok[enough] = np.isfinite(det) & (np.abs(det) > 1e-12)
    if np.any(ok):
        # Fix boolean indexing to maintain proper shape
        ok_indices = np.where(ok)[0]
        if len(ok_indices) > 0:
            # Use list comprehension to solve each system individually
            for i in ok_indices:
                if np.linalg.matrix_rank(G[i]) == 3:  # Check if matrix is invertible
                    try:
                        params[i] = np.linalg.solve(G[i], b[i])
                    except np.linalg.LinAlgError:
                        pass  # Keep as NaN

    lnK = params[:, 0]
    alpha = params[:, 1]
    beta = params[:, 2]

    if lambda_phys is not None:
        beta = np.where(np.isfinite(beta), np.maximum(beta, float(lambda_phys)), beta)

    # Build predictions
    K = np.exp(lnK)
    Ahat = (K[:, None] * (t[None, :] ** alpha[:, None]) * np.exp(-beta[:, None] * t[None, :])).astype(
        np.float64
    )
    # invalid if alpha<=0 or beta<=0 or K not finite
    bad = (~np.isfinite(K)) | (~np.isfinite(alpha)) | (~np.isfinite(beta)) | (alpha <= 0) | (beta <= 0)
    Ahat[bad, :] = np.nan

    tpeak = np.full((N,), np.nan, dtype=np.float64)
    good = ~bad
    tpeak[good] = (alpha[good] / beta[good]).astype(np.float64)

    r2 = r2_score(A.astype(np.float64), Ahat, valid)

    params2 = np.stack([lnK, alpha, beta], axis=1).astype(np.float32)
    return params2, tpeak.astype(np.float32), Ahat.astype(np.float32), r2.astype(np.float32)


def tia_from_gamma_params(params: np.ndarray) -> np.ndarray:
    """
    params: (N,3) lnK, alpha, beta
    Returns TIA (N,) integrating from 0 to inf in same activity units * seconds.
    """
    lnK = params[:, 0].astype(np.float64)
    alpha = params[:, 1].astype(np.float64)
    beta = params[:, 2].astype(np.float64)
    K = np.exp(lnK)
    tia = np.full((params.shape[0],), np.nan, dtype=np.float64)
    ok = np.isfinite(K) & np.isfinite(alpha) & np.isfinite(beta) & (alpha > 0) & (beta > 0)
    if np.any(ok):
        tia[ok] = K[ok] * gamma_func(alpha[ok] + 1.0) / (beta[ok] ** (alpha[ok] + 1.0))
    return tia.astype(np.float32)