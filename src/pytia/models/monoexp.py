from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def fit_monoexp_tail(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit log-linear monoexp tail after measured peak_index per voxel.
    Handles vectorization by masked linear regression on log(A).

    Returns:
      lambda_eff: (N,)
      Ahat: (N,T)
      r2: (N,)
    """
    N, T = A.shape
    t = times.astype(np.float64)

    # Build mask for tail points: j >= peak_index[i]
    j = np.arange(T)[None, :]
    tail_mask = valid & (j >= peak_index[:, None]) & (A > 0)

    w = tail_mask.astype(np.float64)
    y = np.where(tail_mask, np.log(A), 0.0).astype(np.float64)
    x = np.broadcast_to(t[None, :], (N, T)).astype(np.float64)

    Sw = np.sum(w, axis=1)
    Sx = np.sum(w * x, axis=1)
    Sy = np.sum(w * y, axis=1)
    Sxx = np.sum(w * x * x, axis=1)
    Sxy = np.sum(w * x * y, axis=1)

    # slope = (Sw*Sxy - Sx*Sy) / (Sw*Sxx - Sx^2)
    denom = Sw * Sxx - Sx * Sx
    slope = np.full((N,), np.nan, dtype=np.float64)
    ok = (Sw >= 2) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    slope[ok] = (Sw[ok] * Sxy[ok] - Sx[ok] * Sy[ok]) / denom[ok]
    # y ~ a + b x ; for monoexp: ln A = ln A0 - lambda t, so lambda = -slope
    lam = np.full((N,), np.nan, dtype=np.float64)
    lam[ok] = -slope[ok]
    if lambda_phys is not None:
        lam = np.where(np.isfinite(lam), np.maximum(lam, float(lambda_phys)), lam)

    # intercept
    a = np.full((N,), np.nan, dtype=np.float64)
    a[ok] = (Sy[ok] - slope[ok] * Sx[ok]) / Sw[ok]

    Ahat = np.full((N, T), np.nan, dtype=np.float64)
    good = ok & np.isfinite(a) & np.isfinite(lam) & (lam > 0)
    if np.any(good):
        Ahat[good, :] = np.exp(a[good, None] - lam[good, None] * t[None, :])

    r2 = r2_score(A.astype(np.float64), Ahat, valid)
    return lam.astype(np.float32), Ahat.astype(np.float32), r2.astype(np.float32)


def tia_monoexp_with_triangle_uptake(
    A: np.ndarray, times: np.ndarray, valid: np.ndarray, lambda_eff: np.ndarray, peak_index: np.ndarray
) -> np.ndarray:
    """
    TIA = 0.5*A_peak*t_peak + A_peak/lambda_eff
    Where A_peak at measured peak_index. If invalid, NaN.
    """
    N, T = A.shape
    t = times.astype(np.float64)
    # peak time/value
    Apeak = np.take_along_axis(A, peak_index[:, None], axis=1)[:, 0].astype(np.float64)
    tpeak = np.take_along_axis(np.broadcast_to(t[None, :], (N, T)), peak_index[:, None], axis=1)[:, 0]

    ok = np.isfinite(Apeak) & np.isfinite(tpeak) & np.isfinite(lambda_eff) & (lambda_eff > 0)
    ok &= np.take_along_axis(valid, peak_index[:, None], axis=1)[:, 0]

    tia = np.full((N,), np.nan, dtype=np.float64)
    tia[ok] = 0.5 * Apeak[ok] * tpeak[ok] + (Apeak[ok] / lambda_eff[ok].astype(np.float64))
    return tia.astype(np.float32)