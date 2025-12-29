from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def fit_rectangular_uptake_monoexp_washout(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit rectangular uptake + mono-exponential washout model:
    
    For t <= t_max: A(t) = A_max (constant uptake phase)
    For t > t_max: A(t) = A_max * exp(-lambda * (t - t_max))
    
    This model assumes instant uptake to plateau followed by exponential clearance.
    Suitable for tumors with rapid accumulation and slow clearance.
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 2) [A_max, lambda] parameters
        tpeak: (N,) peak time
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 2), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    n_valid = np.sum(valid2, axis=1)
    
    for i in range(N):
        if n_valid[i] < 3:
            continue
        
        t_i = t[valid2[i]]
        A_i = A[i, valid2[i]]
        
        idx_max = np.argmax(A_i)
        t_max = t_i[idx_max]
        A_max = A_i[idx_max]
        
        tpeak[i] = t_max
        
        tail_mask = t_i > t_max
        if np.sum(tail_mask) >= 2:
            t_tail = t_i[tail_mask]
            A_tail = A_i[tail_mask]
            
            log_A_tail = np.log(A_tail)
            poly = np.polyfit(t_tail, log_A_tail, 1)
            lam = -poly[0]
            
            if lambda_phys is not None:
                lam = max(lam, float(lambda_phys))
            
            if lam > 0:
                params[i] = [A_max, lam]
                
                Ahat[i, :] = np.where(
                    t <= t_max,
                    A_max,
                    A_max * np.exp(-lam * (t - t_max))
                )
        else:
            if lambda_phys is not None:
                lam = float(lambda_phys)
            else:
                lam = 1e-4
            
            params[i] = [A_max, lam]
            
            Ahat[i, :] = np.where(
                t <= t_max,
                A_max,
                A_max * np.exp(-lam * (t - t_max))
            )
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2


def fit_rectangular_uptake_monoexp_washout_vectorized(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized implementation of rectangular uptake + mono-exponential washout.
    
    Uses matrix operations for efficient batch processing of all voxels.
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 2) [A_max, lambda] parameters
        tpeak: (N,) peak time
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 2), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    
    A_max = np.take_along_axis(A, peak_index[:, None], axis=1)[:, 0].astype(np.float64)
    t_max = np.take_along_axis(np.broadcast_to(t[None, :], (N, T)), peak_index[:, None], axis=1)[:, 0]
    
    tpeak = t_max
    
    tail_mask = (t[None, :] > t_max[:, None]) & valid2
    n_tail = np.sum(tail_mask, axis=1)
    
    ok_tail = n_tail >= 2
    
    lam = np.full((N,), np.nan, dtype=np.float64)
    
    for i in np.where(ok_tail)[0]:
        t_tail = t[tail_mask[i]]
        A_tail = A[i, tail_mask[i]]
        
        log_A_tail = np.log(A_tail)
        poly = np.polyfit(t_tail, log_A_tail, 1)
        lam[i] = -poly[0]
    
    if lambda_phys is not None:
        lam = np.where(np.isfinite(lam), np.maximum(lam, float(lambda_phys)), lam)
    
    ok = ok_tail & np.isfinite(lam) & (lam > 0)
    
    params[ok, 0] = A_max[ok]
    params[ok, 1] = lam[ok]
    
    for i in np.where(ok)[0]:
        Ahat[i, :] = np.where(
            t <= t_max[i],
            A_max[i],
            A_max[i] * np.exp(-lam[i] * (t - t_max[i]))
        )
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2


def fit_rectangular_uptake_monoexp_washout_matrix(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Matrix-based implementation using weighted least squares for the washout phase.
    
    For the washout phase (t > t_max):
        ln A = ln A_max - lambda * (t - t_max)
        ln A = (ln A_max + lambda * t_max) - lambda * t
    
    This is a linear equation in lambda and intercept.
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 2) [A_max, lambda] parameters
        tpeak: (N,) peak time
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 2), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    
    A_max = np.take_along_axis(A, peak_index[:, None], axis=1)[:, 0].astype(np.float64)
    t_max = np.take_along_axis(np.broadcast_to(t[None, :], (N, T)), peak_index[:, None], axis=1)[:, 0]
    
    tpeak = t_max
    
    tail_mask = (t[None, :] > t_max[:, None]) & valid2 & (A > 0)
    
    w = tail_mask.astype(np.float64)
    y = np.where(tail_mask, np.log(A), 0.0).astype(np.float64)
    x = np.broadcast_to(t[None, :], (N, T)).astype(np.float64)
    
    Sw = np.sum(w, axis=1)
    Sx = np.sum(w * x, axis=1)
    Sy = np.sum(w * y, axis=1)
    Sxx = np.sum(w * x * x, axis=1)
    Sxy = np.sum(w * x * y, axis=1)
    
    denom = Sw * Sxx - Sx * Sx
    slope = np.full((N,), np.nan, dtype=np.float64)
    ok = (Sw >= 2) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    slope[ok] = (Sw[ok] * Sxy[ok] - Sx[ok] * Sy[ok]) / denom[ok]
    
    lam = np.full((N,), np.nan, dtype=np.float64)
    lam[ok] = -slope[ok]
    
    if lambda_phys is not None:
        lam = np.where(np.isfinite(lam), np.maximum(lam, float(lambda_phys)), lam)
    
    ok_final = ok & np.isfinite(lam) & (lam > 0)
    
    params[ok_final, 0] = A_max[ok_final]
    params[ok_final, 1] = lam[ok_final]
    
    for i in np.where(ok_final)[0]:
        Ahat[i, :] = np.where(
            t <= t_max[i],
            A_max[i],
            A_max[i] * np.exp(-lam[i] * (t - t_max[i]))
        )
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2


def tia_rectangular_uptake_monoexp_washout(params: np.ndarray, tpeak: np.ndarray) -> np.ndarray:
    """
    Compute TIA from rectangular uptake + mono-exponential washout parameters.
    
    TIA = A_max * t_max + A_max / lambda
    
    Args:
        params: (N, 2) [A_max, lambda] parameters
        tpeak: (N,) peak time (t_max)
        
    Returns:
        tia: (N,) time-integrated activity values
    """
    A_max = params[:, 0].astype(np.float64)
    lam = params[:, 1].astype(np.float64)
    t_max = tpeak.astype(np.float64)
    
    tia = np.full((params.shape[0],), np.nan, dtype=np.float64)
    
    ok = np.isfinite(A_max) & np.isfinite(lam) & np.isfinite(t_max) & (lam > 0)
    if np.any(ok):
        tia[ok] = A_max[ok] * t_max[ok] + A_max[ok] / lam[ok]
    
    return tia.astype(np.float32)
