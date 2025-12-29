from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def fit_biexp(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit bi-exponential model: A(t) = C * (exp(-lambda1 * t) - exp(-lambda2 * t))
    using nonlinear least squares with vectorized implementation.
    
    This model captures both uptake and clearance phases, suitable for tumor TACs.
    Requires >= 5-6 time points for robust fit.
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 3) [lnC, lambda1, lambda2] parameters
        tpeak: (N,) peak time from fitted parameters
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 3), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    n_valid = np.sum(valid2, axis=1)
    
    for i in range(N):
        if n_valid[i] < 5:
            continue
        
        t_i = t[valid2[i]]
        A_i = A[i, valid2[i]]
        
        if len(t_i) < 5:
            continue
        
        A_max = np.max(A_i)
        t_max = t_i[np.argmax(A_i)]
        
        C_init = A_max * 2.0
        lambda1_init = 1.0 / (t_max + 100.0)
        lambda2_init = 1.0 / (t_max * 0.5 + 10.0)
        
        if lambda_phys is not None:
            lambda1_init = max(lambda1_init, float(lambda_phys))
            lambda2_init = max(lambda2_init, float(lambda_phys))
        
        try:
            popt, _ = _fit_biexp_single(t_i, A_i, C_init, lambda1_init, lambda2_init)
            lnC, lam1, lam2 = popt
            
            if lambda_phys is not None:
                lam1 = max(lam1, float(lambda_phys))
                lam2 = max(lam2, float(lambda_phys))
            
            if lam1 > 0 and lam2 > 0 and lam1 < lam2:
                params[i] = [lnC, lam1, lam2]
                
                Ahat[i, :] = np.exp(lnC) * (np.exp(-lam1 * t) - np.exp(-lam2 * t))
                tpeak[i] = np.log(lam2 / lam1) / (lam2 - lam1)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2


def _fit_biexp_single(
    t: np.ndarray, A: np.ndarray, C_init: float, lam1_init: float, lam2_init: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit bi-exponential model for a single voxel using Levenberg-Marquardt.
    
    Args:
        t: Time points
        A: Activity values
        C_init: Initial guess for C
        lam1_init: Initial guess for lambda1
        lam2_init: Initial guess for lambda2
        
    Returns:
        popt: [lnC, lambda1, lambda2] optimized parameters
        pcov: Covariance matrix
    """
    from scipy.optimize import curve_fit
    
    def biexp_model(t_vec, lnC, lam1, lam2):
        C = np.exp(lnC)
        return C * (np.exp(-lam1 * t_vec) - np.exp(-lam2 * t_vec))
    
    p0 = [np.log(C_init), lam1_init, lam2_init]
    bounds = ([np.log(1e-6), 1e-8, 1e-8], [np.log(1e12), 1.0, 10.0])
    
    popt, pcov = curve_fit(biexp_model, t, A, p0=p0, bounds=bounds, maxfev=5000)
    return popt, pcov


def tia_from_biexp_params(params: np.ndarray) -> np.ndarray:
    """
    Compute TIA from bi-exponential parameters.
    
    TIA = C * (1/lambda1 - 1/lambda2)
    
    Args:
        params: (N, 3) [lnC, lambda1, lambda2]
        
    Returns:
        tia: (N,) time-integrated activity values
    """
    lnC = params[:, 0].astype(np.float64)
    lam1 = params[:, 1].astype(np.float64)
    lam2 = params[:, 2].astype(np.float64)
    
    C = np.exp(lnC)
    tia = np.full((params.shape[0],), np.nan, dtype=np.float64)
    
    ok = np.isfinite(C) & np.isfinite(lam1) & np.isfinite(lam2) & (lam1 > 0) & (lam2 > 0) & (lam1 < lam2)
    if np.any(ok):
        tia[ok] = C[ok] * (1.0 / lam1[ok] - 1.0 / lam2[ok])
    
    return tia.astype(np.float32)


def fit_biexp_linearized(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit bi-exponential model using linearized approach for initial parameter estimation,
    followed by nonlinear refinement.
    
    Linearization: For t >> t_peak, A(t) ≈ C * exp(-lambda1 * t)
    For t << t_peak, A(t) ≈ C * (1 - exp(-(lambda2 - lambda1) * t))
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 3) [lnC, lambda1, lambda2] parameters
        tpeak: (N,) peak time from fitted parameters
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 3), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    
    for i in range(N):
        t_valid = t[valid2[i]]
        A_valid = A[i, valid2[i]]
        
        if len(t_valid) < 5:
            continue
        
        idx_max = np.argmax(A_valid)
        t_max = t_valid[idx_max]
        A_max = A_valid[idx_max]
        
        tail_mask = t_valid > t_max
        if np.sum(tail_mask) >= 3:
            t_tail = t_valid[tail_mask]
            A_tail = A_valid[tail_mask]
            
            log_A_tail = np.log(A_tail)
            poly = np.polyfit(t_tail, log_A_tail, 1)
            lam1_init = -poly[0]
            C_init = np.exp(poly[1]) / (1.0 - np.exp(-(lam1_init - lam1_init * 0.5) * t_max))
        else:
            lam1_init = 1.0 / (t_max + 100.0)
            C_init = A_max * 2.0
        
        lam2_init = lam1_init * 2.0
        
        if lambda_phys is not None:
            lam1_init = max(lam1_init, float(lambda_phys))
            lam2_init = max(lam2_init, float(lambda_phys))
        
        try:
            popt, _ = _fit_biexp_single(t_valid, A_valid, C_init, lam1_init, lam2_init)
            lnC, lam1, lam2 = popt
            
            if lambda_phys is not None:
                lam1 = max(lam1, float(lambda_phys))
                lam2 = max(lam2, float(lambda_phys))
            
            if lam1 > 0 and lam2 > 0 and lam1 < lam2:
                params[i] = [lnC, lam1, lam2]
                
                Ahat[i, :] = np.exp(lnC) * (np.exp(-lam1 * t) - np.exp(-lam2 * t))
                tpeak[i] = np.log(lam2 / lam1) / (lam2 - lam1)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2
