from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def fit_three_phase_exp(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit 3-phase exponential model:
    
    A(t) = C * (1 - exp(-lambda_uptake * t)) * 
           (f_fast * exp(-lambda_fast * t) + (1 - f_fast) * exp(-lambda_slow * t))
    
    This model captures:
    1. Uptake phase: 1 - exp(-lambda_uptake * t)
    2. Fast washout: exp(-lambda_fast * t)
    3. Slow washout: exp(-lambda_slow * t)
    
    Requires >= 6-7 time points for robust fit.
    Suitable for tumors with complex kinetics.
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 5) [lnC, lambda_uptake, lambda_fast, lambda_slow, f_fast] parameters
        tpeak: (N,) peak time from fitted parameters
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 5), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    n_valid = np.sum(valid2, axis=1)
    
    for i in range(N):
        if n_valid[i] < 6:
            continue
        
        t_i = t[valid2[i]]
        A_i = A[i, valid2[i]]
        
        if len(t_i) < 6:
            continue
        
        A_max = np.max(A_i)
        t_max = t_i[np.argmax(A_i)]
        
        C_init = A_max * 1.5
        lam_uptake_init = 1.0 / (t_max * 0.5 + 10.0)
        lam_fast_init = 1.0 / (t_max * 0.3 + 5.0)
        lam_slow_init = 1.0 / (t_max * 2.0 + 100.0)
        f_fast_init = 0.5
        
        if lambda_phys is not None:
            lam_fast_init = max(lam_fast_init, float(lambda_phys))
            lam_slow_init = max(lam_slow_init, float(lambda_phys))
        
        try:
            popt, _ = _fit_three_phase_single(
                t_i, A_i, C_init, lam_uptake_init, lam_fast_init, lam_slow_init, f_fast_init
            )
            lnC, lam_uptake, lam_fast, lam_slow, f_fast = popt
            
            if lambda_phys is not None:
                lam_fast = max(lam_fast, float(lambda_phys))
                lam_slow = max(lam_slow, float(lambda_phys))
            
            if (lam_uptake > 0 and lam_fast > 0 and lam_slow > 0 and 
                lam_fast > lam_slow and 0 <= f_fast <= 1):
                params[i] = [lnC, lam_uptake, lam_fast, lam_slow, f_fast]
                
                C = np.exp(lnC)
                uptake = 1.0 - np.exp(-lam_uptake * t)
                washout = f_fast * np.exp(-lam_fast * t) + (1.0 - f_fast) * np.exp(-lam_slow * t)
                Ahat[i, :] = C * uptake * washout
                
                tpeak[i] = _find_peak_three_phase(C, lam_uptake, lam_fast, lam_slow, f_fast)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2


def _fit_three_phase_single(
    t: np.ndarray, A: np.ndarray, C_init: float, lam_uptake_init: float,
    lam_fast_init: float, lam_slow_init: float, f_fast_init: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit 3-phase exponential model for a single voxel using Levenberg-Marquardt.
    
    Args:
        t: Time points
        A: Activity values
        C_init: Initial guess for C
        lam_uptake_init: Initial guess for lambda_uptake
        lam_fast_init: Initial guess for lambda_fast
        lam_slow_init: Initial guess for lambda_slow
        f_fast_init: Initial guess for f_fast
        
    Returns:
        popt: [lnC, lambda_uptake, lambda_fast, lambda_slow, f_fast] optimized parameters
        pcov: Covariance matrix
    """
    from scipy.optimize import curve_fit
    
    def three_phase_model(t_vec, lnC, lam_uptake, lam_fast, lam_slow, f_fast):
        C = np.exp(lnC)
        uptake = 1.0 - np.exp(-lam_uptake * t_vec)
        washout = f_fast * np.exp(-lam_fast * t_vec) + (1.0 - f_fast) * np.exp(-lam_slow * t_vec)
        return C * uptake * washout
    
    p0 = [np.log(C_init), lam_uptake_init, lam_fast_init, lam_slow_init, f_fast_init]
    bounds = ([np.log(1e-6), 1e-8, 1e-8, 1e-8, 0.0], [np.log(1e12), 1.0, 10.0, 10.0, 1.0])
    
    popt, pcov = curve_fit(three_phase_model, t, A, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov


def _find_peak_three_phase(C: float, lam_uptake: float, lam_fast: float, lam_slow: float, f_fast: float) -> float:
    """
    Find peak time for 3-phase model by numerical optimization.
    
    Args:
        C: Amplitude parameter
        lam_uptake: Uptake rate
        lam_fast: Fast washout rate
        lam_slow: Slow washout rate
        f_fast: Fraction of fast washout
        
    Returns:
        t_peak: Peak time in seconds
    """
    from scipy.optimize import minimize_scalar
    
    def three_phase_model(t_vec):
        uptake = 1.0 - np.exp(-lam_uptake * t_vec)
        washout = f_fast * np.exp(-lam_fast * t_vec) + (1.0 - f_fast) * np.exp(-lam_slow * t_vec)
        return -C * uptake * washout
    
    result = minimize_scalar(three_phase_model, bounds=(0, 100000), method='bounded')
    return result.x


def tia_from_three_phase_params(params: np.ndarray) -> np.ndarray:
    """
    Compute TIA from 3-phase exponential parameters.
    
    TIA = C * [1/lambda_uptake - f_fast/(lambda_uptake+lambda_fast) - (1-f_fast)/(lambda_uptake+lambda_slow)]
    
    Args:
        params: (N, 5) [lnC, lambda_uptake, lambda_fast, lambda_slow, f_fast]
        
    Returns:
        tia: (N,) time-integrated activity values
    """
    lnC = params[:, 0].astype(np.float64)
    lam_uptake = params[:, 1].astype(np.float64)
    lam_fast = params[:, 2].astype(np.float64)
    lam_slow = params[:, 3].astype(np.float64)
    f_fast = params[:, 4].astype(np.float64)
    
    C = np.exp(lnC)
    tia = np.full((params.shape[0],), np.nan, dtype=np.float64)
    
    ok = (np.isfinite(C) & np.isfinite(lam_uptake) & np.isfinite(lam_fast) & 
          np.isfinite(lam_slow) & np.isfinite(f_fast) & 
          lam_uptake > 0 & lam_fast > 0 & lam_slow > 0 & 
          0 <= f_fast <= 1)
    
    if np.any(ok):
        term1 = 1.0 / lam_uptake[ok]
        term2 = f_fast[ok] / (lam_uptake[ok] + lam_fast[ok])
        term3 = (1.0 - f_fast[ok]) / (lam_uptake[ok] + lam_slow[ok])
        tia[ok] = C[ok] * (term1 - term2 - term3)
    
    return tia.astype(np.float32)


def fit_three_phase_exp_linearized(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float | None,
    peak_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit 3-phase exponential model using linearized approach for initial parameter estimation,
    followed by nonlinear refinement.
    
    Strategy:
    1. Estimate uptake phase from early time points
    2. Estimate washout phases from late time points using bi-exponential decomposition
    3. Refine all parameters with nonlinear optimization
    
    Args:
        A: (N, T) activity values
        times: (T,) time points in seconds
        valid: (N, T) boolean mask for valid measurements
        lambda_phys: Physical decay rate (optional constraint)
        peak_index: (N,) index of peak activity per voxel
        
    Returns:
        params: (N, 5) [lnC, lambda_uptake, lambda_fast, lambda_slow, f_fast] parameters
        tpeak: (N,) peak time from fitted parameters
        Ahat: (N, T) predicted activity
        r2: (N,) R-squared goodness of fit
    """
    N, T = A.shape
    t = times.astype(np.float64)
    
    params = np.full((N, 5), np.nan, dtype=np.float64)
    Ahat = np.full((N, T), np.nan, dtype=np.float32)
    tpeak = np.full((N,), np.nan, dtype=np.float32)
    
    valid2 = valid & (A > 0)
    
    for i in range(N):
        t_valid = t[valid2[i]]
        A_valid = A[i, valid2[i]]
        
        if len(t_valid) < 6:
            continue
        
        idx_max = np.argmax(A_valid)
        t_max = t_valid[idx_max]
        A_max = A_valid[idx_max]
        
        early_mask = t_valid <= t_max
        if np.sum(early_mask) >= 2:
            t_early = t_valid[early_mask]
            A_early = A_valid[early_mask]
            
            lam_uptake_init = -np.polyfit(t_early, np.log(1.0 - A_early / A_max), 1)[0]
            lam_uptake_init = max(lam_uptake_init, 1e-6)
        else:
            lam_uptake_init = 1.0 / (t_max * 0.5 + 10.0)
        
        late_mask = t_valid > t_max
        if np.sum(late_mask) >= 4:
            t_late = t_valid[late_mask]
            A_late = A_valid[late_mask]
            
            from scipy.optimize import curve_fit
            
            def biexp_tail(t_vec, lnA_fast, lam_fast, lnA_slow, lam_slow):
                A_fast = np.exp(lnA_fast)
                A_slow = np.exp(lnA_slow)
                return A_fast * np.exp(-lam_fast * t_vec) + A_slow * np.exp(-lam_slow * t_vec)
            
            p0 = [np.log(A_max * 0.5), 1.0 / (t_max * 0.3 + 5.0), 
                  np.log(A_max * 0.5), 1.0 / (t_max * 2.0 + 100.0)]
            bounds = ([np.log(1e-6), 1e-8, np.log(1e-6), 1e-8], 
                      [np.log(1e12), 10.0, np.log(1e12), 10.0])
            
            try:
                popt, _ = curve_fit(biexp_tail, t_late, A_late, p0=p0, bounds=bounds, maxfev=5000)
                lnA_fast, lam_fast_init, lnA_slow, lam_slow_init = popt
                
                if lambda_phys is not None:
                    lam_fast_init = max(lam_fast_init, float(lambda_phys))
                    lam_slow_init = max(lam_slow_init, float(lambda_phys))
                
                A_fast = np.exp(lnA_fast)
                A_slow = np.exp(lnA_slow)
                f_fast_init = A_fast / (A_fast + A_slow)
                C_init = (A_fast + A_slow) / (1.0 - np.exp(-lam_uptake_init * t_max))
            except RuntimeError:
                lam_fast_init = 1.0 / (t_max * 0.3 + 5.0)
                lam_slow_init = 1.0 / (t_max * 2.0 + 100.0)
                f_fast_init = 0.5
                C_init = A_max * 1.5
        else:
            lam_fast_init = 1.0 / (t_max * 0.3 + 5.0)
            lam_slow_init = 1.0 / (t_max * 2.0 + 100.0)
            f_fast_init = 0.5
            C_init = A_max * 1.5
        
        if lambda_phys is not None:
            lam_fast_init = max(lam_fast_init, float(lambda_phys))
            lam_slow_init = max(lam_slow_init, float(lambda_phys))
        
        try:
            popt, _ = _fit_three_phase_single(
                t_valid, A_valid, C_init, lam_uptake_init, lam_fast_init, lam_slow_init, f_fast_init
            )
            lnC, lam_uptake, lam_fast, lam_slow, f_fast = popt
            
            if lambda_phys is not None:
                lam_fast = max(lam_fast, float(lambda_phys))
                lam_slow = max(lam_slow, float(lambda_phys))
            
            if (lam_uptake > 0 and lam_fast > 0 and lam_slow > 0 and 
                lam_fast > lam_slow and 0 <= f_fast <= 1):
                params[i] = [lnC, lam_uptake, lam_fast, lam_slow, f_fast]
                
                C = np.exp(lnC)
                uptake = 1.0 - np.exp(-lam_uptake * t)
                washout = f_fast * np.exp(-lam_fast * t) + (1.0 - f_fast) * np.exp(-lam_slow * t)
                Ahat[i, :] = C * uptake * washout
                
                tpeak[i] = _find_peak_three_phase(C, lam_uptake, lam_fast, lam_slow, f_fast)
        except (RuntimeError, np.linalg.LinAlgError):
            continue
    
    r2 = r2_score(A.astype(np.float64), Ahat.astype(np.float64), valid)
    return params, tpeak, Ahat, r2
