from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def tia_trapz_plus_phys_tail(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float,
    include_t0: bool = True,
    tail_mode: str = "phys",
    min_tail_points: int = 2,
    fit_tail_slope: bool = False,
    lambda_phys_constraint: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Observed trapezoid over valid points + tail extrapolation.
    
    Tail modes:
    - "phys": Physical decay tail = A_last / lambda_phys
    - "fitted": Fit exponential tail from last N points, then extrapolate
    - "hybrid": Use fitted tail if available, otherwise fall back to phys
    
    If include_t0: include (0,0) in integration.

    Returns:
      tia: (N,)
      Ahat: (N,T) piecewise linear prediction at sampled points (=A on valid, NaN on invalid)
      r2: (N,)
      sigma_tail: (N,) uncertainty in tail contribution (NaN if not computed)
    """
    N, T = A.shape
    t = times.astype(np.float64)

    Av = np.where(valid, A, np.nan).astype(np.float64)

    if include_t0:
        t2 = np.concatenate([[0.0], t])
        Av2 = np.concatenate([np.zeros((N, 1), dtype=np.float64), Av], axis=1)
    else:
        t2 = t
        Av2 = Av

    a0 = Av2[:, :-1]
    a1 = Av2[:, 1:]
    dt = np.diff(t2)[None, :]
    seg_ok = np.isfinite(a0) & np.isfinite(a1)
    area_obs = np.nansum(np.where(seg_ok, 0.5 * (a0 + a1) * dt, 0.0), axis=1)

    last_valid = T - 1 - np.argmax(valid[:, ::-1], axis=1)
    Alast = np.take_along_axis(A, last_valid[:, None], axis=1)[:, 0].astype(np.float64)
    tlast = np.take_along_axis(np.broadcast_to(t[None, :], (N, T)), last_valid[:, None], axis=1)[:, 0].astype(np.float64)
    ok_last = np.take_along_axis(valid, last_valid[:, None], axis=1)[:, 0]

    tail = np.full((N,), np.nan, dtype=np.float64)
    sigma_tail = np.full((N,), np.nan, dtype=np.float64)

    if tail_mode == "phys":
        if lambda_phys is not None:
            tail[ok_last] = Alast[ok_last] / float(lambda_phys)
    elif not fit_tail_slope:
        if lambda_phys is not None:
            tail[ok_last] = Alast[ok_last] / float(lambda_phys)
    elif tail_mode in ["fitted", "hybrid"]:
        lam_fit = np.full((N,), np.nan, dtype=np.float64)
        lam_fit_std = np.full((N,), np.nan, dtype=np.float64)

        for i in np.where(ok_last)[0]:
            tail_start_idx = max(0, last_valid[i] - min_tail_points + 1)
            tail_mask = np.zeros(T, dtype=bool)
            tail_mask[tail_start_idx:last_valid[i] + 1] = True
            tail_mask = tail_mask & valid[i]

            if np.sum(tail_mask) >= 2:
                t_tail = t[tail_mask]
                A_tail = A[i, tail_mask]

                log_A_tail = np.log(A_tail)
                poly, cov = np.polyfit(t_tail, log_A_tail, 1, cov=True)
                lam_fit[i] = -poly[0]
                lam_fit_std[i] = np.sqrt(cov[0, 0]) if cov is not None else np.nan

        if lambda_phys_constraint and lambda_phys is not None:
            lam_fit = np.where(np.isfinite(lam_fit), np.maximum(lam_fit, float(lambda_phys)), lam_fit)

        if tail_mode == "fitted":
            ok_fit = ok_last & np.isfinite(lam_fit) & (lam_fit > 0)
            tail[ok_fit] = Alast[ok_fit] / lam_fit[ok_fit]
            if np.any(ok_fit):
                rel_err = lam_fit_std[ok_fit] / lam_fit[ok_fit]
                sigma_tail[ok_fit] = tail[ok_fit] * rel_err
        elif tail_mode == "hybrid":
            ok_fit = ok_last & np.isfinite(lam_fit) & (lam_fit > 0)
            tail[ok_fit] = Alast[ok_fit] / lam_fit[ok_fit]
            if np.any(ok_fit):
                rel_err = lam_fit_std[ok_fit] / lam_fit[ok_fit]
                sigma_tail[ok_fit] = tail[ok_fit] * rel_err
            
            if lambda_phys is not None:
                ok_phys = ok_last & (~ok_fit)
                tail[ok_phys] = Alast[ok_phys] / float(lambda_phys)

    tia = area_obs + tail

    Ahat = Av.astype(np.float64)
    r2 = r2_score(A.astype(np.float64), np.where(np.isfinite(Ahat), Ahat, np.nan), valid)
    return tia.astype(np.float32), Ahat.astype(np.float32), r2.astype(np.float32), sigma_tail.astype(np.float32)