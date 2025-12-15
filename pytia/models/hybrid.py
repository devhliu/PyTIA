from __future__ import annotations

import numpy as np

from ..metrics import r2_score


def tia_trapz_plus_phys_tail(
    A: np.ndarray,
    times: np.ndarray,
    valid: np.ndarray,
    lambda_phys: float,
    include_t0: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Observed trapezoid over valid points + tail = A_last / lambda_phys.
    If include_t0: include (0,0) in integration.

    Returns:
      tia: (N,)
      Ahat: (N,T) piecewise linear prediction at sampled points (=A on valid, NaN on invalid)
      r2: (N,)
    """
    N, T = A.shape
    t = times.astype(np.float64)

    # For trapz we need numeric arrays; set invalid to nan and use nan-aware segment integration
    Av = np.where(valid, A, np.nan).astype(np.float64)

    # Optionally prepend t0=0, A0=0
    if include_t0:
        t2 = np.concatenate([[0.0], t])
        Av2 = np.concatenate([np.zeros((N, 1), dtype=np.float64), Av], axis=1)
    else:
        t2 = t
        Av2 = Av

    # trapezoid with NaNs: integrate only segments where both endpoints finite
    a0 = Av2[:, :-1]
    a1 = Av2[:, 1:]
    dt = np.diff(t2)[None, :]
    seg_ok = np.isfinite(a0) & np.isfinite(a1)
    area_obs = np.nansum(np.where(seg_ok, 0.5 * (a0 + a1) * dt, 0.0), axis=1)

    # Tail uses last valid point
    # get last valid index per voxel in original T
    last_valid = T - 1 - np.argmax(valid[:, ::-1], axis=1)
    Alast = np.take_along_axis(A, last_valid[:, None], axis=1)[:, 0].astype(np.float64)
    ok_last = np.take_along_axis(valid, last_valid[:, None], axis=1)[:, 0]
    tail = np.full((N,), np.nan, dtype=np.float64)
    tail[ok_last] = Alast[ok_last] / float(lambda_phys)

    tia = area_obs + tail

    # Ahat at sampled points: use Av (measured) as proxy for evaluation
    Ahat = Av.astype(np.float64)
    r2 = r2_score(A.astype(np.float64), np.where(np.isfinite(Ahat), Ahat, np.nan), valid)
    return tia.astype(np.float32), Ahat.astype(np.float32), r2.astype(np.float32)