import numpy as np

from pytia.models.gamma_linear import fit_gamma_linear_wls


def test_gamma_linear_peak_estimation():
    # Construct gamma curve with known alpha,beta,K
    K = 5.0
    alpha = 2.0
    beta = 0.4
    t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    A = (K * (t**alpha) * np.exp(-beta * t))[None, :].astype(np.float32)
    valid = np.ones_like(A, dtype=bool)

    params, tpeak, _, _ = fit_gamma_linear_wls(A, t, valid, lambda_phys=None)
    # Expected peak
    expected = alpha / beta
    assert np.isfinite(tpeak[0])
    assert abs(tpeak[0] - expected) < 0.5  # loose tolerance (linearized fit)