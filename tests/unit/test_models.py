import warnings

import numpy as np
from scipy.special import gamma as gamma_func

from pytia.models.gamma_linear import fit_gamma_linear_wls, tia_from_gamma_params
from pytia.models.hybrid import tia_trapz_plus_phys_tail
from pytia.models.monoexp import fit_monoexp_tail, tia_monoexp_with_triangle_uptake


def test_fit_gamma_linear_wls_recovers_clean_curve() -> None:
    times = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    ln_k, alpha, beta = 2.0, 1.5, 0.25
    A = (np.exp(ln_k) * (times**alpha) * np.exp(-beta * times))[None, :].astype(np.float32)
    valid = np.ones_like(A, dtype=bool)

    params, tpeak, Ahat, r2 = fit_gamma_linear_wls(A, times, valid, lambda_phys=None)

    assert params.shape == (1, 3)
    assert Ahat.shape == A.shape
    assert np.isfinite(tpeak[0])
    assert r2[0] > 0.999

    tia = tia_from_gamma_params(params)[0]
    expected = np.exp(ln_k) * gamma_func(alpha + 1.0) / (beta ** (alpha + 1.0))
    np.testing.assert_allclose(tia, expected, rtol=1e-2)


def test_fit_monoexp_tail_and_triangle_tia() -> None:
    times = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
    lam_true = 0.3
    a0 = 10.0
    A = (a0 * np.exp(-lam_true * times))[None, :].astype(np.float32)
    valid = np.ones_like(A, dtype=bool)
    peak_index = np.array([0], dtype=np.int64)

    lam, Ahat, r2 = fit_monoexp_tail(A, times, valid, lambda_phys=None, peak_index=peak_index)

    assert Ahat.shape == A.shape
    assert r2[0] > 0.999
    np.testing.assert_allclose(lam[0], lam_true, rtol=5e-2)

    tia = tia_monoexp_with_triangle_uptake(A, times, valid, lam, peak_index)
    np.testing.assert_allclose(tia[0], a0 / lam_true, rtol=5e-2)


def test_hybrid_trapz_plus_phys_tail_matches_expected_formula() -> None:
    times = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    A = np.array([[6.0, 4.0, 2.0]], dtype=np.float32)
    valid = np.ones_like(A, dtype=bool)
    lambda_phys = 0.2

    tia, Ahat, r2 = tia_trapz_plus_phys_tail(
        A,
        times,
        valid,
        lambda_phys=lambda_phys,
        include_t0=False,
    )

    expected_area = np.trapezoid(A[0], times)
    expected = expected_area + (A[0, -1] / lambda_phys)

    assert Ahat.shape == A.shape
    assert np.isfinite(r2[0])
    np.testing.assert_allclose(tia[0], expected, rtol=1e-6)


def test_fit_gamma_linear_wls_avoids_log_warning_on_nonpositive_invalid_values() -> None:
    times = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    A = np.array([[0.0, 2.0, 3.0]], dtype=np.float32)
    valid = np.array([[False, True, True]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        fit_gamma_linear_wls(A, times, valid, lambda_phys=None)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
