import numpy as np
import pytest

from pytia.models.hybrid import tia_trapz_plus_phys_tail


def test_hybrid_trapz_plus_tail_simple():
    # A(t) = [0, 10] at t=[1,2], include_t0 adds (0,0)
    A = np.array([[0.0, 10.0]], dtype=np.float32)
    t = np.array([1.0, 2.0], dtype=np.float64)
    valid = np.array([[True, True]])
    lambda_phys = 0.5

    tia, _, _, _ = tia_trapz_plus_phys_tail(A, t, valid, lambda_phys=lambda_phys, include_t0=True)
    # trapezoid segments: (0->1):0 ; (1->2): 0.5*(0+10)*1 = 5
    # tail: A_last/lambda = 10/0.5 = 20 => total 25
    assert np.isclose(tia[0], 25.0)


def test_hybrid_tail_mode_phys():
    """Test physical decay tail mode."""
    A = np.array([[0.0, 10.0, 5.0]], dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    valid = np.array([[True, True, True]])
    lambda_phys = 0.5

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=True, tail_mode="phys"
    )

    # Should use physical decay tail
    assert tia[0] > 0
    assert np.isfinite(tia[0])
    # No uncertainty for pure physical tail
    assert np.isnan(sigma_tail[0])


def test_hybrid_tail_mode_fitted():
    """Test fitted tail mode."""
    # Create exponential decay data
    t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    A0 = 100.0
    lam = 0.5
    A = np.array([[A0 * np.exp(-lam * ti) for ti in t]], dtype=np.float32)
    valid = np.array([[True, True, True, True]])

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=0.1, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=2
    )

    # Should use fitted tail
    assert tia[0] > 0
    assert np.isfinite(tia[0])
    # Uncertainty should be computed for fitted tail
    assert np.isfinite(sigma_tail[0])


def test_hybrid_tail_mode_hybrid():
    """Test hybrid tail mode (fitted with physical fallback)."""
    # Create data that might not fit well
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    A = np.array([[10.0, 8.0, 6.0]], dtype=np.float32)
    valid = np.array([[True, True, True]])
    lambda_phys = 0.5

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="hybrid",
        fit_tail_slope=True, min_tail_points=2
    )

    # Should use either fitted or physical tail
    assert tia[0] > 0
    assert np.isfinite(tia[0])


def test_hybrid_tail_mode_none():
    """Test no tail mode."""
    A = np.array([[0.0, 10.0, 5.0]], dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    valid = np.array([[True, True, True]])
    lambda_phys = 0.5

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=True, tail_mode="none"
    )

    # Should only include trapezoidal part (no tail)
    assert tia[0] > 0
    assert np.isfinite(tia[0])
    assert np.isnan(sigma_tail[0])


def test_hybrid_lambda_phys_constraint():
    """Test lambda_phys constraint enforcement."""
    # Create exponential decay with slow rate
    t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    A0 = 100.0
    lam_slow = 0.1
    A = np.array([[A0 * np.exp(-lam_slow * ti) for ti in t]], dtype=np.float32)
    valid = np.array([[True, True, True, True]])
    lambda_phys = 0.5  # Faster physical decay

    tia_constrained, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=2, lambda_phys_constraint=True
    )

    tia_unconstrained, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=2, lambda_phys_constraint=False
    )

    # Constrained version should have smaller TIA (faster decay)
    assert tia_constrained[0] < tia_unconstrained[0]


def test_hybrid_no_lambda_phys_fitted_mode():
    """Test fitted mode without lambda_phys."""
    t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    A0 = 100.0
    lam = 0.5
    A = np.array([[A0 * np.exp(-lam * ti) for ti in t]], dtype=np.float32)
    valid = np.array([[True, True, True, True]])

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=None, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=2
    )

    # Should work with fitted tail even without lambda_phys
    assert tia[0] > 0
    assert np.isfinite(tia[0])
    assert np.isfinite(sigma_tail[0])


def test_hybrid_no_lambda_phys_phys_mode():
    """Test physical mode without lambda_phys should return NaN tail."""
    A = np.array([[0.0, 10.0, 5.0]], dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    valid = np.array([[True, True, True]])

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=None, include_t0=True, tail_mode="phys"
    )

    # Should only include trapezoidal part (tail is NaN)
    assert np.isfinite(tia[0])


def test_hybrid_min_tail_points():
    """Test minimum tail points requirement."""
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    A0 = 100.0
    lam = 0.5
    A = np.array([[A0 * np.exp(-lam * ti) for ti in t]], dtype=np.float32)
    valid = np.array([[True, True, True, True, True]])

    # With min_tail_points=3, should use last 3 points
    tia_3pts, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=0.1, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=3
    )

    # With min_tail_points=2, should use last 2 points
    tia_2pts, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=0.1, include_t0=False, tail_mode="fitted",
        fit_tail_slope=True, min_tail_points=2
    )

    # Both should be finite but may differ slightly
    assert np.isfinite(tia_3pts[0])
    assert np.isfinite(tia_2pts[0])


def test_hybrid_multiple_voxels():
    """Test with multiple voxels."""
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    A = np.array([
        [10.0, 8.0, 6.0],
        [20.0, 15.0, 10.0],
        [5.0, 4.0, 3.0],
    ], dtype=np.float32)
    valid = np.array([
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ])
    lambda_phys = 0.5

    tia, _, _, sigma_tail = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="phys"
    )

    # Should return TIA for all voxels
    assert len(tia) == 3
    assert np.all(np.isfinite(tia))
    # Higher activity should give higher TIA
    assert tia[1] > tia[0]


def test_hybrid_invalid_points():
    """Test handling of invalid points."""
    t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    A = np.array([[10.0, np.nan, 6.0, 5.0]], dtype=np.float32)
    valid = np.array([[True, False, True, True]])
    lambda_phys = 0.5

    tia, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="phys"
    )

    # Should handle invalid points gracefully
    assert np.isfinite(tia[0])


def test_hybrid_include_t0():
    """Test include_t0 parameter."""
    A = np.array([[10.0, 8.0, 6.0]], dtype=np.float32)
    t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    valid = np.array([[True, True, True]])
    lambda_phys = 0.5

    tia_with_t0, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=True, tail_mode="phys"
    )

    tia_without_t0, _, _, _ = tia_trapz_plus_phys_tail(
        A, t, valid, lambda_phys=lambda_phys, include_t0=False, tail_mode="phys"
    )

    # With t0 should be larger (includes segment from 0 to first time point)
    assert tia_with_t0[0] > tia_without_t0[0]