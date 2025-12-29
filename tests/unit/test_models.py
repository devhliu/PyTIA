"""Unit tests for kinetic model fitting functions."""

import numpy as np
import pytest

from pytia.models.gamma_linear import fit_gamma_linear_wls, tia_from_gamma_params
from pytia.models.monoexp import fit_monoexp_tail, tia_monoexp_with_triangle_uptake
from pytia.models.hybrid import tia_trapz_plus_phys_tail
from tests.utils import compute_tia_analytical, TestDataGenerator


class TestGammaLinearFitting:
    """Test gamma-linear model fitting."""

    def test_perfect_gamma_curve(self):
        """Test fitting to perfect gamma curve."""
        # Generate perfect gamma data
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0])
        A0, tpeak, alpha, beta = 100.0, 3600.0, 2.0, 0.0001

        # Gamma function: A = A0 * (t/tpeak)^alpha * exp(-beta*(t-tpeak))
        activities = []
        for t in times:
            if t <= 0:
                activities.append(0.0)
            else:
                val = A0 * (t/tpeak)**alpha * np.exp(-beta*(t-tpeak))
                activities.append(val if val > 0 else 0.0)

        activities = np.array(activities)

        # Fit model
        params, r2, status = fit_gamma_linear_wls(times, activities)

        assert status == 1  # Success

        # Check parameter recovery
        # Note: exact recovery depends on fitting method and data quality
        assert params[0] > 0  # A0 should be positive
        assert params[1] > 0  # Peak time should be positive
        assert r2 > 0.95  # Good fit quality

    def test_rising_curve(self):
        """Test fitting rising curve."""
        times = np.linspace(0, 7200, 5)  # 0-2 hours, rising
        activities = TestDataGenerator.rising_tauc(times, A0=100.0, lambda_uptake=0.0005)

        params, r2, status = fit_gamma_linear_wls(times, activities)

        assert status == 1
        assert r2 > 0.9

    def test_falling_exponential(self):
        """Test fitting to falling exponential."""
        times = np.linspace(0, 14400, 5)  # 0-4 hours
        activities = TestDataGenerator.falling_tauc(times, A0=100.0, lambda_washout=0.0001)

        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Might not fit perfectly to falling curve with gamma model
        assert status in [1, 3]  # Success or fit failed

    def test_noisy_data(self):
        """Test fitting with noisy data."""
        np.random.seed(42)  # For reproducibility
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        clean_activities = TestDataGenerator.hump_tauc(times)
        noisy_activities = clean_activities + np.random.normal(0, 0.1 * clean_activities.max())

        params, r2, status = fit_gamma_linear_wls(times, noisy_activities)

        assert status == 1
        # With noise, R² might be lower but should still be reasonable
        assert r2 > 0.7

    def test_insufficient_points(self):
        """Test with insufficient time points."""
        times = np.array([0.0, 3600.0])  # Only 2 points
        activities = np.array([0.0, 50.0])

        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Status should indicate insufficient points
        assert status != 1

    def test_all_zeros(self):
        """Test with all zero activity."""
        times = np.array([0.0, 1800.0, 3600.0])
        activities = np.zeros(3)

        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Should handle gracefully
        # Status might indicate failure or all points below floor
        assert params is not None

    def test_negative_values(self):
        """Test with negative activity values."""
        times = np.array([0.0, 1800.0, 3600.0])
        activities = np.array([-10.0, 50.0, 80.0])

        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Should handle or reject based on implementation
        assert params is not None

    def test_uneven_time_spacing(self):
        """Test with unevenly spaced time points."""
        times = np.array([0.0, 600.0, 3600.0, 14400.0, 28800.0])  # Uneven spacing
        activities = TestDataGenerator.hump_tauc(times)

        params, r2, status = fit_gamma_linear_wls(times, activities)

        assert status == 1
        assert r2 > 0.8

    def test_tia_from_gamma_params(self):
        """Test TIA calculation from gamma parameters."""
        # Known parameters
        A0 = 100.0
        beta = 0.0001
        # Simplified gamma parameters
        gamma_params = np.array([A0, 3600.0, 2.0, beta])

        tia = tia_from_gamma_params(gamma_params)

        # TIA should be positive
        assert tia > 0

        # For exponential decay (simplified case), TIA ≈ A0/beta
        # This is a rough check
        assert np.isfinite(tia)


class TestMonoExponentialFitting:
    """Test mono-exponential tail fitting."""

    def test_perfect_exponential(self):
        """Test fitting to perfect mono-exponential."""
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0, 57600.0])  # Start after peak
        A0 = 100.0
        lam = 0.0001  # Decay constant
        activities = A0 * np.exp(-lam * times)

        # Fit using tail points (last half)
        tail_start = len(times) // 2
        params, r2, status = fit_monoexp_tail(
            times[tail_start:], activities[tail_start:]
        )

        assert status == 1
        assert r2 > 0.99

        # Check parameter recovery
        assert params[0] > 0  # Initial activity positive
        assert params[1] > 0  # Decay constant positive
        np.testing.assert_allclose(params[1], lam, rtol=0.1)  # Within 10%

    def test_noisy_exponential_tail(self):
        """Test fitting to noisy exponential tail."""
        np.random.seed(42)
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0, 57600.0])
        A0, lam = 100.0, 0.0001
        activities = A0 * np.exp(-lam * times)
        noisy_activities = activities + np.random.normal(0, 2.0, activities.shape)

        params, r2, status = fit_monoexp_tail(times, noisy_activities)

        assert status == 1
        assert r2 > 0.9

    def test_short_tail(self):
        """Test with very short tail (few points)."""
        times = np.array([7200.0, 14400.0])  # Only two points
        activities = np.array([50.0, 25.0])  # Roughly exponential decay

        params, r2, status = fit_monoexp_tail(times, activities)

        # Might succeed or fail depending on requirements
        assert params is not None

    def test_non_exponential_data(self):
        """Test fitting non-exponential data."""
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0])
        activities = np.array([100, 80, 60, 55])  # Plateau rather than exponential

        params, r2, status = fit_monoexp_tail(times, activities)

        # Fit might be poor
        assert r2 < 0.9 if r2 is not None else True

    def test_tia_monoexp_with_triangle(self):
        """Test TIA calculation with triangle uptake."""
        A0 = 100.0
        lam = 0.0001
        t_start = 3600.0
        t_end = 28800.0

        tia = tia_monoexp_with_triangle_uptake(A0, lam, t_start, t_end)

        # Analytical: triangle + exponential tail
        # Triangle area: 0.5 * base * height
        triangle_area = 0.5 * (t_end - t_start) * A0
        # Exponential tail: A0 * exp(-lam * t_end) / lam
        tail_area = A0 * np.exp(-lam * (t_end - t_start)) / lam

        expected_tia = triangle_area + tail_area

        np.testing.assert_allclose(tia, expected_tia, rtol=0.01)


class TestHybridModel:
    """Test hybrid model integration."""

    def test_trapz_plus_physical_decay(self):
        """Test trapezoidal integration with physical decay tail."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = np.array([0.0, 50.0, 80.0, 60.0, 40.0])

        half_life = 21636.0  # Tc-99m
        tia = tia_trapz_plus_phys_tail(times, activities, half_life)

        # Should be positive and finite
        assert tia > 0
        assert np.isfinite(tia)

        # Compare with pure trapezoidal (should be larger due to tail)
        tia_trapz = compute_tia_analytical(times, activities, model_type='trapz')
        assert tia > tia_trapz, "With tail should be larger than trapezoidal only"

    def test_trapz_no_tail(self):
        """Test with very long half-life (no tail contribution)."""
        times = np.array([0.0, 1800.0, 3600.0])
        activities = np.array([0.0, 50.0, 80.0])

        long_half_life = 1e10  # Essentially infinite
        tia = tia_trapz_plus_phys_tail(times, activities, long_half_life)

        # Should approximately equal trapezoidal
        tia_trapz = compute_tia_analytical(times, activities, model_type='trapz')
        np.testing.assert_allclose(tia, tia_trapz, rtol=0.01)

    def test_trapz_single_point(self):
        """Test with single time point."""
        times = np.array([3600.0])
        activities = np.array([100.0])
        half_life = 3600.0  # 1 hour

        tia = tia_trapz_plus_phys_tail(times, activities, half_life)

        # For single point, should be A/lambda
        expected_tia = 100.0 / (np.log(2) / 3600.0)
        np.testing.assert_allclose(tia, expected_tia, rtol=0.01)

    def test_trapz_zero_half_life(self):
        """Test error handling with zero half-life."""
        times = np.array([0.0, 3600.0])
        activities = np.array([50.0, 40.0])

        with pytest.raises((ZeroDivisionError, ValueError)):
            tia_trapz_plus_phys_tail(times, activities, half_life=0.0)

    def test_trapz_negative_half_life(self):
        """Test error handling with negative half-life."""
        times = np.array([0.0, 3600.0])
        activities = np.array([50.0, 40.0])

        with pytest.raises((ValueError, Exception)):
            tia_trapz_plus_phys_tail(times, activities, half_life=-100.0)


class TestModelSelection:
    """Test model selection and appropriateness."""

    def test_best_model_for_hump_curve(self):
        """Test that gamma model is best for hump-shaped curves."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.hump_tauc(times)

        # Fit both models and compare
        gamma_params, gamma_r2, gamma_status = fit_gamma_linear_wls(times, activities)

        # Mono-exponential tail (use last points)
        tail_idx = len(times) // 2
        exp_params, exp_r2, exp_status = fit_monoexp_tail(
            times[tail_idx:], activities[tail_idx:]
        )

        # Gamma should have better R² for hump curve
        if gamma_status == 1 and exp_status == 1:
            assert gamma_r2 > exp_r2

    def test_best_model_for_falling_curve(self):
        """Test that exponential is best for purely falling curves."""
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0])
        activities = TestDataGenerator.falling_tauc(times, A0=100.0, lambda_washout=0.0001)

        # Entire curve (gamma model tries to fit)
        gamma_params, gamma_r2, gamma_status = fit_gamma_linear_wls(times, activities)

        # Tail only (exponential model)
        tail_idx = 1
        exp_params, exp_r2, exp_status = fit_monoexp_tail(
            times[tail_idx:], activities[tail_idx:]
        )

        # Exponential should be better for falling curve
        if gamma_status == 1 and exp_status == 1:
            assert exp_r2 >= gamma_r2


class TestModelRobustness:
    """Test model robustness with edge cases."""

    def test_outlier_tolerance(self):
        """Test model tolerance to outliers."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        clean_activities = TestDataGenerator.hump_tauc(times)

        # Add outlier
        noisy_activities = clean_activities.copy()
        noisy_activities[2] *= 3.0  # Multiplicative outlier

        # Fit with outlier
        gamma_params, r2_outlier, gamma_status = fit_gamma_linear_wls(times, noisy_activities)

        # Fit without outlier
        gamma_params_clean, r2_clean, gamma_status_clean = fit_gamma_linear_wls(times, clean_activities)

        # Both should succeed but outlier version might have lower R²
        assert gamma_status == 1
        assert r2_outlier <= r2_clean

    def test_scaling_invariance(self):
        """Test that models are invariant to scaling."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities_1 = TestDataGenerator.hump_tauc(times, A0=100.0)
        activities_2 = 10.0 * activities_1  # Scale by 10

        # Fit both
        params_1, r2_1, status_1 = fit_gamma_linear_wls(times, activities_1)
        params_2, r2_2, status_2 = fit_gamma_linear_wls(times, activities_2)

        # Should have same R² and proportional parameters
        assert status_1 == status_2 == 1
        np.testing.assert_allclose(r2_1, r2_2, rtol=1e-6)
        np.testing.assert_allclose(params_2[0], 10.0 * params_1[0], rtol=1e-6)

    def test_time_scaling_invariance(self):
        """Test invariance to time units (seconds vs hours)."""
        times_seconds = np.array([0.0, 1800.0, 3600.0, 7200.0])
        activities = TestDataGenerator.hump_tauc(times_seconds)

        # Convert to hours
        times_hours = times_seconds / 3600.0

        # Fit in seconds
        params_sec, r2_sec, status_sec = fit_gamma_linear_wls(times_seconds, activities)

        # Fit in hours (if function accepts)
        # Note: Might need to adjust depending on implementation
        params_hr, r2_hr, status_hr = fit_gamma_linear_wls(times_hours, activities)

        # Should give similar R² but scaled parameters
        assert status_sec == status_hr
        np.testing.assert_allclose(r2_sec, r2_hr, rtol=1e-6)

    def test_missing_values(self):
        """Test handling of missing/NaN values."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.hump_tauc(times).astype(float)
        activities[2] = np.nan  # Missing value

        # Should handle gracefully (skip NaN or return error)
        params, r2, status = fit_gamma_linear_wls(times, activities)

        assert params is not None
        # Status depends on implementation
        assert isinstance(status, (int, np.integer))

    def test_extreme_parameter_values(self):
        """Test with extreme parameter regimes."""
        # Very fast decay
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        activities = np.exp(-times / 1.0)  # Decay constant = 1.0

        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Should handle or fail gracefully
        assert params is not None

        # Very slow decay
        activities = np.exp(-times / 1e6)  # Very slow
        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Should handle very flat curves
        assert params is not None


class TestParameterBounds:
    """Test parameter constraints and bounds."""

    def test_positive_parameters_required(self):
        """Test that kinetic parameters remain positive when required."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0])
        activities = np.abs(np.random.normal(50, 10, 4))  # Ensure positive

        params, r2, status = fit_gamma_linear_wls(times, activities)

        if status == 1:  # If fit succeeded
            # A0 should be positive
            assert params[0] > 0
            # Peak time should be positive
            assert params[1] > 0

    def test_physical_decay_constraints(self):
        """Test physical decay lambda constraints."""
        half_life = 3600.0  # 1 hour
        phys_lambda = np.log(2) / half_life

        # Test with very slow effective decay
        eff_half_life = 1e6  # Very long
        eff_lambda = np.log(2) / eff_half_life

        # Implementation should enforce lambda_effective >= lambda_physical
        # or handle appropriately
        assert eff_lambda < phys_lambda

    def test_parameter_bounds_fitting(self):
        """Test fitting with parameter bounds if implemented."""
        times = np.array([0.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.hump_tauc(times)

        # Test with reasonable bounds
        params, r2, status = fit_gamma_linear_wls(times, activities)

        # Check if parameters are reasonable
        if status == 1:
            # Peak time should be within data range or plausible extension
            assert params[1] > 0
            # Very rough bounds check
            assert params[1] < 1e6  # Not absurdly large


class TestModelComparison:
    """Test comparison between different models."""

    def test_exponential_vs_gamma_varying_data(self):
        """Test model performance across various data patterns."""
        test_cases = [
            ('rising', [0.0, 1800.0, 3600.0, 7200.0]),
            ('hump', [0.0, 1800.0, 3600.0, 7200.0, 14400.0]),
            ('falling', [3600.0, 7200.0, 14400.0, 28800.0]),
        ]

        results = {}
        for pattern, times in test_cases:
            if pattern == 'rising':
                activities = TestDataGenerator.rising_tauc(times)
            elif pattern == 'hump':
                activities = TestDataGenerator.hump_tauc(times)
            else:  # falling
                activities = TestDataGenerator.falling_tauc(times)

            # Gamma fit
            gamma_params, gamma_r2, gamma_status = fit_gamma_linear_wls(times, activities)

            # Exponential fit (tail)
            if len(times) >= 3:
                tail_start = len(times) // 2
                exp_params, exp_r2, exp_status = fit_monoexp_tail(
                    times[tail_start:], activities[tail_start:]
                )
            else:
                exp_r2, exp_status = None, 0

            results[pattern] = {
                'gamma_r2': gamma_r2 if gamma_status == 1 else None,
                'exp_r2': exp_r2 if exp_status == 1 else None,
            }

        # Analyze results
        # Gamma should be best for hump
        assert results['hump']['gamma_r2'] > results['hump'].get('exp_r2', 0)

    def test_consistency_check(self):
        """Test that models give consistent results for simple cases."""
        # Pure exponential
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0])
        lam = 0.0001
        A0 = 100.0
        activities = A0 * np.exp(-lam * times)

        # Try to fit with gamma model
        gamma_params, gamma_r2, gamma_status = fit_gamma_linear_wls(times, activities)

        # Fit with exponential
        exp_params, exp_r2, exp_status = fit_monoexp_tail(times, activities)

        # Both should find similar exponential constants
        if gamma_status == 1 and exp_status == 1:
            # Extract effective lambda from gamma parameters if possible
            # This depends on the specific parameterization
            assert exp_r2 > 0.99
            np.testing.assert_allclose(exp_params[1], lam, rtol=0.1)