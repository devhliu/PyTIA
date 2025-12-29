"""Unit tests for time-activity curve classification."""

import numpy as np
import pytest

from pytia.classify import (
    classify_curves, CLASS_RISING, CLASS_HUMP, CLASS_FALLING
)
from tests.utils import TestDataGenerator


class TestCurveClassification:
    """Test the main curve classification function."""

    def test_rising_curve_classification(self):
        """Test classification of rising curves."""
        # Create typical rising curve
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.rising_tauc(times, A0=100.0, lambda_uptake=0.0005)

        # Classify single curve (add batch dimension)
        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should classify as rising
        assert classes[0] == CLASS_RISING

    def test_hump_curve_classification(self):
        """Test classification of hump-shaped curves."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.hump_tauc(times, A0=100.0, tpeak=3600.0)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should classify as hump
        assert classes[0] == CLASS_HUMP

    def test_falling_curve_classification(self):
        """Test classification of falling curves."""
        times = np.array([3600.0, 7200.0, 14400.0, 28800.0, 57600.0])
        activities = TestDataGenerator.falling_tauc(times, A0=100.0, lambda_washout=0.0001)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should classify as falling
        assert classes[0] == CLASS_FALLING

    def test_multiple_curves(self):
        """Test classification of multiple curves simultaneously."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])

        # Create three different curve types
        rising_curve = TestDataGenerator.rising_tauc(times)
        hump_curve = TestDataGenerator.hump_tauc(times)
        falling_curve = TestDataGenerator.falling_tauc(times)

        activities_2d = np.array([rising_curve, hump_curve, falling_curve])
        times_2d = np.tile(times, (3, 1))

        classes, details = classify_curves(activities_2d, times_2d)

        # Check classifications
        assert len(classes) == 3
        assert classes[0] == CLASS_RISING
        assert classes[1] == CLASS_HUMP
        assert classes[2] == CLASS_FALLING

    def test_noisy_curves(self):
        """Test classification with noisy data."""
        np.random.seed(42)  # For reproducibility
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])

        # Create noisy hump curve
        clean_curve = TestDataGenerator.hump_tauc(times)
        noise = np.random.normal(0, 0.1 * clean_curve.max(), clean_curve.shape)
        noisy_curve = clean_curve + noise
        noisy_curve[noisy_curve < 0] = 0  # Ensure non-negative

        activities_2d = noisy_curve[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should still classify as hump despite noise
        assert classes[0] == CLASS_HUMP

    def test_flat_curve(self):
        """Test classification of flat/constant curve."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = np.full_like(times, 50.0, dtype=float)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Might classify as falling (no increase)
        # Exact behavior depends on implementation
        assert classes[0] in [CLASS_FALLING, CLASS_RISING]

    def test_spike_curve(self):
        """Test classification of curve with a single spike."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = np.array([0.0, 0.0, 100.0, 0.0, 0.0])  # Single spike

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should handle spike appropriately
        # Likely hump or falling
        assert classes[0] in [CLASS_HUMP, CLASS_FALLING]

    def test_negative_values(self):
        """Test classification with negative activity values."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = np.array([-10.0, 20.0, 50.0, 30.0, 10.0])

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should handle negative values
        assert classes[0] in [CLASS_RISING, CLASS_HUMP, CLASS_FALLING]


class TestMinimumPointsRequirement:
    """Test minimum points requirement for classification."""

    def test_two_points_only(self):
        """Test with only two time points."""
        times = np.array([0.0, 3600.0])
        activities = np.array([10.0, 20.0])

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should handle two points (maybe always rising)
        assert classes[0] is not None

    def test_single_point(self):
        """Test with single time point."""
        times = np.array([3600.0])
        activities = np.array([50.0])

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Should handle single point gracefully
        assert classes[0] is not None

    def test_empty_arrays(self):
        """Test with empty time/activity arrays."""
        activities_2d = np.array([[]])
        times_2d = np.array([[]])

        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, IndexError)):
            classify_curves(activities_2d, times_2d)


class TestDifferentials:
    """Test differential calculation for classification."""

    def test_first_derivative_calculation(self):
        """Test first derivative calculation."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # t²

        # Calculate derivatives
        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt

        # For t², derivative at t=0.5, 1.5, 2.5, 3.5 is approximately 1, 3, 5, 7
        expected = np.array([1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(first_deriv, expected, rtol=0.1)

    def test_second_derivative_calculation(self):
        """Test second derivative calculation."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # t²

        # Second derivative should be approximately constant (2)
        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt
        second_deriv = np.diff(first_deriv) / dt[:-1]

        np.testing.assert_allclose(second_deriv, 2.0, rtol=0.1)

    def test_uneven_spacing_derivatives(self):
        """Test derivative calculation with uneven spacing."""
        times = np.array([0.0, 0.5, 2.0, 4.0, 8.0])
        activities = times ** 2  # t²

        # Should handle uneven spacing correctly
        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt

        assert len(first_deriv) == len(times) - 1
        assert np.all(np.isfinite(first_deriv))


class TestClassificationRules:
    """Test the specific classification rule logic."""

    def test_rising_classification_rules(self):
        """Test rules that identify rising curves."""
        # Clear rising case
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # monotonic increase

        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt
        second_deriv = np.diff(first_deriv) / dt[:-1]

        # Classification based on sign of derivatives
        assert np.all(first_deriv > 0)  # Always rising
        assert np.mean(second_deriv) < 0.01  # No strong peak

    def test_hump_classification_rules(self):
        """Test rules that identify hump curves."""
        # Clear hump case
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([1.0, 3.0, 5.0, 3.0, 1.0])  # rise then fall

        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt
        second_deriv = np.diff(first_deriv) / dt[:-1]

        # Should have sign change in first derivative
        assert first_deriv[0] > 0  # Initially rising
        assert first_deriv[-1] < 0  # Eventually falling
        # Strong peak indicated by second derivative

    def test_falling_classification_rules(self):
        """Test rules that identify falling curves."""
        # Clear falling case
        times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        activities = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # monotonic decrease

        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt
        second_deriv = np.diff(first_deriv) / dt[:-1]

        # Should always be negative
        assert np.all(first_deriv < 0)

    def test_ambiguous_classification(self):
        """Test classification of borderline/ambiguous cases."""
        # Very gradual changes
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([1.0, 1.1, 1.2, 1.1, 1.0])  # Slight rise then fall

        dt = np.diff(times)
        da = np.diff(activities)
        first_deriv = da / dt

        # Small positive then negative values
        assert np.any(first_deriv > 0)
        assert np.any(first_deriv < 0)

        # Classification might depend on threshold values
        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]
        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] in [CLASS_RISING, CLASS_HUMP, CLASS_FALLING]


class TestClassificationRobustness:
    """Test robustness of classification to various issues."""

    def test_zero_time_points(self):
        """Test classification when time starts from zero."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0])
        activities = TestDataGenerator.rising_tauc(times)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_nonzero_start_time(self):
        """Test classification when time doesn't start from zero."""
        times = np.array([3600.0, 5400.0, 7200.0, 10800.0])  # Starts at 1 hour
        activities = TestDataGenerator.falling_tauc(times)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_missing_intermediate_timepoints(self):
        """Test with gaps in time points."""
        times = np.array([0.0, 3600.0, 14400.0, 28800.0])  # Large gaps
        activities = TestDataGenerator.hump_tauc(times)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] in [CLASS_RISING, CLASS_HUMP, CLASS_FALLING]

    def test_very_short_time_range(self):
        """Test with very short total time range."""
        times = np.array([0.0, 60.0, 120.0, 180.0])  # Only 3 minutes
        activities = TestDataGenerator.rising_tauc(times, lambda_uptake=0.001)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_very_long_time_range(self):
        """Test with very long total time range."""
        times = np.array([0.0, 86400.0, 172800.0, 259200.0, 345600.0])  # 0-4 days
        activities = TestDataGenerator.falling_tauc(times, lambda_washout=1e-6)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_large_dynamic_range(self):
        """Test with large dynamic range in activity values."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0])
        activities = np.array([1e-3, 1e2, 1e3, 1e2])  # Six orders of magnitude

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_logarithmic_scaling(self):
        """Test classification with log-transformed values."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0])
        activities = np.array([1.0, 10.0, 100.0, 10.0])
        log_activities = np.log10(activities + 1e-10)  # Log-transform

        activities_2d = log_activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        # Should still classify (though might give different result)
        assert classes[0] is not None


class TestClassificationEdgeCases:
    """Test classification edge cases and boundary conditions."""

    def test_identical_consecutive_values(self):
        """Test curves with identical consecutive values."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        activities = np.array([1.0, 1.0, 2.0, 2.0])  # Plateaus

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        assert classes[0] is not None

    def test_single_spike_in_baseline(self):
        """Test single spike within baseline."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([0.1, 0.1, 10.0, 0.1, 0.1])  # Single spike

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        # Likely hump due to spike
        assert classes[0] in [CLASS_HUMP, CLASS_FALLING]

    def test_bidirectional_changes(self):
        """Test curves with multiple direction changes."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        activities = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 2.0])  # Multiple peaks/valleys

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        # Should classify based on dominant pattern
        assert classes[0] is not None

    def test_steps_and_plateaus(self):
        """Test curves with step-like changes."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        activities = np.array([1.0, 1.0, 3.0, 3.0, 5.0])  # Steps up

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, _ = classify_curves(activities_2d, times_2d)
        # Should classify as rising
        assert classes[0] in [CLASS_RISING]


class TestClassificationMetadata:
    """Test metadata and details returned by classification."""

    def test_classification_details_structure(self):
        """Test structure of classification details."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])
        activities = TestDataGenerator.hump_tauc(times)

        activities_2d = activities[np.newaxis, :]
        times_2d = times[np.newaxis, :]

        classes, details = classify_curves(activities_2d, times_2d)

        # Check details structure
        assert isinstance(details, dict)
        # Details might contain derivatives, peak information, etc.
        assert len(details) > 0

    def test_consistent_classification(self):
        """Test that similar curves get same classification."""
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0])

        # Create similar hump curves
        hump1 = TestDataGenerator.hump_tauc(times, A0=100.0)
        hump2 = TestDataGenerator.hump_tauc(times, A0=110.0)  # Slightly different amplitude

        activities_2d = np.array([hump1, hump2])
        times_2d = np.tile(times, (2, 1))

        classes, _ = classify_curves(activities_2d, times_2d)

        # Should get same classification for similar curves
        assert classes[0] == classes[1] == CLASS_HUM