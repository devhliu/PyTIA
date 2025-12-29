"""Unit tests for the core TIA calculation engine."""

import numpy as np
import pytest
import nibabel as nib
from pathlib import Path

from pytia.engine import run_tia, _times_to_seconds, STATUS_LEGEND
from tests.utils import assert_images_close, assert_status_codes, compute_tia_analytical


class TestTimeConversion:
    """Test time unit conversion utilities."""

    def test_seconds_unit(self):
        """Test conversion when unit is seconds."""
        times = [0.0, 3600.0, 7200.0]
        result = _times_to_seconds(times, "seconds")
        np.testing.assert_array_equal(result, times)

    def test_hours_unit(self):
        """Test conversion when unit is hours."""
        times = [0.0, 1.0, 2.0]
        result = _times_to_seconds(times, "hours")
        expected = [0.0, 3600.0, 7200.0]
        np.testing.assert_array_equal(result, expected)

    def test_invalid_unit(self):
        """Test error handling for invalid time unit."""
        times = [0.0, 60.0]
        with pytest.raises(ValueError, match="Unsupported time unit"):
            _times_to_seconds(times, "minutes")


class TestMultiTimepointBasic:
    """Test basic multi-timepoint functionality."""

    def test_basic_two_timepoints(self, temp_dir, synthetic_pet_data):
        """Test with exactly two timepoints."""
        images, times = synthetic_pet_data
        # Use only first two timepoints
        images_subset = images[:2]
        times_subset = times[:2]

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images_subset, times_subset, config=config)

        # Check outputs exist
        assert result.tia_img is not None
        assert result.r2_img is not None
        assert result.status_id_img is not None
        assert result.model_id_img is not None

        # Check reasonable values
        tia_data = np.asarray(result.tia_img.dataobj)
        r2_data = np.asarray(result.r2_img.dataobj)

        # TIA should be positive where activity exists
        assert np.any(tia_data > 0)
        # R² should be between 0 and 1
        assert np.all((r2_data >= 0) | ~np.isfinite(r2_data))

    def test_ascending_time_order(self, temp_dir, synthetic_pet_data):
        """Test with unsorted time points."""
        images, times = synthetic_pet_data
        # Shuffle times and images
        indices = np.array([3, 1, 4, 0, 2])
        images_shuffled = [images[i] for i in indices]
        times_shuffled = times[indices]

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds", "sort_timepoints": True},
        }

        result = run_tia(images_shuffled, times_shuffled, config=config)

        # Should still produce valid results
        assert result.tia_img is not None
        assert np.all(result.times_s == np.sort(times_shuffled))

    def test_mask_processing(self, temp_dir, synthetic_pet_data, roi_mask):
        """Test with provided ROI mask."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "provided", "provided_path": roi_mask.get_filename()},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        # Save mask file
        mask_path = temp_dir / "mask.nii.gz"
        nib.save(roi_mask, mask_path)
        config["mask"]["provided_path"] = str(mask_path)

        result = run_tia(images, times, config=config)

        # Check that processing respects mask
        status_data = np.asarray(result.status_id_img.dataobj)
        mask_data = np.asarray(roi_mask.dataobj)

        # Outside mask should have status 0
        assert np.all(status_data[mask_data == 0] == 0)

    def test_denoising_enabled(self, temp_dir, synthetic_pet_data):
        """Test with denoising enabled."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": True, "method": "masked_gaussian", "sigma_vox": 1.0},
            "time": {"unit": "seconds"},
        }

        result_denoised = run_tia(images, times, config=config)

        # Run without denoising for comparison
        config["denoise"]["enabled"] = False
        result_noisy = run_tia(images, times, config=config)

        # Results should be slightly different
        tia_denoised = np.asarray(result_denoised.tia_img.dataobj)
        tia_noisy = np.asarray(result_noisy.tia_img.dataobj)

        # Should not be identical
        assert not np.allclose(tia_denoised, tia_noisy, rtol=0, atol=0)


class TestPhysicalDecayExtrapolation:
    """Test physical decay tail extrapolation in multi-timepoint mode."""

    def test_phys_tail_enabled(self, temp_dir, synthetic_pet_data):
        """Test with physical decay tail extrapolation."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "fit": {"phys_tail": True},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # TIA should be higher with tail extrapolation
        tia_data = np.asarray(result.tia_img.dataobj)
        assert np.any(tia_data > 0)

    def test_phys_tail_requires_half_life(self, temp_dir, synthetic_pet_data):
        """Test that phys tail requires half-life."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": None},
            "fit": {"phys_tail": True},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Should handle gracefully
        status_data = np.asarray(result.status_id_img.dataobj)
        # Many voxels might fail without half-life
        assert np.any(status_data != 1)


class TestChunkedProcessing:
    """Test memory-efficient chunked processing."""

    def test_small_chunk_size(self, temp_dir, synthetic_pet_data):
        """Test processing with very small chunks."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "performance": {"chunk_size_vox": 100},  # Very small chunks
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Should still produce valid results
        assert result.tia_img is not None
        tia_data = np.asarray(result.tia_img.dataobj)
        assert np.any(np.isfinite(tia_data))

    def test_chunk_size_equal_to_voxels(self, temp_dir, synthetic_pet_data):
        """Test with chunk size equal to number of voxels."""
        images, times = synthetic_pet_data
        n_voxels = np.prod(images[0].shape)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "performance": {"chunk_size_vox": n_voxels},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Should process correctly
        assert result.tia_img is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_voxel(self, temp_dir):
        """Test with single voxel images."""
        shape = (1, 1, 1)
        affine = np.eye(4)
        times = np.array([0.0, 3600.0])

        images = []
        for t in times:
            data = np.full(shape, 100.0 * np.exp(-t/3600), dtype=np.float32)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)
        assert result.tia_img.shape == shape

    def test_uniform_activity(self, temp_dir):
        """Test with uniform activity across all voxels."""
        shape = (5, 5, 5)
        affine = np.eye(4)
        times = np.array([0.0, 1800.0, 3600.0])

        images = []
        for t in times:
            data = np.full(shape, 100.0, dtype=np.float32)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # All non-background TIA values should be identical
        tia_data = np.asarray(result.tia_img.dataobj)
        assert np.all(tia_data[tia_data > 0] == tia_data[tia_data > 0][0])

    def test_all_zero_activity(self, temp_dir):
        """Test with all zero activity."""
        shape = (5, 5, 5)
        affine = np.eye(4)
        times = np.array([0.0, 1800.0, 3600.0])

        images = []
        for t in times:
            data = np.zeros(shape, dtype=np.float32)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # All TIA values should be zero
        tia_data = np.asarray(result.tia_img.dataobj)
        assert np.all(tia_data == 0)

        # All statuses should be 0 (outside/background)
        status_data = np.asarray(result.status_id_img.dataobj)
        assert np.all(status_data == 0)

    def test_negative_activity(self, temp_dir):
        """Test handling of negative activity values."""
        shape = (5, 5, 5)
        affine = np.eye(4)
        times = np.array([0.0, 1800.0, 3600.0])

        images = []
        for t in times:
            # Include some negative values
            data = np.random.normal(50, 20, shape).astype(np.float32)
            data[data < 0] = data[data < 0] * 0.5  # Make some negative
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": True},  # Denoising should handle negatives
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Should not crash and should produce finite values where appropriate
        tia_data = np.asarray(result.tia_img.dataobj)
        assert np.any(np.isfinite(tia_data))


class TestResultsValidation:
    """Test validation of results structure."""

    def test_results_attributes(self, temp_dir, synthetic_pet_data):
        """Test that Results has all expected attributes."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Check all images are nibabel objects
        assert isinstance(result.tia_img, nib.SpatialImage)
        assert isinstance(result.r2_img, nib.SpatialImage)
        assert isinstance(result.sigma_tia_img, nib.SpatialImage)
        assert isinstance(result.model_id_img, nib.SpatialImage)
        assert isinstance(result.status_id_img, nib.SpatialImage)
        assert result.tpeak_img is None or isinstance(result.tpeak_img, nib.SpatialImage)

        # Check metadata
        assert isinstance(result.summary, dict)
        assert isinstance(result.config, dict)
        assert isinstance(result.output_paths, dict)
        assert all(isinstance(p, Path) for p in result.output_paths.values())

    def test_status_codes_valid(self, temp_dir, synthetic_pet_data):
        """Test that all status codes are valid."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # All status codes should be in the legend
        valid_codes = set(STATUS_LEGEND.keys())
        assert_status_codes(result.status_id_img, valid_codes)

    def test_output_paths_created(self, temp_dir, synthetic_pet_data):
        """Test that output files are actually created."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Check that all output files exist
        for path in result.output_paths.values():
            assert path.exists(), f"Output file not created: {path}"


class TestTimingAndPerformance:
    """Test timing and performance aspects."""

    def test_profiling_enabled(self, temp_dir, synthetic_pet_data):
        """Test profiling functionality."""
        images, times = synthetic_pet_data

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "performance": {"enable_profiling": True},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Check timing information in summary
        assert "timing" in result.summary
        timing = result.summary["timing"]
        assert "total_time_seconds" in timing
        assert timing["total_time_seconds"] > 0

    def test_memory_efficiency_large_data(self, temp_dir):
        """Test with moderately large data to check memory usage."""
        # Create larger test data
        shape = (100, 100, 20)  # 200k voxels
        affine = np.eye(4)
        times = np.array([0.0, 3600.0, 7200.0])

        images = []
        for t in times:
            # Simple decay pattern
            data = 100 * np.exp(-t/5000) * np.random.weibull(2, shape).astype(np.float32)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

        config = {
            "io": {"output_dir": str(temp_dir)},
            "physics": {"half_life_seconds": 21636.0},
            "performance": {"chunk_size_vox": 50000},  # 4 chunks
            "mask": {"mode": "otsu"},
            "denoise": {"enabled": False},
            "time": {"unit": "seconds"},
        }

        result = run_tia(images, times, config=config)

        # Should complete successfully
        assert result.tia_img is not None
        assert result.tia_img.shape == shape