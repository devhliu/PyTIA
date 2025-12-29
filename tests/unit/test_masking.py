"""Unit tests for masking operations."""

import numpy as np
import pytest
import nibabel as nib
from pathlib import Path

from pytia.masking import load_mask, make_body_mask, mask_to_bool
from tests.utils import assert_images_close


class TestLoadMask:
    """Test mask loading functionality."""

    def test_load_valid_mask(self, temp_dir, roi_mask):
        """Test loading a valid mask file."""
        # Save mask to file
        mask_path = temp_dir / "test_mask.nii.gz"
        nib.save(roi_mask, mask_path)

        # Load mask
        loaded_mask = load_mask(str(mask_path))

        # Should match original
        assert_images_close(loaded_mask, roi_mask, rtol=1e-6)

    def test_load_mask_different_dtypes(self, temp_dir, affine_matrix):
        """Test loading masks with different data types."""
        shape = (10, 10, 10)

        # Test different data types
        test_cases = [
            (np.uint8, 1),
            (np.int16, 1),
            (np.int32, 2),
            (np.float32, 1.0),
            (np.float64, 1.0),
        ]

        for dtype, value in test_cases:
            data = np.zeros(shape, dtype=dtype)
            data[3:7, 3:7, 3:7] = value

            mask_img = nib.Nifti1Image(data, affine_matrix)
            mask_path = temp_dir / f"mask_{dtype.__name__}.nii.gz"
            nib.save(mask_img, mask_path)

            loaded_mask = load_mask(str(mask_path))
            loaded_data = np.asarray(loaded_mask.dataobj)

            # Should be loaded as boolean
            assert loaded_data.dtype == bool
            assert np.sum(loaded_data) == 64  # 4x4x4 region

    def test_load_mask_with_zeros_ones(self, temp_dir, affine_matrix):
        """Test mask with explicit 0/1 values."""
        shape = (5, 5, 5)
        data = np.zeros(shape, dtype=np.int32)
        data[1:4, 1:4, 1:4] = 1

        mask_img = nib.Nifti1Image(data, affine_matrix)
        mask_path = temp_dir / "mask_01.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        loaded_data = np.asarray(loaded_mask.dataobj)

        assert loaded_data.dtype == bool
        assert np.sum(loaded_data) == 27  # 3x3x3 region

    def test_load_mask_nonexistent(self):
        """Test loading nonexistent mask file."""
        with pytest.raises(FileNotFoundError):
            load_mask("nonexistent_mask.nii.gz")

    def test_load_empty_mask(self, temp_dir, affine_matrix):
        """Test loading an empty mask (all zeros)."""
        shape = (5, 5, 5)
        data = np.zeros(shape, dtype=np.int32)
        mask_img = nib.Nifti1Image(data, affine_matrix)
        mask_path = temp_dir / "empty_mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        loaded_data = np.asarray(loaded_mask.dataobj)

        assert loaded_data.dtype == bool
        assert np.sum(loaded_data) == 0

    def test_load_full_mask(self, temp_dir, affine_matrix):
        """Test loading a full mask (all ones)."""
        shape = (5, 5, 5)
        data = np.ones(shape, dtype=np.int32)
        mask_img = nib.Nifti1Image(data, affine_matrix)
        mask_path = temp_dir / "full_mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        loaded_data = np.asarray(loaded_mask.dataobj)

        assert loaded_data.dtype == bool
        assert np.sum(loaded_data) == 125  # 5x5x5

    def test_load_mask_labels(self, temp_dir, affine_matrix):
        """Test loading mask with multiple labels."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.int32)
        data[:3, :, :] = 1  # Label 1
        data[3:6, :, :] = 2  # Label 2
        data[6:, :, :] = 3   # Label 3

        mask_img = nib.Nifti1Image(data, affine_matrix)
        mask_path = temp_dir / "label_mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        loaded_data = np.asarray(loaded_mask.dataobj)

        # All non-zero should be True
        assert loaded_data.dtype == bool
        assert np.sum(loaded_data) == shape[0] * shape[1] * shape[2]


class TestMakeBodyMask:
    """Test automatic body mask generation."""

    def test_body_mask_simple_phantom(self):
        """Test body mask generation with simple phantom."""
        shape = (20, 20, 10)
        data = np.zeros(shape, dtype=np.float32)

        # Elliptical body
        z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
        center = np.array([shape[0]/2, shape[1]/2, shape[2]/2])
        radius = np.array([shape[0]/3, shape[1]/3, shape[2]/3])

        # Create ellipsoid
        ellipsoid = ((z - center[0])/radius[0])**2 + \
                   ((y - center[1])/radius[1])**2 + \
                   ((x - center[2])/radius[2])**2
        data[ellipsoid <= 1] = 50.0

        img = nib.Nifti1Image(data, np.eye(4))
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        # Check that body region is masked
        assert np.sum(mask_data) > 0
        # Check approximate size
        expected_volume = 4/3 * np.pi * np.prod(radius)
        actual_volume = np.sum(mask_data)
        assert 0.7 * expected_volume < actual_volume < 1.3 * expected_volume

    def test_body_mask_with_noise(self):
        """Test body mask with noisy background."""
        shape = (15, 15, 8)
        data = np.random.normal(5, 2, shape).astype(np.float32)  # Background noise

        # Add high signal region
        data[5:10, 5:10, 2:6] = 100.0

        img = nib.Nifti1Image(data, np.eye(4))
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        # Should mask the high signal region
        assert np.sum(mask_data) == 25  # 5x5x3 region

    def test_body_mask_empty_image(self):
        """Test body mask with empty/zero image."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))

        mask = make_body_mask(img)
        mask_data = np.asarray(mask.dataobj)

        # Should be all False
        assert np.sum(mask_data) == 0

    def test_body_mask_low_signal(self):
        """Test body mask with very low signal."""
        shape = (10, 10, 10)
        data = np.random.normal(1, 0.5, shape).astype(np.float32)  # Very low signal
        img = nib.Nifti1Image(data, np.eye(4))

        mask = make_body_mask(img)
        mask_data = np.asarray(mask.dataobj)

        # Might be empty or very sparse depending on threshold
        assert mask_data.dtype == bool

    def test_body_mask_different_thresholds(self):
        """Test body mask with different threshold settings."""
        shape = (20, 20, 10)
        data = np.zeros(shape, dtype=np.float32)

        # Create pattern with different intensities
        data[5:15, 5:15, 2:8] = 20.0  # Medium signal
        data[8:12, 8:12, 4:6] = 100.0  # High signal
        data[6:14, 6:14, 3:7] += np.random.normal(0, 5, (8, 8, 4))  # Noise

        img = nib.Nifti1Image(data, np.eye(4))

        # Test with different min fraction of max
        masks = []
        thresholds = [0.01, 0.05, 0.1, 0.2]

        for thresh in thresholds:
            mask = make_body_mask(img, min_fraction_of_max=thresh)
            masks.append(np.asarray(mask.dataobj))

        # Higher threshold should give smaller mask
        for i in range(len(thresholds) - 1):
            assert np.sum(masks[i]) >= np.sum(masks[i + 1])

    def test_body_mask_multiple_objects(self):
        """Test body mask with multiple separate objects."""
        shape = (30, 30, 10)
        data = np.zeros(shape, dtype=np.float32)

        # Two separate objects
        data[5:10, 5:10, 3:7] = 80.0  # Object 1
        data[20:25, 20:25, 3:7] = 60.0  # Object 2

        img = nib.Nifti1Image(data, np.eye(4))
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        # Should mask both objects
        expected_sum = 5*5*4 + 5*5*4  # Two 5x5x4 regions
        actual_sum = np.sum(mask_data)
        assert actual_sum == expected_sum

    def test_body_mask_anisotropic_voxels(self):
        """Test body mask with anisotropic voxel sizes."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.float32)
        data[3:7, 3:7, 3:7] = 100.0

        # Anisotropic affine
        affine = np.eye(4)
        affine[0, 0] = 1.0  # 1mm in x
        affine[1, 1] = 2.0  # 2mm in y
        affine[2, 2] = 3.0  # 3mm in z

        img = nib.Nifti1Image(data, affine)
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        # Should still work with anisotropic voxels
        assert np.sum(mask_data) == 64  # 4x4x4 region

    def test_body_mask_very_large_values(self):
        """Test body mask with very large intensity values."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1e6  # Very large value

        img = nib.Nifti1Image(data, np.eye(4))
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        # Should handle large values correctly
        assert np.sum(mask_data) == 64

    def test_body_mask_with_float64(self):
        """Test body mask with float64 data."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.float64)
        data[3:7, 3:7, 3:7] = 100.0

        img = nib.Nifti1Image(data, np.eye(4))
        mask = make_body_mask(img)

        mask_data = np.asarray(mask.dataobj)

        assert mask_data.dtype == bool
        assert np.sum(mask_data) == 64


class TestMaskToBool:
    """Test conversion of masks to boolean type."""

    def test_convert_uint8_mask(self):
        """Test converting uint8 mask to boolean."""
        mask_data = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        bool_mask = mask_to_bool(mask_data)

        assert bool_mask.dtype == bool
        np.testing.assert_array_equal(mask_data.astype(bool), bool_mask)

    def test_convert_float_mask(self):
        """Test converting float mask to boolean."""
        mask_data = np.array([[0.0, 1.0, 0.5], [1.2, 0.8, 0.0]], dtype=np.float32)
        bool_mask = mask_to_bool(mask_data)

        expected = mask_data > 0
        np.testing.assert_array_equal(expected, bool_mask)

    def test_convert_label_mask(self):
        """Test converting multi-label mask to boolean."""
        mask_data = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=np.int32)
        bool_mask = mask_to_bool(mask_data)

        expected = mask_data != 0
        np.testing.assert_array_equal(expected, bool_mask)

    def test_convert_already_bool(self):
        """Test converting already boolean mask."""
        mask_data = np.array([[True, False], [False, True]])
        bool_mask = mask_to_bool(mask_data)

        assert bool_mask is mask_data  # Should return same object

    def test_convert_with_nans(self):
        """Test converting mask with NaN values."""
        mask_data = np.array([[0.0, 1.0, np.nan], [np.inf, 0.5, 0.0]], dtype=np.float32)
        bool_mask = mask_to_bool(mask_data)

        # NaN should be False, inf should be True
        expected = np.array([[False, True, False], [True, True, False]])
        np.testing.assert_array_equal(expected, bool_mask)

    def test_convert_with_negatives(self):
        """Test converting mask with negative values."""
        mask_data = np.array([[-1, 0, 1], [-2, -0.5, 2]], dtype=np.float32)
        bool_mask = mask_to_bool(mask_data)

        # Negative should be False, positive True, zero False
        expected = np.array([[False, False, True], [False, False, True]])
        np.testing.assert_array_equal(expected, bool_mask)

    def test_convert_empty_array(self):
        """Test converting empty array."""
        mask_data = np.array([], dtype=np.int32)
        bool_mask = mask_to_bool(mask_data)

        assert bool_mask.dtype == bool
        assert bool_mask.shape == mask_data.shape

    def test_convert_3d_mask(self):
        """Test converting 3D mask to boolean."""
        mask_data = np.random.randint(0, 3, (5, 5, 5))
        bool_mask = mask_to_bool(mask_data)

        assert bool_mask.dtype == bool
        assert bool_mask.shape == mask_data.shape
        np.testing.assert_array_equal(mask_data != 0, bool_mask)

    def test_convert_very_large_mask(self):
        """Test converting large mask array."""
        shape = (100, 100, 100)
        mask_data = np.random.randint(0, 2, shape, dtype=np.uint8)
        bool_mask = mask_to_bool(mask_data)

        assert bool_mask.dtype == bool
        assert bool_mask.shape == shape
        # Memory should not be significantly increased
        assert bool_mask.nbytes == mask_data.nbytes  # bool should be 1 byte


class TestMaskOperations:
    """Test additional mask operations."""

    def test_mask_intersection(self, temp_dir, affine_matrix):
        """Test intersection of two masks."""
        # Create two masks
        mask1_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask1_data[2:6, 2:6, 2:6] = 1

        mask2_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask2_data[4:8, 4:8, 4:8] = 1

        mask1 = nib.Nifti1Image(mask1_data, affine_matrix)
        mask2 = nib.Nifti1Image(mask2_data, affine_matrix)

        mask1_path = temp_dir / "mask1.nii.gz"
        mask2_path = temp_dir / "mask2.nii.gz"
        nib.save(mask1, mask1_path)
        nib.save(mask2, mask2_path)

        # Load and compute intersection
        loaded_mask1 = load_mask(str(mask1_path))
        loaded_mask2 = load_mask(str(mask2_path))

        bool_mask1 = np.asarray(loaded_mask1.dataobj)
        bool_mask2 = np.asarray(loaded_mask2.dataobj)

        intersection = bool_mask1 & bool_mask2

        # Check intersection region
        expected_size = 2 * 2 * 2  # overlap of 4x4x4 and 4x4x4 regions
        assert np.sum(intersection) == expected_size

    def test_mask_union(self, temp_dir, affine_matrix):
        """Test union of two masks."""
        mask1_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask1_data[1:4, 1:4, 1:4] = 1

        mask2_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask2_data[6:9, 6:9, 6:9] = 1

        mask1 = nib.Nifti1Image(mask1_data, affine_matrix)
        mask2 = nib.Nifti1Image(mask2_data, affine_matrix)

        mask1_path = temp_dir / "mask1.nii.gz"
        mask2_path = temp_dir / "mask2.nii.gz"
        nib.save(mask1, mask1_path)
        nib.save(mask2, mask2_path)

        loaded_mask1 = load_mask(str(mask1_path))
        loaded_mask2 = load_mask(str(mask2_path))

        bool_mask1 = np.asarray(loaded_mask1.dataobj)
        bool_mask2 = np.asarray(loaded_mask2.dataobj)

        union = bool_mask1 | bool_mask2

        # Should have all voxels from both masks
        expected_size = 3 * 3 * 3 * 2  # Two separate 3x3x3 regions
        assert np.sum(union) == expected_size

    def test_mask_inversion(self, temp_dir, affine_matrix):
        """Test mask inversion."""
        mask_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask_data[3:7, 3:7, 3:7] = 1

        mask_img = nib.Nifti1Image(mask_data, affine_matrix)
        mask_path = temp_dir / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        bool_mask = np.asarray(loaded_mask.dataobj)

        # Invert mask
        inverted = ~bool_mask

        # Check that inverted mask has opposite pattern
        assert np.sum(inverted) + np.sum(bool_mask) == 1000  # 10x10x10 total
        # Original region should be False in inverted
        assert not inverted[5, 5, 5]  # Center of original mask
        # Background should be True
        assert inverted[0, 0, 0]

    def test_mask_dilation_simple(self, temp_dir, affine_matrix):
        """Test simple mask dilation."""
        mask_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask_data[4:6, 4:6, 4:6] = 1  # 2x2x2 central region

        mask_img = nib.Nifti1Image(mask_data, affine_matrix)
        mask_path = temp_dir / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        bool_mask = np.asarray(loaded_mask.dataobj)

        # Simple 1-voxel dilation using convolution
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(bool_mask)

        # Dilated mask should be larger
        assert np.sum(dilated) > np.sum(bool_mask)
        # Original region should still be True
        assert dilated[5, 5, 5]

    def test_mask erosion_simple(self, temp_dir, affine_matrix):
        """Test simple mask erosion."""
        mask_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask_data[3:7, 3:7, 3:7] = 1  # 4x4x4 central region

        mask_img = nib.Nifti1Image(mask_data, affine_matrix)
        mask_path = temp_dir / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        bool_mask = np.asarray(loaded_mask.dataobj)

        # Simple erosion
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(bool_mask)

        # Eroded mask should be smaller
        assert np.sum(eroded) < np.sum(bool_mask)

    def test_mask_boundary_voxels(self, temp_dir, affine_matrix):
        """Test identification of mask boundary voxels."""
        mask_data = np.zeros((10, 10, 10), dtype=np.int32)
        mask_data[3:7, 3:7, 3:7] = 1  # 4x4x4 region

        mask_img = nib.Nifti1Image(mask_data, affine_matrix)
        mask_path = temp_dir / "mask.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))
        bool_mask = np.asarray(loaded_mask.dataobj)

        # Find boundary voxels (faces that touch background)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(bool_mask)
        boundary = bool_mask & ~eroded

        # Should have expected number of boundary voxels
        # For 4x4x4 cube: 6 faces × 4×4 voxels - edges × 2 + corners × 3
        expected_boundary = 6 * 16 - 12 * 4 + 8 * 2  # 96 - 48 + 16 = 64
        assert np.sum(boundary) == expected_boundary


class TestMaskConsistency:
    """Test consistency and reproducibility of mask operations."""

    def test_mask_loading_reproducibility(self, temp_dir, roi_mask):
        """Test that mask loading is reproducible."""
        mask_path = temp_dir / "test_mask.nii.gz"
        nib.save(roi_mask, mask_path)

        # Load multiple times
        mask1 = load_mask(str(mask_path))
        mask2 = load_mask(str(mask_path))
        mask3 = load_mask(str(mask_path))

        # Should be identical
        data1 = np.asarray(mask1.dataobj)
        data2 = np.asarray(mask2.dataobj)
        data3 = np.asarray(mask3.dataobj)

        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(data2, data3)

    def test_body_mask_determinism(self):
        """Test that body mask generation is deterministic."""
        shape = (20, 20, 10)
        data = np.zeros(shape, dtype=np.float32)
        data[5:15, 5:15, 2:8] = 50.0
        img = nib.Nifti1Image(data, np.eye(4))

        # Generate mask multiple times
        mask1 = make_body_mask(img)
        mask2 = make_body_mask(img)
        mask3 = make_body_mask(img)

        # Should be identical
        data1 = np.asarray(mask1.dataobj)
        data2 = np.asarray(mask2.dataobj)
        data3 = np.asarray(mask3.dataobj)

        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(data2, data3)

    def test_mask_preserves_geometry(self, temp_dir, affine_matrix):
        """Test that mask operations preserve image geometry."""
        # Create mask with specific geometry
        shape = (15, 20, 25)
        data = np.zeros(shape, dtype=np.int32)
        data[5:10, 8:12, 10:20] = 1

        # Use non-standard affine
        affine = np.array([
            [2.0, 0.5, 0.0, -10.0],
            [0.0, 3.0, 0.0, -20.0],
            [0.0, 0.0, 4.0, -15.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        mask_img = nib.Nifti1Image(data, affine)
        mask_path = temp_dir / "geo_test.nii.gz"
        nib.save(mask_img, mask_path)

        loaded_mask = load_mask(str(mask_path))

        # Geometry should be preserved
        assert loaded_mask.shape == shape
        np.testing.assert_allclose(loaded_mask.affine, affine, rtol=1e-6)

    def test_mask_dtype_consistency(self, temp_dir):
        """Test that mask operations maintain consistent dtypes."""
        # Test with various input dtypes
        dtypes = [np.uint8, np.int16, np.int32, np.float32, np.float64]

        for dtype in dtypes:
            data = np.zeros((5, 5, 5), dtype=dtype)
            data[1:4, 1:4, 1:4] = 1 if dtype.kind in 'iu' else 1.0

            mask_img = nib.Nifti1Image(data, np.eye(4))
            mask_path = temp_dir / f"mask_{dtype.__name__}.nii.gz"
            nib.save(mask_img, mask_path)

            loaded_mask = load_mask(str(mask_path))
            loaded_data = np.asarray(loaded_mask.dataobj)

            # Output should always be boolean
            assert loaded_data.dtype == bool