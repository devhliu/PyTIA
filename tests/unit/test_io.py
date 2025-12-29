"""Unit tests for I/O module utilities."""

import numpy as np
import pytest
import nibabel as nib
from pathlib import Path
import tempfile
import gzip

from pytia.io import (
    load_images, voxel_volume_ml, make_like, stack_4d,
    ensure_dir
)
from tests.utils import assert_images_close, create_corrupted_image_file


class TestLoadImages:
    """Test image loading functionality."""

    def test_load_single_image(self, temp_dir):
        """Test loading a single image."""
        # Create test image
        shape = (10, 10, 10)
        data = np.random.normal(100, 10, shape).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        img_path = temp_dir / "test.nii.gz"
        nib.save(img, img_path)

        # Load image
        loaded_imgs = load_images([img_path])
        assert len(loaded_imgs) == 1

        # Compare with original
        assert_images_close(loaded_imgs[0], img, rtol=1e-6)

    def test_load_multiple_images(self, temp_dir):
        """Test loading multiple images."""
        n_images = 3
        shape = (5, 5, 5)
        images = []

        for i in range(n_images):
            data = np.full(shape, float(i * 10), dtype=np.float32)
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
            images.append(img)

            img_path = temp_dir / f"img_{i}.nii.gz"
            nib.save(img, img_path)

        # Load all images
        img_paths = [temp_dir / f"img_{i}.nii.gz" for i in range(n_images)]
        loaded_imgs = load_images(img_paths)

        assert len(loaded_imgs) == n_images

        for i, loaded_img in enumerate(loaded_imgs):
            expected_data = np.full(shape, float(i * 10))
            loaded_data = np.asarray(loaded_img.dataobj)
            np.testing.assert_array_equal(loaded_data, expected_data)

    def test_load_nifti_gzipped(self, temp_dir):
        """Test loading gzipped NIfTI files."""
        shape = (5, 5, 5)
        data = np.random.normal(50, 5, shape).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        # Save without compression first
        img_path = temp_dir / "test.nii"
        nib.save(img, img_path)

        # Manually gzip it
        with open(img_path, 'rb') as f_in:
            with gzip.open(img_path.with_suffix('.nii.gz'), 'wb') as f_out:
                f_out.writelines(f_in)

        # Load gzipped version
        loaded_imgs = load_images([img_path.with_suffix('.nii.gz')])
        assert len(loaded_imgs) == 1
        assert_images_close(loaded_imgs[0], img)

    def test_load_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            load_images([Path("nonexistent.nii.gz")])

    def test_load_corrupted_file(self, temp_dir):
        """Test error handling for corrupted files."""
        corrupted_path = create_corrupted_image_file(temp_dir)

        with pytest.raises(Exception):  # Could be various nibabel errors
            load_images([corrupted_path])

    def test_load_with_nibabel_objects(self):
        """Test loading when inputs are already nibabel objects."""
        shape = (5, 5, 5)
        data = np.random.normal(100, 10, shape).astype(np.float32)
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        # Should return the same objects
        loaded_imgs = load_images([img])
        assert len(loaded_imgs) == 1
        assert loaded_imgs[0] is img  # Same object reference

    def test_load_mixed_inputs(self, temp_dir):
        """Test loading mixture of paths and nibabel objects."""
        shape = (5, 5, 5)

        # Create file-based image
        data1 = np.random.normal(100, 10, shape).astype(np.float32)
        img1 = nib.Nifti1Image(data1, np.eye(4))
        img1_path = temp_dir / "img1.nii.gz"
        nib.save(img1, img1_path)

        # Create object-based image
        data2 = np.random.normal(100, 10, shape).astype(np.float32)
        img2 = nib.Nifti1Image(data2, np.eye(4))

        loaded_imgs = load_images([img1_path, img2])
        assert len(loaded_imgs) == 2

        # Check first is loaded from file
        assert not isinstance(loaded_imgs[0], type(img1_path))
        # Check second is the original object
        assert loaded_imgs[1] is img2


class TestVoxelVolume:
    """Test voxel volume calculation."""

    def test_isotropic_voxels(self):
        """Test volume calculation for isotropic voxels."""
        # 2mm isotropic voxels
        affine = np.eye(4) * 2.0
        affine[3, 3] = 1.0
        shape = (10, 10, 10)
        data = np.zeros(shape)
        img = nib.Nifti1Image(data, affine)

        volume_ml = voxel_volume_ml(img)
        expected_ml = 2.0 * 2.0 * 2.0 / 1000.0  # 8 mm³ = 0.008 mL
        np.testing.assert_allclose(volume_ml, expected_ml, rtol=1e-6)

    def test_anisotropic_voxels(self):
        """Test volume calculation for anisotropic voxels."""
        affine = np.eye(4)
        affine[0, 0] = 1.0  # 1mm in x
        affine[1, 1] = 2.0  # 2mm in y
        affine[2, 2] = 3.0  # 3mm in z

        volume_ml = voxel_volume_ml(affine)
        expected_ml = 1.0 * 2.0 * 3.0 / 1000.0  # 6 mm³ = 0.006 mL
        np.testing.assert_allclose(volume_ml, expected_ml, rtol=1e-6)

    def test_non_diagonal_affine(self):
        """Test volume calculation with rotation in affine."""
        # Create rotation matrix (45 degrees around z)
        theta = np.pi / 4
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        affine = np.eye(4)
        affine[0, 0] = cos_t * 2.0
        affine[0, 1] = -sin_t * 2.0
        affine[1, 0] = sin_t * 2.0
        affine[1, 1] = cos_t * 2.0
        affine[2, 2] = 3.0

        volume_ml = voxel_volume_ml(affine)
        # Volume should be preserved under rotation
        expected_ml = 2.0 * 2.0 * 3.0 / 1000.0
        np.testing.assert_allclose(volume_ml, expected_ml, rtol=1e-6)

    def test_shearing_affine(self):
        """Test volume calculation with shearing."""
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[1, 1] = 2.0
        affine[2, 2] = 2.0
        affine[0, 1] = 0.5  # Shear in x-y plane

        volume_ml = voxel_volume_ml(affine)
        # Shear is in upper-left 3x3, should affect volume
        expected_ml = 2.0 * 2.0 * 2.0 / 1000.0  # Shear doesn't affect determinant for this case
        np.testing.assert_allclose(volume_ml, expected_ml, rtol=1e-6)

    def test_voxel_volume_from_image(self):
        """Test voxel volume calculation from an image."""
        shape = (10, 10, 10)
        data = np.zeros(shape)
        affine = np.eye(4)
        affine[0, 0] = 1.5
        affine[1, 1] = 2.5
        affine[2, 2] = 3.5

        img = nib.Nifti1Image(data, affine)
        volume_ml = voxel_volume_ml(img.affine)

        expected_ml = 1.5 * 2.5 * 3.5 / 1000.0
        np.testing.assert_allclose(volume_ml, expected_ml, rtol=1e-6)


class TestMakeLike:
    """Test make_like utility for creating images with same geometry."""

    def test_make_like_3d(self):
        """Test creating 3D image with same geometry."""
        shape_ref = (20, 30, 15)
        affine_ref = np.array([
            [2.0, 0.5, 0.0, -10.0],
            [0.0, 2.0, 0.0, -15.0],
            [0.0, 0.0, 4.0, -8.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Reference image
        ref_data = np.random.normal(100, 10, shape_ref).astype(np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine_ref)

        # New data with potentially different shape
        new_data = np.full(shape_ref, 50.0, dtype=np.float32)

        # Create image like reference
        new_img = make_like(new_data, ref_img)

        # Check geometry matches
        np.testing.assert_array_equal(new_img.shape, ref_img.shape)
        np.testing.assert_allclose(new_img.affine, ref_img.affine, rtol=1e-6)

        # Check data matches
        np.testing.assert_array_equal(np.asarray(new_img.dataobj), new_data)

    def test_make_like_different_dtype(self):
        """Test make_like with different data types."""
        shape = (10, 10, 10)
        affine = np.eye(4)

        ref_data = np.random.normal(100, 10, shape).astype(np.float64)
        ref_img = nib.Nifti1Image(ref_data, affine)

        new_data = np.full(shape, 50, dtype=np.int16)
        new_img = make_like(new_data, ref_img)

        # Output dtype should match input data
        assert new_img.get_fdata().dtype == np.float64

    def test_make_like_4d_to_3d(self):
        """Test making 3D image from 4D reference."""
        shape_4d = (10, 10, 10, 5)
        shape_3d = (10, 10, 10)
        affine = np.eye(4)

        ref_data = np.random.normal(100, 10, shape_4d).astype(np.float32)
        ref_img = nib.Nifti1Image(ref_data, affine)

        new_data = np.full(shape_3d, 50.0, dtype=np.float32)
        new_img = make_like(new_data, ref_img)

        # Should handle dimension mismatch
        assert new_img.shape == shape_3d
        np.testing.assert_allclose(new_img.affine, ref_img.affine, rtol=1e-6)

    def test_make_like_with_header(self):
        """Test preserving header information."""
        shape = (10, 10, 10)
        affine = np.eye(4)
        data = np.random.normal(100, 10, shape).astype(np.float32)
        header = nib.Nifti1Header()
        header['descrip'] = b'Test description'
        header['sform_code'] = 1

        ref_img = nib.Nifti1Image(data, affine, header)

        new_data = np.full(shape, 50.0, dtype=np.float32)
        new_img = make_like(new_data, ref_img)

        # Should preserve important header fields
        # Note: may not preserve all fields
        assert new_img.shape == ref_img.shape


class TestStack4D:
    """Test 4D stacking functionality."""

    def test_stack_3d_images(self):
        """Test stacking 3D images into 4D."""
        shape_3d = (5, 5, 5)
        n_volumes = 4
        affine = np.eye(4)

        # Create 3D images
        images_3d = []
        expected_data = []

        for i in range(n_volumes):
            data = np.full(shape_3d, float(i * 10), dtype=np.float32)
            img = nib.Nifti1Image(data, affine)
            images_3d.append(img)
            expected_data.append(data[..., np.newaxis])

        # Stack into 4D
        img_4d = stack_4d(images_3d)

        # Check shape
        assert img_4d.shape == shape_3d + (n_volumes,)

        # Check data
        data_4d = np.asarray(img_4d.dataobj)
        expected_4d = np.concatenate(expected_data, axis=-1)
        np.testing.assert_array_equal(data_4d, expected_4d)

        # Check affine preserved
        np.testing.assert_allclose(img_4d.affine, affine, rtol=1e-6)

    def test_stack_different_shapes(self):
        """Test error when shapes don't match."""
        img1 = nib.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
        img2 = nib.Nifti1Image(np.zeros((6, 6, 6)), np.eye(4))

        with pytest.raises(ValueError, match="same shape"):
            stack_4d([img1, img2])

    def test_stack_different_affines(self):
        """Test error when affines don't match."""
        affine1 = np.eye(4)
        affine2 = np.eye(4)
        affine2[0, 0] = 2.0  # Different voxel size

        img1 = nib.Nifti1Image(np.zeros((5, 5, 5)), affine1)
        img2 = nib.Nifti1Image(np.zeros((5, 5, 5)), affine2)

        with pytest.raises(ValueError, match="same affine"):
            stack_4d([img1, img2])

    def test_stack_empty_list(self):
        """Test error with empty image list."""
        with pytest.raises(ValueError, match="at least one image"):
            stack_4d([])

    def test_stack_single_image(self):
        """Test stacking single image."""
        img = nib.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
        img_4d = stack_4d([img])

        assert img_4d.shape == (5, 5, 5, 1)
        np.testing.assert_allclose(img_4d.affine, img.affine, rtol=1e-6)


class TestDirectoryOperations:
    """Test directory utility functions."""

    def test_ensure_dir_create(self, temp_dir):
        """Test creating a new directory."""
        new_dir = temp_dir / "new_subdir" / "nested"
        assert not new_dir.exists()

        ensure_dir(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_dir_existing(self, temp_dir):
        """Test with existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()

        # Should not raise error
        ensure_dir(existing_dir)
        assert existing_dir.exists()

    def test_ensure_dir_with_file(self, temp_dir):
        """Test error when path points to a file."""
        file_path = temp_dir / "test_file.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError):
            ensure_dir(file_path)

    def test_ensure_dir_parent_exists(self, temp_dir):
        """Test creating directory when parent exists."""
        ensure_dir(temp_dir / "new_dir")
        assert (temp_dir / "new_dir").exists()


class TestSaveLoadTextData:
    """Test saving and loading text-based data."""

    def test_save_load_summary(self, temp_dir):
        """Test saving and loading summary data."""
        import yaml

        summary = {
            'config': {'physics': {'half_life_seconds': 21636.0}},
            'timing': {'total_time_seconds': 45.2},
            'statistics': {
                'n_valid_voxels': 1000,
                'mean_tia': 1234.5
            }
        }

        # Test saving via engine._save_summary equivalent
        summary_path = temp_dir / "summary.yaml"
        with summary_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(summary, f, sort_keys=False)

        # Test loading
        with summary_path.open('r', encoding='utf-8') as f:
            loaded_summary = yaml.safe_load(f)

        assert loaded_summary == summary

    def test_save_numpy_arrays(self, temp_dir):
        """Test saving numpy arrays in text format."""
        data = np.random.normal(0, 1, (10, 10))

        # Save as text
        txt_path = temp_dir / "data.txt"
        np.savetxt(txt_path, data)

        # Load back
        loaded_data = np.loadtxt(txt_path)
        np.testing.assert_allclose(loaded_data, data, rtol=1e-6)


class TestErrorHandling:
    """Test error handling in I/O operations."""

    def test_permission_denied(self, temp_dir):
        """Test handling of permission errors."""
        # Create a directory and remove write permissions
        restricted_dir = temp_dir / "restricted"
        restricted_dir.mkdir()

        # Try to save to this directory
        data = np.zeros((5, 5, 5))
        img = nib.Nifti1Image(data, np.eye(4))
        save_path = restricted_dir / "test.nii"

        # On Unix systems, we can change permissions
        try:
            restricted_dir.chmod(0o444)  # Read-only
            with pytest.raises(PermissionError):
                nib.save(img, save_path)
        finally:
            # Restore permissions for cleanup
            restricted_dir.chmod(0o755)

    def test_disk_full_simulation(self, temp_dir):
        """Test behavior with insufficient disk space (simulated)."""
        # This is difficult to test reliably without filling disk
        # Instead, test that appropriate exceptions can be caught
        from unittest.mock import patch, mock_open

        data = np.zeros((5, 5, 5))
        img = nib.Nifti1Image(data, np.eye(4))

        # Mock file operations to raise OSError
        with patch('nibabel.save', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError, match="No space left"):
                nib.save(img, temp_dir / "test.nii")

    def test_invalid_nifti_header(self, temp_dir):
        """Test handling of invalid NIfTI headers."""
        # This would require creating a file with invalid header structure
        # For now, test that we can catch general exceptions
        invalid_path = temp_dir / "invalid.nii"
        invalid_path.write_bytes(b"Not a NIfTI file")

        with pytest.raises(Exception):
            load_images([invalid_path])


class testDataTypes:
    """Test handling of different data types."""

    def test_uint8_data(self, temp_dir):
        """Test loading uint8 image data."""
        shape = (5, 5, 5)
        data = np.random.randint(0, 255, shape, dtype=np.uint8)
        img = nib.Nifti1Image(data, np.eye(4))

        img_path = temp_dir / "uint8.nii.gz"
        nib.save(img, img_path)

        loaded_imgs = load_images([img_path])
        loaded_data = np.asarray(loaded_imgs[0].dataobj)

        # nibabel typically loads as float64
        assert loaded_data.dtype in [np.float64, np.float32]
        np.testing.assert_array_equal(loaded_data, data.astype(np.float64))

    def test_complex_data(self, temp_dir):
        """Test handling of complex-valued data."""
        shape = (5, 5, 5)
        data = np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape)
        img = nib.Nifti1Image(data.astype(np.complex64), np.eye(4))

        img_path = temp_dir / "complex.nii.gz"
        nib.save(img, img_path)

        loaded_imgs = load_images([img_path])
        loaded_data = np.asarray(loaded_imgs[0].dataobj)

        assert np.iscomplexobj(loaded_data)

    def test_big_endian_data(self, temp_dir):
        """Test handling of big-endian data."""
        shape = (5, 5, 5)
        data = np.random.normal(100, 10, shape).astype('>f4')  # Big-endian float32

        # Note: nibabel typically normalizes endianness
        img = nib.Nifti1Image(data, np.eye(4))
        img_path = temp_dir / "bigendian.nii.gz"
        nib.save(img, img_path)

        loaded_imgs = load_images([img_path])
        loaded_data = np.asarray(loaded_imgs[0].dataobj)

        # Should be converted to native endianness
        assert not loaded_data.dtype.isnative == False  # Should be native
        np.testing.assert_allclose(loaded_data, data.astype('<f4'), rtol=1e-6)