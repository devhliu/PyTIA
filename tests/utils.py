"""Utility functions for PyTIA testing."""

import numpy as np
import nibabel as nib
from pathlib import Path
from contextlib import contextmanager
import tempfile
import yaml
from typing import Any, Dict

from pytia.config import Config


def assert_images_close(img1, img2,
                      rtol: float = 1e-3, atol: float = 1e-6):
    """Assert that two images are close within tolerance.

    Args:
        img1, img2: Nibabel images to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    # Check shapes
    assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    # Check data
    data1 = np.asarray(img1.dataobj)
    data2 = np.asarray(img2.dataobj)

    np.testing.assert_allclose(data1, data2, rtol=rtol, atol=atol,
                               err_msg=f"Image data differs by more than tolerance")

    # Check affines if present
    if hasattr(img1, 'affine') and hasattr(img2, 'affine'):
        np.testing.assert_allclose(img1.affine, img2.affine, rtol=rtol, atol=atol,
                                   err_msg="Affine matrices differ")


def generate_test_phantom(shape: tuple = (30, 30, 30),
                          tacs: Dict[int, Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    """Generate a digital phantom with known time-activity curves.

    Args:
        shape: 3D shape of the phantom
        tacs: Dictionary defining regions and their TACs
              {region_id: {'mask': 3D bool array, 'times': array, 'activities': array}}

    Returns:
        Dictionary with phantom data
    """
    if tacs is None:
        # Default phantom with simple kinetics
        times = np.array([0.0, 1800.0, 3600.0, 7200.0, 14400.0, 28800.0])

        # Region 1: Fast uptake, fast washout
        tac1 = 100 * np.exp(-times/3600.0) * (1 - np.exp(-times/600.0))

        # Region 2: Slow uptake, slow washout
        tac2 = 80 * (1 - np.exp(-times/7200.0)) * np.exp(-times/21600.0)

        # Region 3: Constant (background)
        tac3 = np.full_like(times, 10.0)

        tacs = {
            1: {
                'times': times,
                'activities': tac1,
                'mask': np.zeros(shape, dtype=bool)
            },
            2: {
                'times': times,
                'activities': tac2,
                'mask': np.zeros(shape, dtype=bool)
            },
            3: {
                'times': times,
                'activities': tac3,
                'mask': np.zeros(shape, dtype=bool)
            }
        }

        # Define spatial regions
        tacs[1]['mask'][:shape[0]//3, :, :] = True
        tacs[2]['mask'][shape[0]//3:2*shape[0]//3, :, :] = True
        tacs[3]['mask'][2*shape[0]//3:, :, :] = True

    # Generate 4D data
    n_timepoints = len(tacs[1]['times'])
    data_4d = np.zeros(shape + (n_timepoints,), dtype=np.float32)

    for region_id, region_data in tacs.items():
        for t_idx, activity in enumerate(region_data['activities']):
            data_4d[..., t_idx][region_data['mask']] = activity

    # Add controlled noise
    noise_level = 0.02  # 2% noise
    noise = np.random.normal(0, noise_level, data_4d.shape) * np.abs(data_4d)
    data_4d = data_4d + noise

    # Convert to list of 3D nibabel images
    affine = np.eye(4)
    affine[:3, :3] = 2.0  # 2mm voxels
    images = []
    for t_idx in range(n_timepoints):
        img = nib.Nifti1Image(data_4d[..., t_idx], affine)
        images.append(img)

    return {
        'images': images,
        'times': tacs[1]['times'],
        'tacs': tacs,
        'ground_truth': data_4d
    }


@contextmanager
def temp_config_file(config_dict: Dict[str, Any]):
    """Context manager for temporary configuration files.

    Args:
        config_dict: Configuration dictionary

    Yields:
        Path to temporary YAML config file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
        temp_path = Path(f.name)

    try:
        yield temp_path
    finally:
        temp_path.unlink(missing_ok=True)


def create_test_images_with_different_properties():
    """Create test images with various properties for robustness testing."""
    base_shape = (10, 10, 10)

    # Different headers
    images = {}

    # Standard float32 image
    data_f32 = np.random.normal(100, 10, base_shape).astype(np.float32)
    images['float32'] = nib.Nifti1Image(data_f32, np.eye(4))

    # Float64 image
    data_f64 = data_f32.astype(np.float64)
    images['float64'] = nib.Nifti1Image(data_f64, np.eye(4))

    # Int16 image
    data_i16 = (data_f32 * 10).astype(np.int16)
    images['int16'] = nib.Nifti1Image(data_i16, np.eye(4))

    # Different affine (non-orthogonal)
    affine_aniso = np.array([
        [2.0, 0.5, 0.0, -10.0],
        [0.0, 2.0, 0.0, -10.0],
        [0.0, 0.0, 5.0, -10.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    images['aniso_affine'] = nib.Nifti1Image(data_f32, affine_aniso)

    # Different dimensions
    data_3d = np.random.normal(100, 10, (15, 20, 25)).astype(np.float32)
    images['different_shape'] = nib.Nifti1Image(data_3d, np.eye(4))

    return images


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function (simplified).

    In practice, use memory_profiler or similar for detailed measurement.
    """
    import psutil
    import os

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    result = func(*args, **kwargs)

    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = mem_after - mem_before

    return result, memory_used


def assert_status_codes(status_img, expected_codes: set[int]):
    """Assert that status codes are within expected set.

    Args:
        status_img: Status code image
        expected_codes: Set of acceptable status codes
    """
    status_data = np.asarray(status_img.dataobj)
    unique_codes = np.unique(status_data)

    for code in unique_codes:
        assert code in expected_codes, f"Unexpected status code: {code}"


def create_corrupted_image_file(temp_dir: Path) -> Path:
    """Create a corrupted NIfTI file for testing error handling.

    Args:
        temp_dir: Temporary directory path

    Returns:
        Path to corrupted file
    """
    corrupted_path = temp_dir / "corrupted.nii.gz"

    # Write invalid data
    with corrupted_path.open('wb') as f:
        f.write(b"This is not a valid NIfTI file")

    return corrupted_path


def compute_tia_analytical(times: np.ndarray, activities: np.ndarray,
                          model_type: str = 'trapz') -> float:
    """Compute analytical TIA for known curves for validation.

    Args:
        times: Time points in seconds
        activities: Activity values
        model_type: Type of integration ('trapz', 'exp', 'gamma')

    Returns:
        Exact TIA value
    """
    dt = np.diff(times, prepend=0)

    if model_type == 'trapz':
        # Trapezoidal integration
        return np.sum(0.5 * dt * (activities[:-1] + activities[1:]))
    elif model_type == 'exp':
        # Assume mono-exponential: A = A0 * exp(-lambda * t)
        from scipy.optimize import curve_fit

        def monoexp(t, A0, lam):
            return A0 * np.exp(-lam * t)

        popt, _ = curve_fit(monoexp, times, activities, p0=[activities[0], 1e-4])
        A0, lam = popt
        return A0 / lam
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class TestDataGenerator:
    """Helper class for generating various test data patterns."""

    @staticmethod
    def rising_tauc(times: np.ndarray, A0: float = 100.0, lambda_uptake: float = 1e-4):
        """Generate rising time-activity curve."""
        return A0 * (1 - np.exp(-lambda_uptake * times))

    @staticmethod
    def hump_tauc(times: np.ndarray, A0: float = 100.0,
                  tpeak: float = 3600.0, alpha: float = 2.0, beta: float = 0.0001):
        """Generate hump-shaped (gamma) time-activity curve."""
        return A0 * (times/tpeak)**alpha * np.exp(-beta * (times - tpeak))

    @staticmethod
    def falling_tauc(times: np.ndarray, A0: float = 100.0,
                     lambda_washout: float = 1e-4):
        """Generate falling time-activity curve."""
        return A0 * np.exp(-lambda_washout * times)