"""Pytest configuration and shared fixtures for PyTIA tests."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import nibabel as nib

from pytia.config import default_config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def affine_matrix():
    """Standard affine matrix for test images."""
    return np.array([
        [2.0, 0.0, 0.0, -50.0],
        [0.0, 2.0, 0.0, -50.0],
        [0.0, 0.0, 2.0, -50.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


@pytest.fixture
def synthetic_pet_data(affine_matrix):
    """Generate synthetic PET time-series data for testing.

    Returns:
        tuple: (images, times) where images is a list of nibabel images
               and times is a numpy array of acquisition times in seconds
    """
    shape = (20, 20, 20)  # Small test volume
    times = np.array([0.0, 3600.0, 7200.0, 14400.0, 28800.0])  # hours 0, 1, 2, 4, 8

    # Create realistic time-activity curves
    # Different regions with different kinetics
    images = []

    for t in times:
        data = np.zeros(shape, dtype=np.float32)

        # Region 1: Rapid uptake then washout (organ)
        mask1 = data[:10, :10, :] < 1  # First octant
        tacs = 100 * np.exp(-t/7200) * (1 - np.exp(-t/1800))
        data[mask1] = tacs

        # Region 2: Slow uptake (bone)
        mask2 = (data[10:, :10, :] < 1) & (~mask1)
        tacs = 50 * (1 - np.exp(-t/14400))
        data[mask2] = tacs

        # Region 3: Background noise
        mask3 = np.logical_not(np.logical_or(mask1, mask2))
        data[mask3] = np.random.normal(5, 2, mask3.sum())

        img = nib.Nifti1Image(data, affine_matrix)
        images.append(img)

    return images, times


@pytest.fixture
def single_timepoint_data(affine_matrix):
    """Generate single timepoint PET data for STP testing."""
    shape = (10, 10, 10)
    value = 100.0
    data = np.full(shape, value, dtype=np.float32)
    return nib.Nifti1Image(data, affine_matrix)


@pytest.fixture
def multi_timepoint_config():
    """Base configuration for multi-timepoint analysis."""
    config = default_config()
    config.update({
        "physics": {
            "half_life_seconds": 21636.0,  # Tc-99m
            "enforce_lambda_ge_phys": True,
        },
        "time": {
            "unit": "seconds",
            "sort_timepoints": True,
        },
        "mask": {
            "mode": "otsu",
            "min_fraction_of_max": 0.02,
        },
        "denoise": {
            "enabled": True,
            "method": "masked_gaussian",
            "sigma_vox": 1.2,
        },
        "bootstrap": {
            "enabled": False,
            "n": 100,
            "seed": 42,
        }
    })
    return config


@pytest.fixture
def single_timepoint_config():
    """Base configuration for single-timepoint analysis."""
    config = default_config()
    config.update({
        "physics": {
            "half_life_seconds": 21636.0,  # Tc-99m
        },
        "single_time": {
            "enabled": True,
            "method": "phys",
        },
        "time": {
            "unit": "seconds",
        },
    })
    return config


@pytest.fixture
def roi_mask(affine_matrix):
    """Create a test ROI mask with multiple labels."""
    shape = (20, 20, 20)
    data = np.zeros(shape, dtype=np.int32)

    # Create three regions
    data[0:7, :, :] = 1  # Label 1
    data[7:14, :, :] = 2  # Label 2
    data[14:, :, :] = 3   # Label 3

    return nib.Nifti1Image(data, affine_matrix)


@pytest.fixture
def noise_floor_data(affine_matrix):
    """Generate PET data with controlled noise floor."""
    shape = (10, 10, 10)
    times = np.array([0.0, 3600.0, 7200.0])
    images = []

    for t in times:
        # Signal plus noise
        signal = 50 * np.exp(-t/5000)
        noise_floor = 5.0
        data = np.random.normal(noise_floor, 1.0, shape).astype(np.float32)

        # Add signal to center region
        center = slice(3, 7), slice(3, 7), slice(3, 7)
        data[center] = np.random.normal(signal + noise_floor, 2.0, (4, 4, 4))

        img = nib.Nifti1Image(data, affine_matrix)
        images.append(img)

    return images, times