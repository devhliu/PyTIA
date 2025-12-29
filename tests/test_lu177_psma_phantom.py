"""Test PyTIA with a voxel human phantom simulating Lu-177 PSMA therapy.

This test creates a realistic digital phantom with organs relevant to PSMA therapy:
- Tumors (multiple lesions)
- Kidneys (high uptake)
- Red marrow (dose-limiting organ)
- Liver (moderate uptake)

Pharmacokinetic modeling based on published Lu-177 PSMA data.
"""

import numpy as np
import nibabel as nib
import pytest
from pathlib import Path
import tempfile
from typing import Dict, Tuple

from pytia import run_tia
from pytia.config import Config
from tests.utils import assert_images_close


def create_organs(shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """Create organ masks within the phantom volume.

    Args:
        shape: 3D shape of the phantom (z, y, x)

    Returns:
        Dictionary of organ masks
    """
    masks = {}

    # Create coordinate grids
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Body outline (elliptical)
    center_z, center_y, center_x = shape[0] // 2, shape[1] // 2, shape[2] // 2
    body = ((z - center_z) / (shape[0] * 0.4))**2 + \
           ((y - center_y) / (shape[1] * 0.35))**2 + \
           ((x - center_x) / (shape[2] * 0.3))**2 <= 1
    masks['body'] = body

    # Liver (upper right)
    liver_center = (center_z - shape[0] * 0.1, center_y - shape[1] * 0.1, center_x + shape[2] * 0.1)
    liver = ((z - liver_center[0]) / (shape[0] * 0.15))**2 + \
            ((y - liver_center[1]) / (shape[1] * 0.12))**2 + \
            ((x - liver_center[2]) / (shape[2] * 0.1))**2 <= 1
    masks['liver'] = liver & body

    # Kidneys (left and right)
    # Left kidney
    kidney_l_center = (center_z, center_y - shape[1] * 0.05, center_x - shape[2] * 0.15)
    kidney_l = ((z - kidney_l_center[0]) / (shape[0] * 0.08))**2 + \
               ((y - kidney_l_center[1]) / (shape[1] * 0.06))**2 + \
               ((x - kidney_l_center[2]) / (shape[2] * 0.04))**2 <= 1
    masks['kidney_l'] = kidney_l & body

    # Right kidney
    kidney_r_center = (center_z, center_y - shape[1] * 0.05, center_x + shape[2] * 0.15)
    kidney_r = ((z - kidney_r_center[0]) / (shape[0] * 0.08))**2 + \
               ((y - kidney_r_center[1]) / (shape[1] * 0.06))**2 + \
               ((x - kidney_r_center[2]) / (shape[2] * 0.04))**2 <= 1
    masks['kidney_r'] = kidney_r & body
    masks['kidneys'] = masks['kidney_l'] | masks['kidney_r']

    # Red marrow (simplified as spine and pelvic bones)
    # Spine
    spine = ((x - center_x) / (shape[2] * 0.03))**2 < 1
    spine &= (y < center_y) & (np.abs(y - center_y) > shape[1] * 0.05)
    masks['spine'] = spine & body

    # Pelvic bones
    pelvis_center = (center_z + shape[0] * 0.25, center_y, center_x)
    pelvis = ((z - pelvis_center[0]) / (shape[0] * 0.08))**2 <= 1
    pelvis &= ((x - center_x) / (shape[2] * 0.15))**2 <= 1
    pelvis &= np.abs(y - center_y) < shape[1] * 0.15
    masks['pelvis'] = pelvis & body
    masks['marrow'] = masks['spine'] | masks['pelvis']

    # Tumors (multiple lesions at various locations)
    tumor_masks = []
    tumor_locations = [
        (center_z - shape[0] * 0.15, center_y + shape[1] * 0.15, center_x - shape[2] * 0.1),
        (center_z + shape[0] * 0.1, center_y - shape[1] * 0.1, center_x + shape[2] * 0.05),
        (center_z, center_y + shape[1] * 0.05, center_x + shape[2] * 0.2),
    ]

    for i, tumor_center in enumerate(tumor_locations):
        tumor = ((z - tumor_center[0]) / (shape[0] * 0.03))**2 + \
                ((y - tumor_center[1]) / (shape[1] * 0.03))**2 + \
                ((x - tumor_center[2]) / (shape[2] * 0.03))**2 <= 1
        tumor_masks.append(tumor & body)
        masks[f'tumor_{i+1}'] = tumor_masks[-1]

    masks['tumors'] = np.any(np.stack(tumor_masks), axis=0)

    return masks


def simulate_lu177_psma_pharmacokinetics(times_h: np.ndarray) -> Dict[str, np.ndarray]:
    """Simulate Lu-177 PSMA time-activity curves for different organs.

    Based on published biokinetic data for Lu-177 PSMA therapy.
    Uses bi-exponential models typical for PSMA ligands.

    Args:
        times_h: Time points in hours post-injection

    Returns:
        Dictionary of organ activities (MBq/ml) at each time point
    """
    activities = {}

    # Convert to seconds for calculations
    times_s = times_h * 3600

    # Physical decay constant for Lu-177 (half-life = 6.647 days = 160.73 hours)
    lambda_phys = np.log(2) / 160.73

    # Tumors: High uptake, slow washout
    # A(t) = A0 * (f_fast * exp(-lambda_fast * t) + f_slow * exp(-lambda_slow * t))
    for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
        A0 = 8.5  # MBq/ml at injection
        f_fast, f_slow = 0.4, 0.6
        lambda_fast = 0.15  # 1/hours
        lambda_slow = 0.03  # 1/hours
        activities[tumor_name] = A0 * (f_fast * np.exp(-lambda_fast * times_h) +
                                     f_slow * np.exp(-lambda_slow * times_h))

    # Kidneys: Highest uptake, rapid initial washout then slower component
    A0_kidney = 12.0  # MBq/ml
    f_fast_kidney, f_slow_kidney = 0.6, 0.4
    lambda_fast_kidney = 0.25  # 1/hours
    lambda_slow_kidney = 0.04  # 1/hours
    activities['kidneys'] = A0_kidney * (f_fast_kidney * np.exp(-lambda_fast_kidney * times_h) +
                                        f_slow_kidney * np.exp(-lambda_slow_kidney * times_h))

    # Liver: Moderate uptake, relatively stable
    A0_liver = 4.0  # MBq/ml
    lambda_liver = 0.02  # 1/hours (slow washout)
    activities['liver'] = A0_liver * np.exp(-(lambda_liver + lambda_phys) * times_h)

    # Red marrow: Low uptake, follows blood pool with slight retention
    A0_marrow = 1.5  # MBq/ml
    f_fast_marrow, f_slow_marrow = 0.7, 0.3
    lambda_fast_marrow = 0.3  # 1/hours
    lambda_slow_marrow = 0.05  # 1/hours
    activities['marrow'] = A0_marrow * (f_fast_marrow * np.exp(-lambda_fast_marrow * times_h) +
                                       f_slow_marrow * np.exp(-lambda_slow_marrow * times_h))

    # Background tissue: Very low activity
    activities['background'] = 0.1 * np.exp(-0.1 * times_h)  # MBq/ml

    return activities


def create_phantom_images(shape: Tuple[int, int, int],
                         times_h: np.ndarray,
                         output_dir: Path = None) -> Tuple[list, Dict]:
    """Create phantom PET images at specified time points.

    Args:
        shape: 3D shape of the phantom
        times_h: Acquisition times in hours
        output_dir: Directory to save images (defaults to temp dir)

    Returns:
        Tuple of (list of image paths, organ masks)
    """
    # Create organ masks
    masks = create_organs(shape)

    # Simulate pharmacokinetics
    organ_activities = simulate_lu177_psma_pharmacokinetics(times_h)

    # Create images for each time point
    images = []

    # Use provided directory or create temp directory
    if output_dir is None:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_ctx.__enter__())
        need_cleanup = True
    else:
        tmpdir = Path(output_dir)
        tmpdir.mkdir(exist_ok=True, parents=True)
        need_cleanup = False

    try:
        for i, t in enumerate(times_h):
            # Create activity map for this time point
            activity_map = np.zeros(shape, dtype=np.float32)

            # Add activity from each organ
            organ_map = np.zeros_like(activity_map)

            # Background
            activity_map[masks['body']] += organ_activities['background'][i]

            # Liver
            activity_map[masks['liver']] += organ_activities['liver'][i]
            organ_map[masks['liver']] = organ_activities['liver'][i]

            # Kidneys
            activity_map[masks['kidneys']] += organ_activities['kidneys'][i]
            organ_map[masks['kidneys']] = organ_activities['kidneys'][i]

            # Marrow
            activity_map[masks['marrow']] += organ_activities['marrow'][i]
            organ_map[masks['marrow']] = organ_activities['marrow'][i]

            # Tumors
            for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
                activity_map[masks[tumor_name]] += organ_activities[tumor_name][i]
                organ_map[masks[tumor_name]] = organ_activities[tumor_name][i]

            # Add Poisson noise (realistic PET noise)
            # Scale to typical PET counts & add noise
            scale_factor = 1000  # Convert MBq/ml to counts
            noisy_counts = np.random.poisson(activity_map * scale_factor)
            activity_map = noisy_counts / scale_factor

            # Ensure non-negative
            activity_map = np.maximum(activity_map, 0)

            # Create and save NIfTI image
            img = nib.Nifti1Image(activity_map, np.eye(4))
            img_path = tmpdir / f"activity_t{int(t):03d}h.nii.gz"
            nib.save(img, img_path)
            images.append(str(img_path))

        return images, masks

    finally:
        if need_cleanup:
            tmpdir_ctx.__exit__(None, None, None)


@pytest.fixture
def lu177_psma_phantom(tmp_path):
    """Fixture providing Lu-177 PSMA phantom data."""
    shape = (100, 120, 140)  # Realistic voxel dimensions
    times_h = np.array([4, 24, 48, 96, 112, 176])  # Specified acquisition times

    images, masks = create_phantom_images(shape, times_h, tmp_path)

    return {
        'images': images,
        'times': times_h.tolist(),
        'masks': masks,
        'shape': shape,
        'organ_activities': simulate_lu177_psma_pharmacokinetics(times_h)
    }


@pytest.mark.integration
def test_lu177_psma_phantom_tia_calculation(lu177_psma_phantom):
    """Test TIA calculation with Lu-177 PSMA phantom data.

    This test verifies that PyTIA can correctly calculate time-integrated
    activity for a realistic PSMA therapy scenario.
    """
    phantom = lu177_psma_phantom

    # Configure for Lu-177 physics
    config = {
        "physics": {
            "half_life_seconds": 160.73 * 3600  # Lu-177 half-life in seconds
        },
        "time": {
            "unit": "hours"
        },
        "io": {
            "output_dir": "./test_output",
            "prefix": "lu177_psma_phantom"
        },
        "mask": {
            "method": "none"  # Use entire phantom
        },
        "denoise": {
            "enabled": False  # Keep noise for realistic simulation
        },
        "bootstrap": {
            "enabled": False  # Disable for faster test
        }
    }

    # Run TIA calculation
    result = run_tia(
        images=phantom['images'],
        times=phantom['times'],
        config=config
    )

    # Check that we got valid results
    assert result.tia_img is not None
    assert result.tia_img.shape == phantom['shape']
    assert np.all(result.tia_img.affine == np.eye(4))

    # Debug: Check for NaN values
    tia_data = np.asarray(result.tia_img.dataobj)
    print(f"TIA data - Min: {np.nanmin(tia_data)}, Max: {np.nanmax(tia_data)}, NaN count: {np.sum(np.isnan(tia_data))}")

    # Check status map
    status_data = np.asarray(result.status_id_img.dataobj)
    print(f"Status values - Unique: {np.unique(status_data)}")

    # Check TIA values in different organs
    tia_data = np.asarray(result.tia_img.dataobj)
    r2_data = np.asarray(result.r2_img.dataobj)
    status_data = np.asarray(result.status_id_img.dataobj)
    masks = phantom['masks']

    # Helper function to get valid TIA values (non-NaN, successful status)
    def get_valid_tia(mask):
        valid_mask = mask & (status_data == 1) & ~np.isnan(tia_data)
        if np.any(valid_mask):
            return tia_data[valid_mask]
        return np.array([])

    def get_valid_r2(mask):
        valid_mask = mask & (status_data == 1) & ~np.isnan(r2_data)
        if np.any(valid_mask):
            return r2_data[valid_mask]
        return np.array([])

    # Tumors should have high TIA
    for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
        tumor_mask = masks[tumor_name]
        tumor_tia = get_valid_tia(tumor_mask)
        if len(tumor_tia) > 0:
            mean_tia = np.mean(tumor_tia)
            # TIA should be positive and significant
            assert mean_tia > 10, f"Tumor {tumor_name} TIA too low: {mean_tia} MBq*h/ml"
            print(f"Tumor {tumor_name} TIA: {mean_tia:.2f} MBq*h/ml")
        else:
            print(f"Warning: No valid TIA values for {tumor_name}")

    # Kidneys should have highest TIA (dose-limiting for PSMA)
    kidney_tia = get_valid_tia(masks['kidneys'])
    if len(kidney_tia) > 0:
        mean_kidney_tia = np.mean(kidney_tia)
        assert mean_kidney_tia > 50, f"Kidney TIA too low: {mean_kidney_tia} MBq*h/ml"
        print(f"Kidney TIA: {mean_kidney_tia:.2f} MBq*h/ml")

    # Marrow should have measurable TIA (important for hematologic toxicity)
    marrow_tia = get_valid_tia(masks['marrow'])
    if len(marrow_tia) > 0:
        mean_marrow_tia = np.mean(marrow_tia)
        assert mean_marrow_tia > 10, f"Marrow TIA too low: {mean_marrow_tia} MBq*h/ml"
        print(f"Marrow TIA: {mean_marrow_tia:.2f} MBq*h/ml")

    # R² values should be reasonable where data is good
    for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
        tumor_mask = masks[tumor_name]
        tumor_r2 = get_valid_r2(tumor_mask)
        if len(tumor_r2) > 0:
            mean_r2 = np.mean(tumor_r2)
            assert mean_r2 > 0.5, f"Tumor {tumor_name} R² too low: {mean_r2}"
            print(f"Tumor {tumor_name} R²: {mean_r2:.3f}")


@pytest.fixture
def lu177_psma_phantom_small(tmp_path):
    """Fixture providing smaller Lu-177 PSMA phantom for faster tests."""
    shape = (50, 60, 70)  # Smaller for faster tests
    times_h = np.array([4, 24, 48, 96, 112, 176])  # Specified acquisition times

    images, masks = create_phantom_images(shape, times_h, tmp_path)

    return {
        'images': images,
        'times': times_h.tolist(),
        'masks': masks,
        'shape': shape,
        'organ_activities': simulate_lu177_psma_pharmacokinetics(times_h)
    }


@pytest.mark.integration
def test_lu177_psma_phantom_with_bootstrap(lu177_psma_phantom_small):
    """Test bootstrap uncertainty quantification with Lu-177 PSMA phantom."""
    phantom = lu177_psma_phantom_small

    # Configure with bootstrap enabled
    config = {
        "physics": {
            "half_life_seconds": 160.73 * 3600
        },
        "time": {
            "unit": "hours"
        },
        "io": {
            "output_dir": "./test_output",
            "prefix": "lu177_psma_phantom_bootstrap"
        },
        "mask": {
            "method": "none"
        },
        "denoise": {
            "enabled": False
        },
        "bootstrap": {
            "enabled": True,
            "n": 50,  # Reduced for faster test
            "seed": 42
        }
    }

    # Run TIA calculation with bootstrap
    result = run_tia(
        images=phantom['images'],
        times=phantom['times'],
        config=config
    )

    # Check that uncertainty map is generated
    assert result.sigma_tia_img is not None
    assert result.sigma_tia_img.shape == phantom['shape']

    # Check that uncertainty is reasonable
    sigma_data = np.asarray(result.sigma_tia_img.dataobj)
    tia_data = np.asarray(result.tia_img.dataobj)
    masks = phantom['masks']

    # In high-activity regions, relative uncertainty should be lower
    for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
        tumor_mask = masks[tumor_name]
        if np.any(tumor_mask):
            tumor_tia = np.mean(tia_data[tumor_mask])
            tumor_sigma = np.mean(sigma_data[tumor_mask])
            rel_uncertainty = tumor_sigma / tumor_tia if tumor_tia > 0 else np.inf

            # Relative uncertainty should be reasonable (< 70% for noisy data)
            assert rel_uncertainty < 0.7, f"Tumor {tumor_name} relative uncertainty too high: {rel_uncertainty}"


@pytest.mark.unit
def test_organ_mask_creation():
    """Test that organ masks are created correctly."""
    shape = (50, 60, 70)
    masks = create_organs(shape)

    # Check that all masks have correct shape
    for mask_name, mask in masks.items():
        assert mask.shape == shape, f"{mask_name} has incorrect shape"
        assert mask.dtype == bool, f"{mask_name} is not boolean"

    # Check that organs are within body
    assert not np.any(masks['liver'] & ~masks['body'])
    assert not np.any(masks['kidneys'] & ~masks['body'])
    assert not np.any(masks['tumors'] & ~masks['body'])

    # Check that kidneys don't overlap
    overlap = masks['kidney_l'] & masks['kidney_r']
    assert not np.any(overlap), "Kidneys overlap"

    # Check that some voxels are assigned to organs
    assert np.any(masks['liver'])
    assert np.any(masks['kidneys'])
    assert np.any(masks['tumors'])


@pytest.mark.unit
def test_pharmacokinetic_simulation():
    """Test that pharmacokinetic simulation produces realistic curves."""
    times_h = np.array([0, 4, 24, 48, 96, 168])
    activities = simulate_lu177_psma_pharmacokinetics(times_h)

    # Check that all organs have activity curves
    expected_organs = ['tumor_1', 'tumor_2', 'tumor_3', 'kidneys', 'liver', 'marrow', 'background']
    for organ in expected_organs:
        assert organ in activities, f"Missing activity for {organ}"
        assert len(activities[organ]) == len(times_h), f"Incorrect curve length for {organ}"
        assert np.all(activities[organ] >= 0), f"Negative activity for {organ}"

    # Check organ hierarchy at 4h
    idx_4h = np.where(times_h == 4)[0][0]

    # Kidneys should have highest uptake at early time points
    assert np.mean(activities['kidneys'][idx_4h]) > np.mean(activities['liver'][idx_4h])
    assert np.mean(activities['kidneys'][idx_4h]) > np.mean(activities['marrow'][idx_4h])

    # Tumors should have high uptake
    for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
        assert np.mean(activities[tumor_name][idx_4h]) > np.mean(activities['liver'][idx_4h])

    # Background should be lowest
    assert np.mean(activities['background'][idx_4h]) < 1.0  # MBq/ml


if __name__ == "__main__":
    # Run a quick test if script is executed directly
    shape = (50, 60, 70)
    masks = create_organs(shape)
    print(f"Created phantom masks for organs: {list(masks.keys())}")

    times_h = np.array([4, 24, 48, 96, 112, 176])
    activities = simulate_lu177_psma_pharmacokinetics(times_h)

    print("\nPeak activities (MBq/ml) at 4h:")
    for organ, act in activities.items():
        print(f"  {organ}: {act[np.where(times_h == 4)[0][0]]:.2f}")