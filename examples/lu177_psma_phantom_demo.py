#!/usr/bin/env python
"""
Demonstration of Lu-177 PSMA phantom simulation and TIA calculation.

This example creates a realistic voxel phantom with tumors, kidneys, liver,
and red marrow, then simulates PSMA PET acquisitions at therapeutic
time points and calculates time-integrated activity maps.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from typing import Dict, Tuple

from pytia import run_tia


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
                         output_dir: Path) -> Tuple[list, Dict]:
    """Create phantom PET images at specified time points.

    Args:
        shape: 3D shape of the phantom
        times_h: Acquisition times in hours
        output_dir: Directory to save images

    Returns:
        Tuple of (list of image paths, organ masks)
    """
    # Create organ masks
    masks = create_organs(shape)

    # Simulate pharmacokinetics
    organ_activities = simulate_lu177_psma_pharmacokinetics(times_h)

    # Create images for each time point
    images = []

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for i, t in enumerate(times_h):
        # Create activity map for this time point
        activity_map = np.zeros(shape, dtype=np.float32)

        # Add activity from each organ
        # Background
        activity_map[masks['body']] += organ_activities['background'][i]

        # Liver
        activity_map[masks['liver']] += organ_activities['liver'][i]

        # Kidneys
        activity_map[masks['kidneys']] += organ_activities['kidneys'][i]

        # Marrow
        activity_map[masks['marrow']] += organ_activities['marrow'][i]

        # Tumors
        for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
            activity_map[masks[tumor_name]] += organ_activities[tumor_name][i]

        # Add Poisson noise (realistic PET noise)
        # Scale to typical PET counts & add noise
        scale_factor = 1000  # Convert MBq/ml to counts
        noisy_counts = np.random.poisson(activity_map * scale_factor)
        activity_map = noisy_counts / scale_factor

        # Ensure non-negative
        activity_map = np.maximum(activity_map, 0)

        # Create and save NIfTI image
        img = nib.Nifti1Image(activity_map, np.eye(4))
        img_path = output_dir / f"activity_t{int(t):03d}h.nii.gz"
        nib.save(img, img_path)
        images.append(str(img_path))

    return images, masks


def visualize_phantom(masks, shape, activity_map=None, slice_idx=None):
    """Visualize the phantom anatomy and/or activity.

    Args:
        masks: Dictionary of organ masks
        shape: 3D shape of the phantom
        activity_map: Optional activity distribution to overlay
        slice_idx: Which slice to show (defaults to middle)
    """
    if slice_idx is None:
        slice_idx = shape[0] // 2

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Anatomy slices
    organs_to_show = [
        ('body', 'Body'),
        ('liver', 'Liver'),
        ('kidneys', 'Kidneys'),
        ('marrow', 'Red Marrow'),
        ('tumors', 'Tumors'),
    ]

    for i, (organ_name, label) in enumerate(organs_to_show):
        ax = axes[i]
        slice_data = masks[organ_name][slice_idx, :, :]
        ax.imshow(slice_data.T, cmap='gray', origin='lower')
        ax.set_title(f'{label} (slice {slice_idx})')
        ax.axis('off')

    plt.suptitle('Lu-177 PSMA Phantom - Organ Anatomy', fontsize=14)
    plt.tight_layout()
    return fig


def plot_pharmacokinetics():
    """Plot the simulated time-activity curves."""
    times_h = np.array([0, 4, 24, 48, 96, 112, 176])
    activities = simulate_lu177_psma_pharmacokinetics(times_h)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each organ
    for organ_name in ['tumor_1', 'kidneys', 'liver', 'marrow']:
        if organ_name in activities:
            label = organ_name.replace('_', ' ').title()
            ax.plot(times_h, activities[organ_name], 'o-', linewidth=2, label=label)

    ax.set_xlabel('Time post-injection (hours)', fontsize=12)
    ax.set_ylabel('Activity (MBq/ml)', fontsize=12)
    ax.set_title('Lu-177 PSMA Pharmacokinetic Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    return fig


def main():
    """Run the Lu-177 PSMA phantom demonstration."""
    print("=" * 70)
    print("Lu-177 PSMA Phantom Simulation and TIA Calculation Demo")
    print("=" * 70)

    # Create phantom
    print("\n1. Creating voxel phantom...")
    shape = (100, 120, 140)  # Realistic phantom size
    times_h = np.array([4, 24, 48, 96, 112, 176])  # Clinical acquisition times

    # Create organ masks
    masks = create_organs(shape)
    print(f"   Created phantom with shape: {shape}")
    print(f"   Organ volumes (voxels):")
    for organ_name in ['liver', 'kidneys', 'marrow', 'tumors']:
        n_voxels = np.sum(masks[organ_name])
        print(f"     {organ_name.title()}: {n_voxels}")

    # Visualize anatomy
    if 'DISPLAY' in plt.rcParams:
        print("\n2. Visualizing phantom anatomy...")
        fig = visualize_phantom(masks, shape)
        plt.show()

    # Plot pharmacokinetics
    if 'DISPLAY' in plt.rcParams:
        print("\n3. Plotting simulated pharmacokinetics...")
        fig = plot_pharmacokinetics()
        plt.show()

    # Create PET images at specified time points
    print("\n4. Simulating PET acquisitions...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        images, _ = create_phantom_images(shape, times_h, tmpdir)

        print(f"   Created {len(images)} PET images at times: {times_h} h")

        # Calculate TIA
        print("\n5. Calculating time-integrated activity...")
        config = {
            "physics": {
                "half_life_seconds": 160.73 * 3600  # Lu-177
            },
            "time": {
                "unit": "hours"
            },
            "io": {
                "output_dir": str(tmpdir),
                "prefix": "lu177_psma_demo"
            },
            "mask": {
                "method": "none"
            },
            "bootstrap": {
                "enabled": False
            }
        }

        result = run_tia(
            images=images,
            times=times_h.tolist(),
            config=config
        )

        # Analyze results
        print("\n6. Analyzing TIA results...")
        tia_data = np.asarray(result.tia_img.dataobj)
        status_data = np.asarray(result.status_id_img.dataobj)

        # Helper to get valid TIA values
        def get_valid_tia(mask):
            valid = mask & (status_data == 1) & ~np.isnan(tia_data)
            return tia_data[valid] if np.any(valid) else np.array([])

        # Report organ TIA values
        print("\n   Mean TIA values (MBq*h/ml):")
        for organ_name in ['liver', 'kidneys', 'marrow', 'tumors']:
            organ_tia = get_valid_tia(masks[organ_name])
            if len(organ_tia) > 0:
                mean_tia = np.mean(organ_tia)
                std_tia = np.std(organ_tia)
                print(f"     {organ_name.title()}: {mean_tia:.1f} ± {std_tia:.1f}")

        # Report fit quality
        r2_data = np.asarray(result.r2_img.dataobj)
        print("\n   Mean R² values:")
        for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
            if tumor_name in masks:
                tumor_mask = masks[tumor_name]
                valid = tumor_mask & (status_data == 1) & ~np.isnan(r2_data)
                if np.any(valid):
                    r2_vals = r2_data[valid]
                    print(f"     {tumor_name.title()}: {np.mean(r2_vals):.3f}")

        # Save results for inspection
        print(f"\n7. Results saved to: {result.output_paths}")
        for output_type, path in result.output_paths.items():
            print(f"     {output_type}: {path.name}")

    print("\n" + "=" * 70)
    print("Demo complete! The phantom demonstrates:")
    print("  • Realistic organ anatomies for PSMA therapy")
    print("  • Physiologically-based Lu-177 PSMA pharmacokinetics")
    print("  • Multi-timepoint TIA calculation")
    print("  • Integration with PyTIA's analysis pipeline")
    print("=" * 70)


if __name__ == "__main__":
    main()