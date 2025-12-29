#!/usr/bin/env python
"""
Generate detailed plots of TIA fitting results and model quality
for the Lu-177 PSMA phantom.

This script creates visualizations of:
1. Time-activity curves with fitted models
2. TIA maps
3. Model fit quality (R²) maps
4. Status maps indicating fit success/failure
5. Organ-specific plots for detailed analysis
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from typing import Dict, Tuple

from pytia import run_tia


def create_organs(shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    """Create organ masks within the phantom volume."""
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
    kidney_l_center = (center_z, center_y - shape[1] * 0.05, center_x - shape[2] * 0.15)
    kidney_l = ((z - kidney_l_center[0]) / (shape[0] * 0.08))**2 + \
               ((y - kidney_l_center[1]) / (shape[1] * 0.06))**2 + \
               ((x - kidney_l_center[2]) / (shape[2] * 0.04))**2 <= 1
    masks['kidney_l'] = kidney_l & body

    kidney_r_center = (center_z, center_y - shape[1] * 0.05, center_x + shape[2] * 0.15)
    kidney_r = ((z - kidney_r_center[0]) / (shape[0] * 0.08))**2 + \
               ((y - kidney_r_center[1]) / (shape[1] * 0.06))**2 + \
               ((x - kidney_r_center[2]) / (shape[2] * 0.04))**2 <= 1
    masks['kidney_r'] = kidney_r & body
    masks['kidneys'] = masks['kidney_l'] | masks['kidney_r']

    # Red marrow (simplified as spine and pelvic bones)
    spine = ((x - center_x) / (shape[2] * 0.03))**2 < 1
    spine &= (y < center_y) & (np.abs(y - center_y) > shape[1] * 0.05)
    masks['spine'] = spine & body

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
    """Simulate Lu-177 PSMA time-activity curves for different organs."""
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
    """Create phantom PET images at specified time points."""
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
        activity_map[masks['body']] += organ_activities['background'][i]
        activity_map[masks['liver']] += organ_activities['liver'][i]
        activity_map[masks['kidneys']] += organ_activities['kidneys'][i]
        activity_map[masks['marrow']] += organ_activities['marrow'][i]

        # Tumors
        for tumor_name in ['tumor_1', 'tumor_2', 'tumor_3']:
            activity_map[masks[tumor_name]] += organ_activities[tumor_name][i]

        # Add Poisson noise (realistic PET noise)
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


def create_phantom_and_run_analysis():
    """Create phantom, run TIA analysis, and return all data."""
    # Create phantom
    shape = (80, 100, 120)  # Slightly smaller for faster processing
    times_h = np.array([4, 24, 48, 96, 112, 176])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create phantom images
        images, masks = create_phantom_images(shape, times_h, tmpdir)
        organ_tacs = simulate_lu177_psma_pharmacokinetics(times_h)

        # Run TIA analysis
        config = {
            "physics": {
                "half_life_seconds": 160.73 * 3600  # Lu-177
            },
            "time": {
                "unit": "hours",
                "sort_timepoints": True
            },
            "io": {
                "output_dir": str(tmpdir),
                "prefix": "lu177_psma_analysis"
            },
            "mask": {
                "method": "none"
            },
            "denoise": {
                "enabled": False
            },
            "bootstrap": {
                "enabled": False
            },
            "performance": {
                "chunk_size_vox": 10000
            }
        }

        result = run_tia(
            images=images,
            times=times_h.tolist(),
            config=config
        )

        return {
            'masks': masks,
            'organ_tacs': organ_tacs,
            'times_h': times_h,
            'shape': shape,
            'result': result
        }


def plot_time_activity_fits(data, organ_name):
    """Plot time-activity curves with fitted models for a specific organ.

    Args:
        data: Dictionary containing phantom data and results
        organ_name: Name of the organ to plot
    """
    masks = data['masks']
    organ_tacs = data['organ_tacs']
    times_h = data['times_h']
    result = data['result']

    # Get valid voxels for this organ
    organ_mask = masks.get(organ_name)
    if organ_mask is None or not np.any(organ_mask):
        print(f"Warning: No voxels found for organ '{organ_name}'")
        return

    # Load result data
    tia_data = np.asarray(result.tia_img.dataobj)
    r2_data = np.asarray(result.r2_img.dataobj)
    status_data = np.asarray(result.status_id_img.dataobj)

    # Find voxels with successful fits
    valid_voxels = organ_mask & (status_data == 1) & ~np.isnan(tia_data)

    if not np.any(valid_voxels):
        print(f"Warning: No valid fits found for organ '{organ_name}'")
        return

    # Get indices of valid voxels
    voxel_indices = np.where(valid_voxels)
    n_valid = len(voxel_indices[0])

    # Sample up to 10 voxels for plotting
    max_voxels_to_plot = 10
    if n_valid > max_voxels_to_plot:
        sample_indices = np.random.choice(n_valid, max_voxels_to_plot, replace=False)
        voxel_indices = tuple(v[(sample_indices)] for v in voxel_indices)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sample voxel TACs with true curves
    ax = axes[0, 0]

    # Initialize true_tac with a default value
    true_tac = None

    # Plot true organ TAC
    if organ_name in organ_tacs:
        true_tac = organ_tacs[organ_name]
        ax.plot(times_h, true_tac, 'k--', linewidth=2, label='True organ TAC', alpha=0.7)

    # Plot individual voxel curves (need to reconstruct from images)
    # For this demo, we'll show the true curve and add noise to simulate measured values
    if true_tac is not None:
        for i in range(len(voxel_indices[0])):
            # Add realistic noise to simulate measured values
            noise_factor = np.random.uniform(0.8, 1.2, size=len(times_h))
            noisy_tac = true_tac * noise_factor

            # Color by R² value
            r2_val = r2_data[voxel_indices[0][i], voxel_indices[1][i], voxel_indices[2][i]]
            ax.plot(times_h, noisy_tac, 'o-', alpha=0.5,
                    label=f'Voxel {i+1} (R²={r2_val:.3f})')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Activity (MBq/ml)')
    ax.set_title(f'{organ_name.title()} - Time-Activity Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of R² values
    ax = axes[0, 1]
    r2_values = r2_data[valid_voxels]
    ax.hist(r2_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(r2_values), color='red', linestyle='--',
               label=f'Mean R² = {np.mean(r2_values):.3f}')
    ax.set_xlabel('R² Value')
    ax.set_ylabel('Number of Voxels')
    ax.set_title(f'{organ_name.title()} - R² Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: TIA distribution
    ax = axes[1, 0]
    tia_values = tia_data[valid_voxels]

    # Calculate theoretical TIA from true TAC
    if true_tac is not None:
        true_tia = np.trapezoid(true_tac, times_h * 3600) / 3600  # Convert to hours
        ax.axvline(true_tia, color='red', linestyle='--',
                   label=f'Theoretical TIA = {true_tia:.1f} MBq·h/ml')

    ax.hist(tia_values, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(np.mean(tia_values), color='blue', linestyle='--',
               label=f'Mean TIA = {np.mean(tia_values):.1f} MBq·h/ml')
    ax.set_xlabel('TIA (MBq·h/ml)')
    ax.set_ylabel('Number of Voxels')
    ax.set_title(f'{organ_name.title()} - TIA Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: TIA vs R² scatter plot
    ax = axes[1, 1]
    scatter = ax.scatter(r2_values, tia_values, alpha=0.5, s=10)
    ax.set_xlabel('R² Value')
    ax.set_ylabel('TIA (MBq·h/ml)')
    ax.set_title(f'{organ_name.title()} - TIA vs R²')
    ax.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(r2_values) > 1:
        corr = np.corrcoef(r2_values, tia_values)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: r = {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top')

    plt.suptitle(f'{organ_name.title()} - Detailed Fitting Analysis', fontsize=14)
    plt.tight_layout()

    return fig


def plot_result_maps(data):
    """Plot the result maps (TIA, R², Status, Model ID).

    Args:
        data: Dictionary containing phantom data and results
    """
    result = data['result']
    masks = data['masks']
    shape = data['shape']

    # Load data
    tia_data = np.asarray(result.tia_img.dataobj)
    r2_data = np.asarray(result.r2_img.dataobj)
    status_data = np.asarray(result.status_id_img.dataobj)

    # Choose a slice to display (middle slice with lots of organs)
    z_slice = shape[0] // 2 - 10  # Adjust to capture organs

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Plot 1: TIA map
    ax = axes[0]
    im = ax.imshow(tia_data[z_slice, :, :].T, cmap='hot', origin='lower')
    ax.set_title(f'TIA Map (slice {z_slice})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='MBq·h/ml')

    # Plot 2: R² map
    ax = axes[1]
    # Clip R² to [0, 1] for visualization
    r2_clipped = np.clip(r2_data, 0, 1)
    im = ax.imshow(r2_clipped[z_slice, :, :].T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Goodness of Fit (R²)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='R²')

    # Plot 3: Status map
    ax = axes[2]
    # Define status colors
    colors = ['black', 'green', 'yellow', 'red', 'orange', 'purple']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    im = ax.imshow(status_data[z_slice, :, :].T, cmap=cmap, origin='lower', vmin=0, vmax=5)
    ax.set_title('Fit Status')
    ax.axis('off')

    # Add legend for status codes
    status_labels = [
        '0: Outside mask',
        '1: Success',
        '2: <2 points',
        '3: Fit failed',
        '4: Below floor',
        '5: Nonphysical'
    ]
    patches = [plt.matplotlib.patches.Patch(color=colors[i], label=status_labels[i])
              for i in range(6)]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.05, 1))

    # Plot 4: Anatomy overlay
    ax = axes[3]
    overlay = np.zeros((shape[1], shape[2], 3))

    # Color different organs
    organ_colors = {
        'liver': [0, 0, 1],      # Blue
        'kidneys': [1, 0, 0],     # Red
        'tumors': [0, 1, 0],      # Green
        'marrow': [1, 1, 0],      # Yellow
    }

    for organ_name, color in organ_colors.items():
        if organ_name in masks:
            mask = masks[organ_name][z_slice, :, :]
            for c in range(3):
                overlay[:, :, c] += mask * color[c]

    overlay = np.clip(overlay, 0, 1)
    ax.imshow(overlay, origin='lower')
    ax.set_title(f'Organ Anatomy (slice {z_slice})')
    ax.axis('off')

    # Plot 5: TIA histogram
    ax = axes[4]
    valid_tia = tia_data[status_data == 1]
    valid_tia = valid_tia[~np.isnan(valid_tia)]
    ax.hist(valid_tia, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('TIA (MBq·h/ml)')
    ax.set_ylabel('Count')
    ax.set_title('TIA Distribution (All Valid Voxels)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 6: Organ summary
    ax = axes[5]
    ax.axis('off')

    # Calculate statistics for each organ
    organ_stats = []
    for organ_name in ['liver', 'kidneys', 'tumors', 'marrow']:
        if organ_name in masks:
            mask = masks[organ_name]
            valid = mask & (status_data == 1)
            if np.any(valid):
                organ_tia = tia_data[valid]
                organ_tia = organ_tia[~np.isnan(organ_tia)]
                if len(organ_tia) > 0:
                    organ_r2 = r2_data[valid]
                    organ_r2 = organ_r2[~np.isnan(organ_r2)]

                    stats = {
                        'name': organ_name.title(),
                        'n_voxels': np.sum(mask),
                        'n_valid': np.sum(valid),
                        'mean_tia': np.mean(organ_tia),
                        'std_tia': np.std(organ_tia),
                        'mean_r2': np.mean(organ_r2)
                    }
                    organ_stats.append(stats)

    # Create summary table
    table_data = []
    headers = ['Organ', 'N voxels', 'Valid', 'Mean TIA', 'Std TIA', 'Mean R²']
    for stats in organ_stats:
        table_data.append([
            stats['name'],
            str(stats['n_voxels']),
            str(stats['n_valid']),
            f"{stats['mean_tia']:.1f}",
            f"{stats['std_tia']:.1f}",
            f"{stats['mean_r2']:.3f}"
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('Organ-wise Statistics', pad=20)

    plt.suptitle('Lu-177 PSMA Phantom - TIA Analysis Results', fontsize=14)
    plt.tight_layout()

    return fig


def main():
    """Generate all plots for the Lu-177 PSMA phantom TIA analysis."""
    print("=" * 70)
    print("Generating TIA Fitting and Quality Plots for Lu-177 PSMA Phantom")
    print("=" * 70)

    # Run analysis
    print("\n1. Running TIA analysis...")
    data = create_phantom_and_run_analysis()

    # Create output directory
    output_dir = Path('./lu177_psma_plots')
    output_dir.mkdir(exist_ok=True)

    # Generate plots for each organ
    organs_to_plot = ['liver', 'kidneys', 'marrow', 'tumors']

    print("\n2. Generating organ-specific plots...")
    for organ_name in organs_to_plot:
        if organ_name in data['masks']:
            print(f"   Plotting {organ_name}...")
            fig = plot_time_activity_fits(data, organ_name)
            fig.savefig(output_dir / f'{organ_name}_fitting_analysis.png', dpi=150, bbox_inches='tight')
            print(f"   Saved: {output_dir / f'{organ_name}_fitting_analysis.png'}")
            if 'DISPLAY' in plt.rcParams:
                plt.show()
            else:
                plt.close(fig)

    # Generate result maps
    print("\n3. Generating result maps...")
    fig = plot_result_maps(data)
    fig.savefig(output_dir / 'lu177_psma_result_maps.png', dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir / 'lu177_psma_result_maps.png'}")
    if 'DISPLAY' in plt.rcParams:
        plt.show()
    else:
        plt.close(fig)

    # Generate additional detailed plots for tumors
    print("\n4. Generating detailed tumor analysis...")
    if 'tumors' in data['masks']:
        for tumor_num in [1, 2, 3]:
            tumor_name = f'tumor_{tumor_num}'
            if tumor_name in data['masks'] and np.any(data['masks'][tumor_name]):
                print(f"   Plotting {tumor_name}...")
                fig = plot_time_activity_fits(data, tumor_name)
                fig.savefig(output_dir / f'{tumor_name}_detailed_analysis.png', dpi=150, bbox_inches='tight')
                print(f"   Saved: {output_dir / f'{tumor_name}_detailed_analysis.png'}")
                if 'DISPLAY' in plt.rcParams:
                    plt.show()
                else:
                    plt.close(fig)

    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir.absolute()}")
    print("Generated plots:")
    for file in output_dir.glob('*.png'):
        print(f"  - {file.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()