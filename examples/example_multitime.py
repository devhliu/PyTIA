"""
Example 1: Multi-Timepoint TIA Calculation

This example demonstrates how to:
1. Load multiple activity images
2. Configure PyTIA with multi-timepoint settings
3. Compute TIA using curve fitting and integration
4. Access and visualize results
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile

# Import PyTIA
from pytia import run_tia, load_images, stack_4d, Config


def example_multitime_python_api():
    """Example using Python API directly."""
    print("=" * 70)
    print("EXAMPLE 1: Multi-Timepoint TIA (Python API)")
    print("=" * 70)

    # Create synthetic multi-timepoint data for demonstration
    print("\nStep 1: Create synthetic data")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate 4 timepoints with exponential decay
        timepoints = [0.0, 30.0, 60.0, 120.0]  # seconds
        shape = (10, 10, 10)

        for i, t in enumerate(timepoints):
            # Simulate activity decay: A(t) = 100 * exp(-lambda * t)
            lambda_val = np.log(2.0) / 3600.0  # 1 hour half-life
            activity = 100.0 * np.exp(-lambda_val * t)
            data = np.full(shape, activity, dtype=np.float32)

            # Save as NIfTI
            img = nib.Nifti1Image(data, np.eye(4))
            path = tmpdir / f"activity_t{i}.nii.gz"
            nib.save(img, path)
            print(f"  Created {path.name}: {activity:.2f} Bq/ml at t={t}s")

        # Configure PyTIA
        print("\nStep 2: Configure PyTIA")
        config = {
            "io": {
                "output_dir": str(tmpdir / "output"),
                "prefix": "multitime_example",
            },
            "time": {
                "unit": "seconds",
                "sort_timepoints": True,
            },
            "physics": {
                "half_life_seconds": 3600.0,  # 1 hour
            },
            "mask": {
                "mode": "otsu",  # Automatic body mask
            },
            "denoise": {
                "enabled": True,
                "sigma_vox": 1.2,
            },
            "noise_floor": {
                "enabled": True,
                "mode": "relative",
                "relative_fraction_of_voxel_max": 0.01,
            },
            "bootstrap": {
                "enabled": False,  # Disable for speed
            },
            "model_selection": {
                "mode": "auto",
                "min_points_for_gamma": 3,
            },
        }

        # Collect image paths
        image_paths = sorted(tmpdir.glob("activity_t*.nii.gz"))
        print(f"  Config: {len(timepoints)} timepoints")
        print(f"  Half-life: 3600s (1 hour)")

        # Run TIA estimation
        print("\nStep 3: Run TIA estimation")
        result = run_tia(
            images=[str(p) for p in image_paths],
            times=timepoints,
            config=config,
        )

        # Access results
        print("\nStep 4: Access and analyze results")
        tia_data = np.asarray(result.tia_img.dataobj)
        status_data = np.asarray(result.status_id_img.dataobj)
        model_data = np.asarray(result.model_id_img.dataobj)

        # Get statistics
        valid_mask = status_data == 1
        n_valid = np.sum(valid_mask)
        print(f"  Valid voxels: {n_valid} / {np.prod(shape)}")
        print(f"  Mean TIA: {np.nanmean(tia_data[valid_mask]):.2f} Bq·s/ml")
        print(f"  TIA range: {np.nanmin(tia_data[valid_mask]):.2f} - {np.nanmax(tia_data[valid_mask]):.2f}")

        # Check which models were used
        for model_id in [10, 11, 20, 30]:
            count = np.sum(model_data == model_id)
            if count > 0:
                print(f"  Model {model_id}: {count} voxels")

        # Output paths
        print(f"\nStep 5: Output files saved")
        for key, path in result.output_paths.items():
            print(f"  {key}: {path}")

        print(f"\nSummary:")
        print(f"  Times: {result.summary['times_seconds']}")
        print(f"  Voxel volume: {result.summary['voxel_volume_ml']:.4f} ml")


def example_multitime_config_based():
    """Example using YAML config file."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Multi-Timepoint TIA (Config File)")
    print("=" * 70)

    # This would use: pytia run --config config_multitime.yaml
    # See docs/examples/config_multitime.yaml

    print("\nUsage:")
    print("  pytia run --config examples/config_multitime.yaml")


if __name__ == "__main__":
    example_multitime_python_api()
    example_multitime_config_based()
