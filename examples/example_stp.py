"""
Example 2: Single-Timepoint TIA Calculation

Demonstrates the three STP methods:
1. Physical decay
2. Hänscheid method
3. Prior half-life (global and label-based)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile

from pytia import run_tia


def example_stp_physical_decay():
    """Single-timepoint with physical decay method."""
    print("=" * 70)
    print("EXAMPLE 1: STP - Physical Decay Method")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create single activity image
        print("\nStep 1: Create synthetic activity image")
        activity = 100.0  # Bq/ml
        data = np.full((10, 10, 10), activity, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img_path = tmpdir / "activity.nii.gz"
        nib.save(img, img_path)
        print(f"  Created: {img_path.name}")
        print(f"  Activity: {activity} Bq/ml")

        # Configure for STP - Physical decay
        print("\nStep 2: Configure STP with Physical Decay")
        config = {
            "io": {
                "output_dir": str(tmpdir / "output"),
                "prefix": "stp_phys",
            },
            "physics": {
                "half_life_seconds": 21600.0,  # Tc-99m: 6 hours
            },
            "single_time": {
                "enabled": True,
                "method": "phys",
            },
            "noise_floor": {
                "enabled": True,
                "mode": "relative",
                "relative_fraction_of_voxel_max": 0.01,
            },
        }

        print(f"  Method: phys (Physical Decay)")
        print(f"  Half-life: 21600s (6 hours) - Tc-99m")

        # Run TIA estimation
        print("\nStep 3: Run TIA estimation")
        result = run_tia(
            images=[str(img_path)],
            times=[0.0],
            config=config,
        )

        # Analyze results
        print("\nStep 4: Results")
        tia_data = np.asarray(result.tia_img.dataobj)
        model_data = np.asarray(result.model_id_img.dataobj)

        valid = model_data == 101  # Model ID for phys method
        expected_tia = activity * 21600.0 / np.log(2.0)

        print(f"  Model IDs used: {np.unique(model_data[model_data > 0])}")
        print(f"  Expected TIA: {expected_tia:.2f} Bq·s/ml")
        print(f"  Computed TIA: {tia_data[5, 5, 5]:.2f} Bq·s/ml")

        print(f"\nOutput:")
        for key, path in result.output_paths.items():
            print(f"  {key}: {path}")


def example_stp_haenscheid():
    """Single-timepoint with Hänscheid method."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: STP - Hänscheid Method")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\nStep 1: Create activity image")
        activity = 100.0
        data = np.full((10, 10, 10), activity, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img_path = tmpdir / "activity.nii.gz"
        nib.save(img, img_path)
        print(f"  Activity: {activity} Bq/ml")

        print("\nStep 2: Configure STP with Hänscheid")
        config = {
            "io": {"output_dir": str(tmpdir / "output")},
            "physics": {"half_life_seconds": 6600.0},  # F-18 physical
            "single_time": {
                "enabled": True,
                "method": "haenscheid",
                "haenscheid_eff_half_life_seconds": 7200.0,  # 2 hours effective
            },
        }

        print(f"  Method: haenscheid")
        print(f"  Effective half-life: 7200s (2 hours)")

        result = run_tia(
            images=[str(img_path)],
            times=[0.0],
            config=config,
        )

        tia_data = np.asarray(result.tia_img.dataobj)
        expected_tia = activity * 7200.0 / np.log(2.0)

        print(f"\nResults:")
        print(f"  Expected TIA: {expected_tia:.2f} Bq·s/ml")
        print(f"  Computed TIA: {tia_data[5, 5, 5]:.2f} Bq·s/ml")


def example_stp_prior_global():
    """Single-timepoint with global prior half-life."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: STP - Prior Half-Life (Global)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\nCreate activity image and configure")
        activity = 100.0
        data = np.full((10, 10, 10), activity, dtype=np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img_path = tmpdir / "activity.nii.gz"
        nib.save(img, img_path)

        config = {
            "io": {"output_dir": str(tmpdir / "output")},
            "single_time": {
                "enabled": True,
                "method": "prior_half_life",
                "half_life_seconds": 5400.0,  # 1.5 hours prior
            },
        }

        print(f"  Method: prior_half_life (global)")
        print(f"  Prior half-life: 5400s (1.5 hours)")

        result = run_tia(
            images=[str(img_path)],
            times=[0.0],
            config=config,
        )

        tia_data = np.asarray(result.tia_img.dataobj)
        expected_tia = activity * 5400.0 / np.log(2.0)

        print(f"\nResults:")
        print(f"  Expected TIA: {expected_tia:.2f} Bq·s/ml")
        print(f"  Computed TIA: {tia_data[5, 5, 5]:.2f} Bq·s/ml")


def example_stp_prior_segmentation():
    """Single-timepoint with segmentation-based priors."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: STP - Prior Half-Life (Segmentation)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\nStep 1: Create activity and segmentation images")
        activity = 100.0
        activity_data = np.full((10, 10, 10), activity, dtype=np.float32)
        activity_img = nib.Nifti1Image(activity_data, np.eye(4))
        activity_path = tmpdir / "activity.nii.gz"
        nib.save(activity_img, activity_path)

        # Create label image with 3 regions
        labels_data = np.zeros((10, 10, 10), dtype=np.int32)
        labels_data[0:3, :, :] = 1    # Label 1: Tumor
        labels_data[3:6, :, :] = 2    # Label 2: Liver
        labels_data[6:, :, :] = 3     # Label 3: Kidney
        labels_img = nib.Nifti1Image(labels_data, np.eye(4))
        labels_path = tmpdir / "segmentation.nii.gz"
        nib.save(labels_img, labels_path)

        print(f"  Activity: {activity_path.name}")
        print(f"  Segmentation: {labels_path.name}")

        print("\nStep 2: Configure STP with segmentation-based priors")
        config = {
            "io": {"output_dir": str(tmpdir / "output")},
            "single_time": {
                "enabled": True,
                "method": "prior_half_life",
                "label_map_path": str(labels_path),
                "half_life_seconds": 3600.0,  # Default
                "label_half_lives": {
                    1: 1800.0,   # Tumor: 30 min
                    2: 3600.0,   # Liver: 60 min
                    3: 5400.0,   # Kidney: 90 min
                },
            },
        }

        print(f"  Method: prior_half_life (segmentation)")
        print(f"  Label 1 (Tumor):  1800s (30 min)")
        print(f"  Label 2 (Liver):  3600s (60 min)")
        print(f"  Label 3 (Kidney): 5400s (90 min)")

        result = run_tia(
            images=[str(activity_path)],
            times=[0.0],
            config=config,
        )

        print("\nStep 3: Verify per-label TIA")
        tia_data = np.asarray(result.tia_img.dataobj)

        # Check different regions
        label1_tia = tia_data[1, 5, 5]
        label2_tia = tia_data[4, 5, 5]
        label3_tia = tia_data[7, 5, 5]

        expected_1 = activity * 1800.0 / np.log(2.0)
        expected_2 = activity * 3600.0 / np.log(2.0)
        expected_3 = activity * 5400.0 / np.log(2.0)

        print(f"\nResults by region:")
        print(f"  Label 1 (Tumor):  expected={expected_1:.2f}, computed={label1_tia:.2f} Bq·s/ml")
        print(f"  Label 2 (Liver):  expected={expected_2:.2f}, computed={label2_tia:.2f} Bq·s/ml")
        print(f"  Label 3 (Kidney): expected={expected_3:.2f}, computed={label3_tia:.2f} Bq·s/ml")

        print(f"\nNote: TIA scales with organ-specific half-life!")


if __name__ == "__main__":
    example_stp_physical_decay()
    example_stp_haenscheid()
    example_stp_prior_global()
    example_stp_prior_segmentation()
