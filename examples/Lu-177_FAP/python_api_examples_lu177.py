"""
Python API Examples for Lu-177 FAP TIA Calculation

This file demonstrates Python API usage for calculating Time-Integrated Activity (TIA)
from Lu-177 FAP SPECT images acquired at multiple time points.

Input files (located in examples/Lu-177_FAP/input_multi-points/):
- SPECT-004H_reg.nii.gz (4 hours)
- SPECT-024H_reg.nii.gz (24 hours)
- SPECT-048H_reg.nii.gz (48 hours)
- SPECT-168H_reg.nii.gz (168 hours)

Lu-177 half-life: 6.647 days = 574302.48 seconds
"""

import numpy as np
import nibabel as nib
from pytia import (
    run_tia_from_nifti,
    run_single_timepoint_tia,
    load_nifti_as_array,
    save_array_as_nifti,
    get_nifti_info,
    extract_roi_stats,
    compare_results,
)

# Constants
LU177_HALF_LIFE_SECONDS = 6.647 * 24 * 3600  # 6.647 days in seconds = 574302.48

# Input file paths
INPUT_DIR = "examples/Lu-177_FAP/input_multi-points"
OUTPUT_DIR = "examples/Lu-177_FAP/output_tia"

FILES = {
    "4h": f"{INPUT_DIR}/SPECT-004H_reg.nii.gz",
    "24h": f"{INPUT_DIR}/SPECT-024H_reg.nii.gz",
    "48h": f"{INPUT_DIR}/SPECT-048H_reg.nii.gz",
    "168h": f"{INPUT_DIR}/SPECT-168H_reg.nii.gz",
}

# =============================================================================
# Example 1: TIA Calculation with 3 Time Points [4, 24, 48] hours
# =============================================================================

def example_3_timepoints_basic():
    """
    Basic 3-timepoint TIA calculation for Lu-177 FAP.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp",
    )
    
    print(f"3-timepoint TIA calculation completed")
    print(f"TIA shape: {result.tia_img.shape}")
    print(f"Output paths: {result.output_paths}")
    
    return result


def example_3_timepoints_with_mask(mask_path):
    """
    3-timepoint TIA calculation with mask.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        mask=mask_path,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp_masked",
    )
    
    print(f"3-timepoint TIA calculation with mask completed")
    return result


def example_3_timepoints_with_bootstrap():
    """
    3-timepoint TIA calculation with bootstrap for uncertainty estimation.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        bootstrap=True,
        bootstrap_n=100,
        bootstrap_seed=42,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp_bootstrap",
    )
    
    print(f"3-timepoint TIA calculation with bootstrap completed")
    return result


# =============================================================================
# Example 2: TIA Calculation with 4 Time Points [4, 24, 48, 168] hours
# =============================================================================

def example_4_timepoints_basic():
    """
    Basic 4-timepoint TIA calculation for Lu-177 FAP.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
        times=[4.0, 24.0, 48.0, 168.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_4tp",
    )
    
    print(f"4-timepoint TIA calculation completed")
    print(f"TIA shape: {result.tia_img.shape}")
    print(f"Output paths: {result.output_paths}")
    
    return result


def example_4_timepoints_with_mask(mask_path):
    """
    4-timepoint TIA calculation with mask.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
        times=[4.0, 24.0, 48.0, 168.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        mask=mask_path,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_4tp_masked",
    )
    
    print(f"4-timepoint TIA calculation with mask completed")
    return result


def example_4_timepoints_with_bootstrap():
    """
    4-timepoint TIA calculation with bootstrap for uncertainty estimation.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
        times=[4.0, 24.0, 48.0, 168.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        bootstrap=True,
        bootstrap_n=100,
        bootstrap_seed=42,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_4tp_bootstrap",
    )
    
    print(f"4-timepoint TIA calculation with bootstrap completed")
    return result


# =============================================================================
# Example 3: Advanced Options
# =============================================================================

def example_3_timepoints_chunked():
    """
    3-timepoint TIA calculation with custom chunk size for memory efficiency.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        chunk_size=500000,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp_chunked",
    )
    
    print(f"3-timepoint TIA calculation with chunking completed")
    return result


def example_4_timepoints_no_denoise():
    """
    4-timepoint TIA calculation with denoising disabled.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
        times=[4.0, 24.0, 48.0, 168.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        denoise=False,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_4tp_no_denoise",
    )
    
    print(f"4-timepoint TIA calculation without denoising completed")
    return result


def example_3_timepoints_no_noise_floor():
    """
    3-timepoint TIA calculation with noise floor disabled.
    """
    result = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        noise_floor=False,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp_no_noise_floor",
    )
    
    print(f"3-timepoint TIA calculation without noise floor completed")
    return result


# =============================================================================
# Example 4: Working with Results
# =============================================================================

def example_analyze_results(result):
    """
    Analyze and display TIA calculation results.
    """
    tia_data = result.tia_img.get_fdata()
    r2_data = result.r2_img.get_fdata()
    sigma_tia_data = result.sigma_tia_img.get_fdata()
    
    print(f"\nTIA Statistics:")
    print(f"  Mean: {np.nanmean(tia_data):.2e} Bq·s/voxel")
    print(f"  Std: {np.nanstd(tia_data):.2e} Bq·s/voxel")
    print(f"  Min: {np.nanmin(tia_data):.2e} Bq·s/voxel")
    print(f"  Max: {np.nanmax(tia_data):.2e} Bq·s/voxel")
    
    print(f"\nR² Statistics:")
    print(f"  Mean: {np.nanmean(r2_data):.4f}")
    print(f"  Std: {np.nanstd(r2_data):.4f}")
    print(f"  Min: {np.nanmin(r2_data):.4f}")
    print(f"  Max: {np.nanmax(r2_data):.4f}")
    
    print(f"\nUncertainty Statistics:")
    print(f"  Mean: {np.nanmean(sigma_tia_data):.2e} Bq·s/voxel")
    print(f"  Std: {np.nanstd(sigma_tia_data):.2e} Bq·s/voxel")
    
    print(f"\nSummary:")
    print(f"  Times (s): {result.times_s}")
    print(f"  Status counts: {result.summary.get('status_counts', {})}")
    print(f"  Timing: {result.summary.get('timing_ms', {})}")
    
    return tia_data, r2_data, sigma_tia_data


def example_extract_roi_statistics(result, mask_path):
    """
    Extract TIA statistics for a region of interest.
    """
    stats = extract_roi_stats(
        tia_img=result.tia_img,
        mask_img=mask_path,
        mask_value=1,
    )
    
    print(f"\nROI Statistics:")
    print(f"  Mean TIA: {stats['mean']:.2e} Bq·s/voxel")
    print(f"  Std TIA: {stats['std']:.2e} Bq·s/voxel")
    print(f"  Min TIA: {stats['min']:.2e} Bq·s/voxel")
    print(f"  Max TIA: {stats['max']:.2e} Bq·s/voxel")
    print(f"  Sum TIA: {stats['sum']:.2e} Bq·s")
    print(f"  Count: {stats['count']} voxels")
    
    return stats


def example_compare_3tp_vs_4tp():
    """
    Compare 3-timepoint and 4-timepoint TIA results.
    """
    result_3tp = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"]],
        times=[4.0, 24.0, 48.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_3tp_compare",
    )
    
    result_4tp = run_tia_from_nifti(
        images=[FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
        times=[4.0, 24.0, 48.0, 168.0],
        time_unit="hours",
        half_life_seconds=LU177_HALF_LIFE_SECONDS,
        output_dir=OUTPUT_DIR,
        prefix="Lu177_FAP_4tp_compare",
    )
    
    diff = compare_results(result_3tp, result_4tp, metric="tia")
    
    print(f"\n3TP vs 4TP Comparison:")
    print(f"  Mean difference: {diff['mean_diff']:.2e} Bq·s/voxel")
    print(f"  Std difference: {diff['std_diff']:.2e} Bq·s/voxel")
    print(f"  Min difference: {diff['min_diff']:.2e} Bq·s/voxel")
    print(f"  Max difference: {diff['max_diff']:.2e} Bq·s/voxel")
    print(f"  Absolute mean difference: {diff['abs_mean_diff']:.2e} Bq·s/voxel")
    print(f"  RMSE: {diff['rmse']:.2e} Bq·s/voxel")
    
    return diff


# =============================================================================
# Example 5: Batch Processing
# =============================================================================

def example_batch_processing():
    """
    Process multiple patients or datasets in batch.
    """
    datasets = [
        {
            "id": "patient_001",
            "images": [FILES["4h"], FILES["24h"], FILES["48h"]],
            "times": [4.0, 24.0, 48.0],
        },
        {
            "id": "patient_002",
            "images": [FILES["4h"], FILES["24h"], FILES["48h"], FILES["168h"]],
            "times": [4.0, 24.0, 48.0, 168.0],
        },
    ]
    
    results = []
    for dataset in datasets:
        result = run_tia_from_nifti(
            images=dataset["images"],
            times=dataset["times"],
            time_unit="hours",
            half_life_seconds=LU177_HALF_LIFE_SECONDS,
            output_dir=f"{OUTPUT_DIR}/{dataset['id']}",
            prefix=f"Lu177_FAP_{dataset['id']}",
        )
        results.append(result)
        print(f"Processed {dataset['id']}")
    
    return results


# =============================================================================
# Example 6: Custom Post-Processing
# =============================================================================

def example_threshold_tia(result, threshold_percentile=95):
    """
    Apply threshold to TIA results.
    """
    tia_data = result.tia_img.get_fdata()
    
    threshold = np.nanpercentile(tia_data, threshold_percentile)
    masked_tia = np.where(tia_data > threshold, tia_data, 0)
    
    output_path = f"{OUTPUT_DIR}/thresholded_tia_{threshold_percentile}pct.nii.gz"
    save_array_as_nifti(
        data=masked_tia,
        reference=result.tia_img,
        output_path=output_path,
    )
    
    print(f"Applied threshold: {threshold:.2e} Bq·s/voxel ({threshold_percentile}th percentile)")
    print(f"Saved thresholded TIA to: {output_path}")
    
    return masked_tia


def example_normalize_tia(result):
    """
    Normalize TIA to [0, 1] range.
    """
    tia_data = result.tia_img.get_fdata()
    
    tia_min = np.nanmin(tia_data)
    tia_max = np.nanmax(tia_data)
    normalized_tia = (tia_data - tia_min) / (tia_max - tia_min)
    
    output_path = f"{OUTPUT_DIR}/normalized_tia.nii.gz"
    save_array_as_nifti(
        data=normalized_tia,
        reference=result.tia_img,
        output_path=output_path,
    )
    
    print(f"Normalized TIA to [0, 1] range")
    print(f"Original range: [{tia_min:.2e}, {tia_max:.2e}] Bq·s/voxel")
    print(f"Saved normalized TIA to: {output_path}")
    
    return normalized_tia


# =============================================================================
# Example 7: Visualization
# =============================================================================

def example_visualize_tia(result):
    """
    Visualize TIA results using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        tia_data = result.tia_img.get_fdata()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        mid_slice = tia_data.shape[2] // 2
        
        im1 = axes[0].imshow(tia_data[:, :, mid_slice], cmap='hot')
        axes[0].set_title('TIA (Axial)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(tia_data[:, mid_slice, :], cmap='hot')
        axes[1].set_title('TIA (Coronal)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(tia_data[mid_slice, :, :], cmap='hot')
        axes[2].set_title('TIA (Sagittal)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        output_path = f"{OUTPUT_DIR}/tia_visualization.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Saved visualization to: {output_path}")
    except ImportError:
        print("matplotlib not available for visualization")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Lu-177 FAP TIA Calculation Examples")
    print("=" * 60)
    print(f"Lu-177 half-life: {LU177_HALF_LIFE_SECONDS:.2f} seconds")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    print("\n1. 3-timepoint TIA calculation:")
    print("   result = run_tia_from_nifti(images=[...], times=[4, 24, 48], ...)")
    
    print("\n2. 4-timepoint TIA calculation:")
    print("   result = run_tia_from_nifti(images=[...], times=[4, 24, 48, 168], ...)")
    
    print("\n3. Analyzing results:")
    print("   tia_data = result.tia_img.get_fdata()")
    print("   r2_data = result.r2_img.get_fdata()")
    print("   sigma_tia_data = result.sigma_tia_img.get_fdata()")
    
    print("\n4. Extracting ROI statistics:")
    print("   stats = extract_roi_stats(tia_img, mask_img)")
    
    print("\n5. Comparing results:")
    print("   diff = compare_results(result_3tp, result_4tp)")
    
    print("\nFor detailed examples, see the functions in this file.")
    print("\nTo run specific examples, uncomment the function calls below:")
    
    # Example 1: 3-timepoint TIA calculation
    # result_3tp = example_3_timepoints_basic()
    # example_analyze_results(result_3tp)
    
    # Example 2: 4-timepoint TIA calculation
    # result_4tp = example_4_timepoints_basic()
    # example_analyze_results(result_4tp)
    
    # Example 3: Compare 3TP vs 4TP
    # diff = example_compare_3tp_vs_4tp()
    
    # Example 4: Visualization
    # example_visualize_tia(result_4tp)
