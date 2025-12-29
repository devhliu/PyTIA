"""
Python API Usage Examples for PyTIA

This file demonstrates various ways to use the PyTIA Python API for computing
Time-Integrated Activity (TIA) maps from PET/SPECT imaging data.
"""

import numpy as np
import nibabel as nib
from pytia import (
    run_tia,
    run_tia_from_nifti,
    run_single_timepoint_tia,
    load_nifti_as_array,
    save_array_as_nifti,
    get_nifti_info,
    extract_roi_stats,
    compare_results,
)
from pytia.config import Config

# =============================================================================
# 1. Multi-Timepoint TIA Calculation (2+ images)
# =============================================================================

# Example 1.1: Using run_tia_from_nifti with NIfTI file paths
def example_multi_timepoint_basic():
    """
    Basic multi-timepoint TIA calculation using NIfTI file paths.
    """
    result = run_tia_from_nifti(
        images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
        times=[1.0, 24.0, 72.0],
        time_unit="hours",
        half_life_seconds=21636.0,  # Tc-99m half-life in seconds
        output_dir="./output",
        prefix="patient_001",
    )
    
    print(f"TIA shape: {result.tia_img.shape}")
    print(f"R² shape: {result.r2_img.shape}")
    print(f"Output paths: {result.output_paths}")
    
    return result


# Example 1.2: Using run_tia_from_nifti with nibabel objects
def example_multi_timepoint_nibabel():
    """
    Multi-timepoint TIA calculation using nibabel objects.
    """
    img1 = nib.load("scan1.nii.gz")
    img2 = nib.load("scan2.nii.gz")
    img3 = nib.load("scan3.nii.gz")
    
    result = run_tia_from_nifti(
        images=[img1, img2, img3],
        times=[1.0, 24.0, 72.0],
        time_unit="hours",
        half_life_seconds=21636.0,
        output_dir="./output",
        prefix="patient_001",
    )
    
    return result


# Example 1.3: With mask and bootstrap
def example_multi_timepoint_advanced():
    """
    Multi-timepoint TIA calculation with mask and bootstrap.
    """
    result = run_tia_from_nifti(
        images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
        times=[1.0, 24.0, 72.0],
        time_unit="hours",
        half_life_seconds=21636.0,
        mask="body_mask.nii.gz",
        bootstrap=True,
        bootstrap_n=100,
        bootstrap_seed=42,
        output_dir="./output",
        prefix="patient_001",
    )
    
    return result


# Example 1.4: Using run_tia with custom config
def example_multi_timepoint_config():
    """
    Multi-timepoint TIA calculation with custom configuration.
    """
    config = {
        "inputs": {
            "images": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
            "times": [1.0, 24.0, 72.0],
        },
        "time": {"unit": "hours"},
        "io": {
            "output_dir": "./output",
            "prefix": "patient_001",
        },
        "physics": {"half_life_seconds": 21636.0},
        "mask": {"mode": "provided", "provided_path": "body_mask.nii.gz"},
        "denoise": {"enabled": True, "sigma_vox": 1.2},
        "noise_floor": {"enabled": True, "relative_fraction_of_voxel_max": 0.01},
        "bootstrap": {"enabled": True, "n": 100, "seed": 42},
        "performance": {"chunk_size_vox": 500000},
    }
    
    result = run_tia(
        images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
        times=[1.0, 24.0, 72.0],
        config=config,
        mask="body_mask.nii.gz",
    )
    
    return result


# =============================================================================
# 2. Single-Timepoint TIA Calculation (1 image)
# =============================================================================

# Example 2.1: Physical decay method
def example_single_timepoint_phys():
    """
    Single-timepoint TIA calculation using physical decay method.
    """
    result = run_single_timepoint_tia(
        image="scan1.nii.gz",
        time=24.0,
        method="phys",
        half_life_seconds=21636.0,  # Tc-99m half-life
        time_unit="hours",
        output_dir="./output",
        prefix="patient_001_stp",
    )
    
    print(f"TIA shape: {result.tia_img.shape}")
    return result


# Example 2.2: Hänscheid method (effective half-life)
def example_single_timepoint_haenscheid():
    """
    Single-timepoint TIA calculation using Hänscheid method.
    """
    result = run_single_timepoint_tia(
        image="scan1.nii.gz",
        time=24.0,
        method="haenscheid",
        eff_half_life_seconds=3600.0,  # Effective half-life in seconds
        time_unit="hours",
        output_dir="./output",
        prefix="patient_001_stp",
    )
    
    return result


# Example 2.3: Prior half-life method (global)
def example_single_timepoint_prior_global():
    """
    Single-timepoint TIA calculation using global prior half-life.
    """
    result = run_single_timepoint_tia(
        image="scan1.nii.gz",
        time=24.0,
        method="prior_half_life",
        prior_half_life_seconds=1800.0,  # Prior half-life in seconds
        time_unit="hours",
        output_dir="./output",
        prefix="patient_001_stp",
    )
    
    return result


# Example 2.4: Prior half-life method (segmentation-based)
def example_single_timepoint_prior_segmentation():
    """
    Single-timepoint TIA calculation using segmentation-based priors.
    """
    result = run_single_timepoint_tia(
        image="scan1.nii.gz",
        time=24.0,
        method="prior_half_life",
        label_map="labels.nii.gz",
        label_half_lives={
            1: 1800.0,  # Label 1: 30 minutes
            2: 3600.0,  # Label 2: 60 minutes
            3: 5400.0,  # Label 3: 90 minutes
        },
        time_unit="hours",
        output_dir="./output",
        prefix="patient_001_stp",
    )
    
    return result


# =============================================================================
# 3. Working with Results
# =============================================================================

# Example 3.1: Accessing result data
def example_access_results(result):
    """
    Access and process TIA calculation results.
    """
    tia_data = result.tia_img.get_fdata()
    r2_data = result.r2_img.get_fdata()
    sigma_tia_data = result.sigma_tia_img.get_fdata()
    model_id_data = result.model_id_img.get_fdata()
    status_id_data = result.status_id_img.get_fdata()
    
    print(f"TIA statistics:")
    print(f"  Mean: {np.nanmean(tia_data):.2f}")
    print(f"  Std: {np.nanstd(tia_data):.2f}")
    print(f"  Min: {np.nanmin(tia_data):.2f}")
    print(f"  Max: {np.nanmax(tia_data):.2f}")
    
    print(f"\nR² statistics:")
    print(f"  Mean: {np.nanmean(r2_data):.4f}")
    
    print(f"\nSummary:")
    print(f"  Times (s): {result.times_s}")
    print(f"  Status counts: {result.summary.get('status_counts', {})}")
    print(f"  Timing: {result.summary.get('timing_ms', {})}")
    
    return tia_data, r2_data, sigma_tia_data


# Example 3.2: Saving processed results
def example_save_results(result, output_path):
    """
    Save processed TIA results to NIfTI file.
    """
    tia_data = result.tia_img.get_fdata()
    
    processed_data = tia_data * 1.5  # Example: apply scaling factor
    
    save_array_as_nifti(
        data=processed_data,
        reference=result.tia_img,
        output_path=output_path,
    )
    
    print(f"Saved processed results to: {output_path}")


# =============================================================================
# 4. Utility Functions
# =============================================================================

# Example 4.1: Loading NIfTI files
def example_load_nifti():
    """
    Load NIfTI file as numpy array with metadata.
    """
    data, img = load_nifti_as_array("scan1.nii.gz")
    
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Affine:\n{img.affine}")
    
    return data, img


# Example 4.2: Getting NIfTI file information
def example_get_nifti_info():
    """
    Get comprehensive information about a NIfTI file.
    """
    info = get_nifti_info("scan1.nii.gz")
    
    print(f"Shape: {info['shape']}")
    print(f"Dtype: {info['dtype']}")
    print(f"Voxel sizes: {info['voxel_sizes']}")
    print(f"Voxel volume (ml): {info['voxel_volume_ml']:.4f}")
    
    return info


# Example 4.3: Extracting ROI statistics
def example_extract_roi_stats(tia_result, mask_path):
    """
    Extract TIA statistics for a region of interest.
    """
    stats = extract_roi_stats(
        tia_img=tia_result.tia_img,
        mask_img=mask_path,
        mask_value=1,
    )
    
    print(f"ROI Statistics:")
    print(f"  Mean TIA: {stats['mean']:.2f}")
    print(f"  Std TIA: {stats['std']:.2f}")
    print(f"  Min TIA: {stats['min']:.2f}")
    print(f"  Max TIA: {stats['max']:.2f}")
    print(f"  Sum TIA: {stats['sum']:.2f}")
    print(f"  Count: {stats['count']}")
    
    return stats


# Example 4.4: Comparing two results
def example_compare_results(result1, result2):
    """
    Compare two TIA results and compute difference statistics.
    """
    diff = compare_results(result1, result2, metric="tia")
    
    print(f"Comparison Statistics:")
    print(f"  Mean difference: {diff['mean_diff']:.2f}")
    print(f"  Std difference: {diff['std_diff']:.2f}")
    print(f"  Min difference: {diff['min_diff']:.2f}")
    print(f"  Max difference: {diff['max_diff']:.2f}")
    print(f"  Absolute mean difference: {diff['abs_mean_diff']:.2f}")
    print(f"  RMSE: {diff['rmse']:.2f}")
    
    return diff


# =============================================================================
# 5. Advanced Usage
# =============================================================================

# Example 5.1: Processing multiple patients
def example_batch_processing():
    """
    Process multiple patients in batch.
    """
    patients = [
        {"id": "001", "images": ["p001_s1.nii.gz", "p001_s2.nii.gz", "p001_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
        {"id": "002", "images": ["p002_s1.nii.gz", "p002_s2.nii.gz", "p002_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
        {"id": "003", "images": ["p003_s1.nii.gz", "p003_s2.nii.gz", "p003_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
    ]
    
    results = []
    for patient in patients:
        result = run_tia_from_nifti(
            images=patient["images"],
            times=patient["times"],
            time_unit="hours",
            half_life_seconds=21636.0,
            output_dir=f"./output/patient_{patient['id']}",
            prefix=f"patient_{patient['id']}",
        )
        results.append(result)
        print(f"Processed patient {patient['id']}")
    
    return results


# Example 5.2: Custom post-processing
def example_custom_postprocessing(result):
    """
    Apply custom post-processing to TIA results.
    """
    tia_data = result.tia_img.get_fdata()
    
    threshold = np.nanpercentile(tia_data, 95)
    masked_tia = np.where(tia_data > threshold, tia_data, 0)
    
    save_array_as_nifti(
        data=masked_tia,
        reference=result.tia_img,
        output_path="./output/thresholded_tia.nii.gz",
    )
    
    print(f"Applied threshold: {threshold:.2f}")
    print(f"Saved thresholded TIA to: ./output/thresholded_tia.nii.gz")


# =============================================================================
# 6. Integration with Other Libraries
# =============================================================================

# Example 6.1: Using with matplotlib for visualization
def example_visualization(result):
    """
    Visualize TIA results using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        tia_data = result.tia_img.get_fdata()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        mid_slice = tia_data.shape[2] // 2
        
        axes[0].imshow(tia_data[:, :, mid_slice], cmap='hot')
        axes[0].set_title('TIA (Axial)')
        axes[0].axis('off')
        
        axes[1].imshow(tia_data[:, mid_slice, :], cmap='hot')
        axes[1].set_title('TIA (Coronal)')
        axes[1].axis('off')
        
        axes[2].imshow(tia_data[mid_slice, :, :], cmap='hot')
        axes[2].set_title('TIA (Sagittal)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('./output/tia_visualization.png', dpi=150)
        plt.close()
        
        print("Saved visualization to: ./output/tia_visualization.png")
    except ImportError:
        print("matplotlib not available for visualization")


# =============================================================================
# Main execution example
# =============================================================================

if __name__ == "__main__":
    print("PyTIA Python API Examples")
    print("=" * 60)
    
    print("\n1. Multi-timepoint TIA calculation:")
    print("   result = run_tia_from_nifti(images=[...], times=[...], ...)")
    
    print("\n2. Single-timepoint TIA calculation:")
    print("   result = run_single_timepoint_tia(image='...', time=24.0, ...)")
    
    print("\n3. Accessing results:")
    print("   tia_data = result.tia_img.get_fdata()")
    print("   r2_data = result.r2_img.get_fdata()")
    
    print("\n4. Utility functions:")
    print("   data, img = load_nifti_as_array('scan.nii.gz')")
    print("   info = get_nifti_info('scan.nii.gz')")
    print("   stats = extract_roi_stats(tia_img, mask_img)")
    
    print("\nFor detailed examples, see the functions in this file.")
