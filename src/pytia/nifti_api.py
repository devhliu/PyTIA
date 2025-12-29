"""Convenience API for NIfTI file handling in PyTIA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np

from .engine import Results, run_tia
from .io import load_images, make_like, stack_4d, voxel_volume_ml


def run_tia_from_nifti(
    images: Sequence[str | Path | nib.spatialimages.SpatialImage],
    times: Sequence[float],
    output_dir: str | Path = "./pytia_output",
    prefix: str = "pytia",
    half_life_seconds: float | None = None,
    mask: str | Path | nib.spatialimages.SpatialImage | None = None,
    mask_mode: str = "none",
    time_unit: str = "hours",
    denoise: bool = True,
    noise_floor: bool = True,
    bootstrap: bool = False,
    bootstrap_n: int = 100,
    bootstrap_seed: int = 42,
    chunk_size: int | None = None,
    **kwargs: Any,
) -> Results:
    """
    Run TIA estimation from NIfTI files with simplified API.

    This is a convenience wrapper around run_tia() that provides a simpler
    interface for working with NIfTI files without needing to create a
    configuration file.

    Args:
        images: NIfTI image files or nibabel objects (one or more)
        times: Timepoints for each image
        output_dir: Output directory for results (default: ./pytia_output)
        prefix: Prefix for output files (default: pytia)
        half_life_seconds: Radionuclide half-life in seconds
        mask: Optional mask image file or nibabel object
        mask_mode: Mask mode: "none", "provided", or "auto" (default: none)
        time_unit: Time unit: "hours" or "seconds" (default: hours)
        denoise: Enable spatial denoising (default: True)
        noise_floor: Enable noise floor filtering (default: True)
        bootstrap: Enable bootstrap for uncertainty (default: False)
        bootstrap_n: Number of bootstrap replicates (default: 100)
        bootstrap_seed: Random seed for bootstrap (default: 42)
        chunk_size: Chunk size for voxel processing (default: None)
        **kwargs: Additional configuration options passed to run_tia()

    Returns:
        Results object containing TIA maps and metadata

    Example:
        >>> from pytia.nifti_api import run_tia_from_nifti
        >>> result = run_tia_from_nifti(
        ...     images=["scan1.nii.gz", "scan2.nii.gz"],
        ...     times=[0.0, 24.0],
        ...     half_life_seconds=144540.0,
        ...     output_dir="./output"
        ... )
        >>> print(f"TIA shape: {result.tia_img.shape}")
    """
    config: dict[str, Any] = {
        "inputs": {
            "images": [str(img) if isinstance(img, (str, Path)) else img for img in images],
            "times": list(times),
        },
        "time": {
            "unit": time_unit,
        },
        "io": {
            "output_dir": str(output_dir),
            "prefix": prefix,
        },
        "mask": {
            "mode": mask_mode,
        },
        "denoise": {
            "enabled": denoise,
        },
        "noise_floor": {
            "enabled": noise_floor,
        },
        "bootstrap": {
            "enabled": bootstrap,
            "n": bootstrap_n,
            "seed": bootstrap_seed,
        },
    }

    if half_life_seconds is not None:
        config["physics"] = {"half_life_seconds": float(half_life_seconds)}

    if mask is not None:
        config["mask"]["provided_path"] = str(mask) if isinstance(mask, (str, Path)) else mask

    if chunk_size is not None:
        config["performance"] = {"chunk_size_vox": int(chunk_size)}

    config.update(kwargs)

    return run_tia(images=images, times=times, config=config, mask=mask)


def run_single_timepoint_tia(
    image: str | Path | nib.spatialimages.SpatialImage,
    time: float,
    method: str = "phys",
    output_dir: str | Path = "./pytia_output",
    prefix: str = "pytia",
    half_life_seconds: float | None = None,
    eff_half_life_seconds: float | None = None,
    prior_half_life_seconds: float | None = None,
    label_map: str | Path | nib.spatialimages.SpatialImage | None = None,
    label_half_lives: dict[int, float] | None = None,
    time_unit: str = "hours",
    mask: str | Path | nib.spatialimages.SpatialImage | None = None,
    mask_mode: str = "none",
    denoise: bool = True,
    noise_floor: bool = True,
    **kwargs: Any,
) -> Results:
    """
    Run single-timepoint TIA estimation from NIfTI file.

    Calculates TIA = A(t) / λ_eff using one of three methods:
    - phys: Physical decay from radionuclide half-life
    - haenscheid: Effective half-life in human body
    - prior_half_life: Segmentation-based priors

    Args:
        image: Single NIfTI image file or nibabel object
        time: Timepoint for the image
        method: Method: "phys", "haenscheid", or "prior_half_life" (default: phys)
        output_dir: Output directory for results (default: ./pytia_output)
        prefix: Prefix for output files (default: pytia)
        half_life_seconds: Radionuclide half-life in seconds (for phys method)
        eff_half_life_seconds: Effective half-life in seconds (for haenscheid method)
        prior_half_life_seconds: Prior half-life in seconds (for prior_half_life method)
        label_map: Label map for segmentation-based priors (for prior_half_life method)
        label_half_lives: Mapping of label IDs to half-lives in seconds (for prior_half_life method)
        time_unit: Time unit: "hours" or "seconds" (default: hours)
        mask: Optional mask image file or nibabel object
        mask_mode: Mask mode: "none", "provided", or "auto" (default: none)
        denoise: Enable spatial denoising (default: True)
        noise_floor: Enable noise floor filtering (default: True)
        **kwargs: Additional configuration options

    Returns:
        Results object containing TIA maps and metadata

    Example:
        >>> from pytia.nifti_api import run_single_timepoint_tia
        >>> result = run_single_timepoint_tia(
        ...     image="scan.nii.gz",
        ...     time=24.0,
        ...     method="phys",
        ...     half_life_seconds=144540.0
        ... )
    """
    config: dict[str, Any] = {
        "inputs": {
            "images": [str(image) if isinstance(image, (str, Path)) else image],
            "times": [float(time)],
        },
        "time": {
            "unit": time_unit,
        },
        "io": {
            "output_dir": str(output_dir),
            "prefix": prefix,
        },
        "single_time": {
            "enabled": True,
            "method": method,
        },
        "mask": {
            "mode": mask_mode,
        },
        "denoise": {
            "enabled": denoise,
        },
        "noise_floor": {
            "enabled": noise_floor,
        },
    }

    if method == "phys" and half_life_seconds is not None:
        config["physics"] = {"half_life_seconds": float(half_life_seconds)}

    if method == "haenscheid" and eff_half_life_seconds is not None:
        config["single_time"]["haenscheid_eff_half_life_seconds"] = float(eff_half_life_seconds)

    if method == "prior_half_life":
        if prior_half_life_seconds is not None:
            config["single_time"]["half_life_seconds"] = float(prior_half_life_seconds)
        if label_map is not None:
            config["single_time"]["label_map_path"] = str(label_map) if isinstance(label_map, (str, Path)) else label_map
        if label_half_lives is not None:
            config["single_time"]["label_half_lives"] = {int(k): float(v) for k, v in label_half_lives.items()}

    config.update(kwargs)

    return run_tia(images=[image], times=[time], config=config, mask=mask)


def load_nifti_as_array(path: str | Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    """
    Load a NIfTI file as a numpy array with metadata.

    Args:
        path: Path to NIfTI file

    Returns:
        Tuple of (data_array, nibabel_image)

    Example:
        >>> data, img = load_nifti_as_array("scan.nii.gz")
        >>> print(f"Shape: {data.shape}")
        >>> print(f"Affine: {img.affine}")
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return data, img


def save_array_as_nifti(
    data: np.ndarray,
    reference: str | Path | nib.spatialimages.SpatialImage,
    output_path: str | Path,
    dtype: type | None = None,
) -> nib.spatialimages.SpatialImage:
    """
    Save a numpy array as a NIfTI file using reference image metadata.

    Args:
        data: Data array to save
        reference: Reference NIfTI file or nibabel object for metadata
        output_path: Output file path
        dtype: Optional data type for output (default: same as input)

    Returns:
        nibabel image object

    Example:
        >>> from pytia.nifti_api import save_array_as_nifti
        >>> save_array_as_nifti(tia_data, "reference.nii.gz", "tia.nii.gz")
    """
    if isinstance(reference, (str, Path)):
        ref_img = nib.load(str(reference))
    else:
        ref_img = reference

    if dtype is not None:
        data = data.astype(dtype)

    img = make_like(ref_img, data)
    nib.save(img, str(output_path))
    return img


def get_nifti_info(path: str | Path) -> dict[str, Any]:
    """
    Get information about a NIfTI file.

    Args:
        path: Path to NIfTI file

    Returns:
        Dictionary with file information (shape, affine, voxel sizes, etc.)

    Example:
        >>> info = get_nifti_info("scan.nii.gz")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Voxel sizes: {info['voxel_sizes']}")
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    voxel_sizes = np.abs(img.affine[:3, :3].diagonal())
    voxel_volume_ml = float(np.abs(np.linalg.det(img.affine[:3, :3]))) / 1000.0

    return {
        "shape": data.shape,
        "dtype": str(data.dtype),
        "affine": img.affine.tolist(),
        "voxel_sizes": voxel_sizes.tolist(),
        "voxel_volume_ml": voxel_volume_ml,
        "header": dict(img.header),
    }


def extract_roi_stats(
    tia_img: str | Path | nib.spatialimages.SpatialImage,
    mask_img: str | Path | nib.spatialimages.SpatialImage,
    mask_value: int = 1,
) -> dict[str, float]:
    """
    Extract TIA statistics for a region of interest.

    Args:
        tia_img: TIA NIfTI file or nibabel object
        mask_img: Mask NIfTI file or nibabel object
        mask_value: Value in mask to use as ROI (default: 1)

    Returns:
        Dictionary with statistics (mean, std, min, max, sum, count)

    Example:
        >>> stats = extract_roi_stats("tia.nii.gz", "roi.nii.gz")
        >>> print(f"Mean TIA: {stats['mean']:.2f}")
    """
    if isinstance(tia_img, (str, Path)):
        tia_data = np.asanyarray(nib.load(str(tia_img)).dataobj)
    else:
        tia_data = np.asanyarray(tia_img.dataobj)

    if isinstance(mask_img, (str, Path)):
        mask_data = np.asanyarray(nib.load(str(mask_img)).dataobj)
    else:
        mask_data = np.asanyarray(mask_img.dataobj)

    roi_mask = mask_data == mask_value
    roi_values = tia_data[roi_mask]
    roi_values = roi_values[np.isfinite(roi_values)]

    if len(roi_values) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "sum": np.nan, "count": 0}

    return {
        "mean": float(np.mean(roi_values)),
        "std": float(np.std(roi_values, ddof=1)),
        "min": float(np.min(roi_values)),
        "max": float(np.max(roi_values)),
        "sum": float(np.sum(roi_values)),
        "count": int(len(roi_values)),
    }


def compare_results(
    result1: Results,
    result2: Results,
    metric: str = "tia",
) -> dict[str, float]:
    """
    Compare two TIA results and compute difference statistics.

    Args:
        result1: First Results object
        result2: Second Results object
        metric: Metric to compare: "tia", "r2", "sigma_tia" (default: tia)

    Returns:
        Dictionary with comparison statistics

    Example:
        >>> diff = compare_results(result1, result2, metric="tia")
        >>> print(f"Mean difference: {diff['mean_diff']:.2f}")
    """
    if metric == "tia":
        data1 = result1.tia_img.get_fdata()
        data2 = result2.tia_img.get_fdata()
    elif metric == "r2":
        data1 = result1.r2_img.get_fdata()
        data2 = result2.r2_img.get_fdata()
    elif metric == "sigma_tia":
        data1 = result1.sigma_tia_img.get_fdata()
        data2 = result2.sigma_tia_img.get_fdata()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    diff = data2 - data1
    valid = np.isfinite(diff)

    return {
        "mean_diff": float(np.mean(diff[valid])),
        "std_diff": float(np.std(diff[valid], ddof=1)),
        "min_diff": float(np.min(diff[valid])),
        "max_diff": float(np.max(diff[valid])),
        "abs_mean_diff": float(np.mean(np.abs(diff[valid]))),
        "rmse": float(np.sqrt(np.mean(diff[valid] ** 2))),
    }
