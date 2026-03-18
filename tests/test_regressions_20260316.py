"""Regression tests for fixes implemented on 2026-03-16."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pytia.classify import CLASS_AMBIG, classify_curves
from pytia.config import Config
from pytia.engine import run_tia


def _save_nifti(path: Path, data: np.ndarray) -> Path:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    nib.save(img, path)
    return path


def test_single_timepoint_accepts_scalar_path(tmp_path: Path) -> None:
    """run_tia should accept a scalar path for images input."""
    p = _save_nifti(tmp_path / "t0.nii.gz", np.full((4, 4, 4), 100.0, dtype=np.float32))

    cfg = {
        "io": {"output_dir": str(tmp_path)},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "physics": {"half_life_seconds": 3600.0},
        "single_time": {"enabled": True, "method": "phys"},
        "bootstrap": {"enabled": False},
        "time": {"unit": "seconds"},
    }

    result = run_tia(images=str(p), times=[0.0], config=cfg)
    tia = np.asarray(result.tia_img.dataobj)

    assert np.any(np.isfinite(tia))


def test_chunked_bootstrap_populates_sigma(tmp_path: Path) -> None:
    """Chunked voxel-mode bootstrap should produce finite sigma for fitted voxels."""
    # Hump-like TAC to trigger model fitting path with bootstrap baselines.
    p0 = _save_nifti(tmp_path / "t0.nii.gz", np.full((5, 5, 5), 1.0, dtype=np.float32))
    p1 = _save_nifti(tmp_path / "t1.nii.gz", np.full((5, 5, 5), 3.0, dtype=np.float32))
    p2 = _save_nifti(tmp_path / "t2.nii.gz", np.full((5, 5, 5), 2.0, dtype=np.float32))

    cfg = {
        "io": {"output_dir": str(tmp_path)},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
        "physics": {"half_life_seconds": 3600.0},
        "bootstrap": {"enabled": True, "n": 8, "seed": 7, "reclassify_each_replicate": False},
        "performance": {"chunk_size_vox": 20},
        "model_selection": {"min_points_for_gamma": 3},
        "time": {"unit": "seconds"},
    }

    result = run_tia(images=[p0, p1, p2], times=[1.0, 2.0, 3.0], config=cfg)
    sigma = np.asarray(result.sigma_tia_img.dataobj)
    status = np.asarray(result.status_id_img.dataobj)

    ok = status == 1
    assert np.any(ok)
    assert np.any(np.isfinite(sigma[ok]))


def test_classify_all_invalid_rows_no_warnings() -> None:
    """Classification should handle all-invalid rows without warning-heavy nan reductions."""
    A = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    valid = np.array([[False, False, False], [True, True, True]], dtype=bool)
    cls = classify_curves(A, valid)
    assert cls[0] == CLASS_AMBIG


def test_config_validation_rejects_invalid_enum() -> None:
    """Invalid enum values should fail fast at config-load time."""
    with pytest.raises(ValueError, match="Invalid mask.mode"):
        Config.load({"mask": {"mode": "bad_mode"}})


def test_low_memory_input_mode_matches_default(tmp_path: Path) -> None:
    """Low-memory input path should match default outputs on the same data."""
    p0 = _save_nifti(tmp_path / "lm_t0.nii.gz", np.full((4, 4, 4), 2.0, dtype=np.float32))
    p1 = _save_nifti(tmp_path / "lm_t1.nii.gz", np.full((4, 4, 4), 1.0, dtype=np.float32))
    p2 = _save_nifti(tmp_path / "lm_t2.nii.gz", np.full((4, 4, 4), 0.5, dtype=np.float32))
    images = [p0, p1, p2]
    times = [1.0, 2.0, 3.0]

    base_cfg = {
        "io": {"output_dir": str(tmp_path)},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
        "physics": {"half_life_seconds": 3600.0},
        "bootstrap": {"enabled": False},
        "time": {"unit": "seconds"},
    }
    low_cfg = dict(base_cfg)
    low_cfg["performance"] = {"low_memory_input": True}

    res_default = run_tia(images=images, times=times, config=base_cfg)
    res_low = run_tia(images=images, times=times, config=low_cfg)

    np.testing.assert_allclose(
        np.asarray(res_default.tia_img.dataobj),
        np.asarray(res_low.tia_img.dataobj),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_parallel_bootstrap_is_deterministic(tmp_path: Path) -> None:
    """Parallel bootstrap should remain deterministic for a fixed seed."""
    p0 = _save_nifti(tmp_path / "pb_t0.nii.gz", np.full((4, 4, 4), 1.0, dtype=np.float32))
    p1 = _save_nifti(tmp_path / "pb_t1.nii.gz", np.full((4, 4, 4), 3.0, dtype=np.float32))
    p2 = _save_nifti(tmp_path / "pb_t2.nii.gz", np.full((4, 4, 4), 2.0, dtype=np.float32))

    cfg = {
        "io": {"output_dir": str(tmp_path)},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
        "physics": {"half_life_seconds": 3600.0},
        "bootstrap": {"enabled": True, "n": 8, "seed": 1234, "reclassify_each_replicate": True},
        "performance": {"parallel_bootstrap": True, "parallel_workers": 2, "chunk_size_vox": 10},
        "model_selection": {"min_points_for_gamma": 3},
        "time": {"unit": "seconds"},
    }

    r1 = run_tia(images=[p0, p1, p2], times=[1.0, 2.0, 3.0], config=cfg)
    r2 = run_tia(images=[p0, p1, p2], times=[1.0, 2.0, 3.0], config=cfg)
    np.testing.assert_allclose(
        np.asarray(r1.sigma_tia_img.dataobj),
        np.asarray(r2.sigma_tia_img.dataobj),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )
