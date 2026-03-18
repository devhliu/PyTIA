from pathlib import Path

import nibabel as nib
import numpy as np

from pytia import run_tia


def _sphere_mask(shape: tuple[int, int, int], center: tuple[int, int, int], radius: float) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    return (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= radius**2


def _build_synthetic_phantom_images() -> tuple[list[nib.Nifti1Image], list[float], np.ndarray]:
    shape = (24, 24, 24)
    tumor = _sphere_mask(shape, (10, 12, 12), radius=3.0)
    kidney = _sphere_mask(shape, (15, 12, 12), radius=4.0)

    times_h = [4.0, 24.0, 72.0]
    times_s = np.asarray(times_h) * 3600.0

    images: list[nib.Nifti1Image] = []
    for t in times_s:
        vol = np.zeros(shape, dtype=np.float32)
        vol[tumor] = 80.0 * np.exp(-np.log(2.0) * t / 32_000.0)
        vol[kidney] = 40.0 * np.exp(-np.log(2.0) * t / 18_000.0)
        images.append(nib.Nifti1Image(vol, np.eye(4)))

    return images, times_h, tumor | kidney


def _phantom_cfg(output_dir: Path, bootstrap_enabled: bool) -> dict:
    return {
        "io": {"output_dir": str(output_dir)},
        "time": {"unit": "hours", "sort_timepoints": True},
        "physics": {"half_life_seconds": 6.647 * 24 * 3600.0},
        "mask": {"mode": "otsu", "min_fraction_of_max": 0.02},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
        "integration": {"tail_mode": "phys", "rising_tail_mode": "phys"},
        "bootstrap": {"enabled": bootstrap_enabled, "n": 8, "seed": 123},
        "single_time": {"enabled": False},
    }


def test_synthetic_lu177_like_phantom_runs_end_to_end(tmp_path: Path) -> None:
    images, times_h, roi = _build_synthetic_phantom_images()
    cfg = _phantom_cfg(tmp_path / "out", bootstrap_enabled=False)

    results = run_tia(images=images, times=times_h, config=cfg)

    tia = results.tia_img.get_fdata()
    assert np.isfinite(tia[roi]).all()
    assert results.summary["status_counts"].get("ok", 0) > 0

    for path in results.output_paths.values():
        assert Path(path).exists()


def test_synthetic_lu177_like_phantom_bootstrap_outputs_sigma(tmp_path: Path) -> None:
    images, times_h, roi = _build_synthetic_phantom_images()
    cfg = _phantom_cfg(tmp_path / "out-bootstrap", bootstrap_enabled=True)

    results = run_tia(images=images, times=times_h, config=cfg)

    sigma = results.sigma_tia_img.get_fdata()
    assert np.isfinite(sigma[roi]).sum() > 0
