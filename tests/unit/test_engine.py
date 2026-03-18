from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pytia.engine import MODEL_SINGLE_TIME_PHYS, _times_to_seconds, run_tia


def _save_img(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    aff = np.eye(4) if affine is None else affine
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff), path)
    return path


def _base_cfg(output_dir: Path) -> dict:
    return {
        "io": {"output_dir": str(output_dir)},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
        "bootstrap": {"enabled": False},
        "single_time": {"enabled": False},
    }


def test_times_to_seconds_supports_hours_and_seconds() -> None:
    np.testing.assert_allclose(_times_to_seconds([1.0, 2.0], "seconds"), [1.0, 2.0])
    np.testing.assert_allclose(_times_to_seconds([1.0, 2.0], "hours"), [3600.0, 7200.0])

    with pytest.raises(ValueError, match="Unsupported time unit"):
        _times_to_seconds([1.0], "minutes")


def test_run_tia_single_time_phys_produces_expected_tia(tmp_path: Path) -> None:
    img_path = _save_img(tmp_path / "tp1.nii.gz", np.full((3, 3, 3), 1000.0, dtype=np.float32))
    cfg = _base_cfg(tmp_path / "out-stp")
    cfg["single_time"] = {"enabled": True, "method": "phys"}
    cfg["physics"] = {"half_life_seconds": 3600.0}

    result = run_tia(images=[img_path], times=[1.0], config=cfg)

    lam = np.log(2.0) / 3600.0
    vml = 1.0 / 1000.0
    expected = 1000.0 * vml / lam

    center = float(result.tia_img.get_fdata()[1, 1, 1])
    assert np.isclose(center, expected, rtol=1e-5)
    assert int(result.model_id_img.get_fdata()[1, 1, 1]) == MODEL_SINGLE_TIME_PHYS
    assert result.summary["model_counts"]["single-time phys"] > 0


def test_run_tia_single_time_with_otsu_mask_has_valid_summary(tmp_path: Path) -> None:
    data = np.zeros((5, 5, 5), dtype=np.float32)
    data[2, 2, 2] = 500.0
    img_path = _save_img(tmp_path / "tp1_sparse.nii.gz", data)

    cfg = _base_cfg(tmp_path / "out-otsu")
    cfg["mask"] = {"mode": "otsu", "min_fraction_of_max": 0.02}
    cfg["single_time"] = {"enabled": True, "method": "phys"}
    cfg["physics"] = {"half_life_seconds": 3600.0}

    result = run_tia(images=[img_path], times=[1.0], config=cfg)

    assert "single-time phys" in result.summary["model_counts"]
    assert result.summary["status_counts"].get("ok", 0) >= 1


def test_run_tia_sorts_timepoints_when_enabled(tmp_path: Path) -> None:
    img_late = _save_img(tmp_path / "late.nii.gz", np.full((2, 2, 2), 2.0, dtype=np.float32))
    img_early = _save_img(tmp_path / "early.nii.gz", np.full((2, 2, 2), 4.0, dtype=np.float32))

    cfg = _base_cfg(tmp_path / "out-sort")
    cfg["integration"] = {"tail_mode": "none", "rising_tail_mode": "peak_at_last"}
    cfg["time"] = {"unit": "seconds", "sort_timepoints": True}

    result = run_tia(images=[img_late, img_early], times=[2.0, 1.0], config=cfg)

    np.testing.assert_allclose(result.times_s, [1.0, 2.0])


def test_run_tia_rejects_mismatched_image_shapes(tmp_path: Path) -> None:
    img1 = _save_img(tmp_path / "a.nii.gz", np.zeros((3, 3, 3), dtype=np.float32))
    img2 = _save_img(tmp_path / "b.nii.gz", np.zeros((4, 3, 3), dtype=np.float32))
    cfg = _base_cfg(tmp_path / "out-mismatch")

    with pytest.raises(ValueError, match="same 3D shape"):
        run_tia(images=[img1, img2], times=[1.0, 2.0], config=cfg)
