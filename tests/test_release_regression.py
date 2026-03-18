"""Deterministic release-regression checks for product-level behavior."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from pytia.engine import MODEL_MONOEXP, run_tia
from pytia.version import __version__


def _save_constant_nifti(path: Path, value: float, shape: tuple[int, int, int]) -> Path:
    nib.save(
        nib.Nifti1Image(np.full(shape, value, dtype=np.float32), np.eye(4)),
        path,
    )
    return path


def test_release_golden_falling_curve_distribution() -> None:
    """Falling synthetic TAC should remain monoexp with stable TIA magnitude."""
    shape = (4, 4, 4)
    values = [3.0, 2.0, 1.0]
    times = [3600.0, 7200.0, 14400.0]

    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        image_paths = [
            _save_constant_nifti(work / f"tp_{i}.nii.gz", value, shape)
            for i, value in enumerate(values)
        ]
        cfg = {
            "io": {"output_dir": str(work / "out")},
            "time": {"unit": "seconds", "sort_timepoints": True},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "noise_floor": {"enabled": False},
            "bootstrap": {"enabled": False},
        }

        result = run_tia(images=image_paths, times=times, config=cfg)
        tia = np.asarray(result.tia_img.dataobj)
        model_id = np.asarray(result.model_id_img.dataobj)
        status = np.asarray(result.status_id_img.dataobj)

        np.testing.assert_array_equal(np.unique(status), np.array([1], dtype=np.uint8))
        np.testing.assert_array_equal(
            np.unique(model_id),
            np.array([MODEL_MONOEXP], dtype=np.uint8),
        )
        np.testing.assert_allclose(float(tia[0, 0, 0]), 20.98110580444336, rtol=1e-7, atol=1e-7)
        assert result.summary["status_counts"] == {"ok": int(np.prod(shape))}
        assert result.summary["model_counts"] == {"monoexp": int(np.prod(shape))}


def test_release_summary_contract_fields() -> None:
    """Summary contract should expose version and model legend metadata."""
    shape = (2, 2, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        images = [
            _save_constant_nifti(work / "t0.nii.gz", 10.0, shape),
            _save_constant_nifti(work / "t1.nii.gz", 8.0, shape),
            _save_constant_nifti(work / "t2.nii.gz", 6.0, shape),
        ]
        cfg = {
            "io": {"output_dir": str(work / "out")},
            "time": {"unit": "seconds"},
            "physics": {"half_life_seconds": 3600.0},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "noise_floor": {"enabled": False},
            "bootstrap": {"enabled": False},
        }
        result = run_tia(images=images, times=[1.0, 2.0, 3.0], config=cfg)
        assert result.summary["pytia_version"] == __version__
        assert result.summary["model_legend"][MODEL_MONOEXP] == "monoexp"
