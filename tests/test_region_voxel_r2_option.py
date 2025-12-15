import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from pytia.engine import run_tia


def test_region_voxel_level_r2():
    shape = (2, 1, 1)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    tp1 = np.zeros(shape, np.float32)
    tp2 = np.zeros(shape, np.float32)
    tp1[:, 0, 0] = [10.0, 5.0]
    tp2[:, 0, 0] = [10.0, 5.0]
    img1 = nib.Nifti1Image(tp1, aff)
    img2 = nib.Nifti1Image(tp2, aff)

    labels = np.zeros(shape, np.int16)
    labels[:, 0, 0] = 1
    lab_img = nib.Nifti1Image(labels, aff)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lab_path = d / "labels.nii.gz"
        nib.save(lab_img, str(lab_path))

        cfg = {
            "io": {"output_dir": str(d), "prefix": None},
            "time": {"unit": "seconds", "sort_timepoints": False},
            "physics": {"half_life_seconds": 1000.0, "enforce_lambda_ge_phys": True},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "noise_floor": {"enabled": False},
            "bootstrap": {"enabled": False},
            "regions": {
                "enabled": True,
                "label_map_path": str(lab_path),
                "mode": "roi_aggregate",
                "aggregation": "mean",
                "voxel_level_r2": True,
                "classes": {"1": {"class": "rising", "allowed_models": ["hybrid"], "default_model": "hybrid"}},
                "scaling": {"mode": "tref", "reference_time": "last"},
            },
        }

        res = run_tia(images=[img1, img2], times=[1.0, 2.0], config=cfg)
        r2 = np.asanyarray(res.r2_img.dataobj)
        assert np.all(np.isfinite(r2))