import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from pytia.engine import run_tia


def test_region_roi_scaling_amplitude():
    # 2 voxels in region 1, 2 timepoints
    shape = (2, 1, 1)
    aff = np.diag([1.0, 1.0, 1.0, 1.0])

    # density Bq/ml; voxel volume = 1 mm^3 = 0.001 ml
    # voxel0 has twice amplitude of voxel1 at ref; expect TIA voxel0 ~ 2x voxel1
    tp1 = np.zeros(shape, np.float32)
    tp2 = np.zeros(shape, np.float32)
    tp1[0, 0, 0] = 10.0
    tp1[1, 0, 0] = 5.0
    tp2[0, 0, 0] = 10.0
    tp2[1, 0, 0] = 5.0

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
                "classes": {"1": {"class": "rising", "allowed_models": ["hybrid"], "default_model": "hybrid"}},
                "scaling": {"mode": "tref", "reference_time": "last"},
            },
        }

        res = run_tia(images=[img1, img2], times=[1.0, 2.0], config=cfg)
        tia = np.asanyarray(res.tia_img.dataobj)
        assert np.isfinite(tia[0, 0, 0]) and np.isfinite(tia[1, 0, 0])
        ratio = tia[0, 0, 0] / tia[1, 0, 0]
        assert 1.8 < ratio < 2.2