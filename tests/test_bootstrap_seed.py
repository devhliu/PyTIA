import numpy as np
import nibabel as nib
import tempfile
from pathlib import Path

from pytia.engine import run_tia


def test_bootstrap_reproducible_seed():
    shape = (2, 2, 1)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])

    tp1 = np.ones(shape, np.float32) * 10.0
    tp2 = np.ones(shape, np.float32) * 8.0
    img1 = nib.Nifti1Image(tp1, aff)
    img2 = nib.Nifti1Image(tp2, aff)

    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "io": {"output_dir": d, "prefix": "x"},
            "time": {"unit": "seconds", "sort_timepoints": False},
            "physics": {"half_life_seconds": 1000.0, "enforce_lambda_ge_phys": True},
            "mask": {"mode": "none"},
            "denoise": {"enabled": False},
            "noise_floor": {"enabled": False},
            "bootstrap": {"enabled": True, "n": 10, "seed": 123, "reclassify_each_replicate": True},
        }
        r1 = run_tia(images=[img1, img2], times=[1.0, 2.0], config=cfg)
        s1 = np.asanyarray(r1.sigma_tia_img.dataobj).copy()
        r2 = run_tia(images=[img1, img2], times=[1.0, 2.0], config=cfg)
        s2 = np.asanyarray(r2.sigma_tia_img.dataobj).copy()
        assert np.allclose(s1, s2, equal_nan=True)