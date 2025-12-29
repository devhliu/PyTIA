from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu


def make_body_mask(data4: np.ndarray, min_fraction_of_max: float = 0.02) -> np.ndarray:
    """
    Create mask using Otsu on sum image, then also require >= min_fraction_of_max * max(sum).
    data4: (X,Y,Z,T) activity density or Bq (non-negative recommended).
    """
    s = np.sum(data4, axis=-1)
    flat = s[np.isfinite(s)]
    if flat.size == 0:
        return np.zeros(s.shape, dtype=bool)
    thr = threshold_otsu(flat)
    m = s > thr
    if min_fraction_of_max is not None and min_fraction_of_max > 0:
        m &= s >= (float(np.nanmax(s)) * float(min_fraction_of_max))
    return m


def load_mask(mask_path: str | Path) -> nib.spatialimages.SpatialImage:
    return nib.load(str(mask_path))


def mask_to_bool(mask_img: nib.spatialimages.SpatialImage, ref_shape: tuple[int, int, int]) -> np.ndarray:
    m = np.asanyarray(mask_img.dataobj)
    if m.shape[:3] != ref_shape:
        raise ValueError("Provided mask shape does not match input images.")
    return m.astype(bool)