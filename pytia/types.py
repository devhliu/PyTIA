from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass
class Results:
    tia_img: nib.spatialimages.SpatialImage
    r2_img: nib.spatialimages.SpatialImage
    sigma_tia_img: nib.spatialimages.SpatialImage
    model_id_img: nib.spatialimages.SpatialImage
    status_id_img: nib.spatialimages.SpatialImage
    tpeak_img: nib.spatialimages.SpatialImage | None

    summary: dict[str, Any]
    output_paths: dict[str, Path]
    config: dict[str, Any]
    times_s: np.ndarray