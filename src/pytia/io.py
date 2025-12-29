from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np


def load_images(images: Sequence[str | Path | nib.spatialimages.SpatialImage]) -> list[nib.spatialimages.SpatialImage]:
    out: list[nib.spatialimages.SpatialImage] = []
    for im in images:
        if isinstance(im, (str, Path)):
            out.append(nib.load(str(im)))
        else:
            out.append(im)
    return out


def stack_4d(imgs: Sequence[nib.spatialimages.SpatialImage]) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    if len(imgs) < 2:
        raise ValueError("Need at least 2 timepoints/images.")
    ref = imgs[0]
    shape3 = ref.shape[:3]
    aff = ref.affine
    for im in imgs[1:]:
        if im.shape[:3] != shape3:
            raise ValueError("All images must have same 3D shape.")
        if not np.allclose(im.affine, aff):
            raise ValueError("All images must have same affine.")
    data4 = np.stack([np.asanyarray(im.dataobj).astype(np.float32) for im in imgs], axis=-1)
    return data4, ref


def voxel_volume_ml(img: nib.spatialimages.SpatialImage) -> float:
    # affine encodes mm; |det| gives mm^3; 1 ml = 1000 mm^3
    det = float(np.linalg.det(img.affine[:3, :3]))
    return abs(det) / 1000.0


def make_like(ref: nib.spatialimages.SpatialImage, data: np.ndarray) -> nib.spatialimages.SpatialImage:
    return nib.Nifti1Image(data, affine=ref.affine, header=ref.header)


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path