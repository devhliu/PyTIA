"""Image I/O: loading, stacking, and spatial helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np


def load_images(
    images: str
    | Path
    | nib.spatialimages.SpatialImage
    | Sequence[str | Path | nib.spatialimages.SpatialImage],
) -> list[nib.spatialimages.SpatialImage]:
    if isinstance(images, (str, Path)) or hasattr(images, "shape"):
        images = [images]

    out: list[nib.spatialimages.SpatialImage] = []
    for im in images:
        if isinstance(im, (str, Path)):
            out.append(nib.load(str(im)))
        else:
            out.append(im)

    if len(out) == 0:
        raise ValueError("Need at least 1 timepoint/image.")

    return out


def stack_4d(
    imgs: Sequence[nib.spatialimages.SpatialImage],
) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    if len(imgs) < 1:
        raise ValueError("Need at least 1 timepoint/image.")
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


def make_like(
    ref: nib.spatialimages.SpatialImage, data: np.ndarray
) -> nib.spatialimages.SpatialImage:
    return nib.Nifti1Image(data, affine=ref.affine, header=ref.header)


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
