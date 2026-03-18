from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pytia.io import ensure_dir, load_images, make_like, stack_4d, voxel_volume_ml


def _img(data: np.ndarray, affine: np.ndarray | None = None) -> nib.Nifti1Image:
    return nib.Nifti1Image(data.astype(np.float32), np.eye(4) if affine is None else affine)


def test_load_images_supports_paths_and_objects(tmp_path: Path) -> None:
    path = tmp_path / "tp1.nii.gz"
    nib.save(_img(np.ones((2, 2, 2))), path)

    loaded = load_images([path, _img(np.zeros((2, 2, 2)))])

    assert len(loaded) == 2
    assert all(hasattr(x, "shape") for x in loaded)


def test_load_images_rejects_empty_collection() -> None:
    with pytest.raises(ValueError, match="Need at least 1 timepoint"):
        load_images([])


def test_stack_4d_checks_shape_and_affine() -> None:
    im1 = _img(np.ones((2, 2, 2)), np.diag([1.0, 1.0, 1.0, 1.0]))
    im2 = _img(np.ones((2, 2, 2)), np.diag([1.0, 1.0, 1.0, 1.0]))

    data4, ref = stack_4d([im1, im2])

    assert data4.shape == (2, 2, 2, 2)
    assert ref.shape == (2, 2, 2)

    with pytest.raises(ValueError, match="same 3D shape"):
        stack_4d([im1, _img(np.ones((3, 2, 2)))])

    with pytest.raises(ValueError, match="same affine"):
        stack_4d([im1, _img(np.ones((2, 2, 2)), np.diag([2.0, 1.0, 1.0, 1.0]))])


def test_voxel_volume_ml_and_make_like() -> None:
    affine = np.diag([2.0, 5.0, 10.0, 1.0])
    ref = _img(np.zeros((2, 2, 2)), affine)

    vml = voxel_volume_ml(ref)
    assert np.isclose(vml, 100.0 / 1000.0)

    made = make_like(ref, np.ones((2, 2, 2), dtype=np.float32))
    np.testing.assert_allclose(made.affine, ref.affine)
    assert made.shape == ref.shape


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    out = ensure_dir(tmp_path / "nested" / "out")
    assert out.exists()
    assert out.is_dir()
