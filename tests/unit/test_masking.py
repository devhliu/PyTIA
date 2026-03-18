from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pytia.masking import load_mask, make_body_mask, mask_to_bool


def test_make_body_mask_handles_empty_and_constant_inputs() -> None:
    empty = np.zeros((4, 4, 4, 2), dtype=np.float32)
    mask_empty = make_body_mask(empty)
    assert not mask_empty.any()

    constant = np.ones((4, 4, 4, 3), dtype=np.float32)
    mask_constant = make_body_mask(constant)
    assert mask_constant.all()


def test_make_body_mask_applies_fraction_threshold() -> None:
    data4 = np.zeros((5, 5, 5, 2), dtype=np.float32)
    data4[2, 2, 2, :] = 100.0
    data4[1, 1, 1, :] = 1.0

    mask = make_body_mask(data4, min_fraction_of_max=0.1)

    assert mask[2, 2, 2]
    assert not mask[1, 1, 1]


def test_load_mask_and_mask_to_bool(tmp_path: Path) -> None:
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[1, 1, 1] = 1
    path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)

    mimg = load_mask(path)
    mask = mask_to_bool(mimg, (3, 3, 3))

    assert mask.shape == (3, 3, 3)
    assert mask.dtype == bool
    assert mask[1, 1, 1]


def test_mask_to_bool_rejects_wrong_shape() -> None:
    mimg = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="shape does not match"):
        mask_to_bool(mimg, (3, 3, 3))
