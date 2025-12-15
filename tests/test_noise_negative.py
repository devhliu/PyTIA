import numpy as np

from pytia.noise import clamp_negative_to_zero, compute_noise_floor, valid_mask_from_floor


def test_negative_clamp_and_floor_exclude():
    A = np.array([[-1.0, 0.0, 10.0]], dtype=np.float32)
    A = clamp_negative_to_zero(A)
    assert np.all(A >= 0)

    floor = compute_noise_floor(A, mode="relative", absolute=0.0, rel_frac=0.2)  # 2.0
    valid = valid_mask_from_floor(A, floor)
    assert valid.shape == A.shape
    assert np.array_equal(valid[0], np.array([False, False, True]))