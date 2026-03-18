import numpy as np
import warnings

from pytia.noise import clamp_negative_to_zero, compute_noise_floor, valid_mask_from_floor


def test_negative_clamp_and_floor_exclude():
    A = np.array([[-1.0, 0.0, 10.0]], dtype=np.float32)
    A = clamp_negative_to_zero(A)
    assert np.all(A >= 0)

    floor = compute_noise_floor(A, mode="relative", absolute=0.0, rel_frac=0.2)  # 2.0
    valid = valid_mask_from_floor(A, floor)
    assert valid.shape == A.shape
    assert np.array_equal(valid[0], np.array([False, False, True]))


def test_compute_noise_floor_relative_handles_all_nan_rows_without_warning():
    A = np.array([[np.nan, np.nan], [1.0, np.nan]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        floor = compute_noise_floor(A, mode="relative", absolute=0.0, rel_frac=0.1)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime_warnings
    assert np.isnan(floor[0])
    assert np.isclose(floor[1], 0.1)
