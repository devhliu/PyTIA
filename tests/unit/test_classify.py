import numpy as np

from pytia.classify import (
    CLASS_AMBIG,
    CLASS_FALLING,
    CLASS_HUMP,
    CLASS_RISING,
    classify_curves,
)


def test_classify_curves_detects_core_shapes() -> None:
    A = np.array(
        [
            [10.0, 7.0, 4.0, 1.0],  # falling
            [1.0, 2.0, 4.0, 8.0],  # rising
            [1.0, 5.0, 3.0, 1.0],  # hump
            [2.0, 2.0, 2.0, 2.0],  # ambiguous
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(A, dtype=bool)

    cls = classify_curves(A, valid)

    assert cls.dtype == np.uint8
    np.testing.assert_array_equal(
        cls,
        np.array([CLASS_FALLING, CLASS_RISING, CLASS_HUMP, CLASS_AMBIG], dtype=np.uint8),
    )


def test_classify_curves_respects_valid_mask() -> None:
    A = np.array([[1.0, 3.0, 10.0, 1.0]], dtype=np.float32)
    valid = np.array([[True, True, False, False]])

    cls = classify_curves(A, valid)

    assert cls.shape == (1,)
    assert cls[0] == CLASS_RISING
