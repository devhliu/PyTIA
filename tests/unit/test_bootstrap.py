import numpy as np

from pytia.uncertainty import residual_bootstrap


def test_residual_bootstrap_preserves_shape_and_nonnegativity() -> None:
    A = np.array([[4.0, 6.0, 2.0], [0.5, 1.5, 3.0]], dtype=np.float32)
    Ahat = np.array([[3.0, 5.0, 2.5], [1.0, 1.0, 2.0]], dtype=np.float32)
    valid = np.array([[True, True, True], [True, False, True]])

    out = residual_bootstrap(A, Ahat, valid, np.random.default_rng(42))

    assert out.shape == A.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_residual_bootstrap_is_reproducible_for_same_seed() -> None:
    A = np.array([[1.0, 3.0, 2.0, 4.0]], dtype=np.float32)
    Ahat = np.array([[1.2, 2.8, 2.1, 3.9]], dtype=np.float32)
    valid = np.ones_like(A, dtype=bool)

    out1 = residual_bootstrap(A, Ahat, valid, np.random.default_rng(7))
    out2 = residual_bootstrap(A, Ahat, valid, np.random.default_rng(7))

    np.testing.assert_allclose(out1, out2)


def test_residual_bootstrap_returns_ahat_when_no_valid_points() -> None:
    A = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    Ahat = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    valid = np.zeros_like(A, dtype=bool)

    out = residual_bootstrap(A, Ahat, valid, np.random.default_rng(1))

    np.testing.assert_allclose(out, Ahat)
