import numpy as np

from pytia.models.hybrid import tia_trapz_plus_phys_tail


def test_hybrid_trapz_plus_tail_simple():
    # A(t) = [0, 10] at t=[1,2], include_t0 adds (0,0)
    A = np.array([[0.0, 10.0]], dtype=np.float32)
    t = np.array([1.0, 2.0], dtype=np.float64)
    valid = np.array([[True, True]])
    lambda_phys = 0.5

    tia, _, _ = tia_trapz_plus_phys_tail(A, t, valid, lambda_phys=lambda_phys, include_t0=True)
    # trapezoid segments: (0->1):0 ; (1->2): 0.5*(0+10)*1 = 5
    # tail: A_last/lambda = 10/0.5 = 20 => total 25
    assert np.isclose(tia[0], 25.0)