#!/usr/bin/env python3
"""Test gamma linear fix with dummy data."""

import numpy as np
from pytia.models.gamma_linear import fit_gamma_linear_wls

# Create dummy data similar to what we have
# N voxels, T timepoints
N = 100
T = 4
times = np.array([4.0, 24.0, 48.0, 168.0]) * 3600  # Convert to seconds

# Create synthetic A values (positive activities)
A = np.random.rand(N, T) * 1000 + 100
valid = np.ones((N, T), dtype=bool)

# Add some zeros to test log handling
A[0, 0] = 0
valid[0, 0] = False

print(f"Input shapes: A={A.shape}, times={times.shape}, valid={valid.shape}")
print(f"A min/max: {np.min(A[valid]):.1f} / {np.max(A[valid]):.1f}")

# Test the function
try:
    params, tpk, Ahat, r2 = fit_gamma_linear_wls(A, times, valid, lambda_phys=1e-6)
    print(f"Success! Output shapes: params={params.shape}, tpk={tpk.shape}, Ahat={Ahat.shape}, r2={r2.shape}")
    print(f"Non-NaN TIA outputs: {np.sum(~np.isnan(params))}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()