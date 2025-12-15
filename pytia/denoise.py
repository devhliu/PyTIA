from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def masked_gaussian(data4: np.ndarray, mask3: np.ndarray, sigma_vox: float) -> np.ndarray:
    """
    Smooth each timepoint within mask without bleeding:
      smooth(A*mask)/smooth(mask)
    """
    if sigma_vox is None or sigma_vox <= 0:
        return data4
    mask_f = mask3.astype(np.float32)
    denom = gaussian_filter(mask_f, sigma=sigma_vox)
    denom = np.maximum(denom, 1e-6)
    out = np.empty_like(data4, dtype=np.float32)
    for t in range(data4.shape[-1]):
        num = gaussian_filter(data4[..., t] * mask_f, sigma=sigma_vox)
        out[..., t] = num / denom
        out[..., t] *= mask_f
    return out