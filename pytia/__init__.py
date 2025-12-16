"""
PyTIA: Voxel-wise Time-Integrated Activity maps from PET/SPECT.

Supports:
- Multi-timepoint TIA calculation (2 or more images)
- Single-timepoint TIA calculation (1 image with 3 methods)
  * Physical decay
  * Hänscheid method (effective half-life)
  * Prior half-life (global or segmentation-based)
"""

from .config import Config
from .engine import Results, run_tia
from .io import load_images, voxel_volume_ml, make_like, stack_4d

__version__ = "0.1.0"

__all__ = [
    "run_tia",
    "Results",
    "Config",
    "load_images",
    "voxel_volume_ml",
    "make_like",
    "stack_4d",
]