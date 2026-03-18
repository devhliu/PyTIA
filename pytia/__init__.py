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
from .engine import run_tia
from .io import load_images, make_like, stack_4d, voxel_volume_ml
from .types import Results
from .version import __version__

__all__ = [
    "__version__",
    "Config",
    "Results",
    "load_images",
    "make_like",
    "run_tia",
    "stack_4d",
    "voxel_volume_ml",
]
