# PyTIA: Voxel-Wise Time-Integrated Activity Maps

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PyTIA** is a Python package for computing voxel-wise time-integrated activity (TIA) maps from PET/SPECT imaging data.

## Features

### Multi-Timepoint Analysis
- Supports 2 or more activity images at different timepoints
- Automatic curve classification: rising, hump (gamma), falling (exponential)
- Advanced fitting models with physical decay tail extrapolation
- Bootstrap uncertainty quantification

### Single-Timepoint Analysis
Calculate TIA from a single activity map using one of three methods:

1. **Physical Decay** — Pure radioactive decay extrapolation
2. **Hänscheid Method** — Effective half-life (accounting for biological clearance)
3. **Prior Half-Life** — Global or organ/lesion-specific half-lives from segmentation

### Processing Features
- Automatic masking and denoising
- Noise floor filtering
- Regional ROI aggregation
- Comprehensive status tracking

## Installation

```bash
pip install pytia
```

## Quick Start

### CLI
```bash
pytia run --config config.yaml
```

### Python API
```python
from pytia import run_tia

result = run_tia(
    images=["activity_t0.nii.gz", "activity_t1.nii.gz", "activity_t2.nii.gz"],
    times=[0.0, 30.0, 60.0],
    config={"physics": {"half_life_seconds": 21600.0}, "io": {"output_dir": "./output"}}
)
```

## Documentation

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — Comprehensive usage guide
- **[examples/](examples/)** — Example scripts and configs
- **[docs/CONFIG.md](docs/CONFIG.md)** — Configuration reference

## License

MIT License — see [LICENSE](LICENSE)

PyTIA computes voxel-wise
