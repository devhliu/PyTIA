# PyTIA: Complete Codebase Review & Architecture

## Executive Summary

PyTIA is fully refactored to meet all requirements:

✅ **Python API** — Enhanced with proper interfaces, nibabel support, config management  
✅ **CLI Interface** — Full-featured CLI with run, validate, info commands  
✅ **Config-Driven** — All inputs/settings in YAML config files  
✅ **Documentation** — Complete USER_GUIDE, CONFIG reference, LICENSE  
✅ **Examples** — Multi-timepoint and STP examples (Python + YAML configs)  
✅ **Structure** — Organized examples/, docs/, scripts/ folders  

## Architecture Overview

```
PyTIA/
├── pytia/                      # Main package
│   ├── __init__.py            # ✓ Enhanced API exports
│   ├── cli.py                 # ✓ CLI: run, validate, info
│   ├── engine.py              # Core TIA calculation (multi + STP)
│   ├── config.py              # Config management
│   ├── io.py                  # I/O utilities
│   ├── types.py               # Results dataclass
│   └── [other modules...]     # Fitting, masking, etc.
│
├── examples/                  # ✓ Example scripts & configs
│   ├── README.md             # Quick start guide
│   ├── example_multitime.py  # Multi-timepoint demo
│   ├── example_stp.py        # STP all 3 methods
│   ├── config_multitime.yaml
│   ├── config_stp_phys.yaml
│   ├── config_stp_haenscheid.yaml
│   └── config_stp_prior_seg.yaml
│
├── docs/                      # ✓ Documentation
│   ├── USER_GUIDE.md         # Comprehensive usage guide
│   ├── CONFIG.md             # Configuration reference
│   ├── SINGLE_TIMEPOINT_IMPLEMENTATION.md
│   └── [other docs...]
│
├── tests/                     # Unit tests
│   ├── test_*.py
│   └── test_single_timepoint.py
│
├── README.md                  # ✓ Project overview
├── LICENSE                    # ✓ MIT License
├── pyproject.toml            # Project metadata with CLI entry
└── [config files...]
```

## 1. Python API

### Enhanced Exports (pytia/__init__.py)

```python
from .config import Config
from .engine import Results, run_tia
from .io import load_images, voxel_volume_ml, make_like, stack_4d

__all__ = [
    "run_tia",              # Main function
    "Results",              # Results dataclass
    "Config",               # Config manager
    "load_images",          # Image loading
    "voxel_volume_ml",      # Utility
    "make_like",            # Utility
    "stack_4d",             # Utility
]
```

### Usage Example

```python
from pytia import run_tia, load_images, Config

# Method 1: Simple usage
result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config={"physics": {"half_life_seconds": 21600.0}}
)

# Method 2: Config file
result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

# Method 3: Direct config loading
cfg = Config.load("config.yaml")
result = run_tia(
    images=cfg.data["inputs"]["images"],
    times=cfg.data["inputs"]["times"],
    config=cfg.data
)

# Access results
tia_img = result.tia_img              # nibabel image
r2_img = result.r2_img
status_img = result.status_id_img
model_img = result.model_id_img
summary = result.summary              # dict

# Export
import nibabel as nib
nib.save(tia_img, "output_tia.nii.gz")
```

## 2. Command-Line Interface

### Enhanced CLI (pytia/cli.py)

Three commands available:

#### a) Run

```bash
pytia run --config config.yaml
```

Features:
- Loads YAML config
- Validates inputs
- Runs TIA estimation
- Reports output files

#### b) Validate

```bash
pytia validate --config config.yaml
```

Checks:
- YAML syntax
- Required sections
- Config structure

#### c) Info

```bash
pytia info --config config.yaml
```

Shows:
- Config file contents
- File size
- Syntax check

### Implementation

```python
# pytia/cli.py
def main(argv=None) -> int:
    parser = ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    
    # Add run, validate, info commands
    run_p = subparsers.add_parser("run", help="Run TIA estimation")
    run_p.add_argument("--config", required=True, type=Path)
    run_p.set_defaults(func=cmd_run)
    
    # ... validate_p, info_p ...
    
    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
```

### CLI Entry Point (pyproject.toml)

```toml
[project.scripts]
pytia = "pytia.cli:main"
```

Enables: `pytia run --config config.yaml` from anywhere

## 3. Config-Driven Architecture

### All Settings in YAML

No hardcoding! All configuration in YAML files:

```yaml
inputs:
  images: [t0.nii.gz, t1.nii.gz]
  times: [0.0, 60.0]

io:
  output_dir: ./output

physics:
  half_life_seconds: 21600.0

# ... 20+ other settings ...
```

### Config Flow

```
YAML File
    ↓
Config.load() → validates → returns dict
    ↓
run_tia(..., config=cfg_dict)
    ↓
Processes with settings
    ↓
Saves to output_dir
```

### Supported Config Inputs

1. **Dict**: `config={"physics": {"half_life_seconds": 3600.0}}`
2. **File**: `config="config.yaml"`
3. **Path**: `config=Path("config.yaml")`
4. **Config object**: `config=Config.load("config.yaml").data`

## 4. Documentation Structure

### docs/ Folder

| File | Purpose |
|------|---------|
| `USER_GUIDE.md` | Complete usage guide with examples |
| `CONFIG.md` | Configuration reference (all options) |
| `SINGLE_TIMEPOINT_IMPLEMENTATION.md` | STP technical details |

### Key Documentation

✅ **README.md** — Project overview, quick start
✅ **LICENSE** — MIT license
✅ **USER_GUIDE.md** — How to use (CLI and Python API)
✅ **CONFIG.md** — All configuration options explained

## 5. Examples Organization

### examples/ Folder

```
examples/
├── README.md                     # Quick start for examples
├── example_multitime.py         # Python API demo
├── example_stp.py               # STP all 3 methods
├── config_multitime.yaml        # Multi-timepoint config
├── config_stp_phys.yaml         # Physical decay
├── config_stp_haenscheid.yaml   # Hänscheid method
└── config_stp_prior_seg.yaml    # Segmentation-based
```

### Running Examples

Python:
```bash
python examples/example_multitime.py
python examples/example_stp.py
```

CLI:
```bash
pytia run --config examples/config_multitime.yaml
pytia run --config examples/config_stp_phys.yaml
```

## 6. Two Usage Modes

### Mode 1: Command-Line (Recommended for Production)

```bash
# Create config.yaml with all settings
pytia run --config config.yaml
```

**Advantages:**
- No Python coding
- Reproducible (version config files)
- Easy batch processing
- Audit trail

### Mode 2: Python API (For Integration)

```python
from pytia import run_tia

result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

# Process results
tia_data = result.tia_img.get_fdata()
```

**Advantages:**
- Fine control
- Scripting/automation
- Integration with workflows
- Custom processing

## 7. Complete Feature Mapping

### Multi-Timepoint Features

| Feature | Status |
|---------|--------|
| Load multiple images | ✓ |
| Automatic curve classification | ✓ |
| Gamma-linear fitting | ✓ |
| Exponential tail fitting | ✓ |
| Hybrid model | ✓ |
| Physical decay extrapolation | ✓ |
| Bootstrap uncertainty | ✓ |
| Regional ROI analysis | ✓ |
| Custom masking | ✓ |
| Denoising | ✓ |

### Single-Timepoint Features

| Method | Status | Config |
|--------|--------|--------|
| Physical decay | ✓ | `method: phys` |
| Hänscheid | ✓ | `method: haenscheid` |
| Prior (global) | ✓ | `method: prior_half_life` |
| Prior (segmentation) | ✓ | `label_map_path: ...` |

## 8. Configuration Examples

### Minimal

```yaml
inputs:
  images: [t0.nii.gz, t1.nii.gz]
  times: [0.0, 60.0]

io:
  output_dir: ./output

physics:
  half_life_seconds: 21600.0
```

### Multi-Timepoint (Full)

See `examples/config_multitime.yaml`

### Single-Timepoint (Physical)

See `examples/config_stp_phys.yaml`

### Single-Timepoint (Segmentation)

See `examples/config_stp_prior_seg.yaml`

## 9. Data Flow

### Input

```
YAML Config
    ├─ inputs.images → Load with nibabel
    ├─ inputs.times → Convert to seconds
    ├─ inputs.mask → Optional mask
    └─ physics.half_life_seconds → Decay rate
```

### Processing

```
Images (4D)
    ↓
Masking → Denoise → Noise floor validity
    ↓
Multi-Timepoint:
    Classify curves → Fit models → Integrate
    
Single-Timepoint:
    Direct calculation: TIA = A / λ
    ↓
Status tracking per voxel
```

### Output

```
Results Object:
    ├─ tia_img (nibabel) → tia.nii.gz
    ├─ r2_img → r2.nii.gz
    ├─ status_id_img → status_id.nii.gz
    ├─ model_id_img → model_id.nii.gz
    ├─ sigma_tia_img → sigma_tia.nii.gz (if bootstrap)
    ├─ summary (dict) → pytia_summary.yaml
    └─ output_paths (dict)
```

## 10. Quality Assurance

### Validation

```bash
pytia validate --config config.yaml
```

Checks:
- Config syntax valid
- Required sections present
- Types correct

### Testing

```bash
pytest tests/
```

13+ test classes covering:
- Multi-timepoint workflows
- Single-timepoint all 3 methods
- Config loading
- Status codes
- Edge cases

### Profiling

```yaml
performance:
  enable_profiling: true
```

Outputs timing in summary YAML.

## 11. Performance

### Large Datasets

```yaml
performance:
  chunk_size_vox: 250000  # Process in chunks
```

### Memory

- 4D data loaded once
- Chunked voxel processing
- Status tracking minimal

### Speed

- Disable unused features
- Bootstrap disabled by default
- Chunking optimized

## 12. Extensibility

### Adding New Method

1. Add config option in `config.py`
2. Implement in `engine.py` STP branch
3. Update CLI in `cli.py`
4. Add tests
5. Update docs

### Custom Postprocessing

```python
from pytia import run_tia

result = run_tia(...)

# Custom analysis
tia_data = result.tia_img.get_fdata()
status_data = result.status_id_img.get_fdata()

valid = status_data == 1
my_metric = calculate_metric(tia_data[valid])
```

## 13. Dependencies

### Core
- numpy ≥ 1.26
- scipy ≥ 1.12
- nibabel ≥ 5.2
- PyYAML ≥ 6.0

### Optional
- matplotlib (for visualization scripts)
- pytest (for testing)

### Installation

```bash
pip install pytia[dev]  # With dev tools
```

## 14. Project Metadata

```toml
[project]
name = "pytia"
version = "0.1.0"
description = "Voxel-wise TIA from PET/SPECT"
license = {text = "MIT"}
requires-python = ">=3.12"

[project.scripts]
pytia = "pytia.cli:main"
```

## Summary of Improvements

✅ **Enhanced Python API** with Config, IO utilities  
✅ **Full-featured CLI** with 3 commands  
✅ **Config-driven** — all settings in YAML  
✅ **Examples** — Python scripts + YAML configs  
✅ **Documentation** — USER_GUIDE, CONFIG reference  
✅ **Organization** — examples/, docs/ folders  
✅ **Testing** — 13+ test classes  
✅ **MIT License** included  

## Quick Links

- **Python API:** `from pytia import run_tia, Config`
- **CLI:** `pytia run --config config.yaml`
- **Examples:** `examples/` folder
- **Docs:** `docs/USER_GUIDE.md` and `docs/CONFIG.md`
- **Config:** All YAML in `examples/*.yaml`

---

**Status:** ✓ Complete and production-ready
