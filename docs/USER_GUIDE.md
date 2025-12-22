# PyTIA User Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage Modes](#usage-modes)
4. [Command-Line Interface](#command-line-interface)
5. [Python API](#python-api)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Output](#output)
9. [Troubleshooting](#troubleshooting)

## Overview

PyTIA computes **Time-Integrated Activity (TIA)** maps from PET/SPECT imaging data.

**TIA** represents the integral of activity over time, essential for:
- Absorbed dose calculations
- Biokinetic modeling
- Treatment planning

### Two Analysis Modes

#### Multi-Timepoint (2+ images)
- Fits kinetic models to activity curves
- Estimates TIA through model integration
- Provides model fit quality (R²)
- Uncertainty via bootstrap

#### Single-Timepoint (1 image)
- Direct TIA calculation from single snapshot
- Three methods: physical decay, Hänscheid, prior half-life
- Ideal for clinical routine with time constraints

## Installation

### From PyPI
```bash
pip install pytia
```

### From source
```bash
git clone https://github.com/devhliu/PyTIA.git
cd PyTIA
pip install -e ".[dev]"  # With dev tools
```

## Usage Modes

### Mode 1: Command-Line Interface (Recommended)

Most common and flexible approach:

```bash
# Run analysis
pytia run --config config.yaml

# Validate config before running
pytia validate --config config.yaml

# View config file
pytia info --config config.yaml
```

**Advantages:**
- No Python coding required
- Reproducible (config files can be versioned)
- Batch processing scripts easily

### Mode 2: Python API

Direct Python control:

```python
from pytia import run_tia

result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

# Access results
tia_data = result.tia_img.get_fdata()
```

**Advantages:**
- Fine-grained control
- Scripting and automation
- Integration with other tools

## Command-Line Interface

### Run TIA Estimation

```bash
pytia run --config config.yaml
```

**Arguments:**
- `--config` (required): Path to YAML configuration file

**Output:** Saves results to `io.output_dir` specified in config

**Example:**
```bash
pytia run --config examples/config_multitime.yaml
```

### Validate Configuration

```bash
pytia validate --config config.yaml
```

Checks:
- YAML syntax validity
- Required config sections
- Input file existence

**Example:**
```bash
pytia validate --config config.yaml
✓ Config file is valid

Config structure:
  - inputs
  - io
  - physics
  - single_time
```

### Show Configuration Info

```bash
pytia info --config config.yaml
```

Displays config file contents (useful for verification before running).

## Python API

### Basic Usage

```python
from pytia import run_tia
from pytia import Config

# Method 1: Use config dict
result = run_tia(
    images=["img1.nii.gz", "img2.nii.gz"],
    times=[0.0, 60.0],
    config={"physics": {"half_life_seconds": 3600.0}}
)

# Method 2: Load from YAML
result = run_tia(
    images=["img1.nii.gz", "img2.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

# Method 3: Use Config object
cfg = Config.load("config.yaml")
result = run_tia(
    images=["img1.nii.gz", "img2.nii.gz"],
    times=[0.0, 60.0],
    config=cfg.data
)
```

### Results Object

```python
from pytia import Results

# Access output images (nibabel objects)
tia_img = result.tia_img
r2_img = result.r2_img
model_img = result.model_id_img
status_img = result.status_id_img
sigma_img = result.sigma_tia_img

# Convert to numpy arrays
import numpy as np
tia_data = np.asarray(tia_img.dataobj)

# Summary statistics
summary = result.summary
print(f"Times: {summary['times_seconds']}")
print(f"Valid voxels: {summary['status_counts']['ok']}")

# Output file paths
for key, path in result.output_paths.items():
    print(f"{key}: {path}")
```

### Advanced: Loading Images

```python
from pytia import load_images, stack_4d
import nibabel as nib

# Load images
imgs = load_images(["t0.nii.gz", "t1.nii.gz", "t2.nii.gz"])

# Stack into 4D array
data_4d, ref_img = stack_4d(imgs)
print(f"Shape: {data_4d.shape}")  # (X, Y, Z, T)
```

## Configuration

All settings are in **YAML** config files. No hardcoding!

### Structure

```yaml
inputs:           # Image paths and timepoints
io:               # Output directory
physics:          # Half-life settings
time:             # Unit conversion
mask:             # Masking strategy
denoise:          # Denoising
noise_floor:      # Validity filtering
bootstrap:        # Uncertainty
model_selection:  # Curve fitting
integration:      # Integration parameters
regions:          # Optional ROI analysis
single_time:      # Single-timepoint settings
```

### Minimal Config

```yaml
inputs:
  images:
    - activity_t0.nii.gz
    - activity_t1.nii.gz
  times: [0.0, 60.0]

io:
  output_dir: ./output

physics:
  half_life_seconds: 21600.0
```

### Full Configuration

See [CONFIG.md](CONFIG.md) for detailed reference of all options.

## Examples

### Example 1: Multi-Timepoint (Tc-99m)

```yaml
# config_tc99m.yaml
inputs:
  images: [scan_0h.nii.gz, scan_1h.nii.gz, scan_2h.nii.gz]
  times: [0.0, 3600.0, 7200.0]

io:
  output_dir: ./output/tc99m

physics:
  half_life_seconds: 21600.0  # 6 hours

denoise:
  enabled: true
  sigma_vox: 1.5

noise_floor:
  enabled: true
  relative_fraction_of_voxel_max: 0.01

bootstrap:
  enabled: true
  n: 100
```

Run:
```bash
pytia run --config config_tc99m.yaml
```

### Example 2: Single-Timepoint (Physical Decay)

```yaml
# config_stp_phys.yaml
inputs:
  images: [activity_snapshot.nii.gz]
  times: [0.0]

io:
  output_dir: ./output/stp_phys

physics:
  half_life_seconds: 6600.0  # F-18

single_time:
  enabled: true
  method: phys
```

Run:
```bash
pytia run --config config_stp_phys.yaml
```

### Example 3: Single-Timepoint (Organ-Specific)

```yaml
# config_stp_organs.yaml
inputs:
  images: [activity.nii.gz]
  times: [0.0]

io:
  output_dir: ./output/stp_organs

single_time:
  enabled: true
  method: prior_half_life
  label_map_path: segmentation.nii.gz
  label_half_lives:
    1: 1800.0   # Tumor: 30 min
    2: 3600.0   # Liver: 60 min
    3: 5400.0   # Kidney: 90 min
  half_life_seconds: 3600.0  # Default
```

See `examples/` folder for runnable examples:
- `example_multitime.py` — Multi-timepoint demo
- `example_stp.py` — Single-timepoint demos
- Config files in `examples/*.yaml`

## Output

### Output Files

| File | Content | Notes |
|------|---------|-------|
| `tia.nii.gz` | Time-integrated activity | Bq·s/ml |
| `r2.nii.gz` | Model fit quality | NaN for STP |
| `sigma_tia.nii.gz` | Uncertainty (std dev) | If bootstrap enabled |
| `model_id.nii.gz` | Method ID per voxel | 10/11/20/30/101/102/103 |
| `status_id.nii.gz` | Validity status | 0-5 |
| `pytia_summary.yaml` | Metadata and config | YAML format |

### Status Codes

The `status_id.nii.gz` output map provides a voxel-wise report on the outcome of the TIA calculation. Each voxel is assigned an integer code.

| Code | Status                                  | Description                                                                                             |
| :--- | :-------------------------------------- | :------------------------------------------------------------------------------------------------------ |
| 0    | `outside mask/background`               | The voxel was outside the processing mask and was ignored.                                              |
| 1    | `ok`                                    | A valid TIA value was successfully computed.                                                            |
| 2    | `not applicable: <2 valid points`        | The voxel had fewer than two valid data points after noise filtering, so no model could be fit.         |
| 3    | `fit failed`                            | A model was attempted but failed to converge. This can happen with very noisy data or missing configuration (e.g., `physics.half_life_seconds` for hybrid models). |
| 4    | `all points below noise floor`          | All activity timepoints for this voxel were below the configured noise floor and were excluded.         |
| 5    | `nonphysical parameters`                | The model fit produced parameters that are not physically plausible.                                    |

### Model IDs (Multi-Timepoint)

| Code | Model |
|------|-------|
| 10 | Hybrid (rising) |
| 11 | Hybrid with phys tail |
| 20 | Exponential (falling) |
| 30 | Gamma-linear (hump) |

### Model IDs (Single-Timepoint)

| Code | Method |
|------|--------|
| 101 | Physical decay |
| 102 | Hänscheid effective |
| 103 | Prior half-life |

### Summary File

Example `pytia_summary.yaml`:

```yaml
pytia_version: "0.1.0"
times_seconds: [0.0, 60.0, 120.0]
voxel_volume_ml: 8.0

status_legend:
  0: "outside mask/background"
  1: "ok"
  2: "not applicable: invalid decay rate"
  3: "fit failed"
  4: "all points below noise floor"

status_counts:
  ok: 125400
  outside mask/background: 10000
  all points below noise floor: 234

timing_ms:
  load_sort_ms: 245.3
  mask_denoise_ms: 1523.4
  voxel_fit_ms: 8234.1
  bootstrap_ms: 15023.4
  assemble_ms: 132.1
  save_ms: 234.5
  total_ms: 25393.3
```

## Troubleshooting

### Issue: "Config must contain 'inputs.images'"

**Cause:** Missing `inputs` section in config

**Solution:**
```yaml
inputs:
  images: [activity_t0.nii.gz, activity_t1.nii.gz]
  times: [0.0, 60.0]
```

### Issue: All voxels marked as "FIT_FAILED"

**Cause:** Missing `physics.half_life_seconds`

**Solution:**
```yaml
physics:
  half_life_seconds: 21600.0  # Add this
```

### Issue: TIA values seem wrong (too large/small)

**Cause:** Half-life unit mismatch

**Solution:** Ensure all times in **seconds**:
- 1 hour = 3600 seconds
- 1 minute = 60 seconds
- 110 minutes = 6600 seconds

### Issue: Segmentation labels not working in STP

**Cause:** Label values don't match mapping

**Solution:**
1. Check label image: `nibabel` viewer or ITK-SNAP
2. Match keys in `label_half_lives`:

```python
import nibabel as nib
import numpy as np

img = nib.load("segmentation.nii.gz")
labels = np.unique(np.asarray(img.dataobj))
print(f"Unique labels: {labels}")
```

Then update config:
```yaml
label_half_lives:
  1: 1800.0
  2: 3600.0
  # ... etc
```

### Issue: Memory error with large datasets

**Cause:** Processing all voxels at once

**Solution:** Enable chunking in config:
```yaml
performance:
  chunk_size_vox: 250000  # Reduce if still too large
```

## Tips & Tricks

### Batch Processing

Create a shell script:

```bash
#!/bin/bash
for config in configs/*.yaml; do
    echo "Processing $config..."
    pytia run --config "$config"
done
```

### Python Scripting

Automate config generation:

```python
import yaml
from pytia import run_tia

for patient_id in ["P001", "P002", "P003"]:
    config = {
        "inputs": {
            "images": [f"data/{patient_id}/t0.nii.gz", f"data/{patient_id}/t1.nii.gz"],
            "times": [0.0, 60.0],
        },
        "io": {"output_dir": f"output/{patient_id}"},
        "physics": {"half_life_seconds": 21600.0},
    }
    
    result = run_tia(
        images=config["inputs"]["images"],
        times=config["inputs"]["times"],
        config=config,
    )
    
    print(f"{patient_id}: TIA computed!")
```

### Quality Control

Check summary after running:

```python
import yaml

with open("output/pytia_summary.yaml") as f:
    summary = yaml.safe_load(f)

print(f"Valid voxels: {summary['status_counts']['ok']}")
print(f"Total time: {summary['timing_ms']['total_ms']:.1f} ms")
```

---

**For more information:** See [docs/CONFIG.md](CONFIG.md) for complete configuration reference.
