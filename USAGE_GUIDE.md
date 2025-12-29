# PyTIA Usage Guide

This guide provides comprehensive documentation for using PyTIA, a Python package for computing voxel-wise Time-Integrated Activity (TIA) maps from PET/SPECT imaging data.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Usage](#cli-usage)
4. [Python API Usage](#python-api-usage)
5. [Configuration](#configuration)
6. [Output Interpretation](#output-interpretation)
7. [Examples](#examples)

---

## Installation

### Prerequisites

- Python 3.12 or later
- pip package manager

### Install from PyPI

```bash
pip install pytia
```

### Install from Source

```bash
git clone https://github.com/devhliu/PyTIA.git
cd PyTIA
pip install -e .
```

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Using CLI

```bash
# Multi-timepoint TIA calculation
pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
             --times 1.0 24.0 72.0 \
             --time-unit hours \
             --half-life 21636.0

# Single-timepoint TIA calculation
pytia nifti --images scan1.nii.gz \
             --times 24.0 \
             --single-time \
             --stp-method phys \
             --half-life 21636.0
```

### Using Python API

```python
from pytia import run_tia_from_nifti

# Multi-timepoint TIA calculation
result = run_tia_from_nifti(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    time_unit="hours",
    half_life_seconds=21636.0,
)

# Single-timepoint TIA calculation
from pytia import run_single_timepoint_tia

result = run_single_timepoint_tia(
    image="scan1.nii.gz",
    time=24.0,
    method="phys",
    half_life_seconds=21636.0,
)
```

---

## CLI Usage

### Overview

PyTIA provides a command-line interface (CLI) with three main commands:

- `pytia run` - Run TIA estimation with a configuration file
- `pytia nifti` - Run TIA estimation directly from NIfTI files
- `pytia validate` - Validate a configuration file
- `pytia info` - Show configuration file contents

### Command: `pytia nifti`

The `nifti` command provides a convenient way to run TIA estimation without creating a configuration file.

#### Syntax

```bash
pytia nifti --images <files> --times <times> [options]
```

#### Required Arguments

- `--images`: NIfTI image files (one or more)
- `--times`: Timepoints for each image (space-separated)

#### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--time-unit` | Time unit: `hours` or `seconds` | `hours` |
| `--output-dir` | Output directory | `./pytia_output` |
| `--prefix` | Output file prefix | `pytia` |
| `--half-life` | Radionuclide half-life in seconds | `None` |
| `--mask` | Mask image file | `None` |
| `--mask-mode` | Mask mode: `none`, `provided`, `auto` | `none` |
| `--no-denoise` | Disable denoising | `False` |
| `--no-noise-floor` | Disable noise floor filtering | `False` |
| `--no-bootstrap` | Disable bootstrap | `False` |
| `--bootstrap` | Enable bootstrap with N replicates | `False` |
| `--bootstrap-seed` | Bootstrap random seed | `42` |
| `--chunk-size` | Chunk size for voxel processing | `None` |
| `--single-time` | Enable single-timepoint mode | `False` |
| `--stp-method` | Single-timepoint method | `phys` |
| `--eff-half-life` | Effective half-life for Hänscheid method | `None` |
| `--prior-half-life` | Prior half-life for prior_half_life method | `None` |
| `--label-map` | Label map for segmentation-based priors | `None` |
| `--label-half-lives` | Label half-lives mapping (JSON format) | `None` |

#### Examples

**Multi-timepoint TIA (2+ images):**

```bash
pytia nifti --images img1.nii.gz img2.nii.gz img3.nii.gz \
             --times 1 24 48 \
             --time-unit hours
```

**Single-timepoint TIA (physical decay):**

```bash
pytia nifti --images img1.nii.gz \
             --times 24 \
             --single-time \
             --stp-method phys \
             --half-life 21636.0
```

**Single-timepoint TIA (Hänscheid method):**

```bash
pytia nifti --images img1.nii.gz \
             --times 24 \
             --single-time \
             --stp-method haenscheid \
             --eff-half-life 3600.0
```

**Single-timepoint TIA (segmentation-based priors):**

```bash
pytia nifti --images img1.nii.gz \
             --times 24 \
             --single-time \
             --stp-method prior_half_life \
             --label-map labels.nii.gz \
             --label-half-lives '{"1": 1800.0, "2": 3600.0}'
```

**With mask and custom output:**

```bash
pytia nifti --images img*.nii.gz \
             --times 1 24 48 \
             --mask mask.nii.gz \
             --output-dir ./results \
             --prefix patient1
```

**With bootstrap for uncertainty:**

```bash
pytia nifti --images img*.nii.gz \
             --times 1 24 48 \
             --bootstrap 100 \
             --bootstrap-seed 42
```

### Command: `pytia run`

The `run` command executes TIA estimation using a YAML configuration file.

#### Syntax

```bash
pytia run --config <config_file>
```

#### Example

```bash
pytia run --config config.yaml
```

### Command: `pytia validate`

Validates a configuration file without running the analysis.

#### Syntax

```bash
pytia validate --config <config_file>
```

#### Example

```bash
pytia validate --config config.yaml
```

### Command: `pytia info`

Displays the contents of a configuration file.

#### Syntax

```bash
pytia info --config <config_file>
```

#### Example

```bash
pytia info --config config.yaml
```

---

## Python API Usage

### Overview

PyTIA provides a comprehensive Python API for integrating TIA calculation into custom workflows.

### Main Functions

#### `run_tia_from_nifti()`

Run TIA estimation from NIfTI files with simplified API.

**Parameters:**

- `images`: NIfTI image files or nibabel objects (one or more)
- `times`: Timepoints for each image
- `output_dir`: Output directory for results (default: `./pytia_output`)
- `prefix`: Prefix for output files (default: `pytia`)
- `half_life_seconds`: Radionuclide half-life in seconds
- `mask`: Optional mask image file or nibabel object
- `mask_mode`: Mask mode: `"none"`, `"provided"`, or `"auto"` (default: `"none"`)
- `time_unit`: Time unit: `"hours"` or `"seconds"` (default: `"hours"`)
- `denoise`: Enable spatial denoising (default: `True`)
- `noise_floor`: Enable noise floor filtering (default: `True`)
- `bootstrap`: Enable bootstrap for uncertainty (default: `False`)
- `bootstrap_n`: Number of bootstrap replicates (default: `100`)
- `bootstrap_seed`: Random seed for bootstrap (default: `42`)
- `chunk_size`: Chunk size for voxel processing (default: `None`)
- `**kwargs`: Additional configuration options

**Returns:** `Results` object containing TIA maps and metadata

**Example:**

```python
from pytia import run_tia_from_nifti

result = run_tia_from_nifti(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    time_unit="hours",
    half_life_seconds=21636.0,
    output_dir="./output",
    prefix="patient_001",
)

tia_data = result.tia_img.get_fdata()
print(f"TIA shape: {tia_data.shape}")
```

#### `run_single_timepoint_tia()`

Run single-timepoint TIA estimation from NIfTI file.

**Parameters:**

- `image`: Single NIfTI image file or nibabel object
- `time`: Timepoint for the image
- `method`: Method: `"phys"`, `"haenscheid"`, or `"prior_half_life"` (default: `"phys"`)
- `output_dir`: Output directory for results (default: `./pytia_output`)
- `prefix`: Prefix for output files (default: `pytia`)
- `half_life_seconds`: Radionuclide half-life in seconds (for `phys` method)
- `eff_half_life_seconds`: Effective half-life in seconds (for `haenscheid` method)
- `prior_half_life_seconds`: Prior half-life in seconds (for `prior_half_life` method)
- `label_map`: Label map for segmentation-based priors (for `prior_half_life` method)
- `label_half_lives`: Mapping of label IDs to half-lives in seconds (for `prior_half_life` method)
- `time_unit`: Time unit: `"hours"` or `"seconds"` (default: `"hours"`)
- `mask`: Optional mask image file or nibabel object
- `mask_mode`: Mask mode: `"none"`, `"provided"`, or `"auto"` (default: `"none"`)
- `denoise`: Enable spatial denoising (default: `True`)
- `noise_floor`: Enable noise floor filtering (default: `True`)
- `**kwargs`: Additional configuration options

**Returns:** `Results` object containing TIA maps and metadata

**Example:**

```python
from pytia import run_single_timepoint_tia

result = run_single_timepoint_tia(
    image="scan1.nii.gz",
    time=24.0,
    method="phys",
    half_life_seconds=21636.0,
)

tia_data = result.tia_img.get_fdata()
print(f"TIA shape: {tia_data.shape}")
```

#### `run_tia()`

Run TIA estimation with full configuration control.

**Parameters:**

- `images`: Single image or sequence of images (NIfTI paths or nibabel objects)
- `times`: Timepoints in specified unit (one per image)
- `config`: Config dict, YAML path, or `None` (uses defaults)
- `mask`: Optional mask image (path or nibabel object)

**Returns:** `Results` object containing TIA maps and metadata

**Example:**

```python
from pytia import run_tia

config = {
    "inputs": {
        "images": ["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
        "times": [1.0, 24.0, 72.0],
    },
    "time": {"unit": "hours"},
    "io": {
        "output_dir": "./output",
        "prefix": "patient_001",
    },
    "physics": {"half_life_seconds": 21636.0},
}

result = run_tia(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    config=config,
)
```

### Utility Functions

#### `load_nifti_as_array()`

Load a NIfTI file as a numpy array with metadata.

**Parameters:**

- `path`: Path to NIfTI file

**Returns:** Tuple of `(data_array, nibabel_image)`

**Example:**

```python
from pytia import load_nifti_as_array

data, img = load_nifti_as_array("scan.nii.gz")
print(f"Shape: {data.shape}")
print(f"Affine: {img.affine}")
```

#### `save_array_as_nifti()`

Save a numpy array as a NIfTI file using reference image metadata.

**Parameters:**

- `data`: Data array to save
- `reference`: Reference NIfTI file or nibabel object for metadata
- `output_path`: Output file path
- `dtype`: Optional data type for output (default: same as input)

**Returns:** nibabel image object

**Example:**

```python
from pytia import save_array_as_nifti

save_array_as_nifti(
    data=tia_data,
    reference="reference.nii.gz",
    output_path="tia.nii.gz",
)
```

#### `get_nifti_info()`

Get information about a NIfTI file.

**Parameters:**

- `path`: Path to NIfTI file

**Returns:** Dictionary with file information (shape, affine, voxel sizes, etc.)

**Example:**

```python
from pytia import get_nifti_info

info = get_nifti_info("scan.nii.gz")
print(f"Shape: {info['shape']}")
print(f"Voxel sizes: {info['voxel_sizes']}")
```

#### `extract_roi_stats()`

Extract TIA statistics for a region of interest.

**Parameters:**

- `tia_img`: TIA NIfTI file or nibabel object
- `mask_img`: Mask NIfTI file or nibabel object
- `mask_value`: Value in mask to use as ROI (default: `1`)

**Returns:** Dictionary with statistics (mean, std, min, max, sum, count)

**Example:**

```python
from pytia import extract_roi_stats

stats = extract_roi_stats("tia.nii.gz", "roi.nii.gz")
print(f"Mean TIA: {stats['mean']:.2f}")
```

#### `compare_results()`

Compare two TIA results and compute difference statistics.

**Parameters:**

- `result1`: First `Results` object
- `result2`: Second `Results` object
- `metric`: Metric to compare: `"tia"`, `"r2"`, `"sigma_tia"` (default: `"tia"`)

**Returns:** Dictionary with comparison statistics

**Example:**

```python
from pytia import compare_results

diff = compare_results(result1, result2, metric="tia")
print(f"Mean difference: {diff['mean_diff']:.2f}")
```

### Working with Results

The `Results` object contains:

- `tia_img`: TIA map (nibabel image)
- `r2_img`: R² map (nibabel image)
- `sigma_tia_img`: TIA uncertainty map (nibabel image)
- `model_id_img`: Model ID map (nibabel image)
- `status_id_img`: Status ID map (nibabel image)
- `tpeak_img`: Peak time map (nibabel image, or `None`)
- `summary`: Summary dictionary with metadata
- `output_paths`: Dictionary of output file paths
- `config`: Configuration dictionary
- `times_s`: Timepoints in seconds (numpy array)

**Example:**

```python
import numpy as np

result = run_tia_from_nifti(...)

# Access data arrays
tia_data = result.tia_img.get_fdata()
r2_data = result.r2_img.get_fdata()
sigma_tia_data = result.sigma_tia_img.get_fdata()

# Print statistics
print(f"TIA mean: {np.nanmean(tia_data):.2f}")
print(f"TIA std: {np.nanstd(tia_data):.2f}")
print(f"R² mean: {np.nanmean(r2_data):.4f}")

# Access summary
print(f"Times (s): {result.times_s}")
print(f"Status counts: {result.summary.get('status_counts', {})}")
print(f"Timing: {result.summary.get('timing_ms', {})}")

# Access output paths
print(f"Output paths: {result.output_paths}")
```

---

## Configuration

### Configuration File Structure

PyTIA uses YAML configuration files to control all aspects of TIA calculation.

### Example Configuration File

```yaml
# config.yaml
inputs:
  images:
    - "scan1.nii.gz"
    - "scan2.nii.gz"
    - "scan3.nii.gz"
  times: [1.0, 24.0, 72.0]

time:
  unit: hours
  sort_timepoints: true

io:
  output_dir: ./pytia_output
  prefix: patient_001
  write_summary_yaml: true
  write_status_map: true

physics:
  half_life_seconds: 21636.0  # Tc-99m half-life in seconds
  enforce_lambda_ge_phys: true

mask:
  mode: provided  # options: provided, otsu, none
  provided_path: "body_mask.nii.gz"
  min_fraction_of_max: 0.02

denoise:
  enabled: true
  method: masked_gaussian
  sigma_vox: 1.2

noise_floor:
  enabled: true
  mode: relative  # options: absolute, relative
  absolute_bq_per_ml: 0.0
  relative_fraction_of_voxel_max: 0.01
  behavior: exclude

model_selection:
  mode: auto
  min_points_for_gamma: 3

integration:
  start_time_seconds: 0.0
  tail_mode: phys  # options: phys, fitted, hybrid, none
  rising_tail_mode: phys  # options: phys, peak_at_last
  min_tail_points: 2
  fit_tail_slope: false
  lambda_phys_constraint: true
  include_t0: true

bootstrap:
  enabled: true
  n: 100
  seed: 42
  reclassify_each_replicate: true

performance:
  chunk_size_vox: 500000
  enable_profiling: false

regions:
  enabled: false
  label_map_path: null
  mode: roi_aggregate
  aggregation: mean
  voxel_level_r2: false
  classes: {}
  scaling:
    mode: tref
    reference_time: peak

single_time:
  enabled: false
  method: phys  # options: phys, haenscheid, prior_half_life
  haenscheid_eff_half_life_seconds: null
  half_life_seconds: null
  label_map_path: null
  label_half_lives: {}
```

### Configuration Sections

#### `inputs`

- `images`: List of NIfTI image file paths
- `times`: List of timepoints corresponding to each image

#### `time`

- `unit`: Time unit (`hours` or `seconds`)
- `sort_timepoints`: Whether to sort timepoints (default: `true`)

#### `io`

- `output_dir`: Output directory path
- `prefix`: Prefix for output files
- `write_summary_yaml`: Write summary YAML file (default: `true`)
- `write_status_map`: Write status map (default: `true`)

#### `physics`

- `half_life_seconds`: Radionuclide half-life in seconds
- `enforce_lambda_ge_phys`: Enforce lambda >= lambda_phys (default: `true`)

#### `mask`

- `mode`: Mask mode (`provided`, `otsu`, `none`)
- `provided_path`: Path to provided mask image
- `min_fraction_of_max`: Minimum fraction of max for auto mask (default: `0.02`)

#### `denoise`

- `enabled`: Enable denoising (default: `true`)
- `method`: Denoising method (`masked_gaussian`)
- `sigma_vox`: Gaussian sigma in voxels (default: `1.2`)

#### `noise_floor`

- `enabled`: Enable noise floor filtering (default: `true`)
- `mode`: Mode (`absolute` or `relative`)
- `absolute_bq_per_ml`: Absolute threshold in Bq/ml
- `relative_fraction_of_voxel_max`: Relative threshold (default: `0.01`)
- `behavior`: Behavior (`exclude`)

#### `bootstrap`

- `enabled`: Enable bootstrap (default: `true`)
- `n`: Number of bootstrap replicates (default: `100`)
- `seed`: Random seed (default: `42`)
- `reclassify_each_replicate`: Reclassify each replicate (default: `true`)

#### `performance`

- `chunk_size_vox`: Chunk size for voxel processing (default: `500000`)
- `enable_profiling`: Enable profiling (default: `false`)

#### `single_time`

- `enabled`: Enable single-timepoint mode (default: `false`)
- `method`: Method (`phys`, `haenscheid`, `prior_half_life`)
- `haenscheid_eff_half_life_seconds`: Effective half-life for Hänscheid method
- `half_life_seconds`: Prior half-life for prior_half_life method
- `label_map_path`: Label map path for segmentation-based priors
- `label_half_lives`: Mapping of label IDs to half-lives

---

## Output Interpretation

### Output Files

PyTIA generates the following output files:

1. `{prefix}_tia.nii.gz` - Time-Integrated Activity map (Bq·s/voxel)
2. `{prefix}_r2.nii.gz` - R² goodness-of-fit map
3. `{prefix}_sigma_tia.nii.gz` - TIA uncertainty map (Bq·s/voxel)
4. `{prefix}_model_id.nii.gz` - Model ID map
5. `{prefix}_status_id.nii.gz` - Status ID map
6. `{prefix}_pytia_summary.yaml` - Summary YAML file

### Model ID Codes

| Code | Model |
|------|-------|
| 10 | Hybrid (rising) |
| 11 | Hybrid (hump) |
| 20 | Mono-exponential (falling) |
| 30 | Gamma-variate |
| 101 | Single-timepoint (physical decay) |
| 102 | Single-timepoint (Hänscheid) |
| 103 | Single-timepoint (prior half-life) |

### Status ID Codes

| Code | Status |
|------|--------|
| 0 | Outside mask/background |
| 1 | OK |
| 2 | Not applicable: <2 valid points |
| 3 | Fit failed |
| 4 | All points below noise floor |
| 5 | Nonphysical parameters |

### Summary File

The summary YAML file contains:

- `pytia_version`: PyTIA version
- `times_seconds`: Timepoints in seconds
- `voxel_volume_ml`: Voxel volume in ml
- `status_legend`: Status code legend
- `status_counts`: Count of voxels per status
- `timing_ms`: Timing information for each processing step
- `config`: Configuration used for the analysis

---

## Examples

### Example 1: Basic Multi-Timepoint Analysis

**CLI:**

```bash
pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
             --times 1.0 24.0 72.0 \
             --time-unit hours \
             --half-life 21636.0 \
             --output-dir ./results \
             --prefix patient_001
```

**Python API:**

```python
from pytia import run_tia_from_nifti

result = run_tia_from_nifti(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    time_unit="hours",
    half_life_seconds=21636.0,
    output_dir="./results",
    prefix="patient_001",
)
```

### Example 2: Single-Timepoint Analysis (Physical Decay)

**CLI:**

```bash
pytia nifti --images scan1.nii.gz \
             --times 24.0 \
             --single-time \
             --stp-method phys \
             --half-life 21636.0 \
             --output-dir ./results \
             --prefix patient_001_stp
```

**Python API:**

```python
from pytia import run_single_timepoint_tia

result = run_single_timepoint_tia(
    image="scan1.nii.gz",
    time=24.0,
    method="phys",
    half_life_seconds=21636.0,
    output_dir="./results",
    prefix="patient_001_stp",
)
```

### Example 3: With Mask and Bootstrap

**CLI:**

```bash
pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
             --times 1.0 24.0 72.0 \
             --time-unit hours \
             --half-life 21636.0 \
             --mask body_mask.nii.gz \
             --bootstrap 100 \
             --bootstrap-seed 42 \
             --output-dir ./results \
             --prefix patient_001
```

**Python API:**

```python
from pytia import run_tia_from_nifti

result = run_tia_from_nifti(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    time_unit="hours",
    half_life_seconds=21636.0,
    mask="body_mask.nii.gz",
    bootstrap=True,
    bootstrap_n=100,
    bootstrap_seed=42,
    output_dir="./results",
    prefix="patient_001",
)
```

### Example 4: Batch Processing

**Python API:**

```python
from pytia import run_tia_from_nifti

patients = [
    {"id": "001", "images": ["p001_s1.nii.gz", "p001_s2.nii.gz", "p001_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
    {"id": "002", "images": ["p002_s1.nii.gz", "p002_s2.nii.gz", "p002_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
    {"id": "003", "images": ["p003_s1.nii.gz", "p003_s2.nii.gz", "p003_s3.nii.gz"], "times": [1.0, 24.0, 72.0]},
]

results = []
for patient in patients:
    result = run_tia_from_nifti(
        images=patient["images"],
        times=patient["times"],
        time_unit="hours",
        half_life_seconds=21636.0,
        output_dir=f"./results/patient_{patient['id']}",
        prefix=f"patient_{patient['id']}",
    )
    results.append(result)
    print(f"Processed patient {patient['id']}")
```

### Example 5: Extract ROI Statistics

**Python API:**

```python
from pytia import run_tia_from_nifti, extract_roi_stats

result = run_tia_from_nifti(
    images=["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"],
    times=[1.0, 24.0, 72.0],
    time_unit="hours",
    half_life_seconds=21636.0,
)

stats = extract_roi_stats(
    tia_img=result.tia_img,
    mask_img="tumor_roi.nii.gz",
    mask_value=1,
)

print(f"ROI Statistics:")
print(f"  Mean TIA: {stats['mean']:.2f} Bq·s/voxel")
print(f"  Std TIA: {stats['std']:.2f} Bq·s/voxel")
print(f"  Min TIA: {stats['min']:.2f} Bq·s/voxel")
print(f"  Max TIA: {stats['max']:.2f} Bq·s/voxel")
print(f"  Sum TIA: {stats['sum']:.2f} Bq·s")
print(f"  Count: {stats['count']} voxels")
```

---

## Additional Resources

- [CLI Examples](./cli_examples.py) - Detailed CLI usage examples
- [Python API Examples](./python_api_examples.py) - Detailed Python API examples
- [Configuration Reference](./CONFIG.md) - Complete configuration reference
- [Architecture Documentation](./ARCHITECTURE.md) - Developer-focused architecture guide
- [User Guide](./USER_GUIDE.md) - Detailed user guide

---

## Support

For issues, questions, or contributions, please visit the [PyTIA GitHub repository](https://github.com/devhliu/PyTIA).
