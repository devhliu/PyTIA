# Lu-177 FAP TIA Calculation Examples

This directory contains examples for calculating Time-Integrated Activity (TIA) from Lu-177 FAP SPECT images using PyTIA.

## Dataset Information

**Radionuclide:** Lu-177  
**Half-life:** 6.647 days (574302.48 seconds)  
**Acquisition Time Points:** 4H, 24H, 48H, 168H

## Directory Structure

```
Lu-177_FAP/
├── input_multi-points/          # Input SPECT images
│   ├── SPECT-004H_reg.nii.gz   # 4 hours post-injection
│   ├── SPECT-024H_reg.nii.gz   # 24 hours post-injection
│   ├── SPECT-048H_reg.nii.gz   # 48 hours post-injection
│   └── SPECT-168H_reg.nii.gz   # 168 hours post-injection
├── output_tia/                 # Output directory (created automatically)
├── cli_examples_lu177.sh       # CLI usage examples
├── python_api_examples_lu177.py # Python API usage examples
└── README.md                   # This file
```

## Quick Start

### Using CLI

#### 3-Timepoint TIA [4, 24, 48] hours

```bash
pytia nifti --images input_multi-points/SPECT-004H_reg.nii.gz \
             input_multi-points/SPECT-024H_reg.nii.gz \
             input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --output-dir output_tia \
             --prefix Lu177_FAP_3tp
```

#### 4-Timepoint TIA [4, 24, 48, 168] hours

```bash
pytia nifti --images input_multi-points/SPECT-004H_reg.nii.gz \
             input_multi-points/SPECT-024H_reg.nii.gz \
             input_multi-points/SPECT-048H_reg.nii.gz \
             input_multi-points/SPECT-168H_reg.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --output-dir output_tia \
             --prefix Lu177_FAP_4tp
```

### Using Python API

#### 3-Timepoint TIA

```python
from pytia import run_tia_from_nifti

result = run_tia_from_nifti(
    images=[
        "input_multi-points/SPECT-004H_reg.nii.gz",
        "input_multi-points/SPECT-024H_reg.nii.gz",
        "input_multi-points/SPECT-048H_reg.nii.gz",
    ],
    times=[4.0, 24.0, 48.0],
    time_unit="hours",
    half_life_seconds=574302.48,
    output_dir="output_tia",
    prefix="Lu177_FAP_3tp",
)
```

#### 4-Timepoint TIA

```python
from pytia import run_tia_from_nifti

result = run_tia_from_nifti(
    images=[
        "input_multi-points/SPECT-004H_reg.nii.gz",
        "input_multi-points/SPECT-024H_reg.nii.gz",
        "input_multi-points/SPECT-048H_reg.nii.gz",
        "input_multi-points/SPECT-168H_reg.nii.gz",
    ],
    times=[4.0, 24.0, 48.0, 168.0],
    time_unit="hours",
    half_life_seconds=574302.48,
    output_dir="output_tia",
    prefix="Lu177_FAP_4tp",
)
```

## Examples

### CLI Examples

See [cli_examples_lu177.sh](cli_examples_lu177.sh) for comprehensive CLI examples including:

- Basic 3-timepoint and 4-timepoint TIA calculations
- TIA calculation with mask
- TIA calculation with bootstrap for uncertainty estimation
- Advanced options (chunking, denoising, noise floor)
- Configuration file examples

To run CLI examples:

```bash
# Make the script executable (Linux/Mac)
chmod +x cli_examples_lu177.sh

# Run all examples
./cli_examples_lu177.sh

# Or run individual commands by copying from the file
```

### Python API Examples

See [python_api_examples_lu177.py](python_api_examples_lu177.py) for comprehensive Python API examples including:

- Basic 3-timepoint and 4-timepoint TIA calculations
- TIA calculation with mask and bootstrap
- Advanced options (chunking, denoising, noise floor)
- Result analysis and statistics extraction
- ROI statistics extraction
- Comparison between 3TP and 4TP results
- Batch processing
- Custom post-processing (thresholding, normalization)
- Visualization

To run Python API examples:

```python
# Run the script to see available examples
python python_api_examples_lu177.py

# Or import and use specific functions
from python_api_examples_lu177 import example_3_timepoints_basic

result = example_3_timepoints_basic()
```

## Output Files

PyTIA generates the following output files in the `output_tia/` directory:

| File | Description |
|------|-------------|
| `{prefix}_tia.nii.gz` | Time-Integrated Activity map (Bq·s/voxel) |
| `{prefix}_r2.nii.gz` | R² goodness-of-fit map |
| `{prefix}_sigma_tia.nii.gz` | TIA uncertainty map (Bq·s/voxel) |
| `{prefix}_model_id.nii.gz` | Model ID map |
| `{prefix}_status_id.nii.gz` | Status ID map |
| `{prefix}_pytia_summary.yaml` | Summary YAML file |

## Advanced Usage

### With Mask

```bash
pytia nifti --images input_multi-points/SPECT-*.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --mask mask.nii.gz \
             --output-dir output_tia \
             --prefix Lu177_FAP_masked
```

### With Bootstrap (Uncertainty Estimation)

```bash
pytia nifti --images input_multi-points/SPECT-*.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --bootstrap 100 \
             --bootstrap-seed 42 \
             --output-dir output_tia \
             --prefix Lu177_FAP_bootstrap
```

### With Custom Chunk Size (Memory Efficiency)

```bash
pytia nifti --images input_multi-points/SPECT-*.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --chunk-size 500000 \
             --output-dir output_tia \
             --prefix Lu177_FAP_chunked
```

## Configuration File

You can also use a YAML configuration file:

```yaml
# config_lu177.yaml
inputs:
  images:
    - "input_multi-points/SPECT-004H_reg.nii.gz"
    - "input_multi-points/SPECT-024H_reg.nii.gz"
    - "input_multi-points/SPECT-048H_reg.nii.gz"
    - "input_multi-points/SPECT-168H_reg.nii.gz"
  times: [4.0, 24.0, 48.0, 168.0]

time:
  unit: hours

io:
  output_dir: output_tia
  prefix: Lu177_FAP_4tp

physics:
  half_life_seconds: 574302.48

denoise:
  enabled: true
  sigma_vox: 1.2

noise_floor:
  enabled: true
  mode: relative
  relative_fraction_of_voxel_max: 0.01

bootstrap:
  enabled: true
  n: 100
  seed: 42
```

Run with configuration file:

```bash
pytia run --config config_lu177.yaml
```

## Notes

- **Lu-177 half-life:** 6.647 days = 574302.48 seconds
- **Time unit:** All times are specified in hours
- **Output directory:** Created automatically if it doesn't exist
- **Bootstrap:** Provides uncertainty estimates for TIA values
- **Mask:** Can be used to limit analysis to specific regions (e.g., tumor, organ)

## Additional Resources

- [PyTIA Usage Guide](../../USAGE_GUIDE.md) - Comprehensive usage guide
- [CLI Examples](../../examples/cli_examples.py) - General CLI examples
- [Python API Examples](../../examples/python_api_examples.py) - General Python API examples

## Support

For issues, questions, or contributions, please visit the [PyTIA GitHub repository](https://github.com/devhliu/PyTIA).
