# PyTIA: Voxel-Wise Time-Integrated Activity Maps

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**PyTIA** is a powerful and flexible Python package for computing voxel-wise **Time-Integrated Activity (TIA)** maps from PET/SPECT imaging data. Designed for both researchers and clinicians, it provides a robust, config-driven workflow for biokinetic modeling and absorbed dose calculation.

## Key Features

| Feature                  | Description                                                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Multi-Timepoint Models**   | Automatically classifies time-activity curves (TACs) and fits appropriate kinetic models (**Gamma-variate**, **Mono-exponential**).      |
| **Single-Timepoint (STP)** | Calculates TIA from a single image using multiple methods: **Physical Decay**, **Hänscheid** (effective half-life), or **Prior Half-Life**. |
| **Region-Based Analysis**  | Average TACs within ROIs, fit a single model, and scale results back to voxels for robust parameter estimation.                        |
| **Uncertainty Analysis**   | Quantify TIA uncertainty using **Residual Bootstrap** analysis.                                                                          |
| **Config-Driven Workflow** | Control every aspect of your analysis—from inputs to model parameters—with a single, reproducible YAML configuration file.                 |
| **Extensible & Scriptable**| Use the full-featured **CLI** for batch processing or the **Python API** for seamless integration into custom analysis pipelines.          |
| **Built-in Preprocessing** | Includes automatic body masking, spatial denoising, and noise-floor filtering to improve data quality.                                   |

## How It Works

PyTIA implements a flexible pipeline for TIA calculation, supporting both voxel-wise and region-based analysis.

```
[Input Images & Times] -> [Preprocessing (Mask, Denoise)] -> [Analysis Mode]
                                                                  |
           +------------------------------------------------------+------------------------------------------------------+
           |                                                      |                                                      |
    [Voxel-wise Analysis]                                 [Region-based Analysis]                               [Single-Timepoint Analysis]
           |                                                      |                                                      |
    - For each voxel:                                     - For each ROI:                                      - For each voxel:
    - Classify TAC                                        - Average TAC                                        - Apply formula:
    - Fit kinetic model                                     - Fit single kinetic model                             TIA = A(t) / λ_eff
    - Integrate curve                                     - Scale TIA back to voxels
           |                                                      |                                                      |
           +------------------------------------------------------+------------------------------------------------------+
                                                                  |
                                                         [Output Maps]
                                          (TIA, R², Model ID, Status, Uncertainty)
```

## Installation

PyTIA requires Python 3.12 or later.

```bash
pip install pytia
```

For developers, install with testing and linting tools:
```bash
git clone https://github.com/devhliu/PyTIA.git
cd PyTIA
pip install -e ".[dev]"
```

## Quick Start

Run a multi-timepoint analysis in two steps:

**1. Create a configuration file (`config.yaml`):**

```yaml
# config.yaml
inputs:
  # Paths to your NIfTI images
  images:
    - "activity_t0.nii.gz"
    - "activity_t1.nii.gz"
    - "activity_t2.nii.gz"
  # Time of each scan in hours
  times: [1.0, 24.0, 72.0]

time:
  unit: hours # Specify the unit for the times above

io:
  # Where to save the output maps
  output_dir: ./pytia_output
  # A prefix for all output files
  prefix: patient_01

# Physics of the radionuclide
physics:
  # Half-life of Tc-99m in seconds (6.01 hours)
  half_life_seconds: 21636.0

# Optional: Enable bootstrap for uncertainty
bootstrap:
  enabled: true
  n: 100
  seed: 42
```

**2. Run PyTIA from the command line:**

```bash
pytia run --config config.yaml
```

Output maps (`patient_01_tia.nii.gz`, `patient_01_r2.nii.gz`, etc.) will be saved in the `./pytia_output` directory.

## Documentation

- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md):** Detailed instructions on CLI and API usage, output interpretation, and troubleshooting.
- **[docs/CONFIG.md](docs/CONFIG.md):** A complete reference for every option in the `config.yaml` file.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):** A developer-focused guide to the codebase structure.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
