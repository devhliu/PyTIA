"""
CLI Examples for Lu-177 FAP TIA Calculation

This file demonstrates CLI commands for calculating Time-Integrated Activity (TIA)
from Lu-177 FAP SPECT images acquired at multiple time points.

Input files (located in examples/Lu-177_FAP/input_multi-points/):
- SPECT-004H_reg.nii.gz (4 hours)
- SPECT-024H_reg.nii.gz (24 hours)
- SPECT-048H_reg.nii.gz (48 hours)
- SPECT-168H_reg.nii.gz (168 hours)

Lu-177 half-life: 6.647 days = 574302.48 seconds
"""

# =============================================================================
# Example 1: TIA Calculation with 3 Time Points [4, 24, 48] hours
# =============================================================================

# Basic 3-timepoint TIA calculation:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_3tp

# 3-timepoint TIA calculation with mask:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --mask examples/Lu-177_FAP/mask.nii.gz \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_3tp_masked

# 3-timepoint TIA calculation with bootstrap for uncertainty:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --bootstrap 100 \
             --bootstrap-seed 42 \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_3tp_bootstrap

# =============================================================================
# Example 2: TIA Calculation with 4 Time Points [4, 24, 48, 168] hours
# =============================================================================

# Basic 4-timepoint TIA calculation:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-168H_reg.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_4tp

# 4-timepoint TIA calculation with mask:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-168H_reg.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --mask examples/Lu-177_FAP/mask.nii.gz \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_4tp_masked

# 4-timepoint TIA calculation with bootstrap for uncertainty:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-168H_reg.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --bootstrap 100 \
             --bootstrap-seed 42 \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_4tp_bootstrap

# =============================================================================
# Example 3: Advanced Options
# =============================================================================

# 3-timepoint TIA with custom chunk size for memory efficiency:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --chunk-size 500000 \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_3tp_chunked

# 4-timepoint TIA with denoising disabled:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-168H_reg.nii.gz \
             --times 4.0 24.0 48.0 168.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --no-denoise \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_4tp_no_denoise

# 3-timepoint TIA with noise floor disabled:
pytia nifti --images examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz \
             examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz \
             --times 4.0 24.0 48.0 \
             --time-unit hours \
             --half-life 574302.48 \
             --no-noise-floor \
             --output-dir examples/Lu-177_FAP/output_tia \
             --prefix Lu177_FAP_3tp_no_noise_floor

# =============================================================================
# Example 4: Using Configuration Files
# =============================================================================

# Create a config file for 3 time points:
"""
# config_3tp.yaml
inputs:
  images:
    - "examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz"
    - "examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz"
    - "examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz"
  times: [4.0, 24.0, 48.0]

time:
  unit: hours

io:
  output_dir: examples/Lu-177_FAP/output_tia
  prefix: Lu177_FAP_3tp

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
"""

# Run with config file:
pytia run --config config_3tp.yaml

# Create a config file for 4 time points:
"""
# config_4tp.yaml
inputs:
  images:
    - "examples/Lu-177_FAP/input_multi-points/SPECT-004H_reg.nii.gz"
    - "examples/Lu-177_FAP/input_multi-points/SPECT-024H_reg.nii.gz"
    - "examples/Lu-177_FAP/input_multi-points/SPECT-048H_reg.nii.gz"
    - "examples/Lu-177_FAP/input_multi-points/SPECT-168H_reg.nii.gz"
  times: [4.0, 24.0, 48.0, 168.0]

time:
  unit: hours

io:
  output_dir: examples/Lu-177_FAP/output_tia
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
"""

# Run with config file:
pytia run --config config_4tp.yaml

# =============================================================================
# Notes:
# - Lu-177 half-life: 6.647 days = 574302.48 seconds
# - Output files will be saved in examples/Lu-177_FAP/output_tia/
# - Output files include: TIA map, R² map, sigma_tia map, model_id map, status_id map
# - Bootstrap provides uncertainty estimates for TIA values
# - Mask can be used to limit analysis to specific regions (e.g., tumor, organ)
# =============================================================================
