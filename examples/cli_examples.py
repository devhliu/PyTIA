"""
CLI Usage Examples for PyTIA

This file demonstrates various ways to use the PyTIA CLI for computing
Time-Integrated Activity (TIA) maps from PET/SPECT imaging data.
"""

# =============================================================================
# 1. Multi-Timepoint TIA Calculation (2+ images)
# =============================================================================

# Basic multi-timepoint analysis using NIfTI files directly:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0

# With custom output directory and prefix:
# pytia nifti --images scan*.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --output-dir ./results \
#              --prefix patient_001

# With mask:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --mask body_mask.nii.gz

# With bootstrap for uncertainty estimation:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --bootstrap 100 \
#              --bootstrap-seed 42

# =============================================================================
# 2. Single-Timepoint TIA Calculation (1 image)
# =============================================================================

# Method 1: Physical decay (default)
# pytia nifti --images scan1.nii.gz \
#              --times 24.0 \
#              --single-time \
#              --stp-method phys \
#              --half-life 21636.0

# Method 2: Hänscheid method (effective half-life)
# pytia nifti --images scan1.nii.gz \
#              --times 24.0 \
#              --single-time \
#              --stp-method haenscheid \
#              --eff-half-life 3600.0

# Method 3: Prior half-life (global)
# pytia nifti --images scan1.nii.gz \
#              --times 24.0 \
#              --single-time \
#              --stp-method prior_half_life \
#              --prior-half-life 1800.0

# Method 4: Prior half-life (segmentation-based)
# pytia nifti --images scan1.nii.gz \
#              --times 24.0 \
#              --single-time \
#              --stp-method prior_half_life \
#              --label-map labels.nii.gz \
#              --label-half-lives '{"1": 1800.0, "2": 3600.0, "3": 5400.0}'

# =============================================================================
# 3. Using Configuration Files
# =============================================================================

# Run with a YAML configuration file:
# pytia run --config config.yaml

# Validate a configuration file:
# pytia validate --config config.yaml

# Show configuration file contents:
# pytia info --config config.yaml

# =============================================================================
# 4. Advanced Options
# =============================================================================

# Disable denoising:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --no-denoise

# Disable noise floor filtering:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --no-noise-floor

# Disable bootstrap:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --no-bootstrap

# Set chunk size for memory-efficient processing:
# pytia nifti --images scan1.nii.gz scan2.nii.gz scan3.nii.gz \
#              --times 1.0 24.0 72.0 \
#              --time-unit hours \
#              --half-life 21636.0 \
#              --chunk-size 500000

# =============================================================================
# 5. Example Configuration File (config.yaml)
# =============================================================================

"""
# config.yaml example:
"""
# inputs:
#   images:
#     - "scan1.nii.gz"
#     - "scan2.nii.gz"
#     - "scan3.nii.gz"
#   times: [1.0, 24.0, 72.0]
# 
# time:
#   unit: hours
# 
# io:
#   output_dir: ./pytia_output
#   prefix: patient_001
# 
# physics:
#   half_life_seconds: 21636.0  # Tc-99m half-life in seconds
# 
# mask:
#   mode: provided
#   provided_path: "body_mask.nii.gz"
# 
# denoise:
#   enabled: true
#   sigma_vox: 1.2
# 
# noise_floor:
#   enabled: true
#   mode: relative
#   relative_fraction_of_voxel_max: 0.01
# 
# bootstrap:
#   enabled: true
#   n: 100
#   seed: 42
# 
# performance:
#   chunk_size_vox: 500000
# 
# single_time:
#   enabled: false
#   method: phys
"""

# =============================================================================
# 6. Single-Timepoint Configuration Example
# =============================================================================

"""
# config_stp.yaml example:
"""
# inputs:
#   images:
#     - "scan1.nii.gz"
#   times: [24.0]
# 
# time:
#   unit: hours
# 
# io:
#   output_dir: ./pytia_output
#   prefix: patient_001_stp
# 
# physics:
#   half_life_seconds: 21636.0
# 
# single_time:
#   enabled: true
#   method: phys  # Options: phys, haenscheid, prior_half_life
#   # For haenscheid method:
#   # haenscheid_eff_half_life_seconds: 3600.0
#   # For prior_half_life method:
#   # half_life_seconds: 1800.0
#   # label_map_path: "labels.nii.gz"
#   # label_half_lives:
#   #   1: 1800.0
#   #   2: 3600.0
#   #   3: 5400.0
"""
