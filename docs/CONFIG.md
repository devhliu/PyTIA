# PyTIA Configuration Reference

Complete reference for all YAML configuration options.

## sections

### `inputs` (required)

Input images and timepoints.

```yaml
inputs:
  images:        # List of image paths (string or list of strings)
    - img1.nii.gz
    - img2.nii.gz
    - img3.nii.gz
  times:         # Timepoints (one per image)
    - 0.0        # First image at t=0
    - 60.0       # Second image at t=60s
    - 120.0      # Third image at t=120s
  mask: null     # Optional: path to mask image
```

**Notes:**
- Images must have same spatial dimensions
- Times must be in same unit as `time.unit`
- Mask must match image dimensions (optional)

### `io`

Input/output configuration.

```yaml
io:
  output_dir: ./output          # Directory for output files
  prefix: my_result             # Prefix for output filenames
  save_intermediate: false      # Save intermediate images
  dtype: float32                # Data type for outputs
  write_summary_yaml: true      # Write summary YAML
  write_status_map: true        # Write status map
```

### `time`

Time configuration.

```yaml
time:
  unit: seconds           # Unit of timepoints: seconds | hours
  sort_timepoints: true   # Auto-sort by time
```

### `physics`

Radioactive decay physics.

```yaml
physics:
  half_life_seconds: 21600.0    # Half-life of radionuclide (seconds)
  enforce_lambda_ge_phys: true  # Enforce physical bounds
```

**Common half-lives (seconds):**
- Tc-99m: 21600 (6 hours)
- F-18: 6600 (110 minutes)
- I-131: 692400 (8.02 days)
- C-11: 1320 (20.4 minutes)
- N-13: 600 (10 minutes)
- O-15: 122 (2 minutes)

### `mask`

Masking strategy.

```yaml
mask:
  mode: otsu              # otsu | provided | none
  provided_path: null     # If mode=provided, path to mask image
  min_fraction_of_max: 0.02  # Min threshold for Otsu
```

**Modes:**
- `otsu` — Automatic Otsu thresholding
- `provided` — Use external mask image
- `none` — No masking (all voxels)

### `denoise`

Spatial denoising.

```yaml
denoise:
  enabled: true
  method: masked_gaussian  # Only method available
  sigma_vox: 1.2          # Gaussian kernel sigma (voxels)
```

### `noise_floor`

Noise floor filtering for voxel validity.

```yaml
noise_floor:
  enabled: true
  mode: relative          # absolute | relative
  absolute_bq_per_ml: 0.0     # If mode=absolute
  relative_fraction_of_voxel_max: 0.01  # If mode=relative (1%)
  behavior: exclude       # Only "exclude" supported
```

**Explanation:**
- `relative`: Voxel valid if A > (max_A × fraction)
- `absolute`: Voxel valid if A > threshold_Bq

### `model_selection`

Auto-classification strategy for multi-timepoint.

```yaml
model_selection:
  mode: auto              # Automatic or manual
  min_points_for_gamma: 3  # Min data points for gamma model
```

### `integration`

Integration parameters for curve fitting.

```yaml
integration:
  start_time_seconds: 0.0    # Integration start time
  tail_mode: phys            # phys | none
  rising_tail_mode: phys     # phys | peak_at_last
```

### `bootstrap`

Uncertainty quantification via residual bootstrap.

```yaml
bootstrap:
  enabled: false          # Enable uncertainty estimates
  n: 50                   # Number of bootstrap replicates
  seed: 0                 # Random seed for reproducibility
  reclassify_each_replicate: true  # Reclassify per replicate
```

**Notes:**
- Significantly increases computation time
- Recommended n ≥ 100 for good uncertainty estimates
- Set seed for reproducibility

### `performance`

Performance tuning.

```yaml
performance:
  chunk_size_vox: 500000   # Process voxels in chunks (0 = no chunking)
  enable_profiling: false  # Print timing info
```

### `regions`

Optional regional ROI aggregation.

```yaml
regions:
  enabled: false
  label_map_path: null    # Path to label/ROI image
  mode: roi_aggregate     # Currently only mode available
  aggregation: mean       # Aggregation method
  voxel_level_r2: false   # Compute per-voxel R² in region mode
  
  classes: {}  # Define regions: {label_int: {class, model, ...}}
  
  scaling:
    mode: tref                 # tref | robust_ratio_mean
    reference_time: peak       # peak | last | index:<int>
```

### `single_time`

**Single-Timepoint (STP) configuration.**

```yaml
single_time:
  enabled: false             # Enable STP mode

  method: phys               # Method: phys | haenscheid | prior_half_life
  
  # For haenscheid method:
  haenscheid_eff_half_life_seconds: null
  
  # For prior_half_life method:
  half_life_seconds: null
  label_map_path: null       # Path to segmentation image
  label_half_lives: {}       # Mapping: {label_int: half_life_seconds}
```

#### STP Methods

**Method 1: Physical Decay** (`method: phys`)

Uses `physics.half_life_seconds`:
```
TIA = A(t) / λ
λ = ln(2) / t_half_phys
```

Example:
```yaml
single_time:
  enabled: true
  method: phys

physics:
  half_life_seconds: 21600.0  # Tc-99m
```

**Method 2: Hänscheid** (`method: haenscheid`)

Uses effective half-life:
```
TIA = A(t) / λ_eff
λ_eff = ln(2) / t_half_eff
```

Example:
```yaml
single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 7200.0  # F-18 FDG effective

physics:
  half_life_seconds: 6600.0  # F-18 physical (fallback)
```

**Method 3: Prior Half-Life** (`method: prior_half_life`)

### Global Mode

All voxels same half-life:
```
TIA = A(t) / λ_prior
λ_prior = ln(2) / t_half_prior
```

Example:
```yaml
single_time:
  enabled: true
  method: prior_half_life
  half_life_seconds: 5400.0  # 1.5 hours
```

### Segmentation Mode

Per-label half-life:
```
TIA_v = A_v / λ_L
λ_L = ln(2) / t_half_L (from label_half_lives[L])
```

Example:
```yaml
single_time:
  enabled: true
  method: prior_half_life
  label_map_path: organs.nii.gz
  label_half_lives:
    1: 1800.0    # Label 1: 30 min
    2: 3600.0    # Label 2: 60 min
    3: 5400.0    # Label 3: 90 min
  half_life_seconds: 3600.0  # Default for unmapped labels
```

## Example Configurations

### Example 1: Minimal Multi-Timepoint

```yaml
inputs:
  images: [t0.nii.gz, t1.nii.gz]
  times: [0.0, 60.0]

io:
  output_dir: ./output

physics:
  half_life_seconds: 21600.0
```

### Example 2: Comprehensive Multi-Timepoint

```yaml
inputs:
  images:
    - scan_0h.nii.gz
    - scan_1h.nii.gz
    - scan_2h.nii.gz
    - scan_4h.nii.gz
  times: [0.0, 3600.0, 7200.0, 14400.0]
  mask: body_mask.nii.gz

io:
  output_dir: ./output/multitime
  prefix: patient_001

time:
  unit: seconds
  sort_timepoints: true

physics:
  half_life_seconds: 21600.0

mask:
  mode: provided
  provided_path: body_mask.nii.gz

denoise:
  enabled: true
  sigma_vox: 1.5

noise_floor:
  enabled: true
  mode: relative
  relative_fraction_of_voxel_max: 0.01

bootstrap:
  enabled: true
  n: 100
  seed: 42

performance:
  chunk_size_vox: 250000
  enable_profiling: true

regions:
  enabled: false
```

### Example 3: Single-Timepoint Physical Decay

```yaml
inputs:
  images: [activity_snapshot.nii.gz]
  times: [0.0]

io:
  output_dir: ./output/stp_phys
  prefix: stp_phys

physics:
  half_life_seconds: 21600.0

single_time:
  enabled: true
  method: phys
```

### Example 4: Single-Timepoint Segmentation

```yaml
inputs:
  images: [activity.nii.gz]
  times: [0.0]

io:
  output_dir: ./output/stp_organs
  prefix: stp_organs

single_time:
  enabled: true
  method: prior_half_life
  label_map_path: segmentation.nii.gz
  label_half_lives:
    1: 1800.0   # Tumor
    2: 3600.0   # Liver
    3: 5400.0   # Spleen
    4: 7200.0   # Kidney
    5: 10800.0  # Blood
  half_life_seconds: 3600.0  # Default
```

## Data Types

- **Paths:** Strings (absolute or relative to config location)
- **Numbers:** Integers or floats
- **Booleans:** `true` | `false`
- **Lists:** `[item1, item2, item3]`
- **Dicts:** `{key1: value1, key2: value2}`

## Validation

Validate config before running:

```bash
pytia validate --config config.yaml
```

Checks:
- YAML syntax validity
- Required sections present
- Option types correct
- File paths accessible (if mode=provided)

## Best Practices

1. **Always specify `outputs_dir`** — Prevents accidental overwrites
2. **Use `seed` in bootstrap** — For reproducible results
3. **Enable profiling** during development — Optimize performance
4. **Test with validation** — Before batch processing
5. **Keep configs versioned** — Track changes with git

## Common Issues

### Issue: "Config must contain 'inputs.images'"
**Solution:** Add `inputs.images` list

### Issue: Image dimension mismatch
**Solution:** Ensure all images have same XYZ dimensions

### Issue: Times don't match images
**Solution:** Ensure `len(times) == len(images)`

### Issue: TIA values seem wrong
**Solution:**
1. Check `half_life_seconds` is in seconds (not hours)
2. Verify `time.unit` matches your times list
3. Check noise_floor settings

---

**For more help:** See [USER_GUIDE.md](USER_GUIDE.md)
