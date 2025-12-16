# PyTIA Quick Reference

## Installation

```bash
pip install pytia
```

## Quick Start: 30 Seconds

### Option 1: Command-Line

```bash
# Run with your config
pytia run --config my_config.yaml

# That's it! Outputs in output_dir from config
```

### Option 2: Python

```python
from pytia import run_tia

result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

print(result.summary)
```

## Minimal Config (YAML)

```yaml
inputs:
  images: [t0.nii.gz, t1.nii.gz]
  times: [0.0, 60.0]

io:
  output_dir: ./output

physics:
  half_life_seconds: 21600.0  # 6h for Tc-99m
```

Save as `config.yaml`, then:
```bash
pytia run --config config.yaml
```

## 3 Ways to Use

### Way 1: CLI (Recommended for production)
```bash
pytia run --config config.yaml
```
✓ No coding  
✓ Batch processing  
✓ Reproducible  

### Way 2: Python API (Recommended for development)
```python
from pytia import run_tia, Config

result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz"],
    times=[0.0, 60.0],
    config="config.yaml"
)

tia = result.tia_img
import nibabel as nib
nib.save(tia, "output_tia.nii.gz")
```
✓ Fine control  
✓ Integration  
✓ Custom processing  

### Way 3: Config Dict (For scripting)
```python
from pytia import run_tia

config = {
    "inputs": {
        "images": ["t0.nii.gz", "t1.nii.gz"],
        "times": [0.0, 60.0]
    },
    "io": {"output_dir": "./output"},
    "physics": {"half_life_seconds": 21600.0}
}

result = run_tia(
    images=config["inputs"]["images"],
    times=config["inputs"]["times"],
    config=config
)
```

## CLI Commands

```bash
# Run TIA estimation
pytia run --config config.yaml

# Validate your config file
pytia validate --config config.yaml

# Show config contents
pytia info --config config.yaml
```

## Common Tracers & Half-Lives

| Tracer | Half-Life | Seconds |
|--------|-----------|---------|
| Tc-99m | 6.0 hours | 21,600 |
| F-18 | 110 minutes | 6,600 |
| I-131 | 8.0 days | 691,200 |
| C-11 | 20.4 minutes | 1,224 |
| Ga-68 | 67.9 minutes | 4,074 |

## Typical Config Sections

```yaml
inputs:                    # ← REQUIRED
  images: [t0.nii.gz, t1.nii.gz]
  times: [0.0, 60.0]

io:                        # ← REQUIRED
  output_dir: ./output

physics:                   # ← REQUIRED
  half_life_seconds: 21600.0

# Optional sections:
mask:
  use_mask: true
  mask_path: mask.nii.gz

denoise:
  method: gaussian
  sigma_mm: 2.0

noise_floor:
  enabled: true
  threshold_kbq_ml: 0.5

bootstrap:
  enabled: true
  n_samples: 100

regions:
  enabled: true
  roi_path: roi.nii.gz

single_time:
  method: "phys"
```

## Output Files

After running `pytia run --config config.yaml`:

```
output/
├── tia.nii.gz              # ← Main output (TIA values)
├── r2.nii.gz               # Model fit quality
├── status_id.nii.gz        # Success/failure per voxel
├── model_id.nii.gz         # Which model used (10,11,20,30,101,102,103)
├── sigma_tia.nii.gz        # Uncertainty (if bootstrap enabled)
└── pytia_summary.yaml      # Summary statistics
```

## Load Results in Python

```python
import nibabel as nib

# Load output
tia = nib.load("output/tia.nii.gz")
tia_data = tia.get_fdata()

# Get voxel spacing
voxel_size = tia.header.get_zooms()[:3]

# Access metadata
affine = tia.affine
```

## Examples

Pre-built examples in `examples/` folder:

### Multi-timepoint
```bash
python examples/example_multitime.py
# or
pytia run --config examples/config_multitime.yaml
```

### Single-timepoint (Physical decay)
```bash
pytia run --config examples/config_stp_phys.yaml
```

### Single-timepoint (Hänscheid)
```bash
pytia run --config examples/config_stp_haenscheid.yaml
```

### Single-timepoint (Segmentation-based)
```bash
pytia run --config examples/config_stp_prior_seg.yaml
```

## Status Codes

| Code | Meaning |
|------|---------|
| 1 | ✓ Success |
| 0 | Data not suitable for fitting |
| 2 | Bad fit quality |
| 3 | Masked out |
| 4 | Negative values |
| 5 | Other issues |

## Quick Workflow

1. **Prepare data**
   ```bash
   # Convert to NIfTI if needed
   # Ensure same geometry
   ```

2. **Create config**
   ```yaml
   inputs:
     images: [t0.nii.gz, t1.nii.gz]
     times: [0.0, 60.0]
   io:
     output_dir: ./output
   physics:
     half_life_seconds: 21600.0
   ```

3. **Validate**
   ```bash
   pytia validate --config config.yaml
   ```

4. **Run**
   ```bash
   pytia run --config config.yaml
   ```

5. **Check results**
   ```bash
   ls output/
   # tia.nii.gz, r2.nii.gz, etc.
   ```

## Python Workflow

```python
from pytia import run_tia, load_images
import nibabel as nib

# Load or prepare images
img_list = ["t0.nii.gz", "t1.nii.gz"]
times = [0.0, 60.0]

# Run TIA
result = run_tia(
    images=img_list,
    times=times,
    config="config.yaml"
)

# Access results
tia_img = result.tia_img
r2_img = result.r2_img
summary = result.summary

# Save
nib.save(tia_img, "my_tia.nii.gz")

# Analyze
tia_data = tia_img.get_fdata()
print(f"Mean TIA: {tia_data.mean():.2f}")
print(f"Max TIA: {tia_data.max():.2f}")
```

## Single-Timepoint (STP) Modes

### Physical Decay (Simple)
```yaml
single_time:
  method: "phys"
```
Use when: Direct extrapolation with known half-life

### Hänscheid (F-18 standard)
```yaml
single_time:
  method: "haenscheid"
  effective_half_life_minutes: 110
```
Use when: F-18 FDG or similar with clearance component

### Prior Half-Life (Organ-specific)
```yaml
single_time:
  method: "prior_half_life"
  label_map_path: organs.nii.gz
  label_half_lives:
    1: 120  # tumor
    2: 180  # liver
    3: 150  # kidney
```
Use when: Different tissues have different half-lives

## Getting Help

- **User Guide:** See `docs/USER_GUIDE.md`
- **Config Reference:** See `docs/CONFIG.md`
- **Architecture:** See `docs/ARCHITECTURE.md`
- **Examples:** Run scripts in `examples/` folder
- **Tests:** See `tests/` folder for usage patterns

## Performance Tips

```yaml
performance:
  chunk_size_vox: 250000  # Process in chunks
  enable_profiling: false  # Set true to time operations

bootstrap:
  enabled: false          # Disable if not needed (faster)
  n_samples: 100

denoise:
  method: null            # Disable if not needed
```

## Troubleshooting

**Error: Config file not found**
```bash
# Make sure path is correct
pytia validate --config config.yaml
```

**Error: Image files not found**
```yaml
# Use absolute paths or paths relative to where you run the command
inputs:
  images:
    - /full/path/to/t0.nii.gz
    - /full/path/to/t1.nii.gz
```

**Error: Affinity mismatch**
```bash
# All images must have same geometry
# Use tools like ants to coregister first
```

**Invalid TIA values**
```yaml
# Check noise floor isn't too high
noise_floor:
  threshold_kbq_ml: 0.5  # Adjust if needed

# Check half-life is correct
physics:
  half_life_seconds: 21600.0  # Verify for your tracer
```

## Citation

If you use PyTIA, please cite:
```
PyTIA: Voxel-wise Total Injected Activity from PET/SPECT
```

## License

MIT License - See LICENSE file

---

**For more:** See full documentation in `docs/` folder
