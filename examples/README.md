# PyTIA Examples

This folder contains example scripts and configuration files demonstrating PyTIA usage.

## Quick Start

### 1. Run Multi-Timepoint Example (Python)

```bash
python examples/example_multitime.py
```

This creates synthetic data and runs TIA estimation with curve fitting.

### 2. Run Single-Timepoint Examples (Python)

```bash
python examples/example_stp.py
```

Demonstrates all three STP methods:
- Physical decay
- Hänscheid effective half-life
- Prior half-life (global and segmentation-based)

### 3. Run with Config Files (CLI)

```bash
# Multi-timepoint
pytia run --config examples/config_multitime.yaml

# Single-timepoint with physical decay
pytia run --config examples/config_stp_phys.yaml

# Single-timepoint with Hänscheid
pytia run --config examples/config_stp_haenscheid.yaml

# Single-timepoint with segmentation-based priors
pytia run --config examples/config_stp_prior_seg.yaml
```

## Files

### Python Examples

| File | Description |
|------|-------------|
| `example_multitime.py` | Multi-timepoint TIA with curve fitting |
| `example_stp.py` | Single-timepoint with all 3 methods |
| `demo_stp_calculations.py` | Mathematical demonstrations |

### Configuration Files

| File | Use Case |
|------|----------|
| `config_multitime.yaml` | Multi-timepoint template |
| `config_stp_phys.yaml` | Physical decay method |
| `config_stp_haenscheid.yaml` | Hänscheid effective half-life |
| `config_stp_prior_seg.yaml` | Segmentation-based priors |

## Understanding the Examples

### Multi-Timepoint Flow

1. **Load images** at different timepoints
2. **Configure** physics and fitting parameters
3. **Run TIA estimation** with curve fitting
4. **Access results** (TIA, R², uncertainty, status)

**When to use:**
- Multiple PET/SPECT scans over time
- Quantitative kinetic analysis
- Need model fit quality (R²)

### Single-Timepoint Flow

1. **Create** activity image (single snapshot)
2. **Select method** (phys/haenscheid/prior)
3. **Configure** method-specific parameters
4. **Run TIA calculation** (direct formula)

**When to use:**
- Clinical routine with time constraints
- One image available
- Known or assumed half-life

## Customization

### Adapting Multi-Timepoint Example

```python
# Load your own data
from pytia import run_tia

result = run_tia(
    images=["your_img1.nii.gz", "your_img2.nii.gz"],
    times=[0.0, 60.0],
    config={
        "physics": {"half_life_seconds": 21600.0},
        "io": {"output_dir": "./my_output"}
    }
)
```

### Adapting Config Files

1. Copy relevant `.yaml` file
2. Update `inputs.images` paths
3. Set `io.output_dir`
4. Update `physics.half_life_seconds` if needed
5. Run: `pytia run --config your_config.yaml`

### Batch Processing

Create `batch_process.sh`:

```bash
#!/bin/bash
for patient in data/patient_*/; do
    config="configs/$(basename $patient).yaml"
    if [ -f "$config" ]; then
        pytia run --config "$config"
        echo "✓ Processed $(basename $patient)"
    fi
done
```

Run:
```bash
bash batch_process.sh
```

## Debugging

### Validate Config

```bash
pytia validate --config your_config.yaml
```

### View Config

```bash
pytia info --config your_config.yaml
```

### Check Output

```python
import nibabel as nib
import numpy as np

# Load results
tia = nib.load("output/tia.nii.gz")
status = nib.load("output/status_id.nii.gz")

# Analyze
tia_data = np.asarray(tia.dataobj)
status_data = np.asarray(status.dataobj)

# Statistics
valid = status_data == 1
print(f"Valid voxels: {np.sum(valid)}")
print(f"Mean TIA: {np.nanmean(tia_data[valid]):.2f}")
```

## Tips

### Memory Management

For large datasets, use chunking:

```yaml
performance:
  chunk_size_vox: 250000
```

### Speed

Disable features you don't need:

```yaml
bootstrap:
  enabled: false  # Unless you need uncertainty

denoise:
  enabled: false  # Unless you need smoothing
```

### Reproducibility

Always set seed:

```yaml
bootstrap:
  seed: 42
```

## Getting Help

1. **Check config syntax:** `pytia validate --config config.yaml`
2. **View config:** `pytia info --config config.yaml`
3. **Read documentation:** [docs/USER_GUIDE.md](../docs/USER_GUIDE.md)
4. **Check status codes:** In `output/pytia_summary.yaml`

## Next Steps

1. Try examples with your own data
2. Read [USER_GUIDE.md](../docs/USER_GUIDE.md)
3. Explore [CONFIG.md](../docs/CONFIG.md) for advanced options
4. Check [README.md](../README.md) for overview

---

**Questions?** See [docs/USER_GUIDE.md](../docs/USER_GUIDE.md#troubleshooting)
