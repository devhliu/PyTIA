# Single-Timepoint (STP) TIA Calculation - User Guide

## Quick Start

### 1. Enable STP Mode in Config

Set `single_time.enabled: true` and choose a method:

```yaml
# config.yaml
physics:
  half_life_seconds: 3600.0  # Physical half-life (seconds)

single_time:
  enabled: true
  method: "phys"  # or "haenscheid" or "prior_half_life"
```

### 2. Run with Single Image

```python
from pytia import run_tia

result = run_tia(
    images=["activity_map.nii.gz"],
    times=[0.0],  # Single timepoint
    config="config.yaml"
)

# Access outputs
print(result.tia_img)        # TIA map
print(result.model_id_img)   # Method ID (101/102/103)
print(result.status_id_img)  # Validity status
```

## Methods

### Method 1: Physical Decay (Model ID: 101)

Computes TIA using the radionuclide's **physical half-life**.

**Formula:**
$$\text{TIA} = \frac{A(t)}{\lambda_{\text{phys}}} \quad \text{where} \quad \lambda_{\text{phys}} = \frac{\ln(2)}{t_{1/2,\text{phys}}}$$

**Config:**
```yaml
physics:
  half_life_seconds: 3600.0  # e.g., Tc-99m ≈ 6 hours = 21600 s

single_time:
  enabled: true
  method: phys
```

**Use Case:**
- No biological clearance/metabolism considered
- Pure radioactive decay extrapolation
- Useful for short-lived tracers or simple estimates

---

### Method 2: Hänscheid Method (Model ID: 102)

Uses an **effective half-life** accounting for both radioactive decay and biological clearance.

**Formula:**
$$\text{TIA} = \frac{A(t)}{\lambda_{\text{eff}}} \quad \text{where} \quad \lambda_{\text{eff}} = \frac{\ln(2)}{t_{1/2,\text{eff}}}$$

**Config:**
```yaml
single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 7200.0  # Effective half-life in human body

physics:
  half_life_seconds: 3600.0  # Physical (optional, used as fallback)
```

If `haenscheid_eff_half_life_seconds` is `null`, falls back to `physics.half_life_seconds`.

**Use Case:**
- PET tracers (e.g., F-18 FDG with 2-hour effective clearance)
- SPECT agents with known body clearance
- More realistic dosimetry estimates

---

### Method 3: Prior Half-Life (Model ID: 103)

Uses **prior half-life(s)** from literature or segmentation. Supports two modes:

#### 3a. Global Mode
Same half-life for entire volume.

**Config:**
```yaml
single_time:
  enabled: true
  method: prior_half_life
  half_life_seconds: 5400.0  # 1.5 hours
```

#### 3b. Segmentation-Based Mode
Different half-life per organ/lesion from label map.

**Config:**
```yaml
single_time:
  enabled: true
  method: prior_half_life
  label_map_path: segmentation.nii.gz
  label_half_lives:
    1: 1800.0   # Label 1: 30 min
    2: 3600.0   # Label 2: 60 min
    3: 5400.0   # Label 3: 90 min
  half_life_seconds: 3600.0  # Default for unmapped labels
```

**Formula:**
$$\text{TIA}_v = \frac{A_v}{\lambda_L} \quad \text{where} \quad L = \text{label}(v), \quad \lambda_L = \frac{\ln(2)}{t_{1/2,L}}$$

**Use Case:**
- Organ-specific kinetics (liver clearance ≠ kidney clearance)
- Lesion-specific tracer uptake/clearance
- Multi-organ dosimetry calculations
- Research with known compartment half-lives

---

## Configuration Examples

### Example 1: Tc-99m Dosimetry (Physical Decay)
```yaml
io:
  output_dir: ./output
  
physics:
  half_life_seconds: 21600.0  # Tc-99m: 6 hours

single_time:
  enabled: true
  method: phys

denoise:
  enabled: false
  
noise_floor:
  enabled: true
  mode: relative
  relative_fraction_of_voxel_max: 0.01
```

### Example 2: F-18 FDG (Hänscheid Effective)
```yaml
physics:
  half_life_seconds: 110.0 * 60.0  # F-18: 110 minutes

single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 2.0 * 3600.0  # Effective: 2 hours in body
```

### Example 3: Multi-Organ Dosimetry (Segmentation-Based)
```yaml
single_time:
  enabled: true
  method: prior_half_life
  label_map_path: organs.nii.gz
  half_life_seconds: 3600.0  # Default
  label_half_lives:
    1: 600.0      # Blood: 10 min
    2: 1800.0     # Tumor: 30 min
    3: 3600.0     # Liver: 60 min
    4: 7200.0     # Spleen: 120 min
    5: 5400.0     # Kidney: 90 min
    6: 10800.0    # Intestine: 180 min
```

---

## Output Interpretation

### Output Images

| File | Content | Notes |
|------|---------|-------|
| `tia.nii.gz` | Time-integrated activity | Bq·s/ml or Bq·s/g |
| `model_id.nii.gz` | Method used | 101/102/103 or 0 (background) |
| `status_id.nii.gz` | Voxel validity | See status codes |
| `r2.nii.gz` | Model fit quality | NaN for STP (not applicable) |
| `sigma_tia.nii.gz` | Uncertainty | NaN for STP (no fitting) |

### Status Codes

| Code | Meaning |
|------|---------|
| **0** | `STATUS_OUTSIDE` — Outside mask/background |
| **1** | `STATUS_OK` — Valid TIA computed ✓ |
| **2** | `STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS` — Invalid decay rate (λ ≤ 0 or NaN) |
| **3** | `STATUS_FIT_FAILED` — Missing required configuration (no half-life) |
| **4** | `STATUS_ALL_BELOW_FLOOR` — Activity below noise floor |

### Summary YAML

Example `pytia_summary.yaml`:
```yaml
pytia_version: "0.1.0"
times_seconds: [0.0]
voxel_volume_ml: 8.0
status_legend:
  0: "outside mask/background"
  1: "ok"
  2: "not applicable: invalid decay rate"
  3: "fit failed: missing configuration"
  4: "all points below noise floor"
status_counts:
  ok: 45230
  "outside mask/background": 1000
  "all points below noise floor": 100
config:
  single_time:
    enabled: true
    method: prior_half_life
    # ... rest of config ...
```

---

## Validation & QC

### Check Results
```python
import nibabel as nib
import numpy as np

# Load outputs
tia_img = nib.load("tia.nii.gz")
status_img = nib.load("status_id.nii.gz")
model_img = nib.load("model_id.nii.gz")

tia_data = np.asarray(tia_img.dataobj)
status_data = np.asarray(status_img.dataobj)
model_data = np.asarray(model_img.dataobj)

# Check valid voxels
valid_mask = status_data == 1
print(f"Valid voxels: {np.sum(valid_mask)}")
print(f"Mean TIA: {np.nanmean(tia_data[valid_mask]):.2f} Bq·s/ml")
print(f"TIA range: {np.nanmin(tia_data[valid_mask]):.2f} - {np.nanmax(tia_data[valid_mask]):.2f}")

# Verify method used
print(f"Method distribution:")
for method_id in [101, 102, 103]:
    count = np.sum(model_data == method_id)
    print(f"  Model {method_id}: {count} voxels")
```

---

## Common Issues & Solutions

### Issue: "FIT_FAILED" or all voxels invalid

**Cause:** Missing or invalid half-life configuration.

**Solution:**
- For `phys`: Ensure `physics.half_life_seconds` is set
- For `haenscheid`: Set `single_time.haenscheid_eff_half_life_seconds`
- For `prior_half_life`: Set `single_time.half_life_seconds` or label mappings

### Issue: TIA values seem too large/small

**Cause:** Half-life unit mismatch.

**Solution:** Ensure all half-lives are in **seconds**:
- 1 hour = 3600 seconds
- 1 day = 86400 seconds
- 1 minute = 60 seconds

### Issue: Segmentation-based mode ignoring labels

**Cause:** Label values in `label_half_lives` don't match segmentation file.

**Solution:**
1. Check segmentation file label values: `nilearn.plotting.view_img()`
2. Ensure keys in `label_half_lives` match (as integers)
3. Set `half_life_seconds` for unmapped labels (fallback)

---

## Advanced Usage

### Disable STP for Single Image
If you have a single image but want multi-timepoint fitting:
```yaml
single_time:
  enabled: false  # Bypass STP mode
```
This will attempt multi-timepoint logic (may give unexpected results for T=1).

### Noise Floor with STP
Noise floor filtering is applied the same way as multi-timepoint:
```yaml
noise_floor:
  enabled: true
  mode: relative  # or absolute
  relative_fraction_of_voxel_max: 0.01  # 1% of voxel max
```

### Custom Label Map
Create label map with your own ROIs:
```python
import nibabel as nib
import numpy as np

# Create label map
labels = np.zeros((128, 128, 50), dtype=np.int32)
labels[10:30, 10:30, :] = 1    # Tumor
labels[40:80, 40:80, :] = 2    # Liver
labels[30:40, 30:40, :] = 3    # Kidney

label_img = nib.Nifti1Image(labels, np.eye(4))
nib.save(label_img, "organs.nii.gz")
```

---

## Mathematical Details

### Time-Integrated Activity

For a single timepoint, **TIA** represents the integral of activity from that time to infinity, assuming exponential decay:

$$\text{TIA} = \int_t^{\infty} A(\tau) e^{-\lambda(\tau - t)} d\tau = \frac{A(t)}{\lambda}$$

Where:
- $A(t)$ = Activity at time $t$ (Bq/ml)
- $\lambda$ = Decay constant (s$^{-1}$) = $\ln(2) / t_{1/2}$
- $t_{1/2}$ = Half-life (seconds)

### Effective Half-Life

Combines radioactive decay and biological clearance:

$$\lambda_{\text{eff}} = \lambda_{\text{phys}} + \lambda_{\text{biol}}$$

$$\frac{1}{t_{1/2,\text{eff}}} = \frac{1}{t_{1/2,\text{phys}}} + \frac{1}{t_{1/2,\text{biol}}}$$

For known $t_{1/2,\text{eff}}$ (from literature or prior), use directly without decomposition.

---

## References

- Hänscheid, H. et al. (2023). Dose evaluation in nuclear medicine. *J Nucl Med* 64, 195S–207S.
- ICRP Publication 128: Radiological protection in ion beam therapy
- European guidelines on dosimetry in medical imaging

---

## See Also

- [SINGLE_TIMEPOINT_IMPLEMENTATION.md](./SINGLE_TIMEPOINT_IMPLEMENTATION.md) — Technical implementation details
- `demo_stp_calculations.py` — Mathematical examples and comparisons
- `tests/test_single_timepoint.py` — Comprehensive unit tests

