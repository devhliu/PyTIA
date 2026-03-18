# Single-Timepoint (STP) User Guide

STP mode computes TIA from a single activity image (`T == 1`) using:

`TIA = A(t) / lambda_eff`

Enable STP with:

```yaml
single_time:
  enabled: true
```

## 1) Method selection

### A. `phys` (model ID `101`)

Uses physical decay:

- `lambda_eff = ln(2) / physics.half_life_seconds`

```yaml
physics:
  half_life_seconds: 21600.0

single_time:
  enabled: true
  method: phys
```

### B. `haenscheid` / `hanscheid` (model ID `102`)

Uses effective half-life:

- `lambda_eff = ln(2) / single_time.haenscheid_eff_half_life_seconds`
- Fallback: `physics.half_life_seconds` if STP effective half-life is `null`

```yaml
physics:
  half_life_seconds: 21600.0

single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 14400.0
```

### C. `prior_half_life` / `prior` (model ID `103`)

Two sub-modes:

1. Global half-life
2. Label-map driven half-life

Global example:

```yaml
single_time:
  enabled: true
  method: prior_half_life
  half_life_seconds: 10800.0
```

Label-map example:

```yaml
single_time:
  enabled: true
  method: prior_half_life
  label_map_path: organs.nii.gz
  half_life_seconds: 10800.0
  label_half_lives:
    "1": 1800.0
    "2": 3600.0
    "3": 5400.0
```

`label_half_lives` keys must be integer-like labels.

## 2) Running STP

```bash
pytia validate --config config_stp.yaml
pytia run --config config_stp.yaml
```

Python API:

```python
from pytia import run_tia

result = run_tia(
    images=["activity_single.nii.gz"],
    times=[0.0],
    config="config_stp.yaml",
)
```

## 3) Outputs and IDs

NIfTI outputs:

- `tia.nii.gz`
- `model_id.nii.gz`
- `status_id.nii.gz`
- `r2.nii.gz` (typically NaN in STP)
- `sigma_tia.nii.gz` (typically NaN in STP)

Summary YAML fields include:

- `pytia_version`
- `status_legend` + `status_counts`
- `model_legend` + `model_counts`

Status IDs are shared with multi-timepoint mode:

- `0`: outside mask/background
- `1`: ok
- `2`: not applicable (`<2 valid points` legend; in STP this often indicates invalid effective decay input)
- `3`: fit failed
- `4`: all points below noise floor
- `5`: nonphysical parameters

## 4) Troubleshooting

### `fit failed` for STP voxels

Check method-specific half-life inputs:

- `phys` needs `physics.half_life_seconds`
- `haenscheid` needs `haenscheid_eff_half_life_seconds` or fallback physical half-life
- `prior_half_life` global mode needs `single_time.half_life_seconds`

### All voxels below floor

Relax or disable `noise_floor` for debugging:

```yaml
noise_floor:
  enabled: false
```

### Unexpected method codes

Confirm `single_time.method` and inspect `model_id.nii.gz`:

- `101` phys
- `102` haenscheid/hanscheid
- `103` prior/prior_half_life

## 5) Related docs

- Full config reference: [`CONFIG.md`](CONFIG.md)
- General usage: [`USER_GUIDE.md`](USER_GUIDE.md)
- Runtime architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)
