# PyTIA Quick Start

## 1) Install

```bash
pip install pytia
```

For local development:

```bash
pip install -e ".[dev]"
```

## 2) Create a minimal config

```yaml
inputs:
  images:
    - t0.nii.gz
    - t1.nii.gz
    - t2.nii.gz
  times: [1.0, 24.0, 72.0]

time:
  unit: hours
  sort_timepoints: true

io:
  output_dir: ./output
  prefix: patient_01

physics:
  half_life_seconds: 21600.0
```

## 3) Validate, then run

```bash
pytia validate --config config.yaml
pytia run --config config.yaml
```

Optional:

```bash
pytia info --config config.yaml
pytia --version
```

## 4) Expected outputs

By default, `io.output_dir` contains:

- `tia.nii.gz`
- `r2.nii.gz`
- `sigma_tia.nii.gz`
- `model_id.nii.gz`
- `status_id.nii.gz`
- `pytia_summary.yaml`

If `io.prefix` is set, filenames are prefixed (for example `patient_01_tia.nii.gz`).

## 5) Python API quick example

```python
from pytia import run_tia

result = run_tia(
    images=["t0.nii.gz", "t1.nii.gz", "t2.nii.gz"],
    times=[1.0, 24.0, 72.0],
    config="config.yaml",
)

print(result.summary["status_counts"])
```

## 6) Single-timepoint quick example

```yaml
inputs:
  images: [activity_single.nii.gz]
  times: [0.0]

io:
  output_dir: ./output_stp

physics:
  half_life_seconds: 21600.0

single_time:
  enabled: true
  method: phys
```

Run:

```bash
pytia run --config config_stp.yaml
```

## 7) Common issues

### `Unknown config key(s)...`

PyTIA enforces strict key validation. Check spelling and section names against [`CONFIG.md`](CONFIG.md).

### `Number of times must match number of images`

Ensure `len(inputs.times) == len(inputs.images)`.

### `mask.provided_path is required`

If `mask.mode: provided`, set `mask.provided_path`.

### Unexpected TIA scale

Verify:

- `time.unit` matches `inputs.times`
- `physics.half_life_seconds` is in seconds
- `noise_floor` settings are intentional

## Next

- Full usage: [`USER_GUIDE.md`](USER_GUIDE.md)
- Complete config reference: [`CONFIG.md`](CONFIG.md)
- STP-specific details: [`STP_USER_GUIDE.md`](STP_USER_GUIDE.md)
- Release flow: [`RELEASE.md`](RELEASE.md)
