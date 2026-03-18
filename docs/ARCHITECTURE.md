# PyTIA Architecture

This document describes the runtime architecture and contract surfaces that should remain stable.

## 1) Public entry points

PyTIA exposes two user-facing entry points:

- CLI: `pytia` (`pytia/cli.py`)
  - `pytia run --config ...`
  - `pytia validate --config ...`
  - `pytia info --config ...`
  - `pytia --version`
- Python API (`pytia/__init__.py`)
  - `run_tia(...)`
  - `Config.load(...)`
  - `Results`

Both execution paths converge at `pytia.engine.run_tia(...)`.

## 2) Configuration lifecycle

Configuration is centralized in `pytia.config.Config.load(...)`.

Runtime behavior:

1. Load default schema via `default_config()`
2. Validate user keys (fail fast on unknown keys)
3. Deep-merge user settings onto defaults
4. Validate enums, ranges, and required dependencies

Important behavior:

- Unknown keys are rejected before execution.
- `inputs` is accepted at config load level for CLI/API workflows.
- Single-time aliases are accepted:
  - `haenscheid` and `hanscheid`
  - `prior_half_life` and `prior`

## 3) Core computation pipeline (`run_tia`)

High-level flow:

1. Load images and normalize times to seconds.
2. Build/apply mask (`provided`, `otsu`, or `none`).
3. Optional denoising and negative-value clamping.
4. Convert activity density to per-voxel activity using `voxel_volume_ml`.
5. Apply one of three analysis paths:
   - single-timepoint mode
   - region mode
   - voxel multi-timepoint mode
6. Optional residual bootstrap uncertainty.
7. Assemble outputs, write NIfTI maps, write summary YAML.

## 4) Analysis modes

### A. Single-timepoint mode (`T == 1` and `single_time.enabled`)

Computes `TIA = A(t) / lambda_eff`.

Methods:

- `phys` → model ID `101`
- `haenscheid` / `hanscheid` → model ID `102`
- `prior_half_life` / `prior` → model ID `103`

### B. Region mode (`regions.enabled`)

- Aggregates region TACs by label.
- Fits one region-level kinetic shape.
- Scales region result back to voxel-level amplitudes.
- Optional voxel-level `R²` in region mode.

### C. Voxel multi-timepoint mode

- Curve classification:
  - `CLASS_RISING`
  - `CLASS_HUMP`
  - `CLASS_FALLING`
  - `CLASS_AMBIG`
- Model assignment:
  - hybrid rising: model ID `10`
  - hybrid non-rising: model ID `11`
  - monoexp: model ID `20`
  - gamma: model ID `30`

## 5) Output contract (`pytia.types.Results`)

`Results` fields:

- `tia_img`
- `r2_img`
- `sigma_tia_img`
- `model_id_img`
- `status_id_img`
- `tpeak_img` (optional)
- `summary`
- `output_paths`
- `config`
- `times_s`

Primary output files:

- `tia.nii.gz`
- `r2.nii.gz`
- `sigma_tia.nii.gz`
- `model_id.nii.gz`
- `status_id.nii.gz`
- optional `tpeak.nii.gz`
- `pytia_summary.yaml`

## 6) Status and model semantics

Stable status IDs:

- `0`: outside mask/background
- `1`: ok
- `2`: not applicable: `<2 valid points`
- `3`: fit failed
- `4`: all points below noise floor
- `5`: nonphysical parameters

Stable model IDs:

- `10`, `11`, `20`, `30` for multi-timepoint
- `101`, `102`, `103` for single-timepoint

Summary metadata includes:

- `pytia_version`
- `status_legend` and `status_counts`
- `model_legend` and `model_counts`
- `times_seconds`
- `voxel_volume_ml`
- `timing_ms` (when profiling enabled)

## 7) Local release and regression gates

Local/offline release flow is scriptable:

- `scripts/release_checklist.sh`
- optional strict mode: `scripts/release_checklist.sh --full`
- benchmarks: `scripts/benchmark_local.py`

See [`RELEASE.md`](RELEASE.md) for the current release policy.
