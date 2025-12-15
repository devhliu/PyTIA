# PyTIA — Final Design Document (v1)

## 0. Executive Summary
**PyTIA** is a Python package to compute **voxel-wise Time Integrated Activity (TIA)** maps (**Bq·s per voxel**) from multi-timepoint PET/SPECT activity density images (Bq/ml). It is designed for high noise robustness, supports **N ≥ 2** timepoints, embeds **physical decay constraints**, estimates **peak time** (when applicable), provides **uncertainty** via bootstrap, and supports both:

- **Voxel mode**: vectorized classification + fitting (no voxel Python loops in the core fit path)
- **Region mode**: organ/ROI label map drives a fixed kinetic class per region; fit a mean ROI TAC for shape, then **scale per voxel by voxel amplitude**.

Outputs are written as NIfTI images (nibabel), preserving affine/header alignment.

---

## 1. Scope and Goals

### 1.1 Goals (Requirements mapping)
The package addresses the stated requirements:

1. **Noise level**: denoise (masked Gaussian) + negative clamp + noise floor exclusion; robust modeling and bootstrap uncertainty.
2. **Support ≥2 timepoints**: fallback models exist for N=2; rising/hybrid and falling/exp supported.
3. **Multiple fitting methods + uncertainty**:
   - Gamma variate (linearized, matrix solution)
   - Mono-exponential tail model
   - Hybrid trapezoid + tail
   - Auto selection (voxel mode) or fixed class per region (region mode)
   - Bootstrap uncertainty (default)
4. **Physical decay embedded/constraint**:
   - Enforce clearance rates λ ≥ λ_phys
   - Physical tail extrapolation `A_last / λ_phys` where configured
5. **Model uptake + clearance and estimate peak time**:
   - Gamma model yields analytical `t_peak = α/β` (robust, between timepoints possible)
   - Falling model uses measured peak timepoint
6. **Matrix operations and inversions**:
   - Linearized gamma solution uses batched WLS normal equations (3×3 solves per voxel subset)
   - Exp tail uses batched masked linear regression
7. **Curve classification**:
   - Voxel mode: vectorized classifier (falling/rising/hump/ambiguous)
   - Region mode: label map defines classifications per region (fixed)
8. **Noise floor and reducing noise**:
   - optional denoise + per-voxel noise floor exclusion
9. **Robustness to noise, curve types, timepoint count**:
   - Hybrid fallback ensures stable output
   - Bootstrap provides uncertainty; invalid/insufficient cases are flagged
10. **I/O**:
   - Input: list of NIfTI paths or nibabel images + times
   - Output: TIA (Bq·s), R², sigma_TIA, model_id, status_id, optional tpeak in NIfTI
11. **User selectable methods**:
   - Configurable model selection mode (auto or region-defined)
   - Tail and rising behavior configurable

### 1.2 Non-goals (v1)
- Full compartment modeling
- Motion correction or registration between timepoints
- GPU acceleration
- Automatic organ segmentation (expects ROI labels provided if region mode used)

---

## 2. Definitions and Units

### 2.1 Inputs
- Activity images: **Bq/ml** (activity density)
- Times: seconds internally (`time.unit` supports hours → converted)

### 2.2 Output TIA units: **Bq·s per voxel**
We convert density to voxel activity using voxel volume:

- `V_voxel_ml = |det(affine[:3,:3])| / 1000` (mm³ → ml)
- `A_voxel(t)[Bq] = A_density(t)[Bq/ml] * V_voxel_ml[ml]`
- `TIA[Bq·s] = ∫ A_voxel(t) dt`

### 2.3 Negative values
- Negative activity values are clamped:
  - `A = max(A, 0)` **before** noise-floor exclusion

---

## 3. Data Flow and Pipeline

### 3.1 Pipeline steps
1. **Load** images (paths or nibabel objects)
2. **Validate** consistent 3D shape and affine; stack 4D `(X,Y,Z,T)`
3. **Sort** timepoints if enabled (to align times and images)
4. **Mask**:
   - default: Otsu on sum image (plus min fraction)
   - or provided mask
   - or none (whole volume)
5. **Denoise**:
   - masked Gaussian smoothing (prevents background bleeding)
6. **Clamp negative to zero**
7. **Convert** Bq/ml → Bq per voxel using voxel volume
8. **Noise floor exclusion**:
   - per-voxel floor (absolute or relative)
   - points below floor treated as missing for fitting
9. **Fit** (voxel mode or region mode)
10. **Uncertainty (bootstrap)**:
    - voxel mode: **reclassification per replicate**
    - region mode: refit per replicate but **fixed region classification**
11. **Write outputs** as NIfTI, preserve affine/header
12. **Write summary** YAML: legends + counts + timing

### 3.2 Chunking for large volumes
To limit RAM for large volumes, masked voxels are processed in chunks:

- `performance.chunk_size_vox` (default 500,000)
- Voxel mode fits classification and models chunk-by-chunk
- Bootstrap currently requires holding masked voxel TACs, but chunk-bootstrap is a planned improvement (v1.1)

### 3.3 Speed profiling
Timing (ms) is recorded per stage (load, denoise, fit, bootstrap, save) and written to summary YAML.

---

## 4. Masking and Denoising

### 4.1 Body mask (Otsu)
- `SumImage = Σ_t A(t)`
- Use Otsu threshold to separate body from air
- Apply additional constraint:
  - `SumImage >= min_fraction_of_max * max(SumImage)`

### 4.2 Masked Gaussian smoothing
Within-mask smoothing implemented as:

- `smooth = gaussian(A*mask)/gaussian(mask)`
- Prevents activity bleeding into background region

---

## 5. Noise Floor Exclusion

### 5.1 Floor definitions
- Absolute mode: `floor = absolute_bq_per_ml * voxel_volume_ml`
- Relative mode: `floor_i = frac * max_t A_i(t)` per voxel

### 5.2 Exclusion semantics
- If `A(t) < floor`: point is **excluded** from fitting (missing)
- If less than 2 valid points remain:
  - output is **NOT_APPLICABLE** (see status handling)

---

## 6. Status Handling ("Not applicable message")

NIfTI cannot store strings per voxel. PyTIA encodes not-applicable as:

- output maps store `NaN` for numeric results
- `status_id` map stores a reason code
- summary YAML contains `status_legend` mapping and voxel counts

### 6.1 Status codes
- `0`: OUTSIDE_MASK
- `1`: OK
- `2`: NOT_APPLICABLE_INSUFFICIENT_POINTS (<2 valid)
- `3`: FIT_FAILED
- `4`: ALL_BELOW_FLOOR
- `5`: NONPHYSICAL_PARAMS

---

## 7. Voxel Mode: Vectorized Classification + Fitting

### 7.1 Matrix view
Masked voxel TACs are reshaped into:

- `Y = A` with shape `(N_vox, N_time)`

### 7.2 Vectorized curve classification
Classification uses:
- peak index among valid points
- sign of discrete gradients (with relative epsilon)

Classes:
- Falling: peak at first valid time; decreasing overall
- Rising: peak at last valid time; increasing overall
- Hump: interior peak
- Ambiguous: fallback class

### 7.3 Models

#### Model A: Gamma-Variate (Hump; N ≥ 3)
Model:
- `A(t) = K * t^α * exp(-β t)`

Linearization:
- `ln(A) = ln(K) + α ln(t) - β t`

Vectorized WLS solve using normal equations:
- `G = Xᵀ W X` (3×3 per voxel)
- `b = Xᵀ W y`
- Solve `G θ = b` (batched solve)

Peak time:
- `t_peak = α/β` (analytic, robust between timepoints)

TIA:
- `TIA = ∫0∞ K t^α exp(-β t) dt = K Γ(α+1) / β^(α+1)`

Physical constraint:
- `β >= λ_phys` if enabled

#### Model B: Falling mono-exponential tail
Tail model:
- `ln(A) = a - λ t` after measured peak

Estimate λ via masked linear regression on tail points:
- clamp `λ_eff = max(λ_fit, λ_phys)` when enabled

TIA:
- Uptake approx (triangle): `0.5 * A_peak * t_peak`
- Tail: `A_peak / λ_eff`

#### Model C: Hybrid trapezoid + physical tail (rising + fallback)
Observed integral:
- trapezoid integration over valid points (optionally include `(0,0)`)

Physical tail:
- `A_last / λ_phys`

Rising curves:
- default: trapezoid + physical tail
- optional mode: assume peak at last (affects semantics, not formula)

### 7.4 Hybrid R²: piecewise prediction
For Hybrid, goodness-of-fit is computed using a **piecewise-linear prediction** at sampled times:
- predict `Ahat(t_i)` by linear interpolation between nearest valid samples (uses time values, not indices)
- compute R² on valid points

This is more meaningful than using raw measured values as the “prediction”.

---

## 8. Region Mode: Fixed Classification per Region + ROI Shape Fit + Voxel Amplitude Scaling

Region mode is enabled by providing a label map and region class definitions.

### 8.1 Key rule: region == one classification
Each region label is mapped to one fixed classification/model policy. There is **no** voxel-wise classification inside a region.

### 8.2 ROI-aggregated TAC (mean) per region
For region ℓ with voxel set Ω_ℓ:
- `A_region(t) = mean_{v ∈ Ω_ℓ}(A_voxel(v,t))`

Fit `A_region(t)` using the region’s fixed class/model (gamma/exp/hybrid).

### 8.3 Per-voxel amplitude scaling (required behavior)
The ROI fit provides a kinetic **shape**, each voxel provides amplitude.

Choose reference timepoint `t_ref`:
- default: peak of fitted `Ahat_region` (or last valid)

Let:
- `A_ref = Ahat_region(t_ref)`
- `TIA_ref = ∫ Ahat_region(t) dt` (Bq·s for reference amplitude)

For each voxel:
- `scale_v = A_voxel(v, t_ref) / A_ref` (requires voxel has valid sample at t_ref)
- `TIA_v = scale_v * TIA_ref`

If voxel missing at t_ref → NOT_APPLICABLE for that voxel.

### 8.4 Region voxel-level R² (optional)
Config:
- `regions.voxel_level_r2: true`

Then:
- `Ahat_vox(v,t) = scale_v * Ahat_region(t)`
- compute voxel-level R² against the voxel TAC on valid points

If disabled, broadcast region-level R².

---

## 9. Uncertainty Estimation (Bootstrap; default)

Bootstrap is the default uncertainty method.

### 9.1 Voxel mode bootstrap (reclassification enabled)
Each replicate:
1. compute baseline prediction `Ahat0`
2. residuals `r = A - Ahat0` on valid points
3. sample residuals along time axis → `r*`
4. synthetic `A* = clip(Ahat0 + r*, 0)`
5. reapply noise floor exclusion
6. **reclassify** curves (default true)
7. refit models and compute `TIA*`

Uncertainty:
- `sigma_TIA = std(TIA*)`

### 9.2 Region mode bootstrap
- classification is fixed by region definition
- resample region residuals, refit region curve per replicate → `TIA_ref*`
- voxel sigma is scaled:
  - `sigma_TIA_vox = scale_v * std(TIA_ref*)`

---

## 10. Input/Output

### 10.1 Inputs (Python API)
`run_tia(images, times, config, mask=None)` where:
- `images`: list of paths or nibabel images
- `times`: list of floats; seconds by default
- `config`: YAML path or dict

### 10.2 Outputs
NIfTI maps with same grid/affine/header as the inputs:
- `tia.nii.gz`: float32, Bq·s per voxel
- `r2.nii.gz`: float32
- `sigma_tia.nii.gz`: float32
- `model_id.nii.gz`: uint8
- `status_id.nii.gz`: uint8
- `tpeak.nii.gz`: float32 optional

Sidecar summary:
- `pytia_summary.yaml`: config snapshot, timing, status legend and counts

Filename prefix:
- if `io.prefix` is set, outputs are `{prefix}_tia.nii.gz`, etc.

---

## 11. Configuration Reference (Key)

### 11.1 Minimal config (Python API)
```yaml
physics:
  half_life_seconds: 23040
```

### 11.2 Minimal config (CLI requires inputs)
```yaml
inputs:
  images: ["tp1.nii.gz", "tp2.nii.gz", "tp3.nii.gz"]
  times: [3600, 14400, 86400]

physics:
  half_life_seconds: 23040
```

### 11.3 Region mode config example
```yaml
regions:
  enabled: true
  label_map_path: "./organs_labels.nii.gz"
  voxel_level_r2: true
  classes:
    "1": { class: "falling", allowed_models: ["exp"], default_model: "exp" }
    "2": { class: "hump", allowed_models: ["gamma"], default_model: "gamma" }
    "3": { class: "rising", allowed_models: ["hybrid"], default_model: "hybrid" }
```

---

## 12. Validation

### 12.1 Synthetic validation (unit tests)
- Gamma ground truth peak estimation
- Hybrid trapezoid + physical tail correctness
- negative clamp and noise floor exclusion
- region ROI scaling amplitude behavior
- bootstrap reproducibility by seed

### 12.2 Real-data validation script (ROI comparisons)
`scripts/validate_realdata_rois.py` produces:
- `roi_summary.csv`: per-label mean TIA, mean R², fraction OK
- optional `roi_tacs.csv`: long-form mean TAC per label
- optional ROI bar plot if matplotlib installed

---

## 13. Limitations / Planned Improvements
- Bootstrap in voxel mode currently materializes `A_all` for masked voxels, which can be memory-heavy for very large volumes. Planned: **chunk-bootstrap**.
- Gamma linearization is robust and fast but not as flexible as nonlinear constrained fitting; optional v2: nonlinear refinement using robust loss constraints.
- Hybrid model evaluation is piecewise linear at sampled times; adding a continuous hybrid prediction function is a future extension.

---

## 14. Package Structure
- `pytia/`: core package
- `pytia/models/`: kinetic models
- `pytia/docs/`: design documentation
- `scripts/`: validation utilities
- `tests/`: pytest suite

This design is finalized for v1 implementation.