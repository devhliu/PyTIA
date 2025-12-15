# PyTIA — Design Document (Review Draft)

## 0. Goals & Non-Goals

### Goals
- Compute **voxel-wise Time Integrated Activity (TIA)** from **multi-timepoint activity maps** (PET/SPECT).
- Be robust to:
  - **noise levels** and **noise floor**
  - **small number of timepoints** (minimum **N ≥ 2**)
  - **different uptake/clearance curve shapes**
- Support **multiple fitting methods** with:
  - **auto (recommended) selection**
  - **uncertainty estimation** (parameter covariance and/or bootstrap/residual-based)
- Embed **physical decay** naturally or as a constraint.
- Input/Output:
  - Input: NIfTI paths or `nibabel.Nifti1Image`/`SpatialImage` objects + timepoints
  - Output: TIA map (Bq·s), R² map, uncertainty maps; same grid/affine as input.

### Non-Goals (v1)
- Organ/lesion segmentation tools (assume mask provided or auto-body mask).
- Full kinetic compartment modeling (beyond the specified parametric/semiparametric models).
- GPU acceleration (keep CPU + multiprocessing; allow later extension).

---

## 1. User-Facing API

### Primary entrypoints
1. **Python API**
```python
from pytia import run_tia

out = run_tia(
    images=[img1, img2, img3],          # or file paths
    times=[1*3600, 4*3600, 24*3600],     # seconds recommended
    config="config.yaml"                 # optional; dict also allowed
)
# out is a Results object with nibabel images + numpy arrays
```

2. **CLI**
```bash
pytia run --config config.yaml
```

### Accepted inputs
- A list of either:
  - NIfTI image file paths
  - `nibabel` image objects
- `times`: list/array of acquisition times (seconds; allow hours with config units)
- Optional: explicit mask NIfTI or generated automatically

### Outputs
- `tia.nii.gz`: voxel-wise TIA in **Bq·s**
- `r2.nii.gz`: goodness-of-fit (where applicable; otherwise NaN or computed from piecewise)
- `sigma_tia.nii.gz`: uncertainty (standard deviation) of TIA
- `model_id.nii.gz`: model classification per voxel (uint8 codes)
- Optional intermediate outputs (denoised images, mask, peak time map)

---

## 2. Configuration System (YAML)

### Config principles
- YAML is the single “source of truth” for algorithm choices.
- Allow override by Python dict.
- Validate with schema (Pydantic or jsonschema).

### Example `config.yaml`
```yaml
io:
  output_dir: "./out"
  save_intermediate: false
  dtype: "float32"

time:
  unit: "seconds"     # "hours" allowed; converted internally
  sort_timepoints: true

physics:
  half_life_seconds: 23040   # example: 6.4h
  enforce_lambda_ge_phys: true

mask:
  mode: "otsu"         # "provided" | "otsu" | "none"
  provided_path: null
  min_fraction_of_max: 0.02  # additional floor for mask after otsu

denoise:
  enabled: true
  method: "masked_gaussian"
  sigma_vox: 1.2
  preserve_zeros_outside_mask: true

noise_floor:
  enabled: true
  mode: "relative"     # "absolute" | "relative"
  absolute_bq_per_ml: 0.0
  relative_fraction_of_voxel_max: 0.01
  clamp_below_floor: true

model_selection:
  mode: "auto"         # "auto" | "gamma" | "exp" | "hybrid"
  min_points_for_gamma: 3
  classification:
    eps_relative: 0.02      # used to decide ties/flatness
    hump_requires_drop: 0.05

fitting:
  robust_loss: "soft_l1"    # for least_squares
  max_nfev: 2000
  bounds:
    alpha: [0.05, 10.0]
    beta:  [0.0,  10.0]     # further constrained by physics
  r2_min_valid: 0.0

uncertainty:
  enabled: true
  method: "covariance"   # "covariance" | "bootstrap"
  bootstrap:
    n: 50
    seed: 0

integration:
  start_time_seconds: 0
  tail_mode: "phys"      # "phys" | "fit" | "none"
```

---

## 3. High-Level Pipeline

1. **Load inputs** (paths or nibabel images)  
2. **Validate** dimensions/affines; stack to 4D array: `(X, Y, Z, T)`  
3. **Sort by time** if configured  
4. **Mask generation** (Otsu on sum-image; optional additional threshold)  
5. **Denoising** within mask (masked Gaussian)  
6. **Apply noise floor** (clamp or treat as missing)  
7. **Voxel-wise modeling**:
   - classify each TAC shape
   - select model based on config + classification + #points
   - fit parameters (if parametric)
   - compute TIA and per-voxel metrics
8. **Assemble maps** into nibabel images (preserve affine/header)
9. **Save outputs** + return `Results`

Parallelism: voxel loop parallelized (multiprocessing) with chunking; single-process fallback.

---

## 4. Models (Curve Types) & Classification

### Classification (per voxel)
Given `A(t)` across times `t`:
- `idx_max = argmax(A)`
- **Falling**: `idx_max == 0` and `A` decreases overall
- **Rising**: `idx_max == last` and mostly increasing
- **Hump**: interior peak
- **Flat/Noise**: max below noise floor or low dynamic range → return zeros/NaNs

Additional checks:
- require a minimum relative drop after peak to label “Hump”
- if too noisy/ambiguous → use Hybrid

### Model A — Gamma Variate (Hump, N ≥ 3)
Form:
- `A(t) = K * t^α * exp(-β t)` for `t > 0`
Constraints:
- `K > 0`, `α > 0`
- `β >= λ_phys` if enforcing physics

Outputs:
- `t_peak = α / β`
- `TIA = ∫0^∞ K t^α exp(-β t) dt = K * Γ(α+1) / β^(α+1)`

Fit method:
- `scipy.optimize.least_squares` or `curve_fit` with bounds
- robust loss optional (`soft_l1`)

### Model B — Constrained Mono-exponential Tail (Falling or fallback)
Tail:
- `A(t) = A0 * exp(-λ_eff * t)` (or shifted to peak time)
Constraint:
- `λ_eff = max(λ_fit, λ_phys)` if enabled

Uptake area:
- simple triangular approx from 0 to `t_peak_meas`:
  - `Area_uptake ≈ 0.5 * A(t_peak) * t_peak`
Tail area:
- `Area_tail = A(t_peak) / λ_eff`

Option: if `N >= 3` and there are ≥2 points after peak, estimate λ_fit by linear regression on `log(A)`.

Special case N=2:
- analytic λ from two points if decreasing; else Rising/Hybrid path.

### Model C — Hybrid (Trapezoid + tail)
Observed:
- trapezoid from `t0` to `t_last` (include start_time=0 with assumed 0)
Tail:
- either physical-only: `A_last / λ_phys`  
- or fitted from last two points if configured and stable

This is the most robust fallback when parametric fitting fails.

---

## 5. Uncertainty Estimation

### v1 approach (configurable)
1. **Covariance-based** (fast):
   - from Jacobian returned by least_squares/curve_fit
   - propagate param covariance to TIA via delta method (numerical gradient)

2. **Bootstrap** (more robust, slower):
   - residual bootstrap on TAC: `A_i* = A_fit(t_i) + resampled_residual`
   - refit `n` times
   - uncertainty = std(TIA_boot)

Returned uncertainty maps:
- `sigma_tia` (standard deviation)
Optional:
- `sigma_tpeak` for gamma model

Notes:
- For Hybrid/trapezoid with phys tail: uncertainty can be approximated from measurement noise model or set to NaN/high sentinel; v1: implement a conservative heuristic or bootstrap only.

---

## 6. Noise Handling Strategy

### Body mask
- Otsu threshold on sum image helps exclude air.
- Optional `min_fraction_of_max` avoids selecting extremely low background.

### Denoising
- masked Gaussian with configurable sigma in voxel units
- do not blur across mask boundary (multiply by mask, normalize by blurred mask)

### Noise floor
- clamp below floor to 0 or treat as missing for fitting
- floor can be relative to voxel maximum or absolute Bq/ml

---

## 7. Data Structures & Modules (Package Layout)

Proposed package structure:
- `pytia/`
  - `__init__.py` (exports `run_tia`, `Results`)
  - `config.py` (schema + YAML loader)
  - `io.py` (load images, save outputs, stack 4D)
  - `masking.py` (otsu + utilities)
  - `denoise.py` (masked Gaussian)
  - `models/`
    - `base.py` (interfaces)
    - `gamma_variate.py`
    - `monoexp.py`
    - `hybrid.py`
  - `classify.py` (curve classification)
  - `fit.py` (shared fitting helpers, robust least squares, bounds)
  - `integrate.py` (TIA integrals + tails)
  - `uncertainty.py` (covariance + bootstrap)
  - `engine.py` (voxel loop + parallelization)
  - `cli.py` (argparse/typer)
  - `types.py` (dataclasses)
  - `logging_utils.py`

### Key types
- `Results` dataclass:
  - `tia_img`, `r2_img`, `sigma_tia_img`, `model_id_img`, `tpeak_img?`
  - `meta` (times, config hash, half-life, etc.)
- `FitResult`:
  - params, success, r2, tia, sigma_tia, model_id, tpeak

---

## 8. Numerical & Implementation Details

### Units
- internal time = seconds
- activity unit assumed Bq/ml; package treats values as arbitrary activity density but outputs consistent TIA units (Bq·s/ml if density). In the design we will label output as Bq·s (user expectation); we can optionally store unit notes.

### R² definition
For parametric fits:
- `R² = 1 - SS_res/SS_tot` computed on used points (after floor/masking)
For hybrid:
- compute against piecewise predicted curve at sampled times or set NaN; config decides.

### Physical decay embedding
- `λ_phys = ln(2)/half_life_seconds`
- used as:
  - lower bound for beta or lambda
  - tail extrapolation constant in hybrid

### Robustness rules
- if fit fails / non-finite / negative params → fallback to Hybrid
- if fewer than required valid points after flooring → Hybrid or conservative tail

---

## 9. Performance Plan

- Use numpy arrays for data, avoid nibabel operations inside voxel loop.
- Parallelization:
  - iterate masked voxel indices
  - chunk indices per worker
  - each worker returns arrays (TIA, R², sigma, model_id) for chunk
- Memory considerations:
  - allow processing in Z-slabs to limit peak memory for large volumes.

---

## 10. Validation Plan (design-level)

### Synthetic tests
- Generate TACs with known parameters + Poisson-like noise:
  - Falling, Rising, Hump
  - compare recovered TIA vs analytical truth
- Evaluate sensitivity vs:
  - N=2,3,4+
  - injected noise levels
  - different half-life constraints

### Regression tests
- Ensures stable outputs for fixed seed bootstrap.

### Real-data sanity checks
- Organ ROIs: monotonicity and plausible half-life bounds
- Compare with trapezoid baseline.

---

## 11. Open Questions for Review

1. **Output units**: Do you want TIA per voxel as Bq·s *per ml* (density-integrated) or absolute per voxel (requires voxel volume)?  
   - If absolute, we should multiply by voxel volume (ml) from NIfTI header.
2. **Noise floor behavior**: clamp-to-zero vs exclude from fitting (missing). Which is preferred default?
3. **Rising curves**: Should we always do trapezoid + physical tail, or allow optional “assume peak at last time” tail?
4. **Uncertainty in Hybrid**: acceptable to output NaN/large constant, or do bootstrap only?

---

## 12. “Ready-to-Implement” Acceptance Criteria

The design is ready to implement once you confirm:
- default model selection logic (auto) matches expectations
- decision logic for N=2 vs N≥3 is acceptable
- output unit convention (density vs absolute) is decided
- uncertainty method default (covariance vs bootstrap) is chosen

-------------------------------------------------------------------------------------

make below update for current design in order to generate next version of design document:
1. if the organ & lesion segmentation are provided as input, this package can take good use of this segmentations for new and robust way of fitting [supposing the mean activity in a organ or a lesion has much high SNR, and its fitting may have better constraints for voxel wise fitting in organs or lesions.];
2. the timing after injection are stated in config varible or config files. if the config files used, the nitfi file paths are also stated in config file.
3. for the mask, there are 3 options: inputed mask, histogram based mask, otsu;
4. voxel wise based curve classification - voxels in one class should connected into scatter regions, instead of one voxel isolated;
5. matrix operation (such as inv) based fitting instead of single voxel curve fitting should be used;

confirmation for next steps:

1. **Output units**: TIA per voxel as Bq·s 
2. **Noise floor behavior**:exclude from fitting (missing). 
3. **Rising curves**: always do trapezoid + physical tail, and provide optional “assume peak at last time” tail
4. **Uncertainty in Hybrid**: acceptable to do bootstrap
5. default model selection logic (auto) matches expectations
6. decision logic for N=2 vs N≥3 is acceptable

then output this new design document for my review.


--------------------------------------------------------------------------------------
# PyTIA — Design Document (v1, Ready-to-Implement)

## 0. Confirmed Defaults (from review)

1. **Output units**: **TIA per voxel = Bq·s (absolute per voxel)**  
   - Implementation: convert activity density (Bq/ml) to absolute activity per voxel (Bq) using voxel volume (ml), then integrate over time (s).
   - `TIA_voxel [Bq·s] = ∫ A_density(t)[Bq/ml] * V_voxel[ml] dt`

2. **Noise floor behavior**: **exclude from fitting** (treat as missing)  
   - Points below floor are not used in regression/optimization.
   - If too few valid points remain → fallback to Hybrid or return zero depending on config.

3. **Rising curves**:
   - Default: **trapezoid + physical tail**
   - Option: **assume peak at last time** (alternative tail mode)

4. **Uncertainty in Hybrid**:
   - Default: **bootstrap uncertainty** for Hybrid (and optionally also for other models if enabled)

5. **Auto model selection** and **N=2 vs N≥3 logic**: accepted as designed.

---

## 1. Package Summary

**PyTIA** computes voxel-wise TIA maps from 3D activity maps acquired at multiple timepoints. It supports multiple kinetic models with an auto-selection decision tree, enforces physical decay constraints, reduces noise, estimates peak times when possible, and exports NIfTI maps for TIA, R², uncertainty, and model ID.

---

## 2. Inputs / Outputs

### Inputs
- Activity maps: list of either:
  - NIfTI file paths (`.nii`, `.nii.gz`), or
  - `nibabel` image objects (`nibabel.spatialimages.SpatialImage`)
- Acquisition times: list/array (seconds internally)
- YAML config path or config dict

Optional:
- Mask image (or auto-generated)
- Radionuclide half-life (required for phys tail/constraints)

### Outputs (NIfTI, same affine/grid/header as input)
- `tia.nii.gz` : TIA per voxel **Bq·s**
- `r2.nii.gz` : R² (NaN where undefined)
- `sigma_tia.nii.gz` : bootstrap std dev of TIA
- `model_id.nii.gz` : uint8 model codes (classification + final model used)
- Optional:
  - `tpeak.nii.gz` : estimated peak time (seconds; NaN if not applicable)
  - `denoised_*.nii.gz`, `mask.nii.gz`

---

## 3. Configuration (YAML) — Key Fields

```yaml
io:
  output_dir: "./out"
  save_intermediate: false
  dtype: "float32"

time:
  unit: "seconds"
  sort_timepoints: true

physics:
  half_life_seconds: 23040
  enforce_lambda_ge_phys: true

mask:
  mode: "otsu"              # provided | otsu | none
  provided_path: null
  min_fraction_of_max: 0.02

denoise:
  enabled: true
  method: "masked_gaussian"
  sigma_vox: 1.2

noise_floor:
  enabled: true
  mode: "relative"          # absolute | relative
  absolute_bq_per_ml: 0.0
  relative_fraction_of_voxel_max: 0.01
  behavior: "exclude"       # exclude (missing)

model_selection:
  mode: "auto"              # auto | gamma | exp | hybrid
  min_points_for_gamma: 3

integration:
  start_time_seconds: 0
  rising_tail_mode: "phys"  # phys (default) | peak_at_last
  tail_mode: "phys"         # phys | fit (for non-rising if enabled)

uncertainty:
  enabled: true
  method: "bootstrap"       # bootstrap as default for v1
  bootstrap:
    n: 50
    seed: 0
```

---

## 4. Core Pipeline

1. Load images (paths or nibabel objects)
2. Validate consistent shape + affine; stack to 4D `(X, Y, Z, T)`
3. Convert times into seconds; sort by time if enabled
4. Compute voxel volume from header affine: `V_voxel_ml`
5. Mask generation (Otsu on sum-image, optionally combined with min fraction)
6. Denoising (masked Gaussian, no bleed outside mask)
7. Noise floor exclusion: for each voxel TAC, mark points below floor as invalid
8. For each voxel: classify TAC and select model (auto or user-forced)
9. Fit model (if parametric), integrate to TIA (Bq·s), compute R², estimate uncertainty
10. Assemble full volumes, export NIfTI outputs

Parallelization: multiprocessing over masked voxel indices (chunked).

---

## 5. Models & Auto Selection

### TAC Classification
Given valid points `(t_i, A_i)` after noise-floor exclusion:
- If <2 valid points → return `TIA=0`, metrics NaN (or config option)
- `idx_max = argmax(A_i)`
- **Falling**: max at first valid timepoint
- **Rising**: max at last valid timepoint
- **Hump**: interior max
- Ambiguous/noisy → Hybrid

### Model A — Gamma Variate (Hump, N≥3)
`A(t) = K * t^α * exp(-β t)`
- constraints: `K>0`, `α>0`, `β>=λ_phys` (if enabled)
- peak: `t_peak = α/β`
- TIA (absolute per voxel):
  - integrate density to ∞ analytically: `∫0∞ K t^α exp(-β t) dt`
  - then multiply by voxel volume ml to get Bq·s (or fit directly in Bq)

### Model B — Mono-exponential Tail (Falling or fallback)
- Determine peak time as measured peak (`t_peak_meas`)
- Tail: `A(t)=A_peak * exp(-λ_eff (t - t_peak))`
- λ bounds: `λ_eff >= λ_phys` if enabled
- Uptake area: triangle approx `0.5 * A_peak * t_peak`
- Tail area: `A_peak / λ_eff`
- Sum and multiply by voxel volume → Bq·s

### Model C — Hybrid (Trapezoid + Tail) — Default fallback, and for Rising
Observed:
- trapezoid integration on valid points, include `(0,0)` if `start_time_seconds=0`
Tail:
- Default tail is **physical**: `A_last / λ_phys`
- For Rising curves:
  - default `rising_tail_mode=phys`
  - optional `peak_at_last`: treat last point as peak for tail (same formula but semantics differ; still phys decay rate)

---

## 6. Uncertainty (v1 default = Bootstrap)

Bootstrap approach (per voxel):
- Fit the chosen model (or evaluate piecewise curve for Hybrid)
- Compute residuals at valid timepoints
- Resample residuals and regenerate pseudo-observations
- Recompute TIA `n` times
- `sigma_tia = std(TIA_boot)`
- Outputs NaN if bootstrap cannot run due to insufficient valid points

Notes:
- Bootstrap is used for Hybrid as requested.
- We will ensure reproducibility via seed control.

---

## 7. R² / Goodness of Fit

- For parametric models: R² computed on valid points used in fitting.
- For Hybrid:
  - We can compute R² by evaluating the hybrid predicted values at sampled times (trapezoid piecewise linear + exponential tail if needed only for evaluation).
  - If too few points, R² = NaN.

---

## 8. Model Codes (example)

- 0: background/outside mask
- 1: invalid/insufficient points
- 10: Hybrid (rising)
- 11: Hybrid (general)
- 20: Monoexp (falling)
- 30: Gamma variate (hump)
- 40: Forced user model (optional flags in high bits)

(Exact mapping will be defined in `pytia/models/base.py`.)

---

## 9. Proposed Package Structure (to implement next)

- `pytia/`
  - `__init__.py`
  - `engine.py` (run_tia orchestration)
  - `config.py` (YAML loader + schema)
  - `io.py` (nibabel load/save, stacking, voxel volume)
  - `masking.py` (otsu + provided mask)
  - `denoise.py` (masked Gaussian)
  - `noise.py` (noise floor exclusion utilities)
  - `classify.py`
  - `models/`
    - `base.py`
    - `gamma_variate.py`
    - `monoexp.py`
    - `hybrid.py`
  - `uncertainty.py` (bootstrap)
  - `metrics.py` (R²)
  - `cli.py`
- `tests/` (pytest)
- `docs/` (mkdocs or sphinx, v1 basic usage + config reference)

---

## 10. Implementation Acceptance Criteria (v1)

- Loads NIfTI / nibabel images reliably; preserves affine/header.
- Produces correct units **Bq·s per voxel** using voxel volume conversion.
- Auto model selection works on synthetic falling/rising/hump cases.
- Physical tail and λ constraints applied correctly.
- Bootstrap uncertainty map produced and reproducible with seed.
- CLI + Python API functional.
- Unit tests cover:
  - units conversion
  - each model on synthetic TACs
  - noise floor exclusion behavior
  - rising curve handling
  - bootstrap reproducibility

---

## 11. Remaining Confirmations (small)
1. Should we treat **negative activity values** (can occur after reconstruction) as:
   - clamp to 0 before noise-floor exclusion (recommended), or
   - allow but exclude by floor, or
   - error?
2. Do you want output filenames fixed (`tia.nii.gz`, etc.) or configurable prefix?

(If you confirm these, we can start building the package.)

--------------------------------------------------------------------------------------
confirmation:
- clamp to 0 before noise-floor exclusion for negative activity values;
- configurable prefix for output filenames. if only output directory is provided, use default filename .

please consider below into the new design document:
-----
This is the final design and implementation. It fulfills the requirement for **Matrix Operations** (no pixel-loops), **Robust Peak Estimation**, and **Curve Classification**.

### 1. Design Concept: The Vectorized Kinetic Classifier

To avoid iterating over voxels, we treat the 4D image as a matrix $Y$ of shape $(N_{voxels}, N_{time})$. We solve the kinetics using **Linear Algebra Projections**.

#### 1.1 Robust Peak Time Estimation
Finding the exact physiological peak between sampled timepoints is crucial for accurate uptake integration. We use a **Linearized Gamma Variate** model.

*   **The Model:** $A(t) = K \cdot t^\alpha \cdot e^{-\beta t}$
*   **Linearization:** $\ln(A) = \ln(K) + \alpha \ln(t) - \beta t$
*   **The Matrix Equation:** $Y = X \cdot \theta$
    *   $Y$: Log-activity vector.
    *   $X$: Design matrix with columns $[1, \ln(t), -t]$.
    *   $\theta$: Parameters $[ \ln(K), \alpha, \beta ]^T$.
*   **Peak Estimation:** Once we solve for $\alpha$ and $\beta$ via matrix operations, the peak time is analytically:
    $$ T_{peak} = \frac{\alpha}{\beta} $$
*   **Why this is robust:** It uses *all* data points to determine the curvature, rather than just picking the highest measured pixel.

#### 1.2 Algorithm Workflow (The "Tensor Flow")

1.  **Denoise:** Spatially smooth data to ensure voxel time-curves are continuous.
2.  **Classify (Vectorized):**
    *   Compute Discrete Gradients.
    *   **Class 1 (Falling):** Peak is at $T_1$. (Kidney/Blood).
    *   **Class 2 (Rising):** Peak is at $T_{last}$. (Tumor Retention).
    *   **Class 3 (Hump):** Peak is in between. (Tumor/Liver).
3.  **Phase 1: Peak & Uptake Optimization:**
    *   **For "Humps" ($N \ge 3$):** Apply the **Pseudo-Inverse Projection** of the Time Matrix to the Data Matrix to solve the Gamma parameters instantly for millions of voxels. Calculate exact $T_{peak}$.
    *   **For Others:** Use measured timepoints (fallback).
4.  **Phase 2: Clearance & Integration:**
    *   Apply constrained exponential integration based on the estimated peak.
----