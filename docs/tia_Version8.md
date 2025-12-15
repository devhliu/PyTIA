## 3.2 ROI-aggregated fit (Mean TAC) + Voxel Amplitude Scaling (Final)

If `regions.mode: roi_aggregate`, PyTIA fits one kinetic *shape* per region using the **mean** TAC, then **scales per voxel by each voxel’s activity magnitude**.

### Step A — Build region TAC (mean)
All computations are in **Bq per voxel** time-curves (activity density multiplied by voxel volume early).

For region `ℓ` with voxel set `Ω_ℓ`:
- `A_region(t_j) = mean_{v ∈ Ω_ℓ}(A_voxel(v, t_j))`

### Step B — Fit region kinetics to obtain a shape function
Fit `A_region(t)` using the region’s fixed class/model (falling/rising/hump → exp/hybrid/gamma).

Represent the fitted region curve as:
- `Â_region(t)`

Define a **normalized shape** `s_ℓ(t)` by choosing a reference timepoint `t_ref`:
- `A_ref = Â_region(t_ref)`
- `s_ℓ(t) = Â_region(t) / A_ref`  (dimensionless; `s_ℓ(t_ref)=1`)

Then compute the region shape-integral:
- `TIA_shape_ℓ = ∫ s_ℓ(t) dt`  (units: seconds)

And the region reference TIA (for amplitude = `A_ref`):
- `TIA_ref_ℓ = A_ref * TIA_shape_ℓ`  (units: Bq·s)

### Step C — Scale per voxel by activity magnitude
For each voxel `v ∈ Ω_ℓ`, estimate an amplitude scaling factor:
- Default (simple): `scale_v = A_voxel(v, t_ref) / A_ref`
- Robust option (configurable): `scale_v = mean_j (A_voxel(v, t_j) / Â_region(t_j))` over valid points

Then voxel curve and TIA:
- `Â_v(t) = scale_v * Â_region(t)`
- `TIA_v = scale_v * TIA_ref_ℓ`  (Bq·s per voxel)

### Reference timepoint selection
`t_ref` default:
- the timepoint with **maximum** `Â_region(t)` (region peak), if defined and stable
- otherwise the last valid timepoint

### Output maps in ROI-aggregate mode
- `tpeak` is constant within the region (from region fit) and broadcast.
- `TIA` varies voxel-wise via `scale_v`.
- `R²` can be reported as:
  - region-level goodness of fit (broadcast), and/or
  - voxel-level R² comparing `A_voxel(v,t)` against `Â_v(t)` (optional; compute cost higher).

### Edge cases
- If `A_ref == 0` or insufficient valid points: mark region voxels as NOT_APPLICABLE (status map), outputs NaN.