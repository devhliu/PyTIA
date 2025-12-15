# PyTIA — Design Document (v1, Final Updated After Review)

## 0. Delta From Previous “Final”
This revision incorporates your latest decisions:

1. **Bootstrap:** allow **reclassification per bootstrap replicate** (more realistic; slower).
2. **Valid points < 2:** return a **“not applicable” status/message**, rather than numeric TIA=0.
3. **Vectorized fittings should support region/voxel grouping:** fitting can be applied **per-voxel** or **per-region** (ROI/label-based), with potentially different model strategies per region.

---

## 1. Output “Not Applicable” Behavior (Valid Points < 2)

### 1.1 Constraint: NIfTI outputs must be numeric
NIfTI images cannot store per-voxel strings. Therefore the “not applicable message” will be expressed through:

- Numeric output maps contain:
  - `TIA = NaN`
  - `R² = NaN`
  - `sigma_TIA = NaN`
  - `tpeak = NaN`
- A **status map** and **metadata** provide the message:

#### Status map
`status_id.nii.gz` (uint8), same space as input:
- `0` = outside mask/background
- `1` = OK
- `2` = NOT_APPLICABLE_INSUFFICIENT_POINTS
- `3` = FIT_FAILED
- `4` = ALL_POINTS_BELOW_FLOOR
- `5` = NONPHYSICAL_PARAMS

#### Human-readable message
- In the returned `Results` object: `results.status_legend`
- In sidecar JSON
