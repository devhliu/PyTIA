# Lu-177 PSMA Phantom Simulation

This example demonstrates the use of a realistic voxel phantom for simulating Lu-177 PSMA therapy imaging and dosimetry.

## Overview

The phantom simulates a clinically relevant scenario for PSMA-targeted radioligand therapy with Lu-177, including:
- Multi-timepoint PET acquisitions at therapeutic time points
- Realistic organ pharmacokinetics based on published data
- Dose-limiting organs (kidneys, red marrow) and target lesions

## Phantom Anatomy

The voxel phantom (100×120×140 voxels) includes:
- **Body**: Elliptical outline
- **Liver**: Upper right quadrant
- **Kidneys**: Left and right, typical anatomical positions
- **Red Marrow**: Simplified as spine and pelvic bones
- **Tumors**: 3 spherical lesions at various locations

## Pharmacokinetic Model

Organ-specific time-activity curves are modeled using bi-exponential functions:

### Tumors
- High uptake (A₀ = 8.5 MBq/ml)
- Fast + slow components (40/60% split)
- λ_fast = 0.15 h⁻¹, λ_slow = 0.03 h⁻¹

### Kidneys
- Highest uptake (A₀ = 12.0 MBq/ml)
- Rapid initial washout (60% fast component)
- λ_fast = 0.25 h⁻¹, λ_slow = 0.04 h⁻¹

### Liver
- Moderate uptake (A₀ = 4.0 MBq/ml)
- Slow washout with physical decay
- λ_eff = 0.02 h⁻¹ + λ_phys(Lu-177)

### Red Marrow
- Low uptake (A₀ = 1.5 MBq/ml)
- Blood-pool kinetics with slight retention
- λ_fast = 0.30 h⁻¹, λ_slow = 0.05 h⁻¹

## Usage

### Running the Test

```bash
# Run all phantom tests
pytest tests/test_lu177_psma_phantom.py -v

# Run specific test
pytest tests/test_lu177_psma_phantom.py::test_lu177_psma_phantom_tia_calculation -v
```

### Running the Demo

```bash
python examples/lu177_psma_phantom_demo.py
```

The demo will:
1. Create the voxel phantom with organ masks
2. Simulate PET images at 4, 24, 48, 96, 112, and 176 hours
3. Calculate time-integrated activity (TIA) maps
4. Report mean TIA values for each organ
5. Show model fit quality (R²)

## Expected Results

Typical TIA values (MBq·h/ml):
- Tumors: ~750-800
- Kidneys: ~750 (dose-limiting organ)
- Red Marrow: ~650 (dose-limiting organ)
- Liver: ~600

Model fits should be excellent with R² > 0.95 for tumors.

## Clinical Relevance

This phantom simulates:
- Post-therapy dosimetry imaging schedule
- Key organs for PSMA therapy dose assessment
- Realistic activity distributions with Poisson noise
- Multi-exponential clearance patterns

It provides a testbed for validating TIA calculation algorithms in a clinically relevant context.

## Physics Parameters

- **Radionuclide**: Lu-177
- **Half-life**: 160.73 hours (6.647 days)
- **Energy**: β⁻ emission (max 498 keV), γ emission (113 keV & 208 keV)
- **Clinical use**: PSMA-617 or PSMA-I&T therapy