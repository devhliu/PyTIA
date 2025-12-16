#!/usr/bin/env python
"""
Quick demo/reference script showing single-timepoint TIA calculation.
Demonstrates all three methods with synthetic data.

NOT meant to be run as a test, but as a reference for understanding
the STP calculations.
"""

import numpy as np
from pathlib import Path
import tempfile

# Simulate key calculations without requiring nibabel/pytia installation

def demo_stp_calculations():
    """Demonstrate the three single-timepoint TIA methods."""
    
    print("=" * 70)
    print("SINGLE-TIMEPOINT TIA CALCULATION DEMO")
    print("=" * 70)
    
    # Common parameters
    activity_value = 100.0  # Bq/ml
    print(f"\nBase Activity: {activity_value} Bq/ml")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: PHYSICAL DECAY")
    print("=" * 70)
    print("TIA = A(t) / λ_phys")
    print("λ_phys = ln(2) / half_life_physical")
    print("\nUse Case: Decay based on radionuclide physical half-life")
    print("Config: physics.half_life_seconds")
    
    half_life_phys = 3600.0  # seconds (e.g., Tc-99m ≈ 6 hours)
    lambda_phys = np.log(2.0) / half_life_phys
    tia_phys = activity_value / lambda_phys
    
    print(f"\nExample: Tc-99m (half-life ≈ 6 hours = 21600 s)")
    print(f"  λ_phys = ln(2) / {half_life_phys} = {lambda_phys:.6f} s^-1")
    print(f"  TIA = {activity_value} / {lambda_phys:.6f} = {tia_phys:.2f} Bq·s/ml")
    print(f"  Model ID: 101")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: HÄNSCHEID METHOD (Effective Half-Life)")
    print("=" * 70)
    print("TIA = A(t) / λ_eff")
    print("λ_eff = ln(2) / half_life_effective")
    print("\nUse Case: Account for tracer clearance from human body")
    print("Config: single_time.haenscheid_eff_half_life_seconds")
    print("Fallback: physics.half_life_seconds")
    
    eff_half_life = 7200.0  # seconds (2 hours effective in body)
    lambda_eff = np.log(2.0) / eff_half_life
    tia_haenscheid = activity_value / lambda_eff
    
    print(f"\nExample: F-18 FDG effective clearance (2 hours)")
    print(f"  λ_eff = ln(2) / {eff_half_life} = {lambda_eff:.6f} s^-1")
    print(f"  TIA = {activity_value} / {lambda_eff:.6f} = {tia_haenscheid:.2f} Bq·s/ml")
    print(f"  Model ID: 102")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 3a: PRIOR HALF-LIFE (GLOBAL)")
    print("=" * 70)
    print("TIA = A(t) / λ_prior")
    print("λ_prior = ln(2) / half_life_prior")
    print("\nUse Case: Single global half-life for entire volume")
    print("Config: single_time.half_life_seconds")
    
    prior_half_life = 5400.0  # seconds (1.5 hours)
    lambda_prior = np.log(2.0) / prior_half_life
    tia_prior_global = activity_value / lambda_prior
    
    print(f"\nExample: Unified prior (1.5 hours)")
    print(f"  λ_prior = ln(2) / {prior_half_life} = {lambda_prior:.6f} s^-1")
    print(f"  TIA = {activity_value} / {lambda_prior:.6f} = {tia_prior_global:.2f} Bq·s/ml")
    print(f"  Model ID: 103")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("METHOD 3b: PRIOR HALF-LIFE (SEGMENTATION-BASED)")
    print("=" * 70)
    print("For each voxel v with label L:")
    print("  TIA_v = A_v / λ_L")
    print("  λ_L = ln(2) / half_life_L")
    print("\nUse Case: Organ/lesion-specific half-lives from segmentation")
    print("Config: single_time.label_map_path + single_time.label_half_lives")
    
    # Label mapping example
    label_mapping = {
        1: (1800.0, "Tumor/Lesion"),
        2: (3600.0, "Liver"),
        3: (5400.0, "Kidney"),
    }
    
    print(f"\nExample Segmentation Mapping:")
    print(f"  Label 1 (Tumor)  → {label_mapping[1][0]} s = {label_mapping[1][0]/60:.1f} min")
    print(f"  Label 2 (Liver)  → {label_mapping[2][0]} s = {label_mapping[2][0]/60:.1f} min")
    print(f"  Label 3 (Kidney) → {label_mapping[3][0]} s = {label_mapping[3][0]/60:.1f} min")
    
    print(f"\nPer-voxel TIA calculation:")
    for label, (hl, name) in label_mapping.items():
        lam = np.log(2.0) / hl
        tia_seg = activity_value / lam
        print(f"  Label {label} ({name:12s}): TIA = {tia_seg:8.2f} Bq·s/ml " +
              f"(λ = {lam:.6f} s^-1, t_half = {hl/60:.1f} min)")
    
    print(f"  Model ID: 103 (same for all labels, but different TIA per voxel)")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    
    methods = [
        ("Physical Decay", tia_phys, 101),
        ("Hänscheid (Eff.)", tia_haenscheid, 102),
        ("Prior (Global)", tia_prior_global, 103),
    ]
    
    print(f"\n{'Method':<20} {'TIA (Bq·s/ml)':<20} {'Model ID':<10}")
    print("-" * 50)
    for method, tia, model_id in methods:
        print(f"{method:<20} {tia:<20.2f} {model_id:<10}")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("STATUS CODES")
    print("=" * 70)
    
    status_codes = {
        0: "STATUS_OUTSIDE - Outside mask/background",
        1: "STATUS_OK - Valid TIA computed",
        2: "STATUS_NOT_APPLICABLE_INSUFFICIENT_POINTS - Invalid λ or activity",
        3: "STATUS_FIT_FAILED - Missing required parameters",
        4: "STATUS_ALL_BELOW_FLOOR - Activity below noise floor",
    }
    
    for code, desc in status_codes.items():
        print(f"  {code}: {desc}")
    
    # =========================================================================
    print("\n" + "=" * 70)
    print("YAML CONFIGURATION EXAMPLES")
    print("=" * 70)
    
    yaml_examples = {
        "phys": """
single_time:
  enabled: true
  method: phys
physics:
  half_life_seconds: 21600.0  # Tc-99m: 6 hours
""",
        "haenscheid": """
single_time:
  enabled: true
  method: haenscheid
  haenscheid_eff_half_life_seconds: 7200.0  # Effective: 2 hours
physics:
  half_life_seconds: 21600.0  # Physical (unused in this method)
""",
        "prior_global": """
single_time:
  enabled: true
  method: prior_half_life
  half_life_seconds: 5400.0  # 1.5 hours
""",
        "prior_segmentation": """
single_time:
  enabled: true
  method: prior_half_life
  label_map_path: segmentation.nii.gz
  label_half_lives:
    1: 1800.0   # Lesion: 30 min
    2: 3600.0   # Liver: 60 min
    3: 5400.0   # Kidney: 90 min
  half_life_seconds: 3600.0  # Default
""",
    }
    
    for name, yaml_config in yaml_examples.items():
        print(f"\n{name.upper()}:")
        print(yaml_config)
    
    print("\n" + "=" * 70)
    print("END OF DEMO")
    print("=" * 70)


if __name__ == "__main__":
    demo_stp_calculations()
