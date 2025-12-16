from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]

    @staticmethod
    def load(config: str | Path | dict[str, Any] | None) -> "Config":
        if config is None:
            return Config(default_config())
        if isinstance(config, dict):
            base = default_config()
            deep_update(base, config)
            return Config(base)
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        base = default_config()
        deep_update(base, user)
        return Config(base)


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def default_config() -> dict[str, Any]:
    return {
        "io": {
            "output_dir": "./out",
            "prefix": None,
            "save_intermediate": False,
            "dtype": "float32",
            "write_summary_yaml": True,
            "write_status_map": True,
        },
        "time": {"unit": "seconds", "sort_timepoints": True},
        "physics": {
            "half_life_seconds": None,  # required for phys tail or constraints
            "enforce_lambda_ge_phys": True,
        },
        "mask": {
            "mode": "otsu",  # provided | otsu | none
            "provided_path": None,
            "min_fraction_of_max": 0.02,
        },
        "denoise": {"enabled": True, "method": "masked_gaussian", "sigma_vox": 1.2},
        "noise_floor": {
            "enabled": True,
            "mode": "relative",  # absolute | relative
            "absolute_bq_per_ml": 0.0,
            "relative_fraction_of_voxel_max": 0.01,
            "behavior": "exclude",  # exclude only
        },
        "model_selection": {"mode": "auto", "min_points_for_gamma": 3},
        "integration": {
            "start_time_seconds": 0.0,
            "tail_mode": "phys",  # phys | none
            "rising_tail_mode": "phys",  # phys | peak_at_last
        },
        "bootstrap": {
            "enabled": True,
            "n": 50,
            "seed": 0,
            "reclassify_each_replicate": True,  # voxel-mode behavior
        },
        "performance": {
            "chunk_size_vox": 500_000,
            "enable_profiling": False,
        },
        "regions": {
            "enabled": False,
            "label_map_path": None,
            "mode": "roi_aggregate",  # roi_aggregate (v1 primary)
            "aggregation": "mean",
            "voxel_level_r2": False,  # NEW: voxel-level R2 in region ROI mode
            # each label mapped to one fixed class/model
            "classes": {},
            "scaling": {
                "mode": "tref",  # tref | robust_ratio_mean
                "reference_time": "peak",  # peak | last | index:<int>
            },
        },
        "single_time": {
            # Single-timepoint TIA calculation (when T=1 image provided)
            # Computes TIA = A(t) / λ_eff using one of three methods:
            "enabled": False,
            # Method: phys (physical decay) | haenscheid (effective HL) | prior_half_life (segmentation prior)
            "method": "phys",
            # For haenscheid method: effective half-life in human body (seconds)
            # If None, falls back to physics.half_life_seconds
            "haenscheid_eff_half_life_seconds": None,
            # For prior_half_life method: global half-life (seconds) or default for unmapped labels
            "half_life_seconds": None,
            # For prior_half_life method with label-map: path to label/segmentation image (NIfTI)
            "label_map_path": None,
            # For prior_half_life method with label-map: dict mapping label (int) -> half_life (seconds)
            # Example: {1: 1800.0, 2: 3600.0, 3: 5400.0}
            "label_half_lives": {},
        },
    }