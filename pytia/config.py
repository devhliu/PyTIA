from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_INPUTS_SCHEMA: dict[str, Any] = {
    "images": None,
    "times": None,
    "mask": None,
}

_REGION_CLASS_SCHEMA: dict[str, Any] = {
    "class": None,
    "default_model": None,
    "allowed_models": None,
}


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]

    @staticmethod
    def load(config: str | Path | dict[str, Any] | None) -> Config:
        if config is None:
            cfg = default_config()
            validate_config(cfg)
            return Config(cfg)
        if isinstance(config, dict):
            validate_user_config_keys(config)
            base = default_config()
            deep_update(base, config)
            validate_config(base)
            return Config(base)
        path = Path(config)
        with path.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        validate_user_config_keys(user)
        base = default_config()
        deep_update(base, user)
        validate_config(base)
        return Config(base)


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _require_choice(cfg: dict[str, Any], section: str, key: str, allowed: set[str]) -> None:
    value = cfg[section].get(key)
    if value not in allowed:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"Invalid {section}.{key}: {value!r}. Allowed: {allowed_str}")


def _require_positive(cfg: dict[str, Any], section: str, key: str) -> None:
    value = cfg[section].get(key)
    if value is None:
        return
    if float(value) <= 0:
        raise ValueError(f"{section}.{key} must be > 0, got {value!r}")


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping, got {type(value).__name__}.")
    return dict(value)


def _validate_unknown_keys(
    user_cfg: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    dynamic_map_paths: set[str] | None = None,
) -> None:
    dynamic_paths = dynamic_map_paths or set()
    unknown = sorted(set(user_cfg.keys()) - set(schema.keys()))
    if unknown:
        keys = ", ".join(unknown)
        raise ValueError(f"Unknown config key(s) under {path}: {keys}")

    for key, value in user_cfg.items():
        schema_val = schema[key]
        child_path = f"{path}.{key}"
        if isinstance(schema_val, dict):
            if child_path in dynamic_paths:
                continue
            child_cfg = _require_mapping(value, child_path)
            _validate_unknown_keys(child_cfg, schema_val, child_path, dynamic_paths)


def _require_int_like_keys(mapping: dict[str, Any], path: str) -> None:
    for raw_key in mapping:
        try:
            int(raw_key)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} keys must be integer-like labels, got {raw_key!r}.") from e


def _validate_regions_classes(classes_cfg: Any) -> None:
    classes = _require_mapping(classes_cfg, "config.regions.classes")
    _require_int_like_keys(classes, "config.regions.classes")

    for label, class_def in classes.items():
        class_path = f"config.regions.classes[{label!r}]"
        class_dict = _require_mapping(class_def, class_path)
        _validate_unknown_keys(class_dict, _REGION_CLASS_SCHEMA, class_path)


def _validate_label_half_lives(label_half_lives_cfg: Any) -> None:
    label_half_lives = _require_mapping(label_half_lives_cfg, "config.single_time.label_half_lives")
    _require_int_like_keys(label_half_lives, "config.single_time.label_half_lives")
    for label, half_life in label_half_lives.items():
        if float(half_life) <= 0:
            raise ValueError(
                f"single_time.label_half_lives[{label!r}] must be > 0, got {half_life!r}"
            )


def _validate_inputs_section(inputs_cfg: Any) -> None:
    inputs = _require_mapping(inputs_cfg, "config.inputs")
    _validate_unknown_keys(inputs, _INPUTS_SCHEMA, "config.inputs")

    if "images" in inputs and not isinstance(inputs["images"], list):
        raise ValueError("inputs.images must be a list of image paths.")
    if "times" in inputs and not isinstance(inputs["times"], list):
        raise ValueError("inputs.times must be a list of time values.")
    if "images" in inputs and "times" in inputs and len(inputs["images"]) != len(inputs["times"]):
        n_images = len(inputs["images"])
        n_times = len(inputs["times"])
        raise ValueError(
            "inputs.images and inputs.times must have the same length, "
            f"got {n_images} and {n_times}."
        )


def validate_user_config_keys(user_cfg: Any) -> None:
    user = _require_mapping(user_cfg, "config")

    schema = default_config()
    top_level_schema = dict(schema)
    top_level_schema["inputs"] = _INPUTS_SCHEMA

    _validate_unknown_keys(
        user,
        top_level_schema,
        "config",
        dynamic_map_paths={"config.regions.classes", "config.single_time.label_half_lives"},
    )

    if "inputs" in user:
        _validate_inputs_section(user["inputs"])
    if "regions" in user and "classes" in _require_mapping(user["regions"], "config.regions"):
        _validate_regions_classes(user["regions"]["classes"])
    if "single_time" in user and "label_half_lives" in _require_mapping(
        user["single_time"], "config.single_time"
    ):
        _validate_label_half_lives(user["single_time"]["label_half_lives"])


def validate_config(cfg: dict[str, Any]) -> None:
    _require_choice(cfg, "time", "unit", {"seconds", "hours"})
    _require_choice(cfg, "mask", "mode", {"provided", "otsu", "none"})
    _require_choice(cfg, "noise_floor", "mode", {"absolute", "relative"})
    _require_choice(cfg, "noise_floor", "behavior", {"exclude"})
    _require_choice(cfg, "model_selection", "mode", {"auto"})
    _require_choice(cfg, "integration", "tail_mode", {"phys", "none"})
    _require_choice(cfg, "integration", "rising_tail_mode", {"phys", "peak_at_last"})
    _require_choice(cfg, "regions", "mode", {"roi_aggregate"})
    _require_choice(cfg, "regions", "aggregation", {"mean"})
    scaling_cfg = cfg["regions"].get("scaling", {})
    scaling_mode = scaling_cfg.get("mode")
    if scaling_mode not in {"tref", "robust_ratio_mean"}:
        raise ValueError(
            f"Invalid regions.scaling.mode: {scaling_mode!r}. Allowed: robust_ratio_mean, tref"
        )
    _require_choice(
        cfg,
        "single_time",
        "method",
        {"phys", "haenscheid", "hanscheid", "prior_half_life", "prior"},
    )

    if cfg["mask"]["mode"] == "provided" and not cfg["mask"].get("provided_path"):
        raise ValueError("mask.provided_path is required when mask.mode='provided'")
    if cfg["regions"].get("enabled", False):
        if not cfg["regions"].get("label_map_path"):
            raise ValueError("regions.label_map_path is required when regions.enabled=true")
        if not cfg["regions"].get("classes"):
            raise ValueError("regions.classes is required when regions.enabled=true")
    if cfg["regions"].get("classes"):
        _validate_regions_classes(cfg["regions"]["classes"])

    _require_positive(cfg, "physics", "half_life_seconds")
    _require_positive(cfg, "single_time", "half_life_seconds")
    _require_positive(cfg, "single_time", "haenscheid_eff_half_life_seconds")
    if cfg["single_time"].get("label_half_lives"):
        _validate_label_half_lives(cfg["single_time"]["label_half_lives"])

    if int(cfg["bootstrap"].get("n", 0)) < 1:
        raise ValueError(f"bootstrap.n must be >= 1, got {cfg['bootstrap'].get('n')!r}")
    if int(cfg["performance"].get("chunk_size_vox", 0)) < 0:
        chunk_size = cfg["performance"].get("chunk_size_vox")
        raise ValueError(f"performance.chunk_size_vox must be >= 0, got {chunk_size!r}")
    if int(cfg["performance"].get("parallel_workers", 1)) < 1:
        parallel_workers = cfg["performance"].get("parallel_workers")
        raise ValueError(f"performance.parallel_workers must be >= 1, got {parallel_workers!r}")


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
            "low_memory_input": False,
            "parallel_workers": 1,
            "parallel_bootstrap": False,
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
            # Method: phys | haenscheid | prior_half_life
            "method": "phys",
            # For haenscheid method: effective half-life in human body (seconds)
            # If None, falls back to physics.half_life_seconds
            "haenscheid_eff_half_life_seconds": None,
            # For prior_half_life method: global half-life (seconds) or default for unmapped labels
            "half_life_seconds": None,
            # For prior_half_life method with label-map: path to label/segmentation image (NIfTI)
            "label_map_path": None,
            # For prior_half_life + label-map: dict mapping label -> half-life (seconds)
            # Example: {1: 1800.0, 2: 3600.0, 3: 5400.0}
            "label_half_lives": {},
        },
    }
