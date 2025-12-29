"""Command-line interface for PyTIA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .engine import run_tia


def cmd_run(args: argparse.Namespace) -> int:
    """Run TIA estimation with config file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        return 1

    inp = cfg.get("inputs", {})
    images = inp.get("images")
    times = inp.get("times")
    mask = inp.get("mask")

    if not images or times is None:
        print("ERROR: Config must contain 'inputs.images' and 'inputs.times'")
        return 1

    try:
        print(f"Running TIA estimation...")
        result = run_tia(images=images, times=times, config=cfg, mask=mask)
        print("✓ Complete!")
        for key, path in result.output_paths.items():
            print(f"  {key}: {path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_nifti(args: argparse.Namespace) -> int:
    """Run TIA estimation directly from NIfTI files (no config needed)."""
    images = args.images
    times = args.times
    output_dir = Path(args.output_dir) if args.output_dir else Path("./pytia_output")
    prefix = args.prefix if args.prefix else "pytia"
    
    if not images:
        print("ERROR: Must provide at least one NIfTI image with --images")
        return 1
    
    if times is None:
        print("ERROR: Must provide timepoints with --times")
        return 1
    
    if len(images) != len(times):
        print(f"ERROR: Number of images ({len(images)}) must match number of times ({len(times)})")
        return 1
    
    for img_path in images:
        if not Path(img_path).exists():
            print(f"ERROR: Image not found: {img_path}")
            return 1
    
    config: dict[str, Any] = {
        "inputs": {
            "images": images,
            "times": times,
        },
        "time": {
            "unit": args.time_unit,
        },
        "io": {
            "output_dir": str(output_dir),
            "prefix": prefix,
        },
    }
    
    if args.half_life:
        config["physics"] = {"half_life_seconds": float(args.half_life)}
    
    if args.mask:
        config["mask"] = {"mode": "provided", "provided_path": args.mask}
    else:
        config["mask"] = {"mode": args.mask_mode}
    
    if args.no_denoise:
        config["denoise"] = {"enabled": False}
    
    if args.no_noise_floor:
        config["noise_floor"] = {"enabled": False}
    
    if args.no_bootstrap:
        config["bootstrap"] = {"enabled": False}
    elif args.bootstrap:
        config["bootstrap"] = {"enabled": True, "n": args.bootstrap, "seed": args.bootstrap_seed}
    
    if args.chunk_size:
        config["performance"] = {"chunk_size_vox": int(args.chunk_size)}
    
    if args.single_time:
        config["single_time"] = {"enabled": True, "method": args.stp_method}
        if args.stp_method == "haenscheid" and args.eff_half_life:
            config["single_time"]["haenscheid_eff_half_life_seconds"] = float(args.eff_half_life)
        elif args.stp_method == "prior_half_life":
            if args.prior_half_life:
                config["single_time"]["half_life_seconds"] = float(args.prior_half_life)
            if args.label_map:
                config["single_time"]["label_map_path"] = args.label_map
            if args.label_half_lives:
                config["single_time"]["label_half_lives"] = args.label_half_lives
    
    try:
        print(f"Running TIA estimation from NIfTI files...")
        result = run_tia(images=images, times=times, config=config)
        print("✓ Complete!")
        for key, path in result.output_paths.items():
            print(f"  {key}: {path}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate config file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        return 1

    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        Config.load(cfg)
        print("✓ Config valid")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show config file info."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return 1

    try:
        with config_path.open("r", encoding="utf-8") as f:
            print(f.read())
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pytia",
        description="PyTIA: Time-Integrated Activity from PET/SPECT",
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    run_p = subparsers.add_parser("run", help="Run TIA estimation with config file")
    run_p.add_argument("--config", required=True, type=Path, help="YAML config file")
    run_p.set_defaults(func=cmd_run)

    nifti_p = subparsers.add_parser(
        "nifti",
        help="Run TIA estimation directly from NIfTI files (no config needed)",
        epilog="""Examples:
  Multi-timepoint TIA (2+ images):
    pytia nifti --images img1.nii img2.nii img3.nii --times 1 24 48 --time-unit hours

  Single-timepoint TIA (1 image, physical decay):
    pytia nifti --images img1.nii --times 24 --single-time --stp-method phys

  Single-timepoint TIA (Hänscheid method):
    pytia nifti --images img1.nii --times 24 --single-time --stp-method haenscheid --eff-half-life 3600

  With mask and custom output:
    pytia nifti --images img*.nii --times 1 24 48 --mask mask.nii --output-dir ./results --prefix patient1

  With half-life and bootstrap:
    pytia nifti --images img*.nii --times 1 24 48 --half-life 86200 --bootstrap 100 --bootstrap-seed 42
"""
    )
    nifti_p.add_argument("--images", nargs="+", required=True, help="NIfTI image files (one or more)")
    nifti_p.add_argument("--times", nargs="+", type=float, required=True, help="Timepoints for each image")
    nifti_p.add_argument("--time-unit", default="hours", choices=["hours", "seconds"], help="Time unit (default: hours)")
    nifti_p.add_argument("--output-dir", default="./pytia_output", help="Output directory (default: ./pytia_output)")
    nifti_p.add_argument("--prefix", default="pytia", help="Output file prefix (default: pytia)")
    nifti_p.add_argument("--half-life", type=float, help="Radionuclide half-life in seconds")
    nifti_p.add_argument("--mask", help="Mask image file")
    nifti_p.add_argument("--mask-mode", default="none", choices=["none", "provided", "auto"], help="Mask mode (default: none)")
    nifti_p.add_argument("--no-denoise", action="store_true", help="Disable denoising")
    nifti_p.add_argument("--no-noise-floor", action="store_true", help="Disable noise floor filtering")
    nifti_p.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap")
    nifti_p.add_argument("--bootstrap", type=int, help="Enable bootstrap with N replicates")
    nifti_p.add_argument("--bootstrap-seed", type=int, default=42, help="Bootstrap random seed (default: 42)")
    nifti_p.add_argument("--chunk-size", type=int, help="Chunk size for voxel processing")
    nifti_p.add_argument("--single-time", action="store_true", help="Enable single-timepoint mode (for single image)")
    nifti_p.add_argument("--stp-method", default="phys", choices=["phys", "haenscheid", "prior_half_life"], help="Single-timepoint method (default: phys)")
    nifti_p.add_argument("--eff-half-life", type=float, help="Effective half-life for Hänscheid method (seconds)")
    nifti_p.add_argument("--prior-half-life", type=float, help="Prior half-life for prior_half_life method (seconds)")
    nifti_p.add_argument("--label-map", help="Label map for segmentation-based priors")
    nifti_p.add_argument("--label-half-lives", help="Label half-lives mapping (JSON format)")
    nifti_p.set_defaults(func=cmd_nifti)

    val_p = subparsers.add_parser("validate", help="Validate config")
    val_p.add_argument("--config", required=True, type=Path, help="YAML config file")
    val_p.set_defaults(func=cmd_validate)

    info_p = subparsers.add_parser("info", help="Show config info")
    info_p.add_argument("--config", required=True, type=Path, help="YAML config file")
    info_p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())