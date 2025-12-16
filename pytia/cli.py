"""Command-line interface for PyTIA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    run_p = subparsers.add_parser("run", help="Run TIA estimation")
    run_p.add_argument("--config", required=True, type=Path, help="YAML config file")
    run_p.set_defaults(func=cmd_run)

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