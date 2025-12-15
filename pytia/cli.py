from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .engine import run_tia


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pytia", description="PyTIA: voxel-wise TIA from multi-timepoint activity maps.")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Run TIA estimation")
    runp.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = p.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Expect config to provide input image paths and times
    inp = cfg.get("inputs", {})
    images = inp.get("images", None)
    times = inp.get("times", None)
    mask = inp.get("mask", None)
    if images is None or times is None:
        raise SystemExit("Config must contain inputs.images and inputs.times")

    run_tia(images=images, times=times, config=cfg, mask=mask)
    return 0