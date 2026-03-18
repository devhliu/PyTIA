#!/usr/bin/env python3
"""Local benchmark scenarios for PyTIA (offline, reproducible)."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pytia.engine import run_tia


def _save_nifti(path: Path, data: np.ndarray) -> Path:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)
    return path


def _make_synthetic_dataset(workdir: Path, size: int) -> tuple[list[Path], Path, list[float]]:
    rng = np.random.default_rng(7)
    shape = (size, size, size)
    x, y, z = np.indices(shape, dtype=np.float32)
    cx = (size - 1) / 2.0
    radius = size / 4.0
    radial = ((x - cx) ** 2 + (y - cx) ** 2 + (z - cx) ** 2) / (radius**2)
    base = np.exp(-radial) * 120.0

    # Hump-like TAC with slight noise to exercise model fitting and classification.
    factors = [0.65, 1.35, 0.95]
    times_s = [3600.0, 10800.0, 21600.0]
    image_paths: list[Path] = []
    for i, factor in enumerate(factors):
        noise = rng.normal(0.0, 1.5, size=shape).astype(np.float32)
        data = np.maximum(base * factor + noise, 0.0).astype(np.float32)
        image_paths.append(_save_nifti(workdir / f"tp_{i}.nii.gz", data))

    labels = np.zeros(shape, dtype=np.int16)
    labels[radial <= 1.0] = 1
    label_path = _save_nifti(workdir / "labels.nii.gz", labels)
    return image_paths, label_path, times_s


def _scenario_configs(
    output_root: Path,
    label_path: Path,
    bootstrap_n: int,
) -> list[tuple[str, dict[str, Any]]]:
    common = {
        "time": {"unit": "seconds", "sort_timepoints": True},
        "physics": {"half_life_seconds": 6.647 * 24.0 * 3600.0},
        "mask": {"mode": "none"},
        "denoise": {"enabled": False},
        "noise_floor": {"enabled": False},
    }

    scenarios = [
        (
            "voxel_non_chunked",
            {
                **common,
                "io": {"output_dir": str(output_root / "voxel_non_chunked")},
                "bootstrap": {"enabled": False},
                "performance": {"chunk_size_vox": 0, "enable_profiling": True},
            },
        ),
        (
            "voxel_chunked",
            {
                **common,
                "io": {"output_dir": str(output_root / "voxel_chunked")},
                "bootstrap": {"enabled": False},
                "performance": {"chunk_size_vox": 5000, "enable_profiling": True},
            },
        ),
        (
            "voxel_chunked_bootstrap",
            {
                **common,
                "io": {"output_dir": str(output_root / "voxel_chunked_bootstrap")},
                "bootstrap": {
                    "enabled": True,
                    "n": bootstrap_n,
                    "seed": 17,
                    "reclassify_each_replicate": True,
                },
                "performance": {"chunk_size_vox": 5000, "enable_profiling": True},
            },
        ),
        (
            "region_mode",
            {
                **common,
                "io": {"output_dir": str(output_root / "region_mode")},
                "bootstrap": {"enabled": False},
                "performance": {"chunk_size_vox": 5000, "enable_profiling": True},
                "regions": {
                    "enabled": True,
                    "label_map_path": str(label_path),
                    "mode": "roi_aggregate",
                    "aggregation": "mean",
                    "voxel_level_r2": True,
                    "classes": {
                        "1": {
                            "class": "hump",
                            "default_model": "gamma",
                            "allowed_models": ["gamma", "hybrid"],
                        }
                    },
                    "scaling": {"mode": "tref", "reference_time": "peak"},
                },
            },
        ),
    ]
    return scenarios


def run_benchmark(size: int, bootstrap_n: int, output_json: Path | None) -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"pytia\.models\..*")
    with tempfile.TemporaryDirectory(prefix="pytia-bench-") as tmp:
        workdir = Path(tmp)
        images, label_path, times_s = _make_synthetic_dataset(workdir, size)
        output_root = workdir / "bench_outputs"
        output_root.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        for name, cfg in _scenario_configs(output_root, label_path, bootstrap_n):
            t0 = time.perf_counter()
            res = run_tia(images=images, times=times_s, config=cfg)
            wall_ms = 1000.0 * (time.perf_counter() - t0)
            summary_timing = res.summary.get("timing_ms", {})
            results.append(
                {
                    "scenario": name,
                    "wall_ms": round(wall_ms, 3),
                    "engine_total_ms": float(summary_timing.get("total_ms", np.nan)),
                    "status_counts": res.summary.get("status_counts", {}),
                }
            )

        print("\nPyTIA benchmark summary")
        print("-" * 72)
        print(f"{'Scenario':28s} {'Wall ms':>12s} {'Engine total ms':>16s}")
        print("-" * 72)
        for row in results:
            print(f"{row['scenario']:28s} {row['wall_ms']:12.3f} {row['engine_total_ms']:16.3f}")
        print("-" * 72)
        print("Note: compare against prior runs to detect local performance drift.")

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"Saved benchmark report: {output_json}")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PyTIA benchmark scenarios.")
    parser.add_argument("--size", type=int, default=24, help="Synthetic cube size per axis.")
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=12,
        help="Bootstrap replicates for bootstrap scenario.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write benchmark results as JSON.",
    )
    args = parser.parse_args()
    return run_benchmark(size=args.size, bootstrap_n=args.bootstrap_n, output_json=args.output_json)


if __name__ == "__main__":
    raise SystemExit(main())
