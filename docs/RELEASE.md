# PyTIA Local Release Guide

This guide defines a repeatable, offline/local-first release flow.

## 1) Run the release checklist

```bash
scripts/release_checklist.sh
```

Default checklist stages:
- `ruff check pytia/`
- `ruff format --check pytia/`
- focused regression gates:
  - `tests/test_single_timepoint.py::TestSingleTimePointModelID::test_model_id_phys`
  - `tests/test_region_voxel_r2_option.py`
  - `tests/test_regressions_20260316.py`
  - `tests/test_release_regression.py`
- wheel build + local install smoke
- CLI/API smoke checks (`pytia --version`, `Config.load(None)`)

### Optional strict mode

```bash
scripts/release_checklist.sh --full
```

Runs `pytest tests/` in addition to the focused gates.

## 2) Capture local benchmark baseline

```bash
python scripts/benchmark_local.py --output-json ./benchmarks/latest.json
```

Benchmark scenarios include:
- voxel mode (non-chunked)
- voxel mode (chunked)
- voxel mode (chunked + bootstrap)
- region mode

Track `wall_ms` and `engine_total_ms` over releases to detect regressions.

## 3) Numerical regression policy

- Use deterministic seeds for stochastic paths (bootstrap).
- Prefer tolerance-based assertions (`rtol`/`atol`) for floating-point checks.
- Treat changes to status/model distributions and summary contracts as release-impacting.
