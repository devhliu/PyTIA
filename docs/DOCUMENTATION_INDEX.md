# PyTIA Documentation Index

This index lists the canonical documentation set for PyTIA after legacy cleanup.

## Start here

- [`../README.md`](../README.md): project overview and installation
- [`QUICK_START.md`](QUICK_START.md): shortest path to first successful run

## Canonical documentation set

| Document | Purpose | Audience |
| --- | --- | --- |
| [`QUICK_START.md`](QUICK_START.md) | Minimal setup and first run | New users |
| [`USER_GUIDE.md`](USER_GUIDE.md) | End-to-end usage (CLI + Python API) | Users |
| [`CONFIG.md`](CONFIG.md) | Complete configuration contract | Users / Developers |
| [`STP_USER_GUIDE.md`](STP_USER_GUIDE.md) | Single-timepoint workflow details | Users |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Runtime architecture and contracts | Developers |
| [`RELEASE.md`](RELEASE.md) | Local/offline release checklist and benchmarking | Maintainers |

## Quick task map

### Run TIA from CLI

1. Read [`QUICK_START.md`](QUICK_START.md)
2. Start from an example config in `examples/cli/configs/`
3. Run `pytia validate --config config.yaml`
4. Run `pytia run --config config.yaml`

### Use the Python API

1. Read [`USER_GUIDE.md`](USER_GUIDE.md#python-api)
2. Run `examples/python_api/example_multitime.py` or `examples/python_api/example_stp.py`
3. Adapt the config and inputs for your data

### Configure single-timepoint mode

1. Read [`STP_USER_GUIDE.md`](STP_USER_GUIDE.md)
2. Validate method settings in [`CONFIG.md`](CONFIG.md)
3. Verify `model_id` and `status_id` outputs

### Prepare a local release

1. Run `scripts/release_checklist.sh`
2. Optionally enforce full suite with `scripts/release_checklist.sh --full`
3. Capture performance baseline with `python scripts/benchmark_local.py --output-json ./benchmarks/latest.json`
4. Review [`RELEASE.md`](RELEASE.md)

## Notes on removed legacy documents

Legacy completion reports, design snapshots, and versioned notes were removed from `docs/`
to prevent drift and conflicting guidance. Use only the canonical documents above.
