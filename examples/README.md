# PyTIA Examples

Examples are split by interface:

- `examples/cli/` for command-line workflows (`pytia run/validate/info`)
- `examples/python_api/` for direct `run_tia(...)` usage
- `examples/data/` for example inputs

## Layout

```text
examples/
├── cli/
│   └── configs/
├── python_api/
└── data/
    └── lu177_fap/
```

`examples/data/lu177_fap/input_multi-points/` contains the Lu-177 FAP sample SPECT inputs.

## Verify CLI and Python API are implemented

```bash
pytia --version
python -c "import pytia; print(pytia.__version__, hasattr(pytia, 'run_tia'))"
```

## Python API examples

Optional Lu-177/FAP and phantom demos:

```bash
python examples/python_api/debug_fap.py
MPLBACKEND=Agg python examples/python_api/visualize_fap_results.py
MPLBACKEND=Agg python examples/python_api/lu177_psma_phantom_demo.py
MPLBACKEND=Agg python examples/python_api/lu177_psma_phantom_plotting.py
```

## CLI examples

Validate and run configs:

```bash
pytia validate --config examples/cli/configs/config_lu177_fap.yaml
pytia run --config examples/cli/configs/config_lu177_fap.yaml
```

## Notes

- Keep your own datasets under `examples/data/` or outside the repo, then update `inputs.images`.
- Example outputs are written to `examples/data/lu177_fap/outputs/` as defined in each config.
