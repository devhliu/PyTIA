#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/release_checklist.sh [--full]

Local/offline release checklist:
  1) Lint and format checks
  2) Focused regression gates
  3) Optional full test suite (--full)
  4) Wheel build and local install smoke test
  5) CLI/API smoke checks
EOF
}

RUN_FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) RUN_FULL=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; usage; exit 2 ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/6] ruff check"
ruff check pytia/

echo "[2/6] ruff format check"
ruff format --check pytia/

echo "[3/6] Focused regression tests"
pytest tests/test_single_timepoint.py::TestSingleTimePointModelID::test_model_id_phys -q
pytest tests/test_region_voxel_r2_option.py -q
pytest tests/test_regressions_20260316.py -q
pytest tests/test_release_regression.py -q

if [[ "$RUN_FULL" -eq 1 ]]; then
    echo "[4/6] Full test suite"
    pytest tests/
else
    echo "[4/6] Full test suite skipped"
    echo "       Use --full to enforce pytest tests/."
fi

echo "[5/6] Build wheel and local install smoke"
rm -rf dist/
python -m pip --disable-pip-version-check wheel . --no-deps -w dist/ >/dev/null
WHEEL_PATH="$(ls -1 dist/pytia-*.whl | head -n 1)"
python -m pip --disable-pip-version-check install --force-reinstall --no-deps "$WHEEL_PATH" >/dev/null

echo "[6/6] CLI/API smoke checks"
pytia --version
python - <<'PY'
from pytia import Config, __version__

cfg = Config.load(None).data
assert "physics" in cfg
print(f"PyTIA API smoke OK ({__version__})")
PY

echo "Release checklist completed."
