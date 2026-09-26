#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" -c 'import sys; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)'
if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)'
.venv/bin/python -m pip --version
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
if [ ! -f config/settings.yaml ]; then
  cp config/settings.example.yaml config/settings.yaml
fi
.venv/bin/python scripts/doctor.py
echo "Phase 1 ready. Run: .venv/bin/python run_app.py"
