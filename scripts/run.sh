#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${MRI_PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "First run: python3 -m venv .venv && .venv/bin/pip install -e '.[test]'"
  exit 1
fi
exec "$PYTHON" -m uvicorn mri_doctor.app:app --host 127.0.0.1 --port "${MRI_PORT:-8000}" --no-access-log
