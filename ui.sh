#!/usr/bin/env bash
# Lance l'interface web locale puis ouvre http://127.0.0.1:5000
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Venv absent : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
(sleep 2; xdg-open http://127.0.0.1:${PORT:-5000} >/dev/null 2>&1 || true) &
exec .venv/bin/python app.py
