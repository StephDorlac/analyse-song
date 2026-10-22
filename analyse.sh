#!/usr/bin/env bash
# Analyse tous les MP3 de input/ -> output/. Option : ./analyse.sh --seuil 0.15
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Venv absent : lancer  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg manquant : sudo apt install ffmpeg"; exit 1; }
exec .venv/bin/python analyse.py "$@"
