#!/usr/bin/env bash
# Export MIDI des morceaux déjà analysés (./analyse.sh d'abord).
# Options : --morceau "Nom"  --version brut|quantifie  --sans-controle  --extrait 60:20  --other-program 27
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Venv absent."; exit 1; }
command -v fluidsynth >/dev/null || { echo "fluidsynth manquant : sudo apt install fluidsynth fluid-soundfont-gm"; exit 1; }
exec .venv/bin/python midi.py "$@"
