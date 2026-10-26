#!/usr/bin/env bash
# One-time setup on Mac/Linux. Creates a private Python environment inside this
# folder and installs everything. Usage:  bash setup.sh        (full, for a GPU/Mac)
#                                         bash setup.sh cpu    (light: tests, plots, demo app)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
if [ "${1:-}" = "cpu" ]; then pip install -r requirements-cpu.txt gradio; else pip install -r requirements.txt; fi
python -m pytest -q tests
echo
echo "Done. Each time you open a terminal here, run:  source .venv/bin/activate"
echo "Then try:  python app/app.py --demo     (website with fake images)"
echo "      or:  python scripts/level1_local.py --size 512 --steps 12   (real, needs a GPU or Apple Silicon)"
