#!/usr/bin/env bash
# Level 2: the three controls on the pilot set, then score and plot.
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:-prompts/pilot.csv}
python bench/run.py --controls --prompts "$PROMPTS"
python bench/score.py --prompts "$PROMPTS"
python bench/plot.py
