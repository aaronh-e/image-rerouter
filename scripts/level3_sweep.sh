#!/usr/bin/env bash
# Level 3: both handoff orders over every switch fraction, then score and plot
# the three charts that check the predictions.
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:-prompts/pilot.csv}
python bench/run.py --sweep --prompts "$PROMPTS"
python bench/score.py --prompts "$PROMPTS"
python bench/plot.py --metric pick_score
python bench/plot.py --metric clip_score
python bench/plot.py --metric count_match --groups composition   # prediction 1: Method 1 hurts composition
python bench/plot.py --metric ocr_match --groups text            # prediction 2: Method 2 hurts text
