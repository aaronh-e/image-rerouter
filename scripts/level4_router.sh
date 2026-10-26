#!/usr/bin/env bash
# Level 4: learn the routing policy on 70% of prompts, evaluate on 30%.
# Needs TYPESAFE_API_KEY in the environment for the real Jev; the heuristic
# baseline runs offline.
set -euo pipefail
cd "$(dirname "$0")/.."
PROMPTS=${1:-prompts/pilot.csv}
python bench/decider_eval.py --prompts "$PROMPTS" --provider heuristic
mv results/router_eval.md results/router_eval_heuristic.md
if [ -n "${TYPESAFE_API_KEY:-}" ]; then
  python bench/decider_eval.py --prompts "$PROMPTS" --provider typesafe
else
  echo "TYPESAFE_API_KEY not set: skipped the Jev run (heuristic baseline only)"
fi
