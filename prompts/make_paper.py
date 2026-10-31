#!/usr/bin/env python
"""Build prompts/paper.csv: the prompt set for the paper (753 prompts).

    python prompts/make_paper.py

= our 200 prompts (full.csv: composition, text, hands_faces, dense, product, stylized)
+ GenEval's 553 (geneval_metadata.jsonl, from github.com/djghosh13/geneval, MIT
  licence in GENEVAL_LICENSE). GenEval rows get group `geneval_<task>` and
  expected `geneval=<the metadata line as JSON>`, which bench/score.py checks with
  GenEval's rules. Run a subset with `bench/run.py --groups ...`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    with open(HERE / "full.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(HERE / "geneval_metadata.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            meta = json.loads(line)
            rows.append(dict(id=f"geneval_{i:03d}", group=f"geneval_{meta['tag']}", prompt=meta["prompt"],
                             expected="geneval=" + json.dumps(meta, separators=(",", ":"))))
    with open(HERE / "paper.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "prompt", "expected"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} prompts to {HERE / 'paper.csv'}")


if __name__ == "__main__":
    main()
