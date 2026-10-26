"""Rows produced by the router itself must never feed back into the router's evaluation."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_decider_eval_ignores_router_rows(tmp_path):
    from tests.test_pipeline_dry import _cfg_with_results_dir
    out = tmp_path / "results.csv"
    common = ["--prompts", "prompts/pilot.csv", "--limit", "12", "--dry-run", "--out", str(out), "--images", str(tmp_path / "img")]
    run("bench/run.py", "--controls", "--sweep", *common)
    run("bench/score.py", "--fake", "--results", str(out), "--prompts", "prompts/pilot.csv")
    # plant router rows that would win everything if counted: nearly free and top quality
    df = pd.read_csv(out, dtype={"prompt_id": str})
    planted = df[df.method == "control"].drop_duplicates(["prompt_id", "seed"]).copy()
    planted["method"], planted["gpu_seconds"], planted["pick_score"] = "router", 0.001, 99.0
    pd.concat([df, planted]).to_csv(out, index=False)

    cfg = _cfg_with_results_dir(tmp_path)
    run("bench/decider_eval.py", "--results", str(out), "--provider", "heuristic", "--prompts", "prompts/pilot.csv", "--config", str(cfg))
    key = (tmp_path / "answer_key.csv").read_text()
    policy = (tmp_path / "decider_policy.json").read_text()
    assert "router" not in key and "router" not in policy
    assert "99" not in (tmp_path / "router_eval.md").read_text()


def test_router_plots_as_one_point():
    from bench.plot import aggregate
    rows = [dict(family="router", setting="router", method="router", direction=d, switch_fraction=f, steps=30,
                 gpu_seconds=c, pick_score=q) for d, f, c, q in [("s2l", 0.4, 10, 20), ("l2s", 0.6, 12, 21), ("large", 0.0, 20, 22)]]
    rows.append(dict(family="control", setting="large@30", method="control", direction="large", switch_fraction=0.0,
                     steps=30, gpu_seconds=25, pick_score=22.5))
    agg = aggregate(pd.DataFrame(rows), "pick_score")
    r = agg[agg.family == "router"]
    assert len(r) == 1 and r.iloc[0].n == 3 and r.iloc[0].cost == 14 and r.iloc[0].q == 21
