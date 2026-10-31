"""End-to-end on CPU: dry-run generation -> fake scoring -> plots -> router eval."""
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd=ROOT):
    r = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_dry_run_resume_score_plot_router(tmp_path):
    out = tmp_path / "results.csv"; imgs = tmp_path / "images"; plots = tmp_path / "plots"
    common = ["--prompts", "prompts/pilot.csv", "--limit", "12", "--dry-run", "--out", str(out), "--images", str(imgs)]
    # 12 prompts covers composition (10) + 2 text prompts
    o1 = run("bench/run.py", "--controls", "--sweep", *common)
    rows = list(csv.DictReader(open(out)))
    n_settings = 5 + 8     # controls: large@30, small@30, large@12/18/24 ; sweep: 2 dirs x 4 fractions
    assert len(rows) == 12 * 2 * n_settings, o1
    assert all(Path(r["image_path"]).exists() for r in rows)
    # resume: nothing to do the second time
    o2 = run("bench/run.py", "--controls", "--sweep", *common)
    assert "0 to run" in o2
    assert len(list(csv.DictReader(open(out)))) == len(rows)

    run("bench/score.py", "--fake", "--results", str(out), "--prompts", "prompts/pilot.csv")
    df = pd.read_csv(out)
    assert df["pick_score"].notna().all() and df["clip_score"].notna().all()
    assert df.loc[df.group == "composition", "count_match"].notna().all()
    assert df.loc[df.group == "text", "ocr_match"].notna().all()
    assert list(tmp_path.glob("results.*.csv.bak"))

    o3 = run("bench/plot.py", "--results", str(out), "--out", str(plots))   # default metric = vqa_score
    for f in ("tradeoff_vqa_score.png", "tradeoff_vqa_score_by_group.png", "tradeoff_vqa_score.csv", "heatmap_vqa_score.png",
              "advantage_vqa_score.md", "cost_vs_quality_vqa_score.png", "success_vqa_score.md"):
        assert (plots / f).exists(), f
    assert "Winners per group" in o3

    o4 = run("bench/decider_eval.py", "--results", str(out), "--provider", "heuristic", "--prompts", "prompts/pilot.csv",
             "--config", str(_cfg_with_results_dir(tmp_path)))
    assert (tmp_path / "decider_policy.json").exists() and (tmp_path / "router_eval.md").exists()
    assert "router:" in o4


def _cfg_with_results_dir(tmp_path):
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["paths"]["results_csv"] = str(tmp_path / "results.csv")
    cfg["paths"]["images_dir"] = str(tmp_path / "images")
    cfg["paths"]["plots_dir"] = str(tmp_path / "plots")
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p
