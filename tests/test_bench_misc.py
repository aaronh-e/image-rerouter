"""Small bench fixes: run fingerprint without git, backups, cwd-independent prompts, NMS counting."""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from rerouter import utils

ROOT = Path(__file__).resolve().parents[1]


def test_commit_falls_back_to_a_source_fingerprint(tmp_path, monkeypatch):
    # a zip upload has no .git: rows must still say which code produced them
    for d in ("rerouter", "bench", "app"):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.setattr(utils, "REPO_ROOT", tmp_path)
    a = utils.git_commit()
    assert a.startswith("src-") and len(a) == 14 and a == utils.git_commit()
    # same content with Windows line endings: same fingerprint
    f = tmp_path / "rerouter" / "schedule.py"
    f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert utils.git_commit() == a
    f.write_text(f.read_text() + "\n# changed\n")
    assert utils.git_commit() != a


def _dry_results(tmp_path):
    out = tmp_path / "results.csv"
    subprocess.run([sys.executable, "bench/run.py", "--controls", "--prompts", "prompts/pilot.csv", "--limit", "12",
                    "--dry-run", "--out", str(out), "--images", str(tmp_path / "img")], cwd=ROOT, check=True, capture_output=True)
    return out


def test_every_scoring_run_keeps_its_own_backup(tmp_path):
    out = _dry_results(tmp_path)
    for _ in range(2):
        subprocess.run([sys.executable, "bench/score.py", "--fake", "--force", "--results", str(out)],
                       cwd=ROOT, check=True, capture_output=True)
    assert len(list(tmp_path.glob("results.*.csv.bak"))) == 2


def test_prompt_files_found_from_any_folder(tmp_path):
    out = _dry_results(tmp_path)
    # run from outside the repo: the default prompts/pilot.csv must still be found, or OCR/count get skipped
    r = subprocess.run([sys.executable, str(ROOT / "bench" / "score.py"), "--fake", "--results", str(out)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out)
    assert df.loc[df.group == "text", "ocr_match"].notna().all()
    assert df.loc[df.group == "composition", "count_match"].notna().all()


def test_nms_count_merges_duplicate_boxes():
    from bench.score import nms_count
    boxes = [[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60], [0, 0, 10, 9]]   # three boxes on one cup
    scores = [0.9, 0.8, 0.7, 0.6]
    assert nms_count(boxes, scores, iou=0.5) == 2
    assert nms_count(boxes, scores, iou=1.0) == 4      # nothing suppressed
    assert nms_count([], [], iou=0.5) == 0
