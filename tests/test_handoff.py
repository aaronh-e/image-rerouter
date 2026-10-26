"""Handoff bookkeeping that runs without torch."""
import subprocess
import sys
from pathlib import Path

import yaml

from rerouter import handoff
from rerouter.schedule import Segment

ROOT = Path(__file__).resolve().parents[1]

# SDXL's Euler scheduler: timestep_spacing "leading", steps_offset 1, 1000 train steps.
LEADING_30 = [958 - 33 * i for i in range(30)]   # 958, 925, ..., 1


def first_model_steps_in_diffusers(f, timesteps, num_train=1000):
    """diffusers' own rule for denoising_end=f (SDXL pipeline)."""
    cutoff = int(round(num_train - f * num_train))
    return sum(t >= cutoff for t in timesteps)


def second_model_steps_in_diffusers(f, timesteps, num_train=1000):
    """diffusers' own rule for denoising_start=f (SDXL img2img pipeline)."""
    cutoff = int(round(num_train - f * num_train))
    return sum(t < cutoff for t in timesteps)


def test_naive_fraction_loses_a_step():
    # The bug the review found: passing k/steps straight through runs 11 steps, not 12.
    assert first_model_steps_in_diffusers(12 / 30, LEADING_30) == 11


def test_handoff_fraction_gives_exactly_k_steps_to_each_model():
    for k in range(1, 30):
        f = handoff.handoff_fraction(LEADING_30, k, 1000)
        assert first_model_steps_in_diffusers(f, LEADING_30) == k, k
        assert second_model_steps_in_diffusers(f, LEADING_30) == 30 - k, k


def test_warmup_handles_every_loaded_tier(monkeypatch):
    # "tiny" has no direction in schedule.plan(); warmup must not route through it.
    seen = []
    monkeypatch.setattr(handoff, "generate", lambda bank, *a, segments=None, **k: seen.append(segments))

    class Bank:
        unets = {"large": 0, "small": 0, "tiny": 0}

    handoff.warmup(Bank(), {})
    assert seen == [[Segment(t, 0, 4)] for t in ("large", "small", "tiny")]


def test_benchmark_refuses_twopipe(tmp_path):
    # impl is not part of the run key, so mixing implementations in results.csv would go unnoticed.
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["handoff"]["impl"] = "twopipe"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, "bench/run.py", "--config", str(p), "--controls", "--dry-run", "--limit", "1",
                        "--out", str(tmp_path / "r.csv"), "--images", str(tmp_path / "img")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "swap" in (r.stdout + r.stderr)
    assert not (tmp_path / "r.csv").exists()
