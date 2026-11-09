"""Cost comes from step counts, not the stopwatch. Seen in the pilot: large@30 for three
groups ran in slower Colab sessions (28-30 s vs 25 s), which made every cheaper setting
in those groups look ~10 points cheaper than it was."""
import pandas as pd
import pytest

from bench.plot import load_results

CFG = {"step_seconds": {"large": 0.8, "small": 0.5}, "fixed_seconds": 1.0}


def _csv(tmp_path, rows):
    p = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _row(direction, steps, first, second, n_first, secs, method="control", sw=0.0):
    return dict(prompt_id="p1", group="text", seed=42, method=method, direction=direction, switch_fraction=sw,
                steps=steps, first_model=first, second_model=second, n_first_steps=n_first, gpu_seconds=secs,
                pick_score=22.0, notes="")


def test_cost_ignores_session_speed(tmp_path):
    rows = [_row("large", 30, "large", "large", 30, 29.9),      # slow session
            _row("large", 24, "large", "large", 24, 20.3),      # normal session
            _row("small", 30, "small", "small", 30, 16.4),
            _row("s2l", 30, "small", "large", 6, 26.2, method="handoff", sw=0.2)]
    df = load_results(str(_csv(tmp_path, rows)), CFG)
    assert df.gpu_seconds.tolist() == pytest.approx([25.0, 20.2, 16.0, 1.0 + 6 * 0.5 + 24 * 0.8])
    assert df.measured_seconds.tolist() == [29.9, 20.3, 16.4, 26.2]   # stopwatch kept for reference


def test_unknown_model_fails_loudly(tmp_path):
    rows = [_row("large", 30, "tiny", "tiny", 30, 9.0)]
    with pytest.raises(ValueError, match="tiny"):
        load_results(str(_csv(tmp_path, rows)), CFG)
