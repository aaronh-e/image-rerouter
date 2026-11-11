"""Cost comes from step counts, not the stopwatch, using each row's own GPU. Seen in the
pilot: large@30 for three groups ran in slower Colab sessions (28-30 s vs 25 s), which made
every cheaper setting in those groups look ~10 points cheaper than it was."""
import pandas as pd
import pytest

from bench.plot import load_results
from rerouter import utils

CFG = {"step_seconds": {"Tesla T4": {"large": 0.8, "small": 0.5, "fixed": 1.0},
                        "Fast GPU": {"large": 0.1, "small": 0.05, "fixed": 0.2}}}


def _csv(tmp_path, rows):
    p = tmp_path / "results.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _row(direction, steps, first, second, n_first, secs, method="control", sw=0.0, gpu="Tesla T4", pid="p1"):
    return dict(prompt_id=pid, group="text", seed=42, method=method, direction=direction, switch_fraction=sw,
                steps=steps, first_model=first, second_model=second, n_first_steps=n_first, gpu_seconds=secs,
                gpu_type=gpu, pick_score=22.0, notes="")


def test_cost_ignores_session_speed(tmp_path):
    rows = [_row("large", 30, "large", "large", 30, 29.9),      # slow session
            _row("large", 24, "large", "large", 24, 20.3),      # normal session
            _row("small", 30, "small", "small", 30, 16.4),
            _row("s2l", 30, "small", "large", 6, 26.2, method="handoff", sw=0.2)]
    df = load_results(str(_csv(tmp_path, rows)), CFG)
    assert df.gpu_seconds.tolist() == pytest.approx([25.0, 20.2, 16.0, 1.0 + 6 * 0.5 + 24 * 0.8])
    assert df.measured_seconds.tolist() == [29.9, 20.3, 16.4, 26.2]   # stopwatch kept for reference


def test_each_row_uses_its_own_gpu(tmp_path):
    rows = [_row("large", 30, "large", "large", 30, 25.0), _row("large", 30, "large", "large", 30, 3.0, gpu="Fast GPU", pid="p2")]
    df = load_results(str(_csv(tmp_path, rows)), CFG)
    assert df.gpu_seconds.tolist() == pytest.approx([25.0, 0.2 + 30 * 0.1])


def test_unlisted_gpu_is_fitted_from_its_controls(tmp_path, capsys):
    secs = lambda nl, ns: 0.5 + 0.06 * nl + 0.04 * ns        # an "L4" not in the config
    rows = [_row("large", s, "large", "large", s, secs(s, 0), gpu="NVIDIA L4") for s in (12, 18, 24, 30)]
    rows += [_row("small", 30, "small", "small", 30, secs(0, 30), gpu="NVIDIA L4"),
             _row("s2l", 30, "small", "large", 12, 99.0, method="handoff", sw=0.4, gpu="NVIDIA L4")]
    df = load_results(str(_csv(tmp_path, rows)), CFG)
    assert df.gpu_seconds.iloc[-1] == pytest.approx(secs(18, 12), abs=1e-3)   # stopwatch 99 s ignored
    assert "fitted" in capsys.readouterr().out


def test_unknown_model_fails_loudly(tmp_path):
    rows = [_row("large", 30, "tiny", "tiny", 30, 9.0)]
    with pytest.raises(ValueError, match="tiny"):
        load_results(str(_csv(tmp_path, rows)), CFG)


def test_price_is_per_gpu_never_a_t4_default(monkeypatch):
    monkeypatch.delenv("RR_USD_PER_HOUR", raising=False)
    cfg = {"compute": {"usd_per_hour": {"Tesla T4": 0.35}}}
    assert utils.usd_per_hour(cfg, "Tesla T4") == 0.35
    assert utils.usd_per_hour(cfg, "NVIDIA RTX PRO 6000 Blackwell Server Edition") is None
    monkeypatch.setenv("RR_USD_PER_HOUR", "2.5")                   # the notebook's USD_PER_HOUR
    assert utils.usd_per_hour(cfg, "NVIDIA RTX PRO 6000 Blackwell Server Edition") == 2.5
