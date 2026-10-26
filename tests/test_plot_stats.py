"""Statistics: a prompt's two seeds are one unit, not two independent samples."""
import numpy as np
import pandas as pd

from bench import plot


def _df(deltas_by_prompt, seeds=(42, 1234), base=20.0):
    """Settings A and B on each (prompt, seed); A - B = that prompt's delta for every seed."""
    rows = []
    for i, d in enumerate(deltas_by_prompt):
        for s in seeds:
            for setting, q in (("A", base + d), ("B", base)):
                rows.append(dict(prompt_id=f"p{i}", seed=s, setting=setting, family="s2l", method="handoff",
                                 direction="s2l", switch_fraction=0.4, steps=30, gpu_seconds=10.0, pick_score=q))
    return pd.DataFrame(rows)


def test_paired_delta_counts_prompts_not_images():
    rng = np.random.default_rng(1)
    df = _df(rng.normal(0.2, 1.0, size=10))
    r = plot.paired_delta(df, "pick_score", "A", "B")
    assert r["n"] == 10 and r["n_images"] == 20
    assert r["wins"] + r["losses"] <= 10
    # resampling 20 copies of 10 numbers would understate the spread; per-prompt matches a 10-sample bootstrap
    per_prompt = df[df.setting == "A"].groupby("prompt_id").pick_score.mean() - 20.0
    boots = np.random.default_rng(0).choice(per_prompt.to_numpy(), size=(2000, 10), replace=True).mean(axis=1)
    assert np.isclose(r["ci_lo"], np.percentile(boots, 2.5)) and np.isclose(r["ci_hi"], np.percentile(boots, 97.5))


def test_error_bars_use_prompt_means():
    df = _df([0.0, 1.0, 2.0, 3.0])
    df["family"] = np.where(df.setting == "A", "s2l", "control")
    agg = plot.aggregate(df[df.setting == "A"], "pick_score")
    per_prompt = df[df.setting == "A"].groupby("prompt_id").pick_score.mean()
    assert np.isclose(agg.iloc[0].q_sem, per_prompt.sem())
    assert agg.iloc[0].n == 8                                  # n is still images


def _report(delta_vs_control, spread, ref_steps=30):
    """One handoff setting, its matched large@12 control, and the large@ref reference."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(30):
        d = delta_vs_control + rng.normal(0, spread)
        for s in (42, 1234):
            for setting, fam, direction, sw, steps, cost, q in (
                ("s2l@0.4", "s2l", "s2l", 0.4, 30, 10.0, 21.0 + d),
                ("large@12", "control", "large", 0.0, 12, 10.0, 21.0),
                ("large@30", "control", "large", 0.0, 30, 25.0, 21.0 + d),
                ("large@50", "control", "large", 0.0, 50, 40.0, 30.0)):
                rows.append(dict(prompt_id=f"p{i}", seed=s, group="text", setting=setting, family=fam,
                                 method="control" if fam == "control" else "handoff", direction=direction,
                                 switch_fraction=sw, steps=steps, gpu_seconds=cost, pick_score=q))
    df = pd.DataFrame(rows)
    matched = plot.matched_cost_table(df, "pick_score")
    return plot.success_report(df, "pick_score", matched, ref_steps=ref_steps)


def test_beating_the_control_needs_a_ci_above_zero():
    noisy = _report(delta_vs_control=0.05, spread=3.0)     # mean > 0 but CI straddles 0
    clear = _report(delta_vs_control=1.0, spread=0.1)
    assert "| s2l@0.4 |" in noisy and noisy.split("| s2l@0.4 |")[1].splitlines()[0].rstrip().endswith("| no |")
    assert clear.split("| s2l@0.4 |")[1].splitlines()[0].rstrip().endswith("| YES |")


def test_reference_is_the_configured_step_count():
    # a stray exploratory large@50 must not become the reference
    assert "Reference: large@30" in _report(1.0, 0.1, ref_steps=30)
