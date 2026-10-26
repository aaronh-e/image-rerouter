#!/usr/bin/env python
"""Level 4: build the answer key from the Level 3 sweep, learn a routing policy on
70% of the prompts, and evaluate the Jev router on the held-out 30%.

Because the Level 3 sweep already generated EVERY (prompt, seed, setting), the
router can be evaluated by lookup: route a prompt -> read the cost and quality
of that setting's existing image. No GPU needed. (bench/run.py --method router
also exists if you want the actual routed images.)

    python bench/decider_eval.py                       # Jev via TYPESAFE_API_KEY (config decider.provider)
    python bench/decider_eval.py --provider heuristic  # regex baseline, offline

Outputs: results/decider_policy.json, results/answer_key.csv, results/router_eval.csv,
         results/router_eval.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerouter import utils  # noqa: E402
from rerouter.decider import answers_for_prompts, profile_from_answers  # noqa: E402
from bench.plot import load_results  # noqa: E402

DEFAULT_TOL = {"pick_score": 0.10, "clip_score": 0.50, "hpsv2": 0.005, "ocr_match": 0.0, "count_match": 0.0}


# ------------------------------------------------------------ answer key ----

def per_prompt_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Average the two seeds: one row per (prompt_id, setting)."""
    d = df.dropna(subset=[metric, "gpu_seconds"])
    return d.groupby(["prompt_id", "group", "setting", "method", "direction", "switch_fraction", "steps"], dropna=False) \
            .agg(cost=("gpu_seconds", "mean"), q=(metric, "mean"), n=(metric, "size")).reset_index()


def best_setting(rows: pd.DataFrame, ref_q: float, tol: float) -> pd.Series:
    """Cheapest setting whose quality is within `tol` of the reference; if none,
    the highest-quality one. This is the definition of 'best' for the whole level."""
    ok = rows[rows.q >= ref_q - tol]
    if len(ok):
        return ok.loc[ok.cost.idxmin()]
    return rows.loc[rows.q.idxmax()]


def answer_key(pp: pd.DataFrame, ref_setting: str, tol: float) -> pd.DataFrame:
    out = []
    for pid, rows in pp.groupby("prompt_id"):
        ref = rows[rows.setting == ref_setting]
        if ref.empty:
            continue
        ref_q = float(ref.q.iloc[0]); ref_cost = float(ref.cost.iloc[0])
        b = best_setting(rows, ref_q, tol)
        out.append(dict(prompt_id=pid, group=rows.group.iloc[0], ref_quality=ref_q, ref_cost=ref_cost,
                        best_setting=b.setting, best_quality=b.q, best_cost=b.cost, saving=1 - b.cost / ref_cost))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- policy ----

def split_prompts(prompts: list[dict], train_fraction: float, seed: int) -> tuple[set, set]:
    """Stratified by group so every group appears in the test set."""
    rng = random.Random(seed)
    train, test = set(), set()
    by_group: dict[str, list] = {}
    for p in prompts:
        by_group.setdefault(p["group"], []).append(p["id"])
    for ids in by_group.values():
        ids = sorted(ids); rng.shuffle(ids)
        k = int(round(train_fraction * len(ids)))
        train.update(ids[:k]); test.update(ids[k:])
    return train, test


def learn_policy(pp: pd.DataFrame, profiles: dict[str, str], train_ids: set, ref_setting: str, tol: float) -> tuple[dict, str]:
    """For each profile: among all settings, pick the cheapest whose MEAN quality on
    the training prompts of that profile stays within tol of the reference mean.
    Returns (policy, best_single_fixed_setting)."""
    tr = pp[pp.prompt_id.isin(train_ids)].copy()
    tr["profile"] = tr.prompt_id.map(profiles)

    def choose(sub: pd.DataFrame):
        agg = sub.groupby(["setting", "method", "direction", "switch_fraction", "steps"], dropna=False).agg(cost=("cost", "mean"), q=("q", "mean"), n=("q", "size")).reset_index()
        ref = agg[agg.setting == ref_setting]
        if ref.empty:
            return None
        return best_setting(agg, float(ref.q.iloc[0]), tol)

    policy = {}
    for prof, sub in tr.groupby("profile"):
        b = choose(sub)
        if b is not None:
            policy[prof] = dict(direction=b.direction, switch_fraction=float(b.switch_fraction) if b.method != "control" else 0.0,
                                steps=int(b.steps), method=b.method, setting=b.setting, n_train_prompts=int(sub.prompt_id.nunique()))
    fixed = choose(tr)
    return policy, (fixed.setting if fixed is not None else ref_setting)


def evaluate(pp: pd.DataFrame, key: pd.DataFrame, profiles: dict, policy: dict, fixed_setting: str, ref_setting: str, test_ids: set) -> pd.DataFrame:
    te = pp[pp.prompt_id.isin(test_ids)]
    lookup = te.set_index(["prompt_id", "setting"])
    rows = []
    for pid in sorted(test_ids):
        if (pid, ref_setting) not in lookup.index:
            continue
        g = te[te.prompt_id == pid].group.iloc[0]
        prof = profiles.get(pid, "generic")
        routed = (policy.get(prof) or policy.get("generic") or {}).get("setting", fixed_setting)
        oracle = key[key.prompt_id == pid].best_setting.iloc[0] if (key.prompt_id == pid).any() else ref_setting
        for strategy, s in (("large_full", ref_setting), ("best_fixed", fixed_setting), ("router", routed), ("oracle", oracle)):
            if (pid, s) in lookup.index:
                r = lookup.loc[(pid, s)]
                rows.append(dict(prompt_id=pid, group=g, profile=prof, strategy=strategy, setting=s, cost=float(r.cost), q=float(r.q)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--prompts", default="prompts/pilot.csv")
    ap.add_argument("--metric", default="pick_score")
    ap.add_argument("--tol", type=float, default=None, help="quality slack vs large@full that still counts as 'no loss'")
    ap.add_argument("--provider", default=None, help="override config decider.provider (typesafe|aimlapi|heuristic)")
    args = ap.parse_args(argv)

    cfg = utils.load_config(args.config)
    if args.provider:
        cfg["decider"]["provider"] = args.provider
    tol = args.tol if args.tol is not None else DEFAULT_TOL.get(args.metric, 0.1)
    res_dir = Path(cfg["paths"]["results_csv"]).parent
    prompts = utils.read_prompts(args.prompts)
    df = load_results(args.results or cfg["paths"]["results_csv"])
    df = df[df.prompt_id.isin({p["id"] for p in prompts})]
    ref_setting = f"large@{cfg['generation']['steps']}"

    pp = per_prompt_table(df, args.metric)
    key = answer_key(pp, ref_setting, tol)
    key.to_csv(res_dir / "answer_key.csv", index=False)

    train_ids, test_ids = split_prompts(prompts, cfg["decider"]["train_fraction"], cfg["decider"]["split_seed"])
    answers = answers_for_prompts(prompts, cfg, res_dir / "jev_answers.json")
    profiles = {pid: profile_from_answers(a) for pid, a in answers.items()}
    lat = [a.get("_latency_s", 0) for a in answers.values()]

    policy, fixed = learn_policy(pp, profiles, train_ids, ref_setting, tol)
    (res_dir / "decider_policy.json").write_text(json.dumps(policy, indent=1))
    ev = evaluate(pp, key, profiles, policy, fixed, ref_setting, test_ids)
    ev.to_csv(res_dir / "router_eval.csv", index=False)

    summary = ev.groupby("strategy").agg(cost=("cost", "mean"), quality=("q", "mean"), n_prompts=("prompt_id", "nunique")).reindex(["large_full", "best_fixed", "router", "oracle"])
    summary["saving_vs_large"] = 1 - summary.cost / summary.loc["large_full", "cost"]
    by_group = ev.pivot_table(index="group", columns="strategy", values="q", aggfunc="mean")
    md = ["# Router evaluation (held-out 30%)", "", f"metric = {args.metric}, tolerance = {tol}, provider = {cfg['decider']['provider']}, "
          f"test prompts = {len(test_ids)}, mean Jev latency = {np.mean(lat):.2f}s", "", summary.to_markdown(floatfmt=".3f"), "",
          "## Quality by group", "", by_group.to_markdown(floatfmt=".3f"), "", "## Learned policy", "",
          pd.DataFrame(policy).T.to_markdown() if policy else "(empty)", "", f"Best single fixed setting on train: **{fixed}**", "",
          "## Profile distribution", "", pd.Series(profiles).value_counts().to_markdown()]
    (res_dir / "router_eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nSummary")
    print(f"  answer key -> {res_dir / 'answer_key.csv'}; policy -> {res_dir / 'decider_policy.json'}; eval -> {res_dir / 'router_eval.md'}")
    r, f, o = summary.loc["router"], summary.loc["best_fixed"], summary.loc["oracle"]
    print(f"  router: {r.cost:.2f} GPU-s @ {r.quality:.3f}   best fixed: {f.cost:.2f} @ {f.quality:.3f}   oracle: {o.cost:.2f} @ {o.quality:.3f}")
    print("  (a win = router is cheaper at equal quality, or better at equal cost, than best_fixed; the oracle is the ceiling)")


if __name__ == "__main__":
    main()
