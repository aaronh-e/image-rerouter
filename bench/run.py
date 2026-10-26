#!/usr/bin/env python
"""Run one method (or a sweep) over a prompt file and both seeds.

Examples
    # Level 2: the three controls on the pilot set
    python bench/run.py --controls --prompts prompts/pilot.csv
    # Level 3: one handoff setting
    python bench/run.py --method handoff --direction s2l --switch 0.4 --prompts prompts/pilot.csv
    # Level 3: the whole sweep (both directions x every switch fraction in config.yaml)
    python bench/run.py --sweep --prompts prompts/pilot.csv
    # Level 4: route each prompt with the learned policy
    python bench/run.py --method router --policy results/decider_policy.json --prompts prompts/pilot.csv
    # No GPU? Exercise the whole bookkeeping path with fake images:
    python bench/run.py --controls --prompts prompts/pilot.csv --dry-run --limit 4

Every image gets one row in results/results.csv (append-only; existing runs are
skipped so a crashed Colab session just re-runs the same command).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rerouter import utils  # noqa: E402
from rerouter.schedule import plan, describe  # noqa: E402


# ------------------------------------------------------------------ jobs ----

def build_jobs(args, cfg, prompts) -> list[dict]:
    """Expand CLI flags into a flat list of (prompt, seed, setting) jobs."""
    steps = args.steps or cfg["generation"]["steps"]
    settings: list[dict] = []
    if args.controls:
        settings.append(dict(method="control", direction="large", switch_fraction=0.0, steps=steps))
        settings.append(dict(method="control", direction="small", switch_fraction=0.0, steps=steps))
        for s in cfg["control_step_counts"]:
            if s != steps:
                settings.append(dict(method="control", direction="large", switch_fraction=0.0, steps=s))
    if args.sweep:
        for d in ("s2l", "l2s"):
            for f in cfg["switch_fractions"]:
                if 0.0 < f < 1.0:   # 0 and 1 are the controls; don't duplicate them
                    settings.append(dict(method="handoff", direction=d, switch_fraction=float(f), steps=steps))
    if args.method == "handoff":
        if args.direction not in ("s2l", "l2s") or args.switch is None:
            sys.exit("--method handoff needs --direction s2l|l2s and --switch F")
        settings.append(dict(method="handoff", direction=args.direction, switch_fraction=float(args.switch), steps=steps))
    if args.method == "control":
        if args.direction not in ("large", "small"):
            sys.exit("--method control needs --direction large|small")
        settings.append(dict(method="control", direction=args.direction, switch_fraction=0.0, steps=steps))

    jobs = []
    if args.method == "router":
        from rerouter.decider import answers_for_prompts, route, load_policy
        policy = load_policy(args.policy)
        answers = answers_for_prompts(prompts, cfg, Path(cfg["paths"]["results_csv"]).parent / "jev_answers.json")
        for p in prompts:
            r = route(answers[p["id"]], policy)
            for seed in args.seeds:
                jobs.append(dict(prompt=p, seed=seed, method="router", direction=r["direction"],
                                 switch_fraction=r["switch_fraction"], steps=r.get("steps", steps), profile=r["profile"]))
    for s in settings:
        for p in prompts:
            for seed in args.seeds:
                jobs.append(dict(prompt=p, seed=seed, **s))
    return jobs


# --------------------------------------------------------------- generate ----

class DryRunBank:
    """Stands in for ModelBank when there is no GPU: makes a noise image and
    fake-but-plausible timings so the CSV/plot code can be tested anywhere."""
    gpu_name = "dry-run"
    device = "cpu"

    def generate(self, prompt, seed, direction, switch_fraction, steps, cfg):
        import numpy as np
        from PIL import Image
        from rerouter.handoff import GenResult
        segs = plan(direction, switch_fraction, steps)
        per_step = {"large": 0.80, "small": 0.45, "tiny": 0.30}   # ~T4 seconds/step at 1024^2
        secs = sum(per_step[s.model] * s.n_steps for s in segs) + 0.9  # + VAE decode
        rng = np.random.default_rng(seed + len(prompt))
        img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).resize((256, 256))
        return GenResult(img, secs, 9.5, segs[0].model, segs[-1].model, segs[0].n_steps, describe(segs), "dry")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default="prompts/pilot.csv")
    ap.add_argument("--method", choices=["control", "handoff", "router"], default=None)
    ap.add_argument("--direction", choices=["s2l", "l2s", "large", "small"], default=None)
    ap.add_argument("--switch", type=float, default=None, help="fraction of steps run by the FIRST model")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--controls", action="store_true", help="run the three controls from the plan")
    ap.add_argument("--sweep", action="store_true", help="run both directions over config switch_fractions")
    ap.add_argument("--policy", default="results/decider_policy.json")
    ap.add_argument("--groups", nargs="+", default=None, help="only these prompt groups")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="no GPU: fake images + timings")
    ap.add_argument("--out", default=None, help="results csv (default from config)")
    ap.add_argument("--images", default=None, help="images dir (default from config)")
    args = ap.parse_args(argv)

    cfg = utils.load_config(args.config)
    if cfg["handoff"].get("impl", "swap") != "swap":
        # impl is not part of the run key, so rows from two implementations would mix silently.
        sys.exit("bench/run.py needs handoff.impl: swap in config.yaml (twopipe is a Level 1 comparison only)")
    args.seeds = args.seeds or cfg["seeds"]
    prompts = utils.read_prompts(args.prompts)
    if args.groups:
        prompts = [p for p in prompts if p["group"] in args.groups]
    if args.limit:
        prompts = prompts[: args.limit]
    out_csv = Path(args.out or cfg["paths"]["results_csv"])
    images_dir = Path(args.images or cfg["paths"]["images_dir"])

    jobs = build_jobs(args, cfg, prompts)
    if not jobs:
        sys.exit("nothing to do: pass --controls, --sweep, or --method ...")
    done = utils.existing_keys(out_csv)
    todo = [j for j in jobs if utils.run_key(dict(prompt_id=j["prompt"]["id"], **j)) not in done]
    print(f"{len(jobs)} jobs requested, {len(jobs) - len(todo)} already in {out_csv}, {len(todo)} to run")
    if not todo:
        return

    # --- load models once ---
    if args.dry_run:
        bank = DryRunBank()
        generate = bank.generate
    else:
        from rerouter.models import ModelBank
        from rerouter.handoff import generate as _generate, warmup
        bank = ModelBank(cfg).load(("large", "small"))
        print("loaded:", {k: f"{v:.1f}s" for k, v in bank.load_seconds.items()}, "| UNet params (B):", bank.param_counts())
        warmup(bank, cfg, cfg["handoff"].get("warmup_runs", 1))
        generate = lambda *a: _generate(bank, *a)  # noqa: E731

    commit = utils.git_commit()
    usd_per_hour = float(cfg["compute"]["usd_per_hour"])
    n_ok = n_err = 0
    t_start = time.time()
    for k, j in enumerate(todo, 1):
        p = j["prompt"]
        path = utils.image_path(images_dir, j["method"], j["direction"], j["switch_fraction"], j["steps"], p["id"], j["seed"])
        row = dict(prompt_id=p["id"], group=p["group"], seed=j["seed"], method=j["method"], direction=j["direction"],
                   switch_fraction=f"{j['switch_fraction']:.2f}", steps=j["steps"], gpu_type=bank.gpu_name,
                   image_path=str(path), commit=commit, run_ts=dt.datetime.now().isoformat(timespec="seconds"),
                   prompt=p["prompt"], notes=j.get("profile", ""))
        try:
            res = generate(p["prompt"], j["seed"], j["direction"], j["switch_fraction"], j["steps"], cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            res.image.save(path)
            row.update(res.as_row())
            row["gpu_seconds"] = f"{res.gpu_seconds:.3f}"
            row["peak_mem_gb"] = f"{res.peak_mem_gb:.2f}"
            row["usd_per_1000"] = f"{res.gpu_seconds * usd_per_hour / 3600 * 1000:.4f}"
            n_ok += 1
            print(f"[{k}/{len(todo)}] {p['id']} s{j['seed']} {res.plan:28s} {res.gpu_seconds:6.2f}s {res.peak_mem_gb:5.2f}GB")
        except Exception as e:  # log, don't crash the whole run
            row["notes"] = f"error: {type(e).__name__}: {e}"[:300]
            n_err += 1
            print(f"[{k}/{len(todo)}] {p['id']} s{j['seed']} ERROR {e}")
            traceback.print_exc(limit=2)
            if not args.dry_run and "out of memory" in str(e).lower():
                from rerouter.models import free_cuda
                free_cuda()
        utils.append_row(out_csv, row)

    print("\nSummary")
    print(f"  wrote {n_ok} images and {n_err} error rows to {out_csv} in {(time.time() - t_start) / 60:.1f} min")
    print(f"  images under {images_dir}/<method>/<setting>/<prompt_id>_s<seed>.png")
    print("  next: python bench/score.py  then  python bench/plot.py")


if __name__ == "__main__":
    main()
