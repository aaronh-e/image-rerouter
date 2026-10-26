#!/usr/bin/env python
"""Level 1 on your own computer: large alone, small alone, and one handoff in
each direction, on the same prompt and seed. Picks the GPU automatically
(NVIDIA -> Apple Silicon -> CPU).

    python scripts/level1_local.py                    # defaults: 768px, 20 steps
    python scripts/level1_local.py --size 512 --steps 12   # fastest smoke test
    python scripts/level1_local.py --size 1024 --steps 30  # the real setting (needs a GPU)

Memory: both UNets must be loaded at once (~8 GB of weights at fp16, ~16 GB
at fp32 on CPU). On an Apple Silicon Mac you want 16 GB of unified memory or
more; on a plain CPU expect several minutes per image even at 512px.

Outputs: results/level1_local/*.png and results/level1_local/timing.csv.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerouter import utils  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=768, help="image width and height (SDXL is trained at 1024)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--switch", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default='a storefront sign that says "OPEN 24 HOURS", photo')
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--skip-warmup", action="store_true", help="save time; the first timing will be inflated")
    args = ap.parse_args()

    cfg = copy.deepcopy(utils.load_config())
    cfg["generation"]["width"] = cfg["generation"]["height"] = args.size
    cfg["compute"]["attention_slicing"] = True     # safer on small GPUs / Macs; costs a little speed

    from rerouter.models import ModelBank
    from rerouter.handoff import generate, warmup

    print("loading SDXL + SSD-1B (first time downloads ~10 GB)...")
    t0 = time.time()
    bank = ModelBank(cfg, device=args.device).load(("large", "small"))
    print(f"device={bank.device} dtype={bank.dtype} loaded in {time.time() - t0:.0f}s; UNet params (B): {bank.param_counts()}")
    if not args.skip_warmup:
        print("warm-up run (not timed)...")
        warmup(bank, cfg)

    out = Path("results/level1_local"); out.mkdir(parents=True, exist_ok=True)
    runs = [("large_alone", "large", 0.0), ("small_alone", "small", 0.0),
            (f"method1_s2l_{args.switch:.1f}", "s2l", args.switch), (f"method2_l2s_{args.switch:.1f}", "l2s", args.switch)]
    rows = []
    for name, direction, f in runs:
        print(f"generating {name} ...", flush=True)
        r = generate(bank, args.prompt, args.seed, direction, f, args.steps, cfg)
        r.image.save(out / f"{name}.png")
        rows.append(dict(run=name, plan=r.plan, gpu_seconds=round(r.gpu_seconds, 2), peak_mem_gb=round(r.peak_mem_gb, 2)))
        print(f"   {r.plan:28s} {r.gpu_seconds:7.1f} s")

    import csv
    with open(out / "timing.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    base = rows[0]["gpu_seconds"]
    print("\nSummary")
    print(f"  device: {bank.gpu_name}, {args.size}x{args.size}, {args.steps} steps, seed {args.seed}")
    for r in rows:
        print(f"  {r['run']:22s} {r['gpu_seconds']:7.1f} s   ({1 - r['gpu_seconds'] / base:+.0%} vs large alone)")
    print(f"  images and timing.csv -> {out}/")
    print("  Level 1 gate: both handoff images look clean, and both are faster than large alone.")


if __name__ == "__main__":
    main()
