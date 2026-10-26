"""Methods 1 and 2: run the first N steps with one UNet and the rest with another.

Two implementations, selected by config `handoff.impl`:

  "swap"    One pipeline. A `callback_on_step_end` replaces `pipe.unet` after the
            last step of the first segment. Zero overhead: the latent never leaves
            the loop, and the scheduler state is untouched, so step i of the second
            model sees exactly the noise level it would have seen in a normal run.
            Requires shared text encoders + VAE (true for SDXL/SSD-1B/Vega).

  "twopipe" Model A runs with denoising_end=f and returns the latent; model B's
            img2img pipeline continues with denoising_start=f. Works for any pair
            that shares a latent format, even with different text encoders. Costs
            a second prompt encode. Mirrors Option A in RESEARCH_PLAN.md.

Both return the image plus timing/memory measured the same way, so rows are
comparable across implementations.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from .schedule import Segment, plan, describe


@dataclass
class GenResult:
    image: object                 # PIL.Image
    gpu_seconds: float
    peak_mem_gb: float
    first_model: str
    second_model: str
    n_first_steps: int
    plan: str
    impl: str

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("image")
        return d


def _sync(device: str):
    """GPU work is asynchronous; wait for it so the wall clock is honest."""
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _timer_start(device: str = "cuda"):
    import torch
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def _timer_stop(t0: float, device: str = "cuda"):
    import torch
    _sync(device)
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        peak = torch.mps.driver_allocated_memory() / 1024 ** 3
    else:
        peak = 0.0
    return time.perf_counter() - t0, peak


def generate(bank, prompt: str, seed: int, direction: str, switch_fraction: float,
             steps: int, cfg: dict, segments: list[Segment] | None = None) -> GenResult:
    """Generate one image. `segments` overrides (direction, switch_fraction) for
    sandwich schedules; otherwise the plan is derived from them."""
    import torch

    gen_cfg = cfg["generation"]
    segs = segments or plan(direction, switch_fraction, steps)
    impl = cfg.get("handoff", {}).get("impl", "swap")
    if impl == "twopipe" and len(segs) > 2:
        raise ValueError("twopipe only supports a single handoff; use impl=swap for sandwiches")

    common = dict(
        prompt=prompt,
        negative_prompt=gen_cfg.get("negative_prompt") or None,
        num_inference_steps=steps,
        guidance_scale=gen_cfg["guidance_scale"],
        width=gen_cfg["width"], height=gen_cfg["height"],
    )
    # CPU generator everywhere: identical starting noise on any device, and MPS has no generator of its own.
    g = torch.Generator(device="cpu").manual_seed(seed)

    if impl == "swap" or len(segs) == 1:
        image, secs, peak = _generate_swap(bank, segs, common, g)
    elif impl == "twopipe":
        image, secs, peak = _generate_twopipe(bank, segs, common, g, steps)
    else:
        raise ValueError(f"unknown handoff impl {impl!r}")

    first, second = segs[0].model, (segs[1].model if len(segs) > 1 else segs[0].model)
    return GenResult(image=image, gpu_seconds=secs, peak_mem_gb=peak, first_model=first,
                     second_model=second, n_first_steps=segs[0].n_steps, plan=describe(segs), impl=impl)


def _generate_swap(bank, segs, common, g):
    """One pipeline run; the callback swaps UNets at segment boundaries."""
    # step index -> model that should run the NEXT step
    boundaries = {s.start: s.model for s in segs[1:]}

    def on_step_end(pipe, i, t, kwargs):
        # diffusers calls this after step i finishes; step i+1 is next.
        nxt = boundaries.get(i + 1)
        if nxt is not None:
            pipe.unet = bank.unets[nxt]
        return kwargs

    bank.set_unet(segs[0].model)
    try:
        t0 = _timer_start(bank.device)
        out = bank.pipe(**common, generator=g, callback_on_step_end=on_step_end)
        secs, peak = _timer_stop(t0, bank.device)
    finally:
        bank.restore()
    return out.images[0], secs, peak


def handoff_fraction(timesteps, k: int, num_train_timesteps: int) -> float:
    """The denoising_end/denoising_start value that gives model A exactly the first
    k steps. diffusers keeps steps with t >= round(T - f*T) for model A, and k/steps
    is NOT that: with SDXL's "leading" spacing, 12/30 = 0.4 cuts at t=600 and model A
    runs 11 steps (958, 925, ..., 628), not 12. Cutting at model A's last timestep
    t[k-1] instead gives A steps 0..k-1 and B steps k..N-1, same as the swap path."""
    return 1.0 - float(timesteps[k - 1]) / num_train_timesteps


def _generate_twopipe(bank, segs, common, g, steps):
    """Model A to a latent, model B from that latent (denoising_end / denoising_start)."""
    sched = type(bank.pipe.scheduler).from_config(bank.pipe.scheduler.config)   # a copy: leave the pipe's state alone
    sched.set_timesteps(steps)
    f = handoff_fraction(sched.timesteps, segs[0].end, sched.config.num_train_timesteps)
    img2img = bank.get_img2img()
    t0 = _timer_start(bank.device)
    bank.set_unet(segs[0].model)
    try:
        lat = bank.pipe(**common, generator=g, denoising_end=f, output_type="latent").images
        img2img.unet = bank.unets[segs[1].model]
        out = img2img(prompt=common["prompt"], negative_prompt=common["negative_prompt"],
                      image=lat, denoising_start=f, num_inference_steps=steps,
                      guidance_scale=common["guidance_scale"], generator=g)
    finally:
        bank.restore()
    secs, peak = _timer_stop(t0, bank.device)
    return out.images[0], secs, peak


def warmup(bank, cfg: dict, n: int = 1) -> None:
    """CUDA kernels, cuDNN autotuning and the allocator all warm up on the first
    call; the fairness rule says never time that run."""
    for _ in range(n):
        for tier in bank.unets:
            # explicit segments: schedule.plan() has no direction for a third tier like "tiny"
            generate(bank, "a photo of a cat", 0, tier, 0.0, 4, cfg, segments=[Segment(tier, 0, 4)])
