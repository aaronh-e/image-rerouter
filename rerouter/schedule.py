"""Turn (direction, switch_fraction, steps) into a concrete plan of which model
runs which steps. Pure Python, no torch, so it is unit-tested on CPU.

Vocabulary
----------
direction:
    "s2l"   Method 1, small first then large   (T-Stitch order)
    "l2s"   Method 2, large first then small   (Hybrid SD order)
    "large" control: large model for all steps
    "small" control: small model for all steps
switch_fraction:
    fraction of the TOTAL steps run by the FIRST model.
    s2l @ 0.0 == large alone;  s2l @ 1.0 == small alone (and vice versa for l2s).

A plan is a list of (model_name, first_step_index). Two entries for a plain
handoff; three or more for a "sandwich" (stretch idea), e.g.
    [("large", 0), ("small", 12), ("large", 24)].
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    model: str      # "large" | "small" | "tiny"
    start: int      # first step index this model runs (inclusive)
    end: int        # last step index + 1 (exclusive)

    @property
    def n_steps(self) -> int:
        return self.end - self.start


def steps_for_first(switch_fraction: float, steps: int) -> int:
    """How many steps the first model runs. Rounded to the nearest whole step,
    so 0.4 * 30 = 12 and 0.2 * 30 = 6; at 0.33 * 30 you get 10, not 9.9."""
    if not 0.0 <= switch_fraction <= 1.0:
        raise ValueError(f"switch_fraction must be in [0,1], got {switch_fraction}")
    return int(round(switch_fraction * steps))


def plan(direction: str, switch_fraction: float, steps: int) -> list[Segment]:
    """Build the step plan for one image. Collapses degenerate handoffs
    (0 steps for one model) into a single segment so the swap never fires."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if direction == "large":
        return [Segment("large", 0, steps)]
    if direction == "small":
        return [Segment("small", 0, steps)]
    if direction not in ("s2l", "l2s"):
        raise ValueError(f"unknown direction {direction!r}")

    first, second = ("small", "large") if direction == "s2l" else ("large", "small")
    n_first = steps_for_first(switch_fraction, steps)
    if n_first == 0:
        return [Segment(second, 0, steps)]
    if n_first == steps:
        return [Segment(first, 0, steps)]
    return [Segment(first, 0, n_first), Segment(second, n_first, steps)]


def plan_from_fractions(models: list[str], boundaries: list[float], steps: int) -> list[Segment]:
    """General form for sandwiches: models=["large","small","large"],
    boundaries=[0.4, 0.8] means large for steps [0,12), small [12,24), large [24,30)."""
    if len(boundaries) != len(models) - 1:
        raise ValueError("need one boundary fewer than models")
    cuts = [0] + [int(round(b * steps)) for b in boundaries] + [steps]
    segs = []
    for m, a, b in zip(models, cuts[:-1], cuts[1:]):
        if b > a:
            if segs and segs[-1].model == m:          # merge adjacent same-model segments
                segs[-1] = Segment(m, segs[-1].start, b)
            else:
                segs.append(Segment(m, a, b))
    return segs


def describe(segments: list[Segment]) -> str:
    return " -> ".join(f"{s.model}[{s.start}:{s.end}]" for s in segments)
