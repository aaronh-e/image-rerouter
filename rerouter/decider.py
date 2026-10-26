"""Level 4: a per-prompt decider built on Jev (TypeSafe AI).

Jev reads text and answers *typed* questions with probabilities. It never sees
an image. We ask it a handful of questions about the prompt, bucket the answers
into a "profile", and look the profile up in a policy table that maps
profile -> (direction, switch_fraction). The policy table is learned from the
Level 3 sweep on the 70% training split (bench/decider_eval.py), so Jev only has
to describe the prompt; it never has to know anything about diffusion.

API (verified against TypeSafe's public docs, Oct 2026):
    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    {"model": "jev-latest", "state": <str|obj>, "questions": {id: {type, instructions, criteria?}}}
    -> {"model": ..., "answers": {id: {"type": "noul", "noul": p}
                                   | {"type": "choice", "choice", "probabilities", "confidence"}
                                   | {"type": "score", "score", "probabilities", ...}}, "usage": ...}
Question types: "noul" (yes/no probability), "choice" (pick one), "score" (ordered rubric).
AIMLAPI mirrors it at POST https://api.aimlapi.com/v1/decisions with model "typesafe/jev".
(The plan mentions an OpenRouter id `typesafe/jev-router`; no such listing was found
in Oct 2026, so the direct API is the default.)

Set decider.provider: heuristic in config.yaml to run the whole Level 4 pipeline
offline with keyword rules (useful for testing, and as a baseline: does Jev beat
a regex?).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# ------------------------------------------------------------ questions ----
# Written against the Level 3 predictions: text/hands form late (so the LARGE
# model should finish), composition forms early (so the LARGE model should start).
QUESTIONS = {
    "has_readable_text": {
        "type": "noul",
        "instructions": "Does `prompt` ask for specific readable text, letters, numbers, a sign, a label or a logo with words to appear in the image?",
        "criteria": {"true": "A quoted or specific string, word, number or sign text must be legible in the image",
                     "false": "No specific text needs to be legible"},
    },
    "needs_counting_or_layout": {
        "type": "noul",
        "instructions": "Does `prompt` require an exact number of objects, or exact spatial relations (left of, above, between, inside) between objects?",
    },
    "closeup_hands_or_faces": {
        "type": "noul",
        "instructions": "Is the main subject of `prompt` a close-up of human hands, fingers or a human face where anatomy must look correct?",
    },
    "style": {
        "type": "choice",
        "instructions": "What rendering style does `prompt` ask for?",
        "criteria": {"photoreal": "A photograph, product shot, realistic render or anything that must look like a real photo",
                     "stylized": "Painting, illustration, anime, pixel art, watercolor, 3D cartoon, any non-photographic art style",
                     "unspecified": "No style is implied"},
    },
    "detail_density": {
        "type": "score",
        "instructions": "How many distinct objects, attributes and fine details does `prompt` demand?",
        "criteria": ["One subject, few details", "A few subjects or several attributes", "Many subjects with many specific attributes each"],
    },
}


def profile_from_answers(ans: dict, thr: float = 0.5) -> str:
    """A profile is a short label made from the thresholded answers. Keep it coarse:
    with ~40 training prompts, more than ~6 cells would overfit."""
    if ans["has_readable_text"] >= thr:
        return "text"
    if ans["closeup_hands_or_faces"] >= thr:
        return "hands_faces"
    if ans["needs_counting_or_layout"] >= thr:
        return "layout"
    if ans.get("style") == "stylized":
        return "stylized"
    if float(ans.get("detail_density", 0)) >= 1.5:
        return "dense"
    return "generic"


# Hand-written starting policy, straight from the research plan's predictions.
# bench/decider_eval.py overwrites this with a learned one (results/decider_policy.json).
DEFAULT_POLICY = {
    "text":        {"direction": "s2l", "switch_fraction": 0.4},   # large finishes -> legible text
    "hands_faces": {"direction": "s2l", "switch_fraction": 0.4},   # large finishes -> anatomy
    "layout":      {"direction": "l2s", "switch_fraction": 0.4},   # large starts  -> composition
    "stylized":    {"direction": "l2s", "switch_fraction": 0.2},   # style is forgiving -> cheapest
    "dense":       {"direction": "l2s", "switch_fraction": 0.6},
    "generic":     {"direction": "l2s", "switch_fraction": 0.4},
}


def route(answers: dict, policy: dict | None = None) -> dict:
    """Jev answers -> {"direction", "switch_fraction", "profile"}."""
    policy = policy or DEFAULT_POLICY
    prof = profile_from_answers(answers)
    choice = policy.get(prof) or policy.get("generic") or DEFAULT_POLICY["generic"]
    out = {"direction": choice["direction"], "switch_fraction": float(choice["switch_fraction"]), "profile": prof}
    if "steps" in choice:            # a learned policy may pick a reduced-step control, e.g. large@18
        out["steps"] = int(choice["steps"])
    return out


# ------------------------------------------------------------- providers ----

def _flatten(raw_answers: dict) -> dict:
    """Pull the one number we care about out of each typed answer."""
    out = {}
    for qid, a in raw_answers.items():
        t = a.get("type")
        if t == "noul":
            out[qid] = float(a["noul"])
        elif t == "choice":
            out[qid] = a["choice"]
            out[qid + "_probs"] = a.get("probabilities", {})
        elif t == "score":
            out[qid] = float(a["score"])
    return out


def ask_jev(prompt: str, cfg: dict, retries: int = 4) -> dict:
    """One API call per prompt (all questions share the same state). ~100-500 ms."""
    provider = cfg["decider"].get("provider", "typesafe")
    if provider == "heuristic":
        return heuristic_answers(prompt)
    import requests
    if provider == "typesafe":
        url, key_env, model = "https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", cfg["decider"].get("model", "jev-latest")
    elif provider == "aimlapi":
        url, key_env, model = "https://api.aimlapi.com/v1/decisions", "AIMLAPI_KEY", cfg["decider"].get("model", "typesafe/jev")
    else:
        raise ValueError(f"unknown decider provider {provider!r}")
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"set {key_env} in the environment (or decider.provider: heuristic)")

    body = {"model": model, "state": {"prompt": prompt}, "questions": QUESTIONS}
    delay = 1.0
    for attempt in range(retries):
        t0 = time.perf_counter()
        r = requests.post(url, json=body, timeout=30,
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        if r.status_code in (429, 529) and attempt < retries - 1:
            time.sleep(delay); delay *= 2; continue
        r.raise_for_status()
        data = r.json()
        ans = _flatten(data["answers"])
        ans["_latency_s"] = time.perf_counter() - t0
        ans["_model"] = data.get("model", model)
        return ans
    raise RuntimeError("Jev request kept failing")


# A regex baseline. If Jev can't beat this, the "decider" adds nothing.
_TEXT_RE = re.compile(r"\b(says?|reads?|reading|text|word|letters?|sign|label|logo|caption|headline|written|spell(?:ing|s)?|typography|number plate|license plate)\b|\"[^\"]+\"|'[A-Z0-9][^']*'", re.I)
_LAYOUT_RE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+\w+|\b(left of|right of|above|below|on top of|next to|between|behind|in front of|beside|under)\b", re.I)
_HANDS_RE = re.compile(r"\b(hands?|fingers?|palms?|fist|face|portrait|close[- ]?up|eyes?|smile|wrinkles|headshot)\b", re.I)
_STYLE_RE = re.compile(r"\b(painting|illustration|anime|pixel art|watercolor|watercolour|oil on canvas|ink|sketch|cartoon|woodcut|ukiyo-e|art nouveau|low[- ]poly|vector|comic|charcoal|gouache|linocut|stained glass|isometric)\b", re.I)
_PHOTO_RE = re.compile(r"\b(photo|photograph|photoreal|realistic|dslr|35mm|studio|product shot|8k|macro)\b", re.I)


def heuristic_answers(prompt: str) -> dict:
    words = len(prompt.split())
    commas = prompt.count(",")
    density = 0.0 if words < 12 else (1.0 if words < 30 else 2.0)
    if commas >= 4:
        density = max(density, 1.5)
    return {
        "has_readable_text": 0.9 if _TEXT_RE.search(prompt) else 0.1,
        "needs_counting_or_layout": 0.9 if _LAYOUT_RE.search(prompt) else 0.1,
        "closeup_hands_or_faces": 0.9 if _HANDS_RE.search(prompt) else 0.1,
        "style": "stylized" if _STYLE_RE.search(prompt) else ("photoreal" if _PHOTO_RE.search(prompt) else "unspecified"),
        "detail_density": density,
        "_latency_s": 0.0, "_model": "heuristic",
    }


# ------------------------------------------------------------------ cache ----

def answers_for_prompts(prompts: list[dict], cfg: dict, cache_path: str | Path) -> dict[str, dict]:
    """Ask once per prompt id and cache to JSON so re-runs cost nothing."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    provider = cfg["decider"].get("provider", "typesafe")
    changed = False
    for p in prompts:
        key = f"{provider}:{p['id']}"
        if key in cache:
            continue
        cache[key] = ask_jev(p["prompt"], cfg)
        changed = True
    if changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1))
    return {p["id"]: cache[f"{provider}:{p['id']}"] for p in prompts}


def load_policy(path: str | Path | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return DEFAULT_POLICY
