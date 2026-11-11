#!/usr/bin/env python
"""A small website for playing with the handoff by hand.

    python app/app.py                 # opens http://127.0.0.1:7860 on this machine
    python app/app.py --share         # also prints a public *.gradio.live link (use this on Colab)
    python app/app.py --demo          # no GPU: fake images + estimated timings, to see the UI

Type a prompt, pick Method 1 or 2, drag the switch slider, press Generate. The
page shows the image(s) with GPU-seconds, and saves everything under
results/app/ so nothing is lost. The Sweep tab runs one prompt across many
switch points in both directions at once.

Everything here calls the same rerouter.handoff.generate() the benchmark uses.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from rerouter import utils  # noqa: E402
from rerouter.schedule import plan, describe  # noqa: E402

BASE_CFG = utils.load_config()
OUT = Path("results/app")
STATE = {"bank": None, "demo": False, "device": "auto", "scorers": None}
SCORE_HEADERS = ["VQAScore", "PickScore", "CLIPScore", "OCR text", "count"]
GENEVAL_FILE = Path(__file__).resolve().parents[1] / "prompts" / "geneval_metadata.jsonl"
GENEVAL_TASKS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]
# Gradio draws a lone image as one full-width square, taller than the gallery box,
# so you had to scroll. Cap every grid image so each one is always seen whole.
CSS = ".fit-gallery .grid-wrap .thumbnail-item { max-height: 460px; }"

METHODS = {
    "Method 1: small first, large finishes (s2l)": "s2l",
    "Method 2: large first, small finishes (l2s)": "l2s",
    "Large model only": "large",
    "Small model only": "small",
}


# ----------------------------------------------------------------- engine --

def get_bank(progress=None):
    """Load the two models once, the first time anyone presses Generate."""
    if STATE["bank"] is not None:
        return STATE["bank"]
    if STATE["demo"]:
        from bench.run import DryRunBank
        STATE["bank"] = DryRunBank()
        return STATE["bank"]
    from rerouter.models import ModelBank
    from rerouter.handoff import warmup
    if progress:
        progress(0.05, desc="Loading SDXL + SSD-1B (first time downloads ~10 GB)...")
    bank = ModelBank(BASE_CFG, device=STATE["device"]).load(("large", "small"))
    if progress:
        progress(0.6, desc="Warm-up run (not timed)...")
    warmup(bank, BASE_CFG)
    STATE["bank"] = bank
    return bank


def gen(bank, prompt, seed, direction, switch, steps, cfg):
    if STATE["demo"]:
        return bank.generate(prompt, seed, direction, switch, steps, cfg)
    from rerouter.handoff import generate
    return generate(bank, prompt, seed, direction, switch, steps, cfg)


def usd_per_1000(secs: float, gpu_name: str) -> float | str:
    """Same formula as bench/run.py: gpu_seconds x hourly price / 3600 x 1000. Blank
    when config.yaml has no price for this GPU, rather than a T4's price."""
    price = utils.usd_per_hour(BASE_CFG, gpu_name)
    return "" if price is None else round(secs * price / 3600 * 1000, 3)


NUMBER_WORDS = {w: i for i, w in enumerate("one two three four five six seven eight nine ten".split(), 1)}
# words that end the object name: "three red cups | to the left of ..."
OBJECT_END = {"to", "on", "in", "next", "below", "above", "behind", "under", "near", "beside", "with", "and", "at",
              "of", "around", "between", "from", "by", "for", "over", "into", "onto", "inside", "against", "tied"}


def singular(word: str) -> str:
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def infer_checks(prompt: str):
    """Fill the OCR and count boxes from the prompt, so they rarely need typing.
    Text in quotes -> expected text. A number (word or digit) and the words after it,
    up to a preposition or comma -> object to count, made singular. Quoted text is
    ignored when looking for the number ('ROOM 101' is not a count). Returns
    (expected_text, count_object, count_n); blanks mean 'skip that check'."""
    import re
    quoted = re.search(r'["“](.+?)["”]', prompt)
    text = quoted.group(1).strip() if quoted else ""
    words = re.sub(r'["“].+?["”]', " ", prompt).lower().replace(",", " , ").replace(".", " . ").split()
    for i, w in enumerate(words):
        n = NUMBER_WORDS.get(w) or (int(w) if w.isdigit() else 0)
        if not n:
            continue
        obj = []
        for x in words[i + 1:i + 4]:   # object names are at most 3 words ("red fire hydrant")
            if x in OBJECT_END or x in ",." or x.endswith("ing"):   # "-ing" = a verb: "dogs sitting on"
                break
            obj.append(x)
        if obj:
            obj[-1] = singular(obj[-1]) if n > 1 else obj[-1]
            return text, " ".join(obj), n
    return text, "", 0


def quality(name, prompt, expected_text, count_object, count_n):
    """The benchmark's own scorers (bench/score.py) on one saved image, run after
    the timer stops. Blank where a check wasn't asked for or its package is missing."""
    if STATE["scorers"] is None:
        from bench.score import Scorers
        STATE["scorers"] = Scorers(BASE_CFG, fake=STATE["demo"])
    sc, path = STATE["scorers"], str(OUT / name)
    na = "unavailable"   # the scorer failed to load; scorer_note() says why
    vqa, pick, clip = sc.vqa_score(path, prompt), sc.pick_score(path, prompt), sc.clip_score(path, prompt)
    ocr = count = ""
    if expected_text.strip():
        m, cer, _ = sc.ocr(path, expected_text.strip())
        ocr = na if m is None else f"{'yes' if m else 'no'} (CER {cer:.2f})"
    if count_object.strip() and count_n:
        _, det = sc.count(path, count_object.strip(), int(count_n))
        count = na if det is None else f"{det}/{int(count_n)}"
    scores = [na if vqa is None else vqa, na if pick is None else pick, na if clip is None else clip, ocr, count]
    append_row(OUT / "scores.csv",
               ["image", "prompt", "vqa_score", "pick_score", "clip_score", "ocr", "expected_text", "count", "count_object"],
               [name, prompt, *scores[:4], expected_text.strip(), count, count_object.strip()])
    return scores


def append_row(path: Path, header: list, row: list) -> None:
    """Append to a csv log. A log written with older columns is renamed aside, not mixed."""
    if path.exists():
        with open(path, newline="") as f:
            old = next(csv.reader(f), None)
        if old != header:
            path.rename(path.with_name(f"{path.stem}.{dt.datetime.now():%Y%m%d_%H%M%S}.old.csv"))
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def scoring_room():
    """VQAScore (~7 GB) does not fit beside SDXL + SSD-1B on a small GPU, so park the
    image models on CPU while scoring. Returns a function that brings them back."""
    bank = STATE["bank"]
    if STATE["demo"] or bank is None or getattr(bank, "device", "cpu") != "cuda":
        return lambda: None
    import torch
    if torch.cuda.get_device_properties(0).total_memory >= 24e9:
        return lambda: None
    bank.move("cpu")
    return lambda: bank.move("cuda")


def score_all(names, prompt, expected_text, count_object, count_n, progress, labels):
    """Score a job's images only after ALL of them are generated, then park the scorers
    off the GPU, so they never sit beside SDXL during generation (T4 OOM)."""
    restore = scoring_room()
    try:
        out = []
        for name, label in zip(names, labels):
            progress(1.0, desc=f"Scoring: {label}")
            out.append(quality(name, prompt, expected_text, count_object, count_n))
        return out
    finally:
        if STATE["scorers"] is not None:
            STATE["scorers"].park()
        restore()


def scorer_note() -> str:
    off = (STATE["scorers"].disabled if STATE["scorers"] else {})
    return "".join(f"\n\n**{k} unavailable:** `{v[:200]}`" for k, v in off.items())


def ints(steps, seed):
    if seed is None or steps is None:
        raise gr.Error("Seed and total steps need a number.")
    return int(steps), int(seed)


def cfg_for(size: int, guidance: float) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    cfg["generation"]["width"] = cfg["generation"]["height"] = int(size)
    cfg["generation"]["guidance_scale"] = float(guidance)
    return cfg


def save(image, prompt, seed, direction, switch, steps, res):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]   # ms: same-second runs don't overwrite
    name = f"{stamp}_{direction}_sw{switch:.2f}_st{steps}_s{seed}.png"
    image.save(OUT / name)
    new = not (OUT / "log.csv").exists()
    with open(OUT / "log.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "prompt", "seed", "direction", "switch_fraction", "steps", "plan", "gpu_seconds", "peak_mem_gb", "image"])
        w.writerow([stamp, prompt, seed, direction, f"{switch:.2f}", steps, res.plan, f"{res.gpu_seconds:.2f}", f"{res.peak_mem_gb:.2f}", name])
    return name


# ------------------------------------------------------------------ tabs ---

def run_single(prompt, seed, steps, size, guidance, methods, switch,
               score, expected_text, count_object, count_n, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    if not methods:
        raise gr.Error("Tick at least one method.")
    cfg = cfg_for(size, guidance)
    steps, seed = ints(steps, seed)
    bank = get_bank(progress)
    # always run in the METHODS order, whatever order the boxes were ticked in
    jobs = [(m, d, float(switch) if d in ("s2l", "l2s") else 0.0) for m, d in METHODS.items() if m in methods]
    gallery, rows = [], []
    for i, (label, d, f) in enumerate(jobs):
        progress((i + 1) / (len(jobs) + 1), desc=f"Generating: {label}")
        res = gen(bank, prompt, seed, d, f, steps, cfg)
        name = save(res.image, prompt, seed, d, f, steps, res)
        gallery.append((res.image, f"{label}  |  {res.gpu_seconds:.1f} s  |  {res.plan}"))
        rows.append([label, res.plan, round(res.gpu_seconds, 2), round(res.peak_mem_gb, 2), usd_per_1000(res.gpu_seconds, bank.gpu_name), name])
    scores = (score_all([r[5] for r in rows], prompt, expected_text, count_object, count_n, progress, [r[0] for r in rows])
              if score else [[""] * len(SCORE_HEADERS)] * len(rows))
    base = next((r[2] for r in rows if r[0] == "Large model only"), None)
    if base:
        for r in rows:
            r.append(f"{1 - r[2] / base:+.0%}")
    else:
        for r in rows:
            r.append("")
    for r, s in zip(rows, scores):
        r.extend(s)
    note = f"Saved to {OUT}/ (log.csv has every run). Device: {bank.gpu_name}."
    if STATE["demo"]:
        note = "DEMO MODE: these are noise images, estimated timings and random scores. Run without --demo on a GPU for real ones."
    return gallery, rows, note + scorer_note()


def run_sweep(prompt, seed, steps, size, guidance, fractions, both,
              score, expected_text, count_object, count_n, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Type a prompt first.")
    try:
        fr = sorted({round(float(x), 2) for x in fractions.replace(";", ",").split(",") if x.strip()})
    except ValueError:
        raise gr.Error("Switch points must be numbers between 0 and 1, separated by commas, e.g. 0.2, 0.4, 0.6")
    if any(not 0 <= f <= 1 for f in fr):
        raise gr.Error("Switch points must be between 0 and 1.")
    cfg = cfg_for(size, guidance)
    steps, seed = ints(steps, seed)
    bank = get_bank(progress)
    jobs = [("large", 0.0), ("small", 0.0)]
    for d in (("s2l", "l2s") if both else ("s2l",)):
        jobs += [(d, f) for f in fr if 0 < f < 1]
    gallery, rows, names = [], [], []
    for i, (d, f) in enumerate(jobs):
        progress((i + 1) / (len(jobs) + 1), desc=f"{d} @ {f:.2f}")
        res = gen(bank, prompt, seed, d, f, steps, cfg)
        names.append(save(res.image, prompt, seed, d, f, steps, res))
        label = {"large": "large only", "small": "small only"}.get(d, f"{d} @ {f:.2f}")
        gallery.append((res.image, f"{label}  |  {res.gpu_seconds:.1f} s"))
        rows.append([label, res.plan, round(res.gpu_seconds, 2), round(res.peak_mem_gb, 2), usd_per_1000(res.gpu_seconds, bank.gpu_name)])
    scores = (score_all(names, prompt, expected_text, count_object, count_n, progress, [r[0] for r in rows])
              if score else [[""] * len(SCORE_HEADERS)] * len(rows))
    base = rows[0][2]
    for r, s in zip(rows, scores):
        r.append(f"{1 - r[2] / base:+.0%}")
        r.extend(s)
    note = f"{len(rows)} images saved to {OUT}/." + (" DEMO MODE: fake images and scores." if STATE["demo"] else "")
    return gallery, rows, note + scorer_note()


def geneval_prompts(per_task: int) -> list[dict]:
    """The first `per_task` GenEval prompts of each task, in GenEval's file order."""
    with open(GENEVAL_FILE, encoding="utf-8") as f:
        metas = [json.loads(l) for l in f if l.strip()]
    return [m for t in GENEVAL_TASKS for m in [m for m in metas if m["tag"] == t][:per_task]]


def run_geneval(seed, steps, size, guidance, methods, switch, per_task, progress=gr.Progress()):
    if not methods:
        raise gr.Error("Tick at least one method.")
    if not per_task or per_task < 1:
        raise gr.Error("Prompts per task needs to be at least 1.")
    cfg = cfg_for(size, guidance)
    steps, seed = ints(steps, seed)
    bank = get_bank(progress)
    runs = [(m, d, float(switch) if d in ("s2l", "l2s") else 0.0) for m, d in METHODS.items() if m in methods]
    jobs = [(meta, run) for meta in geneval_prompts(int(per_task)) for run in runs]
    gallery, done = [], []
    for i, (meta, (label, d, f)) in enumerate(jobs):
        progress((i + 1) / (len(jobs) + 1), desc=f"Generating {i + 1}/{len(jobs)}: {label}")
        res = gen(bank, meta["prompt"], seed, d, f, steps, cfg)
        done.append((meta, label, res, save(res.image, meta["prompt"], seed, d, f, steps, res)))
    # score only after every image exists, then park (same rule as score_all)
    if STATE["scorers"] is None:
        from bench.score import Scorers
        STATE["scorers"] = Scorers(BASE_CFG, fake=STATE["demo"])
    rows = []
    try:
        for i, (meta, label, res, name) in enumerate(done):
            progress(1.0, desc=f"GenEval check {i + 1}/{len(done)}")
            ok, why = STATE["scorers"].geneval(str(OUT / name), meta)
            verdict = "unavailable" if ok is None else "pass" if ok else "fail"
            gallery.append((res.image, f"{verdict}  |  {label}  |  {meta['prompt']}"))
            rows.append([meta["tag"], meta["prompt"], label, round(res.gpu_seconds, 2), round(res.peak_mem_gb, 2),
                         usd_per_1000(res.gpu_seconds, bank.gpu_name), verdict, why])
            append_row(OUT / "geneval.csv", ["image", "task", "prompt", "run", "gpu_seconds", "peak_mem_gb", "result", "reason"],
                       [name, meta["tag"], meta["prompt"], label, f"{res.gpu_seconds:.2f}", f"{res.peak_mem_gb:.2f}", verdict, why])
    finally:
        STATE["scorers"].park()
    return gallery, geneval_summary(rows, [r[0] for r in runs], bank.gpu_name), rows, (
        f"{len(rows)} images; per-image results in {OUT}/geneval.csv."
        + (" DEMO MODE: noise images, random pass/fail." if STATE["demo"] else "") + scorer_note())


def geneval_summary(rows, labels, gpu_name):
    """One row per method: cost, then GenEval score = mean of the per-task pass rates
    (GenEval's own overall number), then each task's pass rate."""
    out = []
    for label in labels:
        mine = [r for r in rows if r[2] == label and r[6] != "unavailable"]
        allmine = [r for r in rows if r[2] == label]
        secs = sum(r[3] for r in allmine) / len(allmine)
        rates = {}
        for t in GENEVAL_TASKS:
            got = [r[6] == "pass" for r in mine if r[0] == t]
            if got:
                rates[t] = sum(got) / len(got)
        out.append([label, len(allmine), round(secs, 2), max(r[4] for r in allmine), usd_per_1000(secs, gpu_name),
                    round(sum(rates.values()) / len(rates), 3) if rates else "unavailable",
                    *[f"{rates[t]:.0%}" if t in rates else "" for t in GENEVAL_TASKS]])
    base = next((r[2] for r in out if r[0] == "Large model only"), None)
    for r in out:
        r.insert(5, f"{1 - r[2] / base:+.0%}" if base else "")
    return out


def explain(methods, switch, steps):
    lines = []
    for m, d in METHODS.items():
        if m not in (methods or []):
            continue
        segs = plan(d, float(switch), int(steps))
        if len(segs) == 1:
            lines.append(f"- {m}: all {int(steps)} steps on the **{segs[0].model}** model.")
        else:
            a, b = segs
            lines.append(f"- {m}: steps 1 to {a.end} ({a.n_steps}) on the **{a.model}** model, "
                         f"then {b.start + 1} to {b.end} ({b.n_steps}) on the **{b.model}** model. Plan: `{describe(segs)}`")
    return "\n".join(lines) or "Tick at least one method."


# -------------------------------------------------------------------- ui ----

def build():
    with gr.Blocks(title="Image Re-router") as demo:
        gr.Markdown("# Image Re-router\nSplit one image between a small and a large diffusion model and see what it costs. "
                    "Same code as the benchmark (`rerouter/handoff.py`).")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value='a storefront sign that says "OPEN 24 HOURS", photo', scale=4, lines=2)
            seed = gr.Number(label="Seed", value=42, precision=0, scale=1)
        with gr.Row():
            steps = gr.Slider(4, 50, value=BASE_CFG["generation"]["steps"], step=1, label="Total steps")
            size = gr.Dropdown([512, 768, 1024], value=1024, label="Size (px)")
            guidance = gr.Slider(1.0, 12.0, value=BASE_CFG["generation"]["guidance_scale"], step=0.5, label="Guidance scale")
        with gr.Accordion("Quality checks (same scorers as the benchmark)", open=True):
            score = gr.Checkbox(value=False, label="Score each image: VQAScore + PickScore + CLIPScore (first use downloads ~13 GB of scoring models)")
            gr.Markdown("The boxes below fill themselves from the prompt (quoted text; a number and the object after it). Edit them if they guessed wrong.")
            t0, o0, n0 = infer_checks(prompt.value)
            with gr.Row():
                expected_text = gr.Textbox(label="Expected text, for the OCR check (blank = skip)", value=t0, scale=2)
                count_object = gr.Textbox(label="Object to count (blank = skip)", value=o0, scale=1)
                count_n = gr.Number(label="Expected count", value=n0, precision=0, scale=1)
            prompt.change(infer_checks, prompt, [expected_text, count_object, count_n])
        checks = [score, expected_text, count_object, count_n]

        with gr.Tab("One setting"):
            default_methods = [list(METHODS)[0], "Large model only", "Small model only"]
            method = gr.CheckboxGroup(list(METHODS), value=default_methods,
                                      label="Methods (tick any mix; each makes one image with the same seed)")
            switch = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Switch point: fraction of steps run by the FIRST model (Methods 1 and 2)")
            note = gr.Markdown(explain(default_methods, 0.4, BASE_CFG["generation"]["steps"]))
            go = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Images (click to enlarge)", columns=4, height=520, object_fit="contain",
                                 elem_classes="fit-gallery")
            table = gr.Dataframe(headers=["run", "plan", "GPU-seconds", "peak GB", "$/1000 images", "file", "vs large", *SCORE_HEADERS],
                                 label="Cost and quality", interactive=False)
            status = gr.Markdown()
            for c in (method, switch, steps):
                c.change(explain, [method, switch, steps], note)
            go.click(run_single, [prompt, seed, steps, size, guidance, method, switch, *checks], [gallery, table, status],
                      concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab("Sweep"):
            gr.Markdown("One prompt across many switch points. Images are shown in the order they ran; the table gives the cost of each.")
            fractions = gr.Textbox(label="Switch points (comma separated)", value="0.2, 0.4, 0.6, 0.8")
            both = gr.Checkbox(value=True, label="Both directions (Method 1 and Method 2)")
            go2 = gr.Button("Run sweep", variant="primary")
            gallery2 = gr.Gallery(label="Images (click to enlarge)", columns=5, height=520, object_fit="contain",
                                  elem_classes="fit-gallery")
            table2 = gr.Dataframe(headers=["setting", "plan", "GPU-seconds", "peak GB", "$/1000 images", "vs large", *SCORE_HEADERS],
                                  label="Cost and quality", interactive=False)
            status2 = gr.Markdown()
            go2.click(run_sweep, [prompt, seed, steps, size, guidance, fractions, both, *checks], [gallery2, table2, status2],
                      concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab("GenEval"):
            gr.Markdown("GenEval's own prompts (`a photo of two clocks`), checked with GenEval's pass/fail rules for "
                        "objects, counting, colours and position. Uses the seed, steps, size and guidance above; ignores the prompt box. "
                        "Our detector (OWLv2 + CLIP) stands in for GenEval's, so compare methods with each other, not with published GenEval numbers.")
            with gr.Row():
                method3 = gr.CheckboxGroup(list(METHODS), value=default_methods, label="Methods", scale=3)
                per_task = gr.Number(label="Prompts per task (6 tasks; GenEval has 80-100 each)", value=2, precision=0, scale=1)
            switch3 = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Switch point: fraction of steps run by the FIRST model (Methods 1 and 2)")
            go3 = gr.Button("Run GenEval", variant="primary")
            summary3 = gr.Dataframe(headers=["run", "images", "GPU-seconds (mean)", "peak GB", "$/1000 images", "vs large",
                                             "GenEval score", *GENEVAL_TASKS], label="Per method", interactive=False)
            gallery3 = gr.Gallery(label="Images (click to enlarge)", columns=6, height=520, object_fit="contain",
                                  elem_classes="fit-gallery")
            table3 = gr.Dataframe(headers=["task", "prompt", "run", "GPU-seconds", "peak GB", "$/1000 images", "result", "why it failed"],
                                  label="Per image", interactive=False)
            status3 = gr.Markdown()
            go3.click(run_geneval, [seed, steps, size, guidance, method3, switch3, per_task],
                      [gallery3, summary3, table3, status3], concurrency_id="gpu", concurrency_limit=1)

        with gr.Tab("Help"):
            gr.Markdown("""
**Switch point** is the share of the total steps that the *first* model runs. With 30 steps, 0.4 means the first model
does 12 steps and the second does 18. At 0.0 the second model does everything; at 1.0 the first model does everything.

**Method 1 (s2l)** = small model first, large model finishes. Prediction: layout mistakes on counting/position prompts.
**Method 2 (l2s)** = large model first, small model finishes. Prediction: worse text and hands.

**Seed** fixes the starting noise, so the same prompt + seed + settings always gives the same image. Keep it fixed when
comparing settings; change it to see a different composition.

**GPU-seconds** is wall-clock time of the generation on the GPU. The 'vs large' column is the saving against the large
model alone at the same step count. **$/1000 images** = GPU-seconds x the hourly price in `config.yaml` / 3600 x 1000.

**peak GB** is the most GPU memory the generation used.

**Quality checks** (open the accordion under the sliders; scoring runs after the timer stops, so it never adds to GPU-seconds):
- **VQAScore**: does the image show the prompt? A vision-language model (Qwen2.5-VL-3B) is asked "Does this figure show
  '<prompt>'? yes or no"; the score is its probability of "Yes", 0 to 1. Much better than CLIPScore at counts, colours and
  positions. On GPUs under 24 GB the image models move to CPU while it runs (a few extra seconds, not counted).
- **PickScore**: predicted human preference (a CLIP model trained on which image people picked). Higher is better; usually 18-24, and a gap under ~0.3 is small.
- **CLIPScore**: how well the image matches the prompt (100 x CLIP cosine similarity). Higher is better; 25-35 is typical.
- **OCR text**: easyocr reads the image; *yes* if the expected text is found. CER = character error rate, 0 is perfect.
- **count**: OWLv2 counts the object; shown as detected/expected.

**GenEval tab**: GenEval's prompts in six tasks (one object, two objects, counting, colours, position, two colours on two
objects). Each image passes or fails by GenEval's rules; *why it failed* says what the detector saw. **GenEval score** is the
mean of the six task pass rates, as in the GenEval paper. Our detector differs from GenEval's, so it is for comparing
methods here, not with published numbers.

**What the benchmark measures** (RESEARCH_PLAN.md): how much quality each % of cost saving gives up compared with the large
model alone, per prompt type, and whether a handoff beats just running the large model with fewer steps at the same cost.

Every image and its timing is appended to `results/app/log.csv`; quality scores go to `results/app/scores.csv`,
GenEval results to `results/app/geneval.csv`.
""")
    return demo


def parse_auth(value):
    """'user:password' -> (user, password); blank -> no login."""
    if not value:
        return None
    if ":" not in value:
        raise SystemExit("--auth must look like user:password")
    user, pw = value.split(":", 1)
    return user, pw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="store_true", help="print a public link (needed on Colab)")
    ap.add_argument("--demo", action="store_true", help="no GPU: fake images and estimated timings")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--auth", default=os.environ.get("RR_AUTH", ""),
                    help="user:password to require a login (recommended with --share); or set RR_AUTH")
    a = ap.parse_args()
    STATE["demo"], STATE["device"] = a.demo, a.device
    print("DEMO MODE: fake images." if a.demo else "Models load on the first Generate click.")
    if a.share and not a.auth:
        print("WARNING: the public link has no login; anyone with it can use this GPU. Add --auth user:password.")
    build().launch(share=a.share, server_port=a.port, show_error=True, css=CSS, auth=parse_auth(a.auth))
