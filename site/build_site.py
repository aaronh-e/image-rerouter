#!/usr/bin/env python
"""Build the results website's data: site/data.json and one image strip per
(prompt, seed) in site/sprites/. The page is site/index.html; publish the three together.

    python site/build_site.py --run path/to/unzipped/results-folder   # folder that holds results/results.csv

Every number on the page comes from this file's output: per-image scores, and per
(metric, category, setting) the paired change vs the full large model with a 95% CI
(a prompt's two seeds are averaged first, so n = prompts).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.plot import load_results  # noqa: E402
from rerouter import utils  # noqa: E402

STEPS = 30
# Display order of the 13 settings; also the order of tiles in each image strip.
SETTINGS = (["large@30", "large@24", "large@18", "large@12", "small@30"]
            + [f"l2s@{f}" for f in ("0.8", "0.6", "0.4", "0.2")]
            + [f"s2l@{f}" for f in ("0.2", "0.4", "0.6", "0.8")])
GROUPS = {
    "composition": ("Composition", "Counting, colours and positions"),
    "text": ("Text", "Readable words in the image"),
    "hands_faces": ("Hands & faces", "Close-up people"),
    "product": ("Product", "Photoreal objects"),
    "stylized": ("Stylized", "Art styles"),
    "dense": ("Dense", "Long, detailed prompts"),
}
METRICS = {
    "pick_score": ("PickScore", "Predicted human preference", "points"),
    "vqa_score": ("VQAScore", "Does the image match the prompt? (0 to 1)", "score"),
    "clip_score": ("CLIPScore", "Image-text similarity", "points"),
    "ocr_match": ("Text accuracy", "Share of text prompts whose words come out right", "rate"),
}
TILE = 512


def t95(df: int) -> float:
    """Two-sided 95% t quantile (Cornish-Fisher expansion; within 0.001 for df >= 3)."""
    z, v = 1.959964, float(df)
    return (z + (z**3 + z) / (4 * v) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * v**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * v**3))


def setting_id(r) -> str:
    if r.method == "control":
        return f"{r.direction}@{int(r.steps)}"
    return f"{r.direction}@{float(r.switch_fraction):.1f}"


def setting_info(sid: str) -> dict:
    kind, x = sid.split("@")
    if kind in ("large", "small"):
        segs = [[kind, 0, int(x)]]
    else:
        n_first = round(float(x) * STEPS)
        first, second = ("large", "small") if kind == "l2s" else ("small", "large")
        segs = [[first, 0, n_first], [second, n_first, STEPS]]
    family = {"large": "full" if sid == "large@30" else "fewer", "small": "small",
              "l2s": "large_first", "s2l": "small_first"}[kind]
    return {"id": sid, "family": family, "segments": segs}


def ci(d: pd.Series) -> dict | None:
    d = d.dropna()
    if len(d) < 2:
        return None
    half = t95(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"mean": round(float(d.mean()), 4), "lo": round(float(d.mean() - half), 4),
            "hi": round(float(d.mean() + half), 4), "n": int(len(d)),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder containing results/results.csv and results/images/")
    ap.add_argument("--out", default=str(ROOT / "site"))
    ap.add_argument("--quality", type=int, default=74)
    ap.add_argument("--rebuild-images", action="store_true", help="remake strips that already exist")
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    (out / "sprites").mkdir(parents=True, exist_ok=True)

    cfg = utils.load_config()
    df = load_results(str(run / "results" / "results.csv"), cfg["compute"])   # gpu_seconds = step-count cost
    df["setting"] = [setting_id(r) for r in df.itertuples()]
    gpus = df.gpu_type.unique()
    assert len(gpus) == 1, f"one GPU per site, got {gpus}"
    missing = set(SETTINGS) - set(df.setting)
    assert not missing, f"settings missing from results: {missing}"

    full_sec = float(df[df.setting == "large@30"].gpu_seconds.mean())
    settings = []
    for sid in SETTINGS:
        info = setting_info(sid)
        sec = float(df[df.setting == sid].gpu_seconds.mean())
        info.update(seconds=round(sec, 4), saved=round(1 - sec / full_sec, 4))
        settings.append(info)

    prompts = pd.read_csv(ROOT / "prompts" / "pilot.csv", dtype=str).fillna("")
    prompts = prompts[prompts.id.isin(df.prompt_id)]

    # metric -> group ("all" + each) -> setting -> paired change vs large@30
    summary = {}
    for m in METRICS:
        summary[m] = {}
        for g in ["all", *GROUPS]:
            sub = (df if g == "all" else df[df.group == g]).dropna(subset=[m])
            if sub.empty:
                continue
            pp = sub.pivot_table(index="prompt_id", columns="setting", values=m, aggfunc="mean")
            res = {}
            for s in SETTINGS[1:]:
                if s in pp and (r := ci(pp[s] - pp["large@30"])):
                    res[s] = r
            summary[m][g] = {"full_mean": round(float(pp["large@30"].mean()), 4), "settings": res}

    # equal-cost head-to-head of the two orders: l2s@(1-f) and s2l@f run the same steps per model
    h2h = {}
    for m in ("pick_score", "vqa_score"):
        pp = df.dropna(subset=[m]).pivot_table(index="prompt_id", columns="setting", values=m, aggfunc="mean")
        rows = []
        for f in (0.2, 0.4, 0.6, 0.8):
            a, b = f"l2s@{1 - f:.1f}", f"s2l@{f:.1f}"
            r = ci(pp[a] - pp[b])
            rows.append({"large_first": a, "small_first": b, **r})
        h2h[m] = rows

    # each handoff vs the fewer-steps control of nearly the same cost (within 3 points of % saved)
    saved = {s["id"]: s["saved"] for s in settings}
    fewer = [s for s in SETTINGS if s.startswith("large@") and s != "large@30"]
    matched = {}
    for m in ("pick_score", "vqa_score"):
        pp = df.dropna(subset=[m]).pivot_table(index="prompt_id", columns="setting", values=m, aggfunc="mean")
        rows = []
        for h in (s for s in SETTINGS if s[:3] in ("l2s", "s2l")):
            c = min(fewer, key=lambda s: abs(saved[s] - saved[h]))
            if abs(saved[c] - saved[h]) <= 0.03:
                rows.append({"handoff": h, "control": c, **ci(pp[h] - pp[c])})
        matched[m] = rows

    # per-image scores + one strip per (prompt, seed)
    images = {}
    idx = {s: i for i, s in enumerate(SETTINGS)}
    for (pid, seed), g in df.groupby(["prompt_id", "seed"]):
        key = f"{pid}_s{int(seed)}"
        strip_path = out / "sprites" / f"{key}.webp"
        make_strip = args.rebuild_images or not strip_path.exists()
        strip = Image.new("RGB", (TILE * len(SETTINGS), TILE)) if make_strip else None
        scores = [None] * len(SETTINGS)
        for r in g.itertuples():
            i = idx[r.setting]
            if make_strip:
                img = Image.open(run / r.image_path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
                strip.paste(img, (i * TILE, 0))
            ocr = None if pd.isna(r.ocr_match) else int(r.ocr_match)
            text = str(r.ocr_text)[:80] if ocr is not None and isinstance(r.ocr_text, str) else None
            scores[i] = [round(r.pick_score, 3), round(r.vqa_score, 3), round(r.clip_score, 2), ocr, text]
        assert all(s is not None for s in scores), f"{key} is missing a setting"
        if make_strip:
            strip.save(strip_path, "WEBP", quality=args.quality, method=6)
        images[key] = scores

    data = {
        "meta": {"gpu": str(gpus[0]), "run_start": df.run_ts.min(), "run_end": df.run_ts.max(), "steps": STEPS,
                 "n_prompts": int(df.prompt_id.nunique()), "seeds": sorted(int(s) for s in df.seed.unique()),
                 "n_images": int(len(df)), "full_seconds": round(full_sec, 4), "tile": TILE,
                 "step_seconds": cfg["compute"]["step_seconds"].get(str(gpus[0])),
                 "models": {"large": "SDXL base 1.0", "small": "Segmind SSD-1B"}},
        "settings": settings,
        "groups": [{"id": k, "label": v[0], "blurb": v[1]} for k, v in GROUPS.items()],
        "metrics": [{"id": k, "label": v[0], "blurb": v[1], "unit": v[2]} for k, v in METRICS.items()],
        "prompts": [{"id": r.id, "group": r.group, "text": r.prompt, "expected": r.expected} for r in prompts.itertuples()],
        "summary": summary, "head_to_head": h2h, "matched": matched, "images": images,
        "score_fields": ["pick_score", "vqa_score", "clip_score", "ocr_match", "ocr_text"],
    }
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    size = sum(p.stat().st_size for p in (out / "sprites").glob("*.webp"))
    print(f"wrote {out / 'data.json'} ({(out / 'data.json').stat().st_size / 1e3:.0f} kB) "
          f"and {len(images)} strips ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
