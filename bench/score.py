#!/usr/bin/env python
"""Add quality columns to results.csv.

    python bench/score.py                       # score every row that lacks scores
    python bench/score.py --only clip ocr       # just some scorers
    python bench/score.py --manual results/manual_counts.csv   # merge hand counts
    python bench/score.py --fake                # CPU smoke test: random scores

Scorers (each is optional; missing packages are reported, not fatal):
    vqa_score     VQAScore recipe with Qwen2.5-VL-3B: P("Yes") that the image shows the prompt, 0-1
    pick_score    PickScore_v1 (predicted human preference, higher is better)
    hpsv2         HPSv2 (same idea, different model; off by default, big download)
    clip_score    100 * cosine(CLIP image, CLIP text) with openai/clip-vit-large-patch14
    ocr           easyocr reads the image; ocr_match=1 if expected text is found
                  (normalised, substring); ocr_cer = character error rate vs best line
    count         OWLv2 open-vocabulary detector counts `object`; count_match=1 if equal
    geneval       GenEval's pass/fail rules (counting, colours, position) for prompts whose
                  `expected` is geneval=<json>; OWLv2 + CLIP stand in for GenEval's detector

The pure helpers (normalise_text, char_error_rate, text_match, count_match) have
no model dependencies and are unit-tested in tests/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerouter import utils  # noqa: E402

SCORE_COLS = ["vqa_score", "pick_score", "hpsv2", "clip_score", "ocr_match", "ocr_cer", "ocr_text", "count_match", "count_detected",
              "geneval_pass", "geneval_reason"]


# ---------------------------------------------------------- pure helpers ----

def normalise_text(s: str) -> str:
    """Upper-case, drop everything but letters/digits/spaces, squeeze spaces.
    OCR confuses 0/O and 1/I/l, so we map those too; the task is 'is the sign
    readable', not 'is the font pixel-perfect'."""
    s = s.upper().translate(str.maketrans({"0": "O", "1": "I", "|": "I", "5": "S"}))
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def char_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein distance / len(ref), on normalised strings. 0 = perfect."""
    hyp, ref = normalise_text(hyp), normalise_text(ref)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def text_match(ocr_lines: list[str], expected: str) -> tuple[int, float]:
    """(exact_match, best_cer). Exact match = expected appears inside the joined OCR
    output after normalisation. CER is the best over single lines and the join."""
    exp = normalise_text(expected)
    joined = normalise_text(" ".join(ocr_lines))
    exact = int(bool(exp) and exp in joined)
    candidates = [joined] + [normalise_text(l) for l in ocr_lines]
    cer = min((char_error_rate(c, exp) for c in candidates), default=1.0)
    return exact, round(cer, 3)


def count_match(detected: int, expected: int) -> int:
    return int(detected == expected)


def nms_count(boxes: list, scores: list, iou: float) -> int:
    return len(nms_keep(boxes, scores, iou))


def nms_keep(boxes: list, scores: list, iou: float) -> list[int]:
    """Indices left after non-maximum suppression, most confident first. OWLv2's
    post_process_object_detection does no NMS, so one cup can come back as several
    overlapping boxes and inflate the count. Greedy: keep the best box, drop boxes
    overlapping it by more than `iou`."""
    def overlap(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0
    kept = []
    for i in sorted(range(len(boxes)), key=lambda i: -scores[i]):
        if all(overlap(boxes[i], boxes[k]) <= iou for k in kept):
            kept.append(i)
    return kept


# GenEval (Ghosh et al. 2023, MIT licence), ported from evaluation/evaluate_images.py
# so the rules match; only the detector differs (OWLv2 here, Mask2Former there).
GENEVAL_COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]


def relative_position(box_a, box_b, threshold: float = 0.1) -> set[str]:
    """Position of box A relative to box B ('left of', 'above', ...), ignoring offsets
    smaller than `threshold` x the boxes' combined size. Boxes are x0, y0, x1, y1."""
    ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    off = (ca[0] - cb[0], ca[1] - cb[1])
    dims = (abs(box_a[2] - box_a[0]) + abs(box_b[2] - box_b[0]), abs(box_a[3] - box_a[1]) + abs(box_b[3] - box_b[1]))
    rev = [max(abs(o) - threshold * d, 0) * (1 if o > 0 else -1 if o < 0 else 0) for o, d in zip(off, dims)]
    if all(abs(r) < 1e-3 for r in rev):
        return set()
    norm = (off[0] ** 2 + off[1] ** 2) ** 0.5
    dx, dy = rev[0] / norm, rev[1] / norm
    rels = set()
    if dx < -0.5: rels.add("left of")
    if dx > 0.5: rels.add("right of")
    if dy < -0.5: rels.add("above")
    if dy > 0.5: rels.add("below")
    return rels


def geneval_judge(meta: dict, objects: dict, color_of) -> tuple[bool, str]:
    """GenEval's pass/fail rule for one image. `objects` maps class -> boxes, most
    confident first; `color_of(class, boxes)` returns one GENEVAL_COLORS name per box.
    'include' clauses must all hold; any 'exclude' clause that holds fails the image."""
    correct, reason, matched_groups = True, [], []
    for req in meta.get("include", []):
        cls, matched = req["class"], True
        found = objects.get(cls, [])[:req["count"]]
        if len(found) < req["count"]:
            correct = matched = False
            reason.append(f"expected {cls}>={req['count']}, found {len(found)}")
        else:
            if "color" in req:
                colors = color_of(cls, found)
                if colors.count(req["color"]) < req["count"]:
                    correct = matched = False
                    reason.append(f"expected {req['color']} {cls}, found " + ", ".join(colors))
            if "position" in req and matched:
                rel, target = req["position"]
                if matched_groups[target] is None:
                    correct = matched = False
                    reason.append(f"no target for {cls} to be {rel}")
                else:
                    for box in found:
                        for tbox in matched_groups[target]:
                            got = relative_position(box, tbox)
                            if rel not in got:
                                correct = matched = False
                                reason.append(f"expected {cls} {rel} target, found {' and '.join(got) or 'overlapping'}")
                                break
                        if not matched:
                            break
        matched_groups.append(found if matched else None)
    for req in meta.get("exclude", []):
        if len(objects.get(req["class"], [])) >= req["count"]:
            correct = False
            reason.append(f"expected {req['class']}<{req['count']}, found {len(objects[req['class']])}")
    return correct, "; ".join(reason)


# --------------------------------------------------------------- scorers ----

def owl_post_process(proc, out, threshold: float, target_sizes) -> dict:
    """Boxes and scores for one image. transformers 5 removed
    Owlv2Processor.post_process_object_detection (it raised AttributeError, so the
    counter switched itself off and count_match came back empty); the image
    processor has had the method since 4.x, so call it there."""
    return proc.image_processor.post_process_object_detection(out, threshold=threshold, target_sizes=target_sizes)[0]


class Scorers:
    """Lazy wrappers so a missing package only disables that one metric."""

    def __init__(self, cfg: dict, fake: bool = False):
        self.cfg, self.fake = cfg, fake
        self._pick = self._hps = self._clip = self._ocr = self._det = self._vqa = None
        self.disabled: dict[str, str] = {}

    @staticmethod
    def _device() -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def park(self) -> None:
        """Move the loaded torch scorers to CPU and release their GPU memory. The
        website calls this after each job so ~4 GB of scorers never sit on a 15 GB T4
        while SDXL generates; each scorer moves itself back on its next call.
        (easyocr's reader stays put: it is ~0.1 GB and has no .to().)"""
        for loaded in (self._pick, self._clip, self._det, self._vqa):
            if loaded is not None:
                loaded[0].to("cpu")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- PickScore
    def pick_score(self, path: str, prompt: str):
        """PickScore_v1: a CLIP-H model fine-tuned on Pick-a-Pic human preferences.
        Score = logit_scale * cosine(image, text), as in the official example; ~18-24
        for typical images, higher = people would prefer it. fp16 on GPU (~2 GB):
        at fp32 it is ~4 GB and would not fit next to SDXL + SSD-1B on a 15 GB T4."""
        if self.fake:
            import random; return round(random.uniform(18, 23), 3)
        if "pick_score" in self.disabled: return None
        try:
            import torch
            from PIL import Image
            if self._pick is None:
                from transformers import AutoModel, AutoProcessor
                dtype = torch.float16 if self._device() == "cuda" else torch.float32
                model = AutoModel.from_pretrained(self.cfg["scoring"]["pickscore_model"], torch_dtype=dtype).eval()
                self._pick = (model, AutoProcessor.from_pretrained(self.cfg["scoring"]["pickscore_processor"]), dtype)
            model, proc, dtype = self._pick
            dev = self._device()
            model.to(dev)   # no-op unless park() moved it to CPU
            inputs = proc(text=[prompt], images=Image.open(path).convert("RGB"), padding=True, truncation=True,
                          max_length=77, return_tensors="pt").to(dev)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                # logits_per_image = logit_scale.exp() * cosine(image, text): exactly PickScore
                return round(float(model(**inputs).logits_per_image[0, 0]), 4)
        except Exception as e:
            self.disabled["pick_score"] = str(e); return None

    # -- VQAScore
    def vqa_score(self, path: str, prompt: str):
        """VQAScore (Lin et al. 2024): P("Yes") when a vision-language model is asked
        'Does this figure show "{prompt}"? Please answer yes or no.' 0-1, higher = the
        image matches the prompt. The paper's CLIP-FlanT5 needs t2v_metrics, which pins
        transformers 4.49 / torch 2.5.1; this is the same recipe with Qwen2.5-VL-3B (~7 GB)."""
        if self.fake:
            import random; return round(random.uniform(0.3, 0.95), 4)
        if "vqa_score" in self.disabled: return None
        try:
            import torch
            from PIL import Image
            dev = self._device()
            if self._vqa is None:
                from transformers import AutoModelForImageTextToText, AutoProcessor
                mid = self.cfg["scoring"]["vqa_model"]
                # T4 has no native bf16 and Qwen2.5-VL can overflow in fp16; torch emulates bf16 there
                dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (
                    torch.float16 if dev == "cuda" else torch.float32)
                model = AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=dtype).eval()
                proc = AutoProcessor.from_pretrained(mid)
                self._vqa = (model, proc, proc.tokenizer.encode("Yes", add_special_tokens=False)[0])
            model, proc, yes = self._vqa
            model.to(dev)
            question = f'Does this figure show "{prompt}"? Please answer yes or no.'
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[Image.open(path).convert("RGB")], return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            return round(float(torch.softmax(logits, -1)[yes]), 4)
        except Exception as e:
            self.disabled["vqa_score"] = str(e); return None

    # -- HPSv2
    def hpsv2(self, path: str, prompt: str):
        if self.fake:
            import random; return round(random.uniform(0.2, 0.3), 4)
        if "hpsv2" in self.disabled or not self.cfg["scoring"].get("hpsv2"): return None
        try:
            import hpsv2
            return round(float(hpsv2.score(path, prompt, hps_version="v2.1")[0]), 4)
        except Exception as e:
            self.disabled["hpsv2"] = str(e); return None

    # -- CLIPScore
    def clip_score(self, path: str, prompt: str):
        if self.fake:
            import random; return round(random.uniform(20, 35), 3)
        if "clip_score" in self.disabled: return None
        try:
            from PIL import Image
            i, t = self._clip_embed([prompt], [Image.open(path).convert("RGB")])
            return round(float(100 * (i * t).sum()), 3)
        except Exception as e:
            self.disabled["clip_score"] = str(e); return None

    def _clip_embed(self, texts: list[str], images: list):
        """Unit-length CLIP embeddings: (images x dim, texts x dim)."""
        import torch
        if self._clip is None:
            from transformers import CLIPModel, CLIPProcessor
            mid = self.cfg["scoring"]["clip_model"]
            self._clip = (CLIPModel.from_pretrained(mid).eval(), CLIPProcessor.from_pretrained(mid))
        model, proc = self._clip
        dev = self._device()
        model.to(dev)
        inputs = proc(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.no_grad():
            out = model(**inputs)
        return (out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True),
                out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True))

    # -- OCR
    def ocr(self, path: str, expected_text: str):
        if self.fake:
            import random
            lines = [expected_text if random.random() < 0.5 else "OPFN 24 H0URS"]
        else:
            if "ocr" in self.disabled: return None, None, None
            try:
                if self._ocr is None:
                    import easyocr
                    self._ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
                lines = [t for (_box, t, _conf) in self._ocr.readtext(path)]
            except Exception as e:
                self.disabled["ocr"] = str(e); return None, None, None
        exact, cer = text_match(lines, expected_text)
        return exact, cer, " | ".join(lines)[:200]

    # -- object counting
    def count(self, path: str, obj: str, expected: int):
        if self.fake:
            import random
            det = max(0, expected + random.choice([-1, 0, 0, 0, 1]))
            return count_match(det, expected), det
        if "count" in self.disabled or self.cfg["scoring"].get("counter") == "manual":
            return None, None
        try:
            from PIL import Image
            det = len(self._detect(Image.open(path).convert("RGB"), obj))
            return count_match(det, expected), det
        except Exception as e:
            self.disabled["count"] = str(e); return None, None

    def _detect(self, img, obj: str) -> list:
        """OWLv2 boxes (x0, y0, x1, y1) for `obj` after NMS, most confident first."""
        import torch
        if self._det is None:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            mid = self.cfg["scoring"]["owlv2_model"]
            self._det = (Owlv2ForObjectDetection.from_pretrained(mid).eval(), Owlv2Processor.from_pretrained(mid))
        model, proc = self._det
        dev = self._device()
        model.to(dev)
        inputs = proc(text=[[f"a photo of a {obj}"]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs)
        # OWLv2 pads to square; target size is the padded square side.
        side = max(img.size)
        res = owl_post_process(proc, out, self.cfg["scoring"]["owlv2_threshold"], torch.tensor([[side, side]]).to(dev))
        boxes, scores = res["boxes"].tolist(), res["scores"].tolist()
        return [boxes[i] for i in nms_keep(boxes, scores, self.cfg["scoring"].get("owlv2_nms_iou", 0.5))]

    # -- GenEval
    def geneval(self, path: str, meta: dict):
        """(passed, reason) by GenEval's rules for one image; `meta` is a line of GenEval's
        evaluation_metadata.jsonl. Colours are CLIP zero-shot on each box's crop, with
        GenEval's three templates (no mask: OWLv2 gives boxes only)."""
        if self.fake:
            import random
            ok = random.random() < 0.6
            return ok, "" if ok else "fake failure"
        if "geneval" in self.disabled: return None, ""
        try:
            from PIL import Image
            img = Image.open(path).convert("RGB")
            classes = {r["class"] for r in meta.get("include", []) + meta.get("exclude", [])}
            objects = {c: b for c in classes if (b := self._detect(img, c))}

            def color_of(cls, boxes):
                texts = [t.format(c=c, cls=cls) for c in GENEVAL_COLORS
                         for t in ("a photo of a {c} {cls}", "a photo of a {c}-colored {cls}", "a photo of a {c} object")]
                i, t = self._clip_embed(texts, [img.crop(tuple(b)) for b in boxes])
                t = t.reshape(len(GENEVAL_COLORS), 3, -1).mean(1)
                t = t / t.norm(dim=-1, keepdim=True)
                return [GENEVAL_COLORS[k] for k in (i @ t.T).argmax(1).tolist()]

            return geneval_judge(meta, objects, color_of)
        except Exception as e:
            self.disabled["geneval"] = str(e); return None, ""


# ------------------------------------------------------------------ main ----

def score_frame(df: pd.DataFrame, cfg: dict, only: list[str] | None, fake: bool, force: bool,
                expected_lookup: dict | None = None) -> pd.DataFrame:
    """`expected_lookup` maps prompt_id -> parsed `expected` dict (from the prompt files)."""
    sc = Scorers(cfg, fake=fake)
    expected_lookup = expected_lookup or {}
    for c in SCORE_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    # a column that is all-NaN loads as float64 and then refuses strings
    for c in ("ocr_text", "geneval_reason"):
        df[c] = df[c].astype(object)
    for c in ("vqa_score", "geneval_pass", "pick_score", "hpsv2", "clip_score", "ocr_match", "ocr_cer", "count_match", "count_detected"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    want = set(only or ["vqa", "pick_score", "hpsv2", "clip", "ocr", "count", "geneval"])
    n = 0
    for idx, row in df.iterrows():
        if str(row.get("notes", "")).startswith("error") or not Path(str(row["image_path"])).exists():
            continue
        exp = expected_lookup.get(str(row["prompt_id"]), {})
        prompt, path = str(row["prompt"]), str(row["image_path"])

        def missing(col):
            return force or pd.isna(row.get(col))

        if "vqa" in want and missing("vqa_score"):
            df.at[idx, "vqa_score"] = sc.vqa_score(path, prompt)
        if "pick_score" in want and missing("pick_score"):
            df.at[idx, "pick_score"] = sc.pick_score(path, prompt)
        if "hpsv2" in want and missing("hpsv2") and (fake or cfg["scoring"].get("hpsv2")):
            df.at[idx, "hpsv2"] = sc.hpsv2(path, prompt)
        if "clip" in want and missing("clip_score"):
            df.at[idx, "clip_score"] = sc.clip_score(path, prompt)
        if "ocr" in want and "text" in exp and missing("ocr_match"):
            m, cer, txt = sc.ocr(path, exp["text"])
            df.at[idx, "ocr_match"], df.at[idx, "ocr_cer"], df.at[idx, "ocr_text"] = m, cer, txt
        if "count" in want and "count" in exp and missing("count_match"):
            m, det = sc.count(path, exp.get("object", ""), int(exp["count"]))
            df.at[idx, "count_match"], df.at[idx, "count_detected"] = m, det
        if "geneval" in want and "geneval" in exp and missing("geneval_pass"):
            ok, why = sc.geneval(path, json.loads(exp["geneval"]))
            df.at[idx, "geneval_pass"] = None if ok is None else int(ok)
            df.at[idx, "geneval_reason"] = why
        n += 1
        if n % 25 == 0:
            print(f"  scored {n} rows ...")
    for k, v in sc.disabled.items():
        print(f"  [warn] scorer '{k}' disabled: {v[:120]}")
    return df


def merge_manual(df: pd.DataFrame, manual_csv: str) -> pd.DataFrame:
    """results/manual_counts.csv: image_path,count_match[,count_detected][,notes]
    Hand counts win over the detector."""
    m = pd.read_csv(manual_csv)
    m = m.set_index("image_path")
    for col in ("count_match", "count_detected"):
        if col in m.columns:
            df[col] = df["image_path"].map(m[col]).combine_first(df[col]) if col in df.columns else df["image_path"].map(m[col])
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--prompts", default="prompts/pilot.csv", help="for the `expected` column")
    ap.add_argument("--only", nargs="+", choices=["vqa", "pick_score", "hpsv2", "clip", "ocr", "count", "geneval"], default=None)
    ap.add_argument("--manual", default=None, help="csv of hand counts to merge")
    ap.add_argument("--fake", action="store_true", help="random scores (CPU smoke test)")
    ap.add_argument("--force", action="store_true", help="re-score rows that already have scores")
    args = ap.parse_args(argv)

    cfg = utils.load_config(args.config)
    results = Path(args.results or cfg["paths"]["results_csv"])
    df = pd.read_csv(results, dtype={"prompt_id": str})
    # attach `expected` from the prompt file(s); full.csv is a superset of pilot
    exp_lookup = {}
    for pf in {args.prompts, "prompts/pilot.csv", "prompts/full.csv", "prompts/paper.csv"}:
        if utils.repo_path(pf).exists():
            for p in utils.read_prompts(pf):
                exp_lookup.setdefault(p["id"], utils.parse_expected(p["expected"]))

    df = score_frame(df, cfg, args.only, args.fake, args.force, exp_lookup)
    if args.manual:
        df = merge_manual(df, args.manual)

    # never lose data: a new timestamped backup every run, so two bad runs can't erase the original
    bak = results.with_name(f"{results.stem}.{dt.datetime.now():%Y%m%d_%H%M%S_%f}.csv.bak")
    shutil.copy(results, bak)
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore").to_csv(results, index=False)

    print("\nSummary")
    ok = df[~df["notes"].astype(str).str.startswith("error")]
    for c in ("vqa_score", "pick_score", "hpsv2", "clip_score", "ocr_match", "count_match", "geneval_pass"):
        if c in ok and ok[c].notna().any():
            print(f"  {c:13s} filled for {ok[c].notna().sum():4d} rows, mean {pd.to_numeric(ok[c], errors='coerce').mean():.3f}")
    print(f"  wrote {results} (backup at {bak})")
    print("  next: python bench/plot.py")


if __name__ == "__main__":
    main()
