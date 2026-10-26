#!/usr/bin/env python
"""Add quality columns to results.csv.

    python bench/score.py                       # score every row that lacks scores
    python bench/score.py --only clip ocr       # just some scorers
    python bench/score.py --manual results/manual_counts.csv   # merge hand counts
    python bench/score.py --fake                # CPU smoke test: random scores

Scorers (each is optional; missing packages are reported, not fatal):
    pick_score    PickScore_v1 (predicted human preference, higher is better)
    hpsv2         HPSv2 (same idea, different model; off by default, big download)
    clip_score    100 * cosine(CLIP image, CLIP text) with openai/clip-vit-large-patch14
    ocr           easyocr reads the image; ocr_match=1 if expected text is found
                  (normalised, substring); ocr_cer = character error rate vs best line
    count         OWLv2 open-vocabulary detector counts `object`; count_match=1 if equal

The pure helpers (normalise_text, char_error_rate, text_match, count_match) have
no model dependencies and are unit-tested in tests/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerouter import utils  # noqa: E402

SCORE_COLS = ["pick_score", "hpsv2", "clip_score", "ocr_match", "ocr_cer", "ocr_text", "count_match", "count_detected"]


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


# --------------------------------------------------------------- scorers ----

class Scorers:
    """Lazy wrappers so a missing package only disables that one metric."""

    def __init__(self, cfg: dict, fake: bool = False):
        self.cfg, self.fake = cfg, fake
        self._pick = self._hps = self._clip = self._ocr = self._det = None
        self.disabled: dict[str, str] = {}

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
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if dev == "cuda" else torch.float32
                model = AutoModel.from_pretrained(self.cfg["scoring"]["pickscore_model"], torch_dtype=dtype).to(dev).eval()
                self._pick = (model, AutoProcessor.from_pretrained(self.cfg["scoring"]["pickscore_processor"]), dev, dtype)
            model, proc, dev, dtype = self._pick
            inputs = proc(text=[prompt], images=Image.open(path).convert("RGB"), padding=True, truncation=True,
                          max_length=77, return_tensors="pt").to(dev)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                # logits_per_image = logit_scale.exp() * cosine(image, text): exactly PickScore
                return round(float(model(**inputs).logits_per_image[0, 0]), 4)
        except Exception as e:
            self.disabled["pick_score"] = str(e); return None

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
            import torch
            from PIL import Image
            if self._clip is None:
                from transformers import CLIPModel, CLIPProcessor
                mid = self.cfg["scoring"]["clip_model"]
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._clip = (CLIPModel.from_pretrained(mid).to(dev).eval(), CLIPProcessor.from_pretrained(mid), dev)
            model, proc, dev = self._clip
            inputs = proc(text=[prompt], images=Image.open(path).convert("RGB"), return_tensors="pt",
                          padding=True, truncation=True).to(dev)
            with torch.no_grad():
                out = model(**inputs)
                i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
                t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return round(float(100 * (i * t).sum()), 3)
        except Exception as e:
            self.disabled["clip_score"] = str(e); return None

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
            import torch
            from PIL import Image
            if self._det is None:
                from transformers import Owlv2ForObjectDetection, Owlv2Processor
                mid = self.cfg["scoring"]["owlv2_model"]
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._det = (Owlv2ForObjectDetection.from_pretrained(mid).to(dev).eval(), Owlv2Processor.from_pretrained(mid), dev)
            model, proc, dev = self._det
            img = Image.open(path).convert("RGB")
            inputs = proc(text=[[f"a photo of a {obj}"]], images=img, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inputs)
            # OWLv2 pads to square; target size is the padded square side.
            side = max(img.size)
            res = proc.post_process_object_detection(out, threshold=self.cfg["scoring"]["owlv2_threshold"],
                                                     target_sizes=torch.tensor([[side, side]]).to(dev))[0]
            det = int(len(res["scores"]))
            return count_match(det, expected), det
        except Exception as e:
            self.disabled["count"] = str(e); return None, None


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
    df["ocr_text"] = df["ocr_text"].astype(object)
    for c in ("pick_score", "hpsv2", "clip_score", "ocr_match", "ocr_cer", "count_match", "count_detected"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    want = set(only or ["pick_score", "hpsv2", "clip", "ocr", "count"])
    n = 0
    for idx, row in df.iterrows():
        if str(row.get("notes", "")).startswith("error") or not Path(str(row["image_path"])).exists():
            continue
        exp = expected_lookup.get(str(row["prompt_id"]), {})
        prompt, path = str(row["prompt"]), str(row["image_path"])

        def missing(col):
            return force or pd.isna(row.get(col))

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
    ap.add_argument("--only", nargs="+", choices=["pick_score", "hpsv2", "clip", "ocr", "count"], default=None)
    ap.add_argument("--manual", default=None, help="csv of hand counts to merge")
    ap.add_argument("--fake", action="store_true", help="random scores (CPU smoke test)")
    ap.add_argument("--force", action="store_true", help="re-score rows that already have scores")
    args = ap.parse_args(argv)

    cfg = utils.load_config(args.config)
    results = Path(args.results or cfg["paths"]["results_csv"])
    df = pd.read_csv(results, dtype={"prompt_id": str})
    # attach `expected` from the prompt file(s); full.csv is a superset of pilot
    exp_lookup = {}
    for pf in {args.prompts, "prompts/pilot.csv", "prompts/full.csv"}:
        if Path(pf).exists():
            for p in utils.read_prompts(pf):
                exp_lookup.setdefault(p["id"], utils.parse_expected(p["expected"]))

    df = score_frame(df, cfg, args.only, args.fake, args.force, exp_lookup)
    if args.manual:
        df = merge_manual(df, args.manual)

    # never lose data: keep a backup of the previous file before rewriting in place
    shutil.copy(results, results.with_suffix(".csv.bak"))
    df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore").to_csv(results, index=False)

    print("\nSummary")
    ok = df[~df["notes"].astype(str).str.startswith("error")]
    for c in ("pick_score", "hpsv2", "clip_score", "ocr_match", "count_match"):
        if c in ok and ok[c].notna().any():
            print(f"  {c:13s} filled for {ok[c].notna().sum():4d} rows, mean {pd.to_numeric(ok[c], errors='coerce').mean():.3f}")
    print(f"  wrote {results} (backup at {results.with_suffix('.csv.bak')})")
    print("  next: python bench/plot.py")


if __name__ == "__main__":
    main()
