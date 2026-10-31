#!/usr/bin/env python
"""Regenerate the two notebooks from Python source (easier to review in git than raw .ipynb JSON)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.metadata["accelerator"] = "GPU"
    return n


SETUP = '''# --- Setup (Colab: Runtime > Change runtime type > T4 GPU or better) ---
import os, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
REPO_URL = ""   # paste your GitHub URL here once the repo is pushed; leave empty to upload the zip instead
if IN_COLAB and not os.path.exists("image-rerouter"):
    if REPO_URL:
        !git clone -q $REPO_URL
    else:
        from google.colab import files
        import glob, shutil
        print("Pick the image-rerouter zip (any name, e.g. image-rerouter_2026-10-26.zip) in the file dialog below.")
        ZIP = next(iter(files.upload()))
        !rm -rf /content/_unzip && unzip -q -o "$ZIP" -d /content/_unzip && rm "$ZIP"
        # the code is wherever config.yaml sits next to rerouter/, whatever the zip's folder is called
        src = next(os.path.dirname(c) for c in glob.glob("/content/_unzip/**/config.yaml", recursive=True)
                   if os.path.isdir(os.path.join(os.path.dirname(c), "rerouter")))
        shutil.move(src, "/content/image-rerouter")
if IN_COLAB:
    %cd image-rerouter
    !pip install -q -r requirements.txt
    !pip uninstall -q -y torchao   # Colab's preinstalled torchao is too old for new diffusers (FqnToConfig import error); unused here
# optional: faster downloads + higher rate limits
# from huggingface_hub import login; login()
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
from rerouter import utils
cfg = utils.load_config()
cfg'''

L1 = [
("md", """# Level 1: Can models hand off at all?

This notebook answers one yes/no question per pair of models: **can model A start an image mid-generation and model B finish it, without the result breaking?**

What it does, in order:

1. Prints the **format table** (VAE config, weight fingerprints, scheduler) for SDXL, SSD-1B, Vega, Playground v2.5 and SD 1.5. No GPU needed.
2. Loads SDXL + SSD-1B once (shared text encoders + fp16-fix VAE) and times each alone.
3. **Verifies VAE compatibility empirically**: decodes an SSD-1B latent with three different VAE copies and compares.
4. Does one handoff in each direction at 40% (Method 1 small→large, Method 2 large→small) and builds the timing table.
5. Checks that the one-pipeline UNet swap and the two-pipeline `denoising_end/denoising_start` route give the same image.
6. Same-company pair (SSD-1B ↔ Vega), then the pairs that should **break** (SDXL → Playground v2.5, SD 1.5 → SDXL) and the **pixel bridge** that rescues them.

**Gate:** a clean handoff image in each direction, a timing table, and a list of which pairs work.

Time on a T4: about 25 minutes for steps 1–5 (plus ~15 GB of downloads the first time). Step 6 is optional and downloads another ~15 GB.
"""),
("code", SETUP),
("md", """## 1. Format table (no GPU)

Analytic check from the Hub. Spoiler, already measured tensor by tensor (see `docs/compat_report.md`):

- **SSD-1B's shipped VAE is `madebyollin/sdxl-vae-fp16-fix`** (max abs difference 0.015 = fp16 rounding). The fp16-fix VAE is SDXL's VAE with rescaled internals, so the **latent space is the same**. The file hash differs from SDXL's, which is exactly the "subtle" case the plan warns about, but here it's benign.
- SSD-1B's and Vega's text encoders differ from SDXL's only by rounding noise (mean |Δ| ≈ 6e-5, a bf16 round-trip). Sharing SDXL's encoders is safe.
- Vega and Playground v2.5 ship SDXL's VAE byte for byte, but Playground uses `scaling_factor` 0.5 (not 0.13025) **and** an EDM schedule.
- SD 1.5 uses a different VAE (and 512px), so its latents are a different language."""),
("code", """!python bench/compat_check.py
from IPython.display import Markdown, display
display(Markdown(open("docs/compat_report.md").read()))"""),
("md", "## 2. Load SDXL + SSD-1B once, time each alone"),
("code", """from rerouter.models import ModelBank
from rerouter.handoff import generate, warmup
bank = ModelBank(cfg).load(("large", "small"))
print("load seconds:", bank.load_seconds)
print("UNet params (billions):", bank.param_counts())
print(f"GPU memory after load: {torch.cuda.memory_allocated()/1e9:.1f} GB")
warmup(bank, cfg)   # never time the first run"""),
("code", """PROMPT = 'a storefront sign that says "OPEN 24 HOURS", photo'
SEED = 42
STEPS = cfg["generation"]["steps"]
runs = {}
runs["large alone"] = generate(bank, PROMPT, SEED, "large", 0.0, STEPS, cfg)
runs["small alone"] = generate(bank, PROMPT, SEED, "small", 0.0, STEPS, cfg)
for k, r in runs.items():
    print(f"{k:12s} {r.gpu_seconds:6.2f} s  peak {r.peak_mem_gb:.2f} GB  plan {r.plan}")"""),
("code", """import matplotlib.pyplot as plt
def grid(images, titles, cols=None, size=4):
    cols = cols or len(images)
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    for ax, im, t in zip(axes.flat if hasattr(axes, "flat") else [axes], images, titles):
        ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.show()
grid([r.image for r in runs.values()], list(runs))"""),
("md", """## 3. VAE compatibility, empirically

Run SSD-1B to a **latent** (no decode), then decode that one latent with:

- (a) the fp16-fix VAE the bench uses,
- (b) the VAE shipped inside `segmind/SSD-1B`,
- (c) the stock SDXL VAE (in fp32, since it NaNs in fp16).

If all three give the same picture (PSNR > 30 dB), SSD-1B and SDXL speak the same latent language and a mid-generation handoff is meaningful."""),
("code", """import numpy as np
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

g = torch.Generator("cuda").manual_seed(SEED)
bank.set_unet("small")
lat = bank.pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=cfg["generation"]["guidance_scale"],
                generator=g, output_type="latent").images
bank.restore()
print("latent shape:", tuple(lat.shape), "(1, 4, 128, 128 expected)")

def decode_with(vae, lat):
    vae = vae.to("cuda")
    with torch.no_grad():
        x = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype)).sample
    img = VaeImageProcessor().postprocess(x.float().cpu(), output_type="pil")[0]
    return img

vae_fix  = bank.pipe.vae
vae_ssd  = AutoencoderKL.from_pretrained(cfg["models"]["small"], subfolder="vae", torch_dtype=torch.float32)
vae_sdxl = AutoencoderKL.from_pretrained(cfg["models"]["large"], subfolder="vae", torch_dtype=torch.float32)
decodes = {"fp16-fix VAE (bench)": decode_with(vae_fix, lat),
           "VAE shipped with SSD-1B": decode_with(vae_ssd, lat),
           "stock SDXL VAE (fp32)": decode_with(vae_sdxl, lat)}
del vae_ssd, vae_sdxl; torch.cuda.empty_cache()

def psnr(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
ref = decodes["fp16-fix VAE (bench)"]
for k, im in decodes.items():
    print(f"{k:26s} PSNR vs bench VAE: {psnr(ref, im):5.1f} dB")
grid(list(decodes.values()), list(decodes))
print("Verdict:", "SAME latent space" if all(psnr(ref, im) > 30 for im in decodes.values()) else "DIFFERENT - stop and investigate")"""),
("md", """## 4. One handoff in each direction at 40%

`switch_fraction` = fraction of the total steps run by the **first** model.
- Method 1 `s2l@0.4`: SSD-1B runs steps 0–11, SDXL runs 12–29.
- Method 2 `l2s@0.4`: SDXL runs steps 0–11, SSD-1B runs 12–29."""),
("code", """runs["Method 1: small->large @0.4"] = generate(bank, PROMPT, SEED, "s2l", 0.4, STEPS, cfg)
runs["Method 2: large->small @0.4"] = generate(bank, PROMPT, SEED, "l2s", 0.4, STEPS, cfg)
import pandas as pd
timing = pd.DataFrame([{"run": k, "gpu_seconds": round(r.gpu_seconds, 2), "peak_mem_gb": round(r.peak_mem_gb, 2),
                        "plan": r.plan, "impl": r.impl} for k, r in runs.items()])
timing["saving_vs_large"] = (1 - timing.gpu_seconds / timing.gpu_seconds.iloc[0]).map("{:+.0%}".format)
timing.to_csv("results/level1_timing.csv", index=False)
display(timing)
grid([r.image for r in runs.values()], list(runs), cols=2, size=5)
for k, r in runs.items():
    r.image.save(f"results/level1_{k.split(':')[0].replace(' ', '_').lower()}.png")"""),
("md", """## 5. Does the UNet swap equal the two-pipeline route?

Same seed, same handoff point; the only difference is the plumbing. The two images should be nearly identical (small differences come from fp16 and the `denoising_end` timestep rounding). This validates `impl: swap` (which we use because it is cheaper) against the textbook Option A."""),
("code", """import copy
cfg2 = copy.deepcopy(cfg); cfg2["handoff"]["impl"] = "twopipe"
a = generate(bank, PROMPT, SEED, "s2l", 0.4, STEPS, cfg)
b = generate(bank, PROMPT, SEED, "s2l", 0.4, STEPS, cfg2)
print(f"swap: {a.gpu_seconds:.2f}s   twopipe: {b.gpu_seconds:.2f}s   PSNR between them: {psnr(a.image, b.image):.1f} dB")
grid([a.image, b.image], ["impl=swap", "impl=twopipe"])"""),
("md", """## 6. Other pairs

### 6a. Same company: SSD-1B ↔ Vega (should work)
Vega is a smaller distillation that also uses SDXL's VAE. On a 15 GB T4 we free the SDXL UNet first."""),
("code", """RUN_OPTIONAL = True   # set False to skip the heavy downloads in 6a-6c
if RUN_OPTIONAL:
    import gc
    # drop EVERY reference to the SDXL UNet (the dict, the pipeline, the restore slot) or it stays in GPU memory
    del bank.unets["large"]; bank.pipe.unet = bank._original_unet = bank.unets["small"]
    gc.collect(); torch.cuda.empty_cache()
    from diffusers import UNet2DConditionModel
    bank.unets["tiny"] = UNet2DConditionModel.from_pretrained(cfg["models"]["tiny"], subfolder="unet",
                                                               torch_dtype=torch.float16, variant="fp16").to("cuda")
    bank._check_unets_agree()
    from rerouter.schedule import Segment
    pair = {"small alone": generate(bank, PROMPT, SEED, "small", 0.0, STEPS, cfg),
            "tiny alone (Vega)": generate(bank, PROMPT, SEED, "small", 0.0, STEPS, cfg, segments=[Segment("tiny", 0, STEPS)]),
            "small->tiny @0.4": generate(bank, PROMPT, SEED, "s2l", 0.4, STEPS, cfg, segments=[Segment("small", 0, 12), Segment("tiny", 12, STEPS)]),
            "tiny->small @0.4": generate(bank, PROMPT, SEED, "l2s", 0.4, STEPS, cfg, segments=[Segment("tiny", 0, 12), Segment("small", 12, STEPS)])}
    for k, r in pair.items(): print(f"{k:20s} {r.gpu_seconds:6.2f}s")
    grid([r.image for r in pair.values()], list(pair), cols=2, size=5)"""),
("md", """### 6b. Different schedule: SDXL → Playground v2.5 (should break; pixel bridge works)

Playground v2.5 uses the same VAE weights but `scaling_factor=0.5` and an **EDM** schedule. Handing it an SDXL latent at "step 12" means nothing to it: it reads the wrong noise level. The naive handoff below should come out blurry/burnt. The **pixel bridge** (decode → img2img at strength 0.6) works for any pair but costs an extra decode+encode, and Playground redraws rather than continues."""),
("code", """if RUN_OPTIONAL:
    bank.free(); del bank; gc.collect(); torch.cuda.empty_cache()
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16)
    sdxl = StableDiffusionXLPipeline.from_pretrained(cfg["models"]["large"], vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
    g = torch.Generator("cuda").manual_seed(SEED)
    sdxl_lat = sdxl(PROMPT, num_inference_steps=STEPS, denoising_end=0.4, output_type="latent", generator=g).images
    sdxl_img = sdxl(PROMPT, num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    del sdxl; gc.collect(); torch.cuda.empty_cache()
    pg = StableDiffusionXLImg2ImgPipeline.from_pretrained("playgroundai/playground-v2.5-1024px-aesthetic", torch_dtype=torch.float16, variant="fp16").to("cuda")
    # naive: feed the SDXL latent straight in (rescale 0.13025 -> 0.5 so at least the magnitude is right)
    naive = pg(PROMPT, image=sdxl_lat * (0.5 / 0.13025), denoising_start=0.4, num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    # pixel bridge: decode SDXL's own finished image, let Playground redraw 60% of it
    bridge = pg(PROMPT, image=sdxl_img, strength=0.6, num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    grid([sdxl_img, naive, bridge], ["SDXL alone", "naive latent handoff -> Playground (expect broken)", "pixel bridge -> Playground"])
    del pg; gc.collect(); torch.cuda.empty_cache()"""),
("md", """### 6c. Different format: SD 1.5 → SDXL (impossible in latent space; pixel bridge works)

SD 1.5's latent for a 512px image is 64×64×4 in a *different* VAE's language. Decoding it with SDXL's VAE gives garbage. The pixel bridge (decode with SD 1.5's VAE → upscale → SDXL img2img) is the only route."""),
("code", """if RUN_OPTIONAL:
    from diffusers import StableDiffusionPipeline
    sd15 = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda")
    g = torch.Generator("cuda").manual_seed(SEED)
    lat15 = sd15(PROMPT, num_inference_steps=STEPS, output_type="latent", generator=g).images
    img15 = VaeImageProcessor().postprocess(sd15.vae.decode(lat15 / sd15.vae.config.scaling_factor).sample.float().cpu(), output_type="pil")[0]
    del sd15; gc.collect(); torch.cuda.empty_cache()
    # wrong-VAE decode: SDXL's VAE reading SD1.5's latent
    vae = AutoencoderKL.from_pretrained(cfg["models"]["vae"], torch_dtype=torch.float16).to("cuda")
    with torch.no_grad():
        wrong = VaeImageProcessor().postprocess(vae.decode(lat15.half() / 0.13025).sample.float().cpu(), output_type="pil")[0]
    sdxl_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(cfg["models"]["large"], vae=vae, torch_dtype=torch.float16, variant="fp16").to("cuda")
    bridge15 = sdxl_i2i(PROMPT, image=img15.resize((1024, 1024)), strength=0.5, num_inference_steps=STEPS, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
    grid([img15, wrong, bridge15], ["SD 1.5 alone (512px)", "SD1.5 latent decoded by SDXL VAE (garbage)", "pixel bridge -> SDXL img2img"])
    del sdxl_i2i, vae; gc.collect(); torch.cuda.empty_cache()"""),
("md", """## 7. Gate check

Fill in from what you saw above, then save. This is the Level 1 deliverable."""),
("code", """pairs = pd.DataFrame([
    ("SSD-1B -> SDXL (Method 1)", "latent", "WORKS", "same VAE (fp16-fix), same schedule"),
    ("SDXL -> SSD-1B (Method 2)", "latent", "WORKS", "same VAE (fp16-fix), same schedule"),
    ("SSD-1B <-> Vega",            "latent", "WORKS?", "fill in from 6a"),
    ("SDXL -> Playground v2.5",    "latent", "BREAKS?", "EDM schedule + scaling 0.5; fill in from 6b"),
    ("SDXL -> Playground v2.5",    "pixel bridge", "WORKS?", "extra decode/encode; redraws rather than continues"),
    ("SD 1.5 -> SDXL",             "latent", "IMPOSSIBLE", "different VAE, 64x64 vs 128x128"),
    ("SD 1.5 -> SDXL",             "pixel bridge", "WORKS?", "fill in from 6c"),
], columns=["pair", "route", "result", "why"])
pairs.to_csv("results/level1_pairs.csv", index=False)
display(pairs)
print("Timing table -> results/level1_timing.csv ; images -> results/level1_*.png")"""),
]

L2 = [
("md", """# Levels 2–4 on Colab: controls, methods, decider

Runs the benchmark scripts end to end and keeps results on Google Drive so a disconnect costs nothing (every run is append-only and resumable).

Budget on a T4 (≈1.3 s/step for SDXL at 1024²... measured, not assumed; your numbers go in `results.csv`):

| stage | images | rough T4 time |
|---|---|---|
| Level 2 controls (pilot) | 60 prompts × 2 seeds × 5 settings = 600 | ~5 h |
| Level 3 sweep (pilot) | 60 × 2 × 8 settings = 960 | ~6 h |
| Scoring | all | ~30 min |

Run a `--limit 6` smoke test first; then let each stage run in chunks (`--groups`) across sessions."""),
("code", """import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive; drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/image-rerouter"      # results persist here between sessions
    REPO_URL = ""   # paste your GitHub URL once pushed; empty = upload the zip the first time
    if not os.path.exists(WORK):
        if REPO_URL:
            !git clone -q $REPO_URL $WORK
        else:
            from google.colab import files
            import glob, shutil
            print("Pick the image-rerouter zip (any name). This only happens once; after that it lives on your Drive.")
            ZIP = next(iter(files.upload()))
            !rm -rf /content/_unzip && unzip -q -o "$ZIP" -d /content/_unzip && rm "$ZIP"
            src = next(os.path.dirname(c) for c in glob.glob("/content/_unzip/**/config.yaml", recursive=True)
                       if os.path.isdir(os.path.join(os.path.dirname(c), "rerouter")))
            shutil.move(src, WORK)
    %cd $WORK
    if REPO_URL:   # a ! command must start its own line, or Colab reports invalid syntax
        !git pull -q
    else:
        print(f"Using the code already in {WORK} (your first upload). To update it, set REPO_URL, or delete that folder and re-run.")
    !pip install -q -r requirements.txt
    !pip uninstall -q -y torchao   # Colab's preinstalled torchao is too old for new diffusers (FqnToConfig import error); unused here
import torch; print(torch.cuda.get_device_name(0))"""),
("code", """# GPU-seconds convert to dollars with compute.usd_per_hour in config.yaml: edit it there for your provider.
# (Rewriting config.yaml from here would strip its comments.) The GPU name is logged in every row automatically.
import yaml
print("GPU:", torch.cuda.get_device_name(0), "| price used for $/1000 images:", yaml.safe_load(open("config.yaml"))["compute"]["usd_per_hour"], "$/h")
# Hugging Face token (optional but avoids throttled downloads): Colab > Secrets > HF_TOKEN
try:
    from google.colab import userdata; os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception: pass"""),
("md", "## Smoke test (3 prompts, everything end to end, ~10 min)"),
("code", """!python bench/run.py --controls --prompts prompts/pilot.csv --limit 3 --out results/smoke.csv --images results/smoke_images
!python bench/score.py --results results/smoke.csv --only vqa pick_score
!python bench/plot.py --results results/smoke.csv --out results/smoke_plots
from IPython.display import Image, display; display(Image("results/smoke_plots/tradeoff_vqa_score.png"))"""),
("md", """## Level 2: controls
Large@30 (ceiling), small@30 (floor), large@12/18/24 (the line to beat). Re-run the same cell after any disconnect; finished rows are skipped."""),
("code", """!python bench/run.py --controls --prompts prompts/pilot.csv"""),
("code", """!python bench/score.py --prompts prompts/pilot.csv
!python bench/plot.py --metric vqa_score
!python bench/plot.py --metric pick_score
display(Image("results/plots/tradeoff_vqa_score.png"))
display(Image("results/plots/tradeoff_vqa_score_by_group.png"))
from IPython.display import Markdown
display(Markdown(open("results/plots/advantage_vqa_score.md").read()))"""),
("md", "### Spot check: 10 random images against their scores (the plan's human sanity check)"),
("code", """import pandas as pd, matplotlib.pyplot as plt
from PIL import Image as PILImage
df = pd.read_csv("results/results.csv")
df = df[~df.notes.astype(str).str.startswith("error")].sample(10, random_state=0)
fig, axes = plt.subplots(2, 5, figsize=(20, 9))
for ax, (_, r) in zip(axes.flat, df.iterrows()):
    ax.imshow(PILImage.open(r.image_path)); ax.axis("off")
    ax.set_title(f"{r.setting if 'setting' in r else r.direction}@{r.steps} VQA={r.vqa_score:.2f} PS={r.pick_score:.2f}\\n{r.prompt[:40]}", fontsize=8)
plt.tight_layout(); plt.show()
# Write what you see vs. the scores in results/spot_check.md"""),
("md", """## Level 3: the two methods
Both orders at switch fractions 0.2/0.4/0.6/0.8 (0 and 1 are the controls). Then the prediction checks:
- Method 1 (small first) should hurt **composition** → `count_match` on the composition group.
- Method 2 (large first) should hurt **text/hands** → `ocr_match` on the text group; hands by eye."""),
("code", """!python bench/run.py --sweep --prompts prompts/pilot.csv"""),
("code", """!python bench/score.py --prompts prompts/pilot.csv
!python bench/plot.py --metric vqa_score
!python bench/plot.py --metric pick_score
!python bench/plot.py --metric count_match --groups composition
!python bench/plot.py --metric ocr_match --groups text
display(Image("results/plots/tradeoff_vqa_score.png"))
display(Image("results/plots/tradeoff_vqa_score_by_group.png"))
display(Image("results/plots/heatmap_vqa_score.png"))
from IPython.display import Markdown
display(Markdown(open("results/plots/advantage_vqa_score.md").read()))
display(Markdown(open("results/plots/success_vqa_score.md").read()))
display(pd.read_csv("results/plots/order_head_to_head_vqa_score.csv"))"""),
("md", """### Hands by eye
OCR and counting are automatic; hand quality is not. Show the hands_faces prompts at the same cost (s2l@0.4 vs l2s@0.6) side by side and tally which looks right."""),
("code", """df = pd.read_csv("results/results.csv"); df = df[(df.group == "hands_faces") & (df.seed == 42)]
a = df[(df.direction == "s2l") & (df.switch_fraction.round(1) == 0.4)].set_index("prompt_id")
b = df[(df.direction == "l2s") & (df.switch_fraction.round(1) == 0.6)].set_index("prompt_id")
ids = sorted(set(a.index) & set(b.index))
fig, axes = plt.subplots(len(ids), 2, figsize=(8, 4 * len(ids)))
for row, pid in zip(axes, ids):
    row[0].imshow(PILImage.open(a.loc[pid].image_path)); row[0].set_title(f"{pid} s2l@0.4 (large finishes)", fontsize=8)
    row[1].imshow(PILImage.open(b.loc[pid].image_path)); row[1].set_title(f"{pid} l2s@0.6 (small finishes)", fontsize=8)
    for ax in row: ax.axis("off")
plt.tight_layout(); plt.show()"""),
("md", """## Level 4: the Jev decider
`decider_eval.py` builds the answer key from the sweep, splits prompts 70/30, asks Jev its typed questions, learns a profile→setting policy on the 70%, and scores the router on the 30% **by lookup** (every setting was already generated). The regex baseline runs first so you can see whether Jev adds anything."""),
("code", """try:
    from google.colab import userdata; os.environ["TYPESAFE_API_KEY"] = userdata.get("TYPESAFE_API_KEY")
except Exception: print("no TYPESAFE_API_KEY secret; only the heuristic baseline will run")
!python bench/decider_eval.py --prompts prompts/pilot.csv --provider heuristic
!cp results/router_eval.md results/router_eval_heuristic.md
if os.environ.get("TYPESAFE_API_KEY"):
    !python bench/decider_eval.py --prompts prompts/pilot.csv --provider typesafe
display(Markdown(open("results/router_eval.md").read()))"""),
("code", """# Optional: actually generate the routed images (adds Jev latency to wall clock, logged in notes) and put the router on the chart.
# !python bench/run.py --method router --policy results/decider_policy.json --prompts prompts/pilot.csv
# !python bench/score.py && python bench/plot.py"""),
("md", """## Full set
Only after the pilot looks right. Same commands with `prompts/full.csv` (200 prompts): roughly 4× the time."""),
("code", """# !python bench/run.py --controls --prompts prompts/full.csv
# !python bench/run.py --sweep --prompts prompts/full.csv
# !python bench/score.py --prompts prompts/full.csv && python bench/plot.py
# !python bench/decider_eval.py --prompts prompts/full.csv"""),
]

APP = [
("md", """# The website, on Colab's GPU

Runs `app/app.py` on a Colab GPU and gives you a public link. Open the link in a new tab, type a prompt, drag the switch slider, press **Generate**. Keep this tab open while you use it; closing it stops the server.

Setup: Runtime > Change runtime type > **T4 GPU** > Save. Then Runtime > Run all. The first Generate click loads the models (~10 GB download, 3-5 minutes); later clicks take about 20-40 seconds per image on a T4."""),
("code", SETUP),
("code", """APP_PASSWORD = ""   # set one so only people you give it to can use your GPU (user name: guest); blank = no login
os.environ["RR_AUTH"] = f"guest:{APP_PASSWORD}" if APP_PASSWORD else ""
!python app/app.py --share"""),
]

(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb(L1), ROOT / "notebooks/model_handoff_test.ipynb")
nbf.write(nb(L2), ROOT / "notebooks/colab_bench.ipynb")
nbf.write(nb(APP), ROOT / "notebooks/colab_app.ipynb")
print("wrote notebooks")
