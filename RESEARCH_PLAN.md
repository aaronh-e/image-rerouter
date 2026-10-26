# Image Model Re-router: Research Plan

> Context file for Claude Code. Read this whole file before writing any code. Work level by level: finish a level's checklist and pass its gate before starting the next level.

## 1. Project in one paragraph

This is a class research project. It asks whether a cheap "decider" can route parts of the image-generation process between a **small** and a **large** diffusion model, to cut compute without a noticeable loss in quality. It's inspired by **Jev** (TypeSafe AI), a text-only decision model that routes chatbot requests between cheap and expensive LLMs. Jev can't see images or generate them. It reads text and returns probabilities for typed questions (yes/no, pick-one, score). This project applies the same idea *inside* a diffusion model.

**Central goal:** spend less compute per image without quality dropping substantially.

**Main metric:** GPU-seconds per image that passes the quality bar, compared with the large model doing all the work.

## 2. The two methods

| # | Name | What happens | Prior work |
|---|------|--------------|------------|
| 1 | **Small-to-Large Step Handoff** | The small model runs the early denoising steps (layout). The large model runs the late steps (detail). | T-Stitch, https://arxiv.org/abs/2402.14167 |
| 2 | **Large-to-Small Step Handoff** | The large model runs the early steps (composition). The small model runs the late steps (texture). | Hybrid SD, https://arxiv.org/abs/2408.06646 |

> **Out of scope for now:**
> - Segmentation-guided regional refinement (a former Method 3).
> - Closed frontier models (e.g. Nano Banana, GPT Image). Their APIs never expose a half-finished image, so they can't do mid-generation handoffs.
>
> Don't build anything for either unless this file is updated to bring it back.

Related work for context: CATImage, prompt-level routing between models (https://arxiv.org/abs/2506.14753). DDSM, a different model size for each step (https://arxiv.org/abs/2310.03337); it found the best order depends on the kind of images.

### Predictions to test

- **Method 1:** the small model sets the composition, so **composition prompts should suffer** (wrong object counts or positions). The large model can't fix the layout later.
- **Method 2:** composition should be good, but **text and hands should suffer**, because they form in the late steps that the small model now handles.

## 3. Technical setup

### Models (open weights, Hugging Face)

- **Large:** `stabilityai/stable-diffusion-xl-base-1.0` (SDXL)
- **Small:** `segmind/SSD-1B`. Segmind describes it as distilled from SDXL, 50% smaller and 60% faster, under the Apache 2.0 license.
- **Optional smaller tier:** `segmind/Segmind-Vega`
- **VAE:** `madebyollin/sdxl-vae-fp16-fix`, which avoids black images at fp16 on a T4.
- **Decider (novel step):** Jev via OpenRouter (`typesafe/jev-router`) or the Jev API. It reads text only.

### Handoff compatibility rule

Two models can pass a *half-finished* image to each other only if they agree on two things: **the compressed image format** and **the noise schedule**. Being from the same company doesn't matter.

#### What the compressed image format (VAE latent) is

A 1024×1024 image has about 3 million numbers (1 million pixels × red, green, blue). That's too many to run a big model over at every step. So each model family uses a separate small network called a **VAE** (variational autoencoder):

- **The encoder** squeezes an image into a much smaller grid called the **latent**. For SDXL, a 1024×1024 image becomes a 128×128 grid with 4 numbers per spot, about 48 times fewer numbers.
- **The decoder** expands a latent back into a full picture.

The diffusion model does all its work on the latent, and the decoder turns the result into the final image.

**Each VAE invents its own "language" for those numbers.** One VAE might use its first number for something like brightness. Another, trained separately, might use the same position for something unrelated. So model A's latent is meaningless to model B unless both use the same VAE. It's like a ZIP file: it only opens with a program that understands that format.

Formats can differ obviously (SD 1.5 and SDXL use 4 numbers per spot, FLUX uses 16) or subtly: the same shape, but a different VAE or scaling. **The subtle case is the dangerous one.** Nothing crashes, and the image just comes out garbled.

#### What the noise schedule is

Generation starts from pure static, and each step removes some of it. The **noise schedule** is the rule for **how much noise is left at each step**, e.g. "at step 10 of 30, the image is about 60% noise." Each model is trained on one schedule. In effect it learns: "at step 10, I expect exactly this much noise, and I remove the right amount."

If model A hands over at step 12, model B must read "step 12" as the same amount of noise. If B was trained on a different schedule, it misjudges how noisy the image is and removes too much or too little. The result looks blurry, washed out, or burnt. Some models also define the noise itself differently (Playground v2.5 uses the "EDM" convention; FLUX and SD 3.5 use "flow matching"), which is another way two models can disagree.

#### Relay-race analogy

- **Compressed format:** both runners have to hold the same kind of baton.
- **Noise schedule:** both have to agree on where the handoff point on the track is.

| Pair | Same format? | Same schedule? | Mid-generation handoff? |
|---|---|---|---|
| SDXL ↔ SSD-1B (Stability ↔ Segmind) | Expected yes (distilled from SDXL), verify | Yes | **Should work** |
| SD 1.5 → SDXL | No (different VAE) | Yes | **Breaks** |
| SDXL → Playground v2.5 | Same shape | No (EDM) | **Breaks** |
| SDXL → FLUX (any version) | No (4 vs 16+ numbers per spot) | No | **Impossible**, shapes don't even match |

When either test fails, the only option is a **pixel bridge**: decode model A's result to a normal image, then let model B redraw it with img2img. This works for any pair, but it costs an extra decode and encode, and model B may drift from A's layout.

**How to check a pair before testing it:** compare the VAE config (`latent_channels`, `scaling_factor`, `shift_factor`, `latents_mean`), the VAE weight fingerprint (SHA-256 on Hugging Face), and the scheduler config (`_class_name`, `beta_schedule`, `prediction_type`). The format table in `model_handoff_test.ipynb` does this automatically. SSD-1B's model card doesn't say outright that it uses SDXL's VAE, so **Level 1 must verify this**.

### How to do a handoff in diffusers

Option A: two pipelines. The first stops early and returns the latent, and the second continues from it.

```python
lat = small_pipe(prompt, num_inference_steps=30, denoising_end=0.4,
                 output_type="latent", generator=g).images
img = big_img2img(prompt, image=lat, denoising_start=0.4,
                  num_inference_steps=30, generator=g).images[0]
```

Option B: one pipeline, swap the UNet mid-run. This only works when both models share text encoders and VAE.

```python
def swap(pipe, i, t, kw):
    if i == n_first - 1:
        pipe.unet = unet_second
    return kw
pipe(prompt, num_inference_steps=30, callback_on_step_end=swap)
```

### Compute

There's no local GPU. Use **Google Colab (T4 or better)** or an hourly cloud GPU. Record the GPU type and hourly price in `config.yaml`, so GPU-seconds can be converted to dollars.

## 4. Benchmark design (shared by every method)

### Prompts

- **Pilot:** 60 prompts, 10 per group. **Full:** about 200 prompts. Run every prompt with **2 seeds** (e.g. 42 and 1234).
- A seed fixes the starting noise, so the same prompt and seed always produce the same image. Two seeds show whether a result is consistent or luck.
- The six groups:
  1. **composition:** counting, positions, colors (GenEval style), e.g. "three red cups left of a blue plate"
  2. **text:** readable text, e.g. "a storefront sign that says OPEN 24 HOURS"
  3. **hands_faces:** close-up hands and faces
  4. **product:** photoreal products
  5. **stylized:** stylized art
  6. **dense:** long, dense prompts (DPG-Bench style)
- For the **composition** and **text** prompts, store the expected answer: the object count for counting prompts, the exact string for text prompts.

### Fairness rule

Every method uses the same prompts, seeds, step count (30 by default), resolution (1024×1024), guidance scale, scheduler and GPU. Time runs one image at a time, and skip the first warm-up run.

### Cost measurements

- `gpu_seconds`: wall-clock generation time, with `torch.cuda.synchronize()` before and after
- `peak_mem_gb`: `torch.cuda.max_memory_allocated()`, reset before each run
- `usd_per_1000`: `gpu_seconds × hourly_price / 3600 × 1000`

### Quality measurements

- **Automatic:** PickScore or HPSv2 (predicted human preference) and CLIPScore (prompt match). PickScore replaced ImageReward on 2026-10-26, because ImageReward no longer runs with current libraries (see docs/NOTES.md).
- **Objective:** an OCR exact-match or character-error rate for `text` prompts, and an object-count match for `composition` prompts (counting by hand is fine for the pilot)
- **Human (optional):** blind side-by-side pairs, at least 100 per comparison, analyzed with a sign test

### Controls (every method is compared against these)

1. **Large model alone at full steps:** the quality ceiling
2. **Small model alone at full steps:** the cost floor
3. **Large model alone with fewer steps** (e.g. 12, 18, 24): **the line to beat**. If cutting the large model's steps matches a method at the same cost, the routing adds nothing.

### Main chart

A cost-vs-quality curve: GPU-seconds on the x axis, quality score on the y axis. Plot one line per method, sweeping its setting, plus the controls. Make it overall and per prompt group.

### Success criterion

At least **30% fewer GPU-seconds** than the large model alone, with no significant quality loss, and **better than control #3** at the same cost.

## 5. Scaffolded levels

Each level has a goal, tasks, a gate, and a fallback result that can be reported if time runs out there.

### Level 0: Basics (Week 1, no code)

- [ ] Watch the image-model video (https://www.youtube.com/watch?v=x2GRE-RzmD8) and 3Blue1Brown's video on attention in transformers.
- [ ] Read the abstract and main results table of T-Stitch and Hybrid SD.
- **Gate:** you can explain what a denoising step is, why early steps set the layout and late steps add detail, and why a handoff needs a shared VAE.

### Level 1: Can models hand off at all? (Weeks 1–2)

Level 1 answers a yes/no question for each pair of open models: can model A start an image mid-generation and model B finish it, without the result breaking?

- [ ] Load SDXL and SSD-1B in diffusers (fp16, fp16-fix VAE).
- [ ] Generate the same prompt and seed with each, and time both.
- [ ] **Verify compatibility:** decode an SSD-1B latent with SDXL's VAE, or run `model_handoff_test.ipynb`. The notebook also tests same-company pairs (SSD-1B ↔ Vega), cross-company pairs (SSD-1B ↔ SDXL, SDXL → Playground v2.5), and a different-format pair (SD 1.5 → SDXL).
- [ ] Do one handoff in each direction at 40%: Method 1 (small → large) and Method 2 (large → small).
- **Gate:** a clean handoff image in each direction, a timing table (large alone, small alone, both handoffs), and a short list of which tested pairs work and which break.
- **Fallback result:** a working demo that two models can share one image mid-generation, plus a map of which models can hand off to which, with measured speeds.

### Level 2: Test bench and controls (Weeks 2–3)

- [ ] Create `prompts/pilot.csv` (60 prompts) with columns `id, group, prompt, expected`.
- [ ] Write `bench/run.py` to run any method over a prompt file and both seeds, saving images and logging one row per image.
- [ ] Write `bench/score.py` to add PickScore/HPSv2, CLIPScore, OCR and count checks.
- [ ] Run the three controls.
- [ ] Spot-check 10 images by eye against their scores.
- **Gate:** a cost-vs-quality chart showing the three controls.
- **Fallback result:** how fast quality falls as steps are cut or the model shrinks (diminishing returns of compute).

### Level 3: Methods 1 and 2 (Weeks 3–4)

- [ ] Method 1: switch from small to large at 0, 20, 40, 60, 80 and 100% of the steps.
- [ ] Method 2: the same switch points, from large to small.
- [ ] Add both curves to the control chart.
- [ ] Make a table of the winning order for each prompt group at the same cost.
- [ ] Check the predictions: composition accuracy for Method 1, OCR and hand quality for Method 2.
- **Gate:** one chart with both curves and the controls, the per-group winner table, and an answer on whether either method beats control #3.
- **Fallback result:** T-Stitch's order vs. Hybrid SD's order, head to head on the same models and prompts. This is the project's main result.

### Level 4: Novel step, a per-prompt decider (Week 5)

- [ ] From the Level 3 results, find the best order and switch point for each prompt (the answer key).
- [ ] Split the prompts 70/30. Set the rules on the 70% and test only on the held-back 30%.
- [ ] Ask Jev 4–6 typed questions per prompt (e.g. readable text? counting or positions? close-up hands?), and turn the answers into an order and switch point.
- [ ] Compare the Jev router with the best single fixed setting on the held-back prompts.
- **Gate:** the router's result on the cost-vs-quality chart next to the best fixed setting. A win or a loss both count as a finding.

### Stretch ideas

- **Switch when ready:** switch models once the predicted image stops changing between steps, instead of at a fixed step.
- **Sandwich schedules:** try large → small → large, or small → large → small, using the same sweep code.
- **Third tier:** add Segmind Vega as an even smaller model, and test three-model schedules.

## 6. Suggested repo layout

```
image-rerouter/
├── RESEARCH_PLAN.md          # this file
├── config.yaml               # model IDs, steps, resolution, seeds, GPU type + $/hr
├── prompts/
│   ├── pilot.csv             # 60 prompts: id, group, prompt, expected
│   └── full.csv              # ~200 prompts
├── rerouter/
│   ├── models.py             # load pipelines/UNets once, shared text encoders + VAE
│   ├── handoff.py            # method 1 and 2 (switch_fraction, direction)
│   └── decider.py            # Jev questions -> (direction, switch_fraction)
├── bench/
│   ├── run.py                # python bench/run.py --method handoff --direction s2l --switch 0.4 --prompts prompts/pilot.csv
│   ├── score.py              # adds quality columns to results
│   └── plot.py               # cost-vs-quality curves, per-group tables
├── notebooks/
│   └── model_handoff_test.ipynb
└── results/
    ├── images/               # {method}/{setting}/{prompt_id}_s{seed}.png
    └── results.csv
```

### `results.csv` schema (one row per image)

`prompt_id, group, seed, method, direction, switch_fraction, steps, gpu_type, gpu_seconds, peak_mem_gb, usd_per_1000, image_path, pick_score, hpsv2, clip_score, ocr_match, count_match, notes`

## 7. Rules for Claude Code

- **Don't train or fine-tune models.** Every method reuses pretrained weights and only changes which model runs which steps. Training is only allowed in a stretch goal.
- **Keep runs reproducible:** fixed seeds, settings in `config.yaml`, and the git commit hash logged with every results file.
- **Never overwrite results.** Append to `results.csv`, and skip runs that already exist, so a crashed Colab session can resume.
- **Load models once and reuse them.** Share the text encoders and VAE between SDXL and SSD-1B to save GPU memory, and free models (`del`, `gc.collect()`, `torch.cuda.empty_cache()`) before loading others.
- **Start every level on the pilot set.** Move to `full.csv` only after the pilot results look right.
- **Log failures as rows** (with `notes="error: ..."`) instead of crashing the whole run.
- Write the code so a beginner can follow it: short functions, comments that explain *why*, and a plain-English summary printed at the end of every script.

## 8. Known risks

- Hard prompts can look easy in text, so Jev may misroute them.
- Automatic scores miss things people notice. Keep a human spot check.
- FLOP savings often shrink once measured as real time, so always report wall-clock GPU-seconds.
- SSD-1B and SDXL VAE compatibility is not yet confirmed (Level 1).
- Jev details: TypeSafe names its training method (RLCD) and a "parallel sampler" but hasn't published how they work. Measured latency in public tests was about 0.3–0.5 seconds per decision.
