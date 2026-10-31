# Design notes, findings so far, and open questions

Companion to RESEARCH_PLAN.md. This is the "brainstorm" half: why the code is shaped the
way it is, what we already know without a GPU, and what to watch for.

## 1. Findings so far (no GPU needed, Oct 2026)

Measured with `bench/compat_check.py` and a tensor-by-tensor comparison of the safetensors
files on the Hub (full numbers in `compat_report.md`).

1. **SSD-1B ships `madebyollin/sdxl-vae-fp16-fix` as its VAE.** Its VAE file hash differs
   from SDXL's, and 244 of 248 tensors differ, which looks alarming. But against the
   fp16-fix VAE the max abs difference is 0.015 (fp16 rounding of an fp32 file) and no
   element differs by more than 0.1%. The fp16-fix VAE is SDXL's VAE with rescaled
   internals, so **SDXL and SSD-1B share one latent space**. The plan's Level 1 "verify"
   item is answered analytically; the notebook still confirms it empirically (PSNR between
   decodes of one latent with three VAE copies).
2. **Segmind's text encoders are SDXL's up to rounding noise** (mean |Δ| ≈ 6e-5, max ≈ 0.05,
   consistent with a bf16 round-trip). Sharing SDXL's encoders across both UNets, as
   `rerouter/models.py` does, is safe. SSD-1B's copies also contain a `position_ids`
   buffer that SDXL's don't, which explains the different file sizes.
3. **Vega and Playground v2.5 ship SDXL's VAE byte for byte.** Playground still can't take
   an SDXL latent mid-run because it uses `scaling_factor` 0.5 and an EDM schedule. So the
   plan's table row "SDXL → Playground: same format, different schedule" is slightly off:
   the *weights* are the same, the *scale and schedule* differ. Either way, it breaks.
4. Both SDXL and SSD-1B ship `EulerDiscreteScheduler`, `scaled_linear`, `epsilon`,
   `timestep_spacing: leading`. Identical schedule, as expected.
5. The OpenRouter id `typesafe/jev-router` in the plan does not appear in any public
   listing. Jev's documented API is `POST https://api.typesafe.ai/v1/systemone` (model
   `jev-latest`), mirrored by AIMLAPI. `rerouter/decider.py` targets that.

## 2. Design decisions

**UNet swap, not two pipelines.** `handoff.impl: swap` keeps one pipeline and replaces
`pipe.unet` inside `callback_on_step_end`. The latent and the scheduler state never leave
the loop, so step 12 of the second model sees exactly what step 12 of a normal run would.
The two-pipeline route (`denoising_end` / `denoising_start`) re-encodes the prompt and
cuts the schedule by timestep, not step count: passing k/steps gave model A one step too
few (11 of 30 at 0.4 with SDXL's "leading" spacing), so `handoff_fraction()` now cuts at
model A's last timestep and both routes split at the same step. It is kept as
`impl: twopipe` for pairs with different encoders, and the Level 1 notebook checks the two
agree (PSNR). bench/run.py refuses twopipe: `impl` is not in the run key, so mixing would
go unnoticed.

**switch_fraction counts steps for the first model.** With this definition the endpoints
of each sweep equal the controls (`s2l@0` = large alone, `s2l@1` = small alone).
`schedule.plan()` collapses those endpoints to the controls' exact plan (unit-tested in
`tests/test_schedule.py::test_endpoints_collapse_to_controls`), so the sweep skips 0 and 1
and the controls stand in for them.

**Cost is linear in steps, so `s2l@f` and `l2s@(1-f)` cost the same.** This is the
cleanest head-to-head of T-Stitch order vs Hybrid SD order: same models, same steps per
model, same prompts, same seeds, only the order differs. `plot.py` pairs them by image
and runs a bootstrap CI plus an exact sign test. If one order wins on composition and the
other on text, that's the Level 3 story, and it's the signal Level 4's router needs.

**`peak_mem_gb` includes both UNets for every method,** even "small alone", because both
stay loaded so a handoff costs nothing. It measures this setup, not the memory of a
deployment that only ever runs the small model; footnote it if you report it.

**Same guidance scale for both models (7.5).** Segmind recommends 9.0 for SSD-1B alone.
The fairness rule wins, but note it as a possible handicap for the small model. A cheap
sensitivity check: rerun `small@30` at 9.0 on 10 prompts.

**"Same cost" vs control 3.** Two mechanisms: (a) the winners table picks the best setting
under each control-3 budget (±5%); (b) the matched-cost table pairs each handoff setting
with the nearest large-only control within ±15% and reports the paired delta. If no
control is within 15%, the cell says so rather than interpolating.

**Statistics.** Pair by (prompt, seed) to remove prompt difficulty, then average each
prompt's seeds: the two seeds of a prompt are correlated, so the unit is the prompt. Report
mean delta with a 2000-sample bootstrap 95% CI over prompts and an exact two-sided sign test
on per-prompt deltas; chart error bars are the SEM of per-prompt means. "Beats the matched
control" needs that CI entirely above 0, and the success reference is large@config steps. On the pilot, each
setting has 120 images; a mean PickScore delta of ~0.1 is roughly the smallest thing
you can hope to see. On the full set (400 images per setting) halve that. (This was worked
out for ImageReward; re-check it against the pilot's PickScore spread once real data exists.
The router's `pick_score` tolerance of 0.10 in decider_eval.py carries the same caveat.)

**Quality scorers.** PickScore is the default y-axis (human-preference proxy, strongest
signal per image). CLIPScore is weaker but cheap. OCR uses easyocr with 0/O, 1/I, 5/S
folded together, exact match as a substring plus a character error rate. Counting uses
OWLv2 zero-shot detection on `object`, which is noisy for plurals and colours; the plan
allows hand counts on the pilot, so `score.py --manual results/manual_counts.csv` merges
a hand-filled CSV and the hand count wins.

**PickScore replaced ImageReward (2026-10-26).** image-rerouter needs transformers 5.x on
Colab (current diffusers), and image-reward 1.5 (2023) was written for 4.x. Tested locally
with Colab's versions: it needs OpenAI's `clip` from GitHub (requirements had the wrong
package), then fails on at least six transformers APIs that moved or were removed
(modeling_utils helpers, `additional_special_tokens_ids`, `init_weights` tying,
`invert_attention_mask`, `get_extended_attention_mask`, `get_head_mask`). Pinning
transformers 4.x instead breaks current diffusers via huggingface_hub. PickScore_v1
(Kirstain et al. 2023, Pick-a-Pic) measures the same thing, is a plain CLIP-H checkpoint,
and loads with stock transformers. Score = `logits_per_image` = logit_scale x cosine,
identical to the official get_*_features recipe (22.7211 vs 22.7210 on a test sign).
Loaded fp16 on GPU (~2 GB, vs ~4 GB at fp32). Even so, PickScore + CLIP-L (fp32, ~1.7 GB)
+ OWLv2 (~0.6 GB) next to both UNets (~9.6 GB) leaves no room for SDXL activations at 1024 px
on a 15 GB T4, so the website scores only after a job's images are all generated and then
parks the scorers on CPU (`Scorers.park()`). bench/score.py runs in its own process, without
the UNets, so it needs neither. Sanity check on a sign
reading OPEN 24 HOURS: matching prompt 22.72, same prompt with "CLOSED" 19.57, unrelated
prompt 17.20.

**VQAScore and GenEval without their official packages (2026-10-28).** Both are now in
bench/score.py and the website. Official VQAScore (t2v_metrics 3.0) pins torch==2.5.1,
transformers==4.49.0, numpy==1.26.4, Pillow==10.0.0 and httpx==0.24.0 (Gradio 6 needs newer),
plus apex/deepspeed, so it can't share an environment with this app (same problem as
ImageReward above). Official GenEval scores with mmdet 2.x / mmcv-full 1.x. So: VQAScore =
the paper's recipe (P("Yes") to 'Does this figure show "{prompt}"? Please answer yes or no.')
with Qwen2.5-VL-3B-Instruct; GenEval = its 553 prompts (prompts/geneval_metadata.jsonl, MIT)
and its pass/fail rules ported line for line (`geneval_judge`, `relative_position`), with OWLv2
+ NMS for Mask2Former and CLIP-L on box crops (no masks) for colours. One OWLv2 threshold for
every task; GenEval uses 0.9 for counting because Mask2Former's scores run higher. Valid for
comparing our methods with each other, not with published numbers; plan: hand-label ~100
images and report each scorer's agreement. Qwen at ~7 GB doesn't fit beside both UNets on a
T4, so on GPUs under 24 GB the website moves the image models to CPU while scoring
(`ModelBank.move`, needs ~10 GB of system RAM). Untested on a GPU so far.

**Level 4 is a lookup, not a rerun.** Because Level 3 generates every (prompt, seed,
setting), routing a test prompt just means reading the row of the chosen setting. The
router's cost and quality on the held-out 30% are exact, no GPU, and it's compared with
(i) large@30, (ii) the best single fixed setting chosen on the training 70%, and (iii) the
per-prompt oracle from the answer key. `bench/run.py --method router` exists if you want
the actual routed images (and Jev latency in the wall clock).

**Profiles are coarse on purpose.** Jev's five answers are collapsed into six profiles
(text, hands_faces, layout, stylized, dense, generic). With ~42 training prompts, a finer
policy table would memorize noise. The regex baseline (`--provider heuristic`) answers
"does Jev beat a keyword match?", which is the honest version of the novelty claim.

**Resumability.** Every row is appended and fsynced; the run key is
(prompt_id, seed, method, direction, switch_fraction, steps). Error rows stay in the
file as a record and are retried on the next run.

## 3. What results would mean

- **Method 1 (small first) hurts composition** → `count_match` on composition drops as
  `s2l` switch fraction rises, while `l2s` stays flat. Prediction confirmed.
- **Method 2 (large first) hurts text/hands** → `ocr_match` on text drops as `l2s`
  switch fraction rises (more small-model late steps); hands judged by eye in the notebook.
- **Neither beats control 3** → the honest finding is that step reduction already captures
  the easy savings; routing only helps if it reorders *which* steps are cheap. DDSM found
  the best order depends on image type, which is exactly what makes a per-prompt router
  worth testing.
- **Router wins** → it should sit above the `best_fixed` point at equal cost, or left of it
  at equal quality, on the chart. Report the gap to the oracle as the headroom left.

## 4. Open questions and risks to watch

- **T4 memory at 1024².** Both UNets resident is ~9.6 GB of weights. The fp16-fix VAE
  decode at 1024² can spike. If OOM: attention slicing, VAE tiling, or run `small`-only
  jobs with the large UNet moved to CPU (but never time a run that includes a device move).
- **SSD-1B speed claim.** Segmind says 60% faster. On a T4 at fp16 expect less (memory
  bandwidth, not FLOPs, often bounds the small model). The dry-run cost model assumes
  0.80 s/step large vs 0.45 s/step small; replace with measured numbers from Level 1.
- **Jev latency is CPU/network time, not GPU time.** Log it separately (it's in
  `jev_answers.json` as `_latency_s`); don't add it to `gpu_seconds`.
- **Hard prompts look easy in text.** A "simple" product prompt can still need the large
  model for a logo. The oracle-vs-router gap per group tells you where Jev misroutes.
- **Automatic scores miss what people see.** Keep the 10-image spot check and the hands
  side-by-side in the notebook; write what you saw in `results/spot_check.md`.

## 5. Stretch ideas already supported by the code

- **Sandwich schedules:** `rerouter.schedule.plan_from_fractions(["large","small","large"],
  [0.4, 0.8], 30)` and pass `segments=` to `handoff.generate`. The swap callback handles
  any number of boundaries.
- **Third tier (Vega):** `ModelBank.load(("large","small","tiny"))`; the Level 1 notebook
  shows SSD-1B ↔ Vega.
- **Switch when ready:** in the callback, compute the predicted clean image
  `x0 = latents - sigma * noise_pred` each step and switch to the small model once
  `||x0_t - x0_{t-1}||` falls below a threshold. The callback already receives the
  latents; add `noise_pred` via `callback_on_step_end_tensor_inputs`.
- **Per-prompt cost model:** since cost ≈ n_large·c_L + n_small·c_S + c_decode, Level 1's
  two timings predict the cost of any schedule. Use it to pick sweep points at
  exactly-equal cost to the control-3 step counts (e.g. the `l2s` fraction that costs the
  same as large@18).
