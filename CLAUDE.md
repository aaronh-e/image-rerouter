# image-rerouter: notes for Claude Code (local)

Read RESEARCH_PLAN.md (the brief) and docs/NOTES.md (decisions + findings) first.
This project started in a cloud session; the code, tests and docs are complete and
committed. Levels 2 and 3 have run on the 60-prompt pilot twice: on Colab T4s over four
sessions (Nov 2-8, 2026) and in one session on an RTX PRO 6000 Blackwell (Nov 9). The
Blackwell run is the clean dataset (~/Downloads/results-20261110T061803Z-1-001.zip). In
the T4 run, the 35 large@30 images from the Nov 2 session came from a different software
environment and are different pictures for the same seed; don't compare against them.
This machine has no GPU.

## State of play
- Level 1's VAE question is answered analytically (docs/compat_report.md): SSD-1B ships
  the sdxl-vae-fp16-fix VAE, so SDXL and SSD-1B share one latent space.
- 69 CPU tests pass: `python -m pytest -q tests` (uses --dry-run fakes, no GPU).
- Plots and decider_eval compute cost from step counts (config.yaml compute.step_seconds,
  per GPU name; an unlisted GPU is fitted from its own controls), not the stopwatch: Colab
  speed varies ~20% between sessions. Prices are per GPU too (compute.usd_per_hour, or
  RR_USD_PER_HOUR from the notebook); an unpriced GPU gets blank dollars. Why: docs/NOTES.md.
- The OWLv2 counter calls proc.image_processor.post_process_object_detection: transformers 5
  removed the processor-level method, which silently disabled count_match (empty in both pilot runs).
- The website (app/app.py) has optional quality checks that reuse bench/score.py's Scorers
  (VQAScore, PickScore, CLIPScore, OCR, count) plus a $/1000 column; scores go to results/app/scores.csv.
  Its GenEval tab runs GenEval prompts with GenEval's rules (OWLv2 + CLIP, not the official detector).
- prompts/paper.csv (built by prompts/make_paper.py) = full.csv's 200 + GenEval's 553.
  VQAScore uses Qwen2.5-VL-3B, not t2v_metrics. Why: docs/NOTES.md.
- PickScore replaced ImageReward (broken on transformers 5); column is `pick_score`. Why: docs/NOTES.md.
- Colab: the notebooks uninstall torchao after pip install (Colab's old torchao breaks
  new diffusers' import with a FqnToConfig error). Edit docs/make_notebooks.py, not the .ipynb files.
- site/ is the results website. site/build_site.py turns an unzipped results folder into
  site/data.json and site/sprites/ (git-ignored, ~33 MB); site/index.html reads them.
  It is published as a claude.ai artifact: https://claude.ai/artifact/XzgZtoKGDmtt7NYgJB6kUc.
  Copy style the user asked for: short plain sentences, no small label text, pictures first.
- Next steps: rescore counts on the pilot (`bench/score.py --only count`, needs Colab),
  then Level 4 (`bench/decider_eval.py`), then the full prompt set.

## Conventions (do not change silently)
- switch_fraction = fraction of total steps run by the FIRST model; s2l@0 == large alone.
- results/results.csv is append-only; run key = (prompt_id, seed, method, direction,
  switch_fraction, steps). Never overwrite results.
- All settings live in config.yaml. Log the git commit in every row (utils.git_commit).
- No training or fine-tuning. Same prompts, seeds, steps, resolution, guidance for every method.

## Commands
    python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
    pip install -r requirements.txt                       # or requirements-cpu.txt without a GPU
    python app/app.py --demo                              # the website, fake images
    python scripts/level1_local.py --size 512 --steps 12  # Level 1 for real (needs GPU / Apple Silicon)
    python bench/run.py --controls --dry-run && python bench/score.py --fake && python bench/plot.py
    python site/build_site.py --run <unzipped results folder>   # rebuild the website's data and images
    python -m http.server 8781 --directory site                # preview the website at http://localhost:8781
