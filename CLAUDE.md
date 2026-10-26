# image-rerouter: notes for Claude Code (local)

Read RESEARCH_PLAN.md (the brief) and docs/NOTES.md (decisions + findings) first.
This project started in a cloud session; the code, tests and docs are complete and
committed. Nothing has run on a GPU yet.

## State of play
- Level 1's VAE question is answered analytically (docs/compat_report.md): SSD-1B ships
  the sdxl-vae-fp16-fix VAE, so SDXL and SSD-1B share one latent space.
- 27 CPU tests pass: `python -m pytest -q tests` (uses --dry-run fakes, no GPU).
- The website (app/app.py) has optional quality checks that reuse bench/score.py's Scorers
  (PickScore, CLIPScore, OCR, count) plus a $/1000 column; scores go to results/app/scores.csv.
- PickScore replaced ImageReward (broken on transformers 5); column is `pick_score`. Why: docs/NOTES.md.
- Colab: the notebooks uninstall torchao after pip install (Colab's old torchao breaks
  new diffusers' import with a FqnToConfig error). Edit docs/make_notebooks.py, not the .ipynb files.
- Next real step: run Level 1 on a GPU (scripts/level1_local.py on this machine if it
  has one, else notebooks/model_handoff_test.ipynb on Colab), then Levels 2-4 via
  scripts/level2_controls.sh, level3_sweep.sh, level4_router.sh.

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
