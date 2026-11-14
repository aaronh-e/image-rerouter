# image-rerouter

Can switching models halfway make AI image generation cheaper?

Diffusion models make an image by starting from random noise and removing a little of it
at each step. Our project uses 30 denoising steps. The early steps decide the layout and
the later steps refine the detail. So we had a big model (SDXL) and a smaller, faster one
(SSD-1B) split the steps, with one model handing its in-progress image to the other
partway through. Then we measured how much cheaper each handoff made an image and how
much quality dropped.

**Results website:** https://aaronh-e.github.io/image-rerouter/

## Our findings (60-prompt pilot)

- **A handoff can produce an image that is 20% cheaper, with no clear loss in quality.**
  The big model does the first 12 steps and the small model does the last 18. The cost
  of generation is GPU time, so cutting time cuts cost.
- **The handoff beats fewer steps of the big model.** Running the big model alone for
  24 steps saves about the same time but scores 0.14 lower on PickScore.
- **The model that starts the handoff sets the layout and the ending model refines detail.**
- **The best handoff depends on the type of prompt.** Small-model-first is worse for
  composition and text prompts and better for dense ones. That's why the next step is a
  router that classifies each prompt and picks the best handoff.

The full plan is in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Notes on decisions and results
are in [docs/NOTES.md](docs/NOTES.md).

## The two models

- **SDXL base 1.0** is the big model (2.6B-parameter UNet). It's our quality benchmark.
- **Segmind SSD-1B** is a distilled SDXL (1.3B). Each step takes about 35% less time.
  It's our cost benchmark.

SSD-1B keeps SDXL's text encoders and VAE, so both models work in the same latent space.
That's what makes a handoff possible: one model can stop after any step and pass the
latent to the other.

We test the handoff both ways:
- **Method 1, `s2l`:** small model starts, big model finishes (the order T-Stitch uses).
- **Method 2, `l2s`:** big model starts, small model finishes (the order Hybrid SD uses).

## Running it

On a laptop (no GPU needed):

```bash
pip install -r requirements-cpu.txt
python -m pytest -q tests                # all tests run on CPU with fake models
python bench/compat_check.py             # Level 1: checks the two models are compatible
```

On a GPU (Colab T4 or better):

```bash
pip install -r requirements.txt
python bench/run.py --controls --prompts prompts/pilot.csv      # Level 2: baselines
python bench/run.py --sweep    --prompts prompts/pilot.csv      # Level 3: every handoff
python bench/score.py --prompts prompts/pilot.csv               # score the images
python bench/plot.py                                            # charts and tables
python bench/decider_eval.py --provider heuristic               # Level 4 with a keyword router
TYPESAFE_API_KEY=... python bench/decider_eval.py               # Level 4 with Jev
```

Or use the notebooks: `notebooks/model_handoff_test.ipynb` for Level 1 and
`notebooks/colab_bench.ipynb` for Levels 2 to 4 (results save to Drive).

To rebuild the results website from a finished run:

```bash
python site/build_site.py --run path/to/results-folder
python -m http.server 8781 --directory site     # preview at http://localhost:8781
```

## Try a handoff yourself

```bash
python app/app.py            # needs a GPU, opens http://127.0.0.1:7860
python app/app.py --demo     # any laptop, fake images
```

On Colab, open `notebooks/colab_app.ipynb`, pick a T4, run all, and click the public link.
You type a prompt, pick a method and a switch point, and see the images and how long they took.

## What's where

```
config.yaml           all the settings: models, steps, seeds, switch points, GPU prices, scorers
prompts/pilot.csv     60 prompts, 10 per type
prompts/full.csv      200 prompts (30-40 per type)
rerouter/             the handoff itself, model loading, and the Jev router
bench/                running, scoring, plotting, and evaluating the router
app/                  the try-it-yourself app (Gradio)
site/                 the results website
notebooks/            Colab notebooks
tests/                CPU tests, including a full fake run end to end
docs/                 notes and the model compatibility report
results/              outputs (not in git)
```

## A few rules we stuck to

- **`switch_fraction`** is the share of the 30 steps run by the *first* model. `s2l@0.4`
  means SSD-1B does steps 0-11 and SDXL does steps 12-29.
- **Same everything else.** Every setting uses the same prompts, seeds, step count,
  resolution and guidance. For a given prompt and seed, the only thing that changes is
  which model runs each step.
- **Results are append-only.** `results/results.csv` has one row per image, and rerunning
  a command skips images that are already done.
- **Cost comes from step counts.** Colab speed varies between sessions, so we time each
  model's step once and work out each image's cost from how many steps each model ran.

## Memory

Both UNets stay on the GPU (SDXL 5.1 GB + SSD-1B 2.7 GB), about 9.6 GB with the encoders
and VAE. If a 15 GB T4 runs out of memory at 1024x1024, set
`compute.attention_slicing: true` in `config.yaml`.
