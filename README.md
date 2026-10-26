# image-rerouter

Can a cheap text-only decider route *parts of* image generation between a small and a
large diffusion model, cutting GPU-seconds without a visible quality loss?
Full plan: [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Design notes, findings so far and open
questions: [docs/NOTES.md](docs/NOTES.md).

**Models:** SDXL base 1.0 (large) and Segmind SSD-1B (small), both at fp16 with the
fp16-fix VAE. **Methods:** Method 1 `s2l` = small model first, large finishes
(T-Stitch order). Method 2 `l2s` = large first, small finishes (Hybrid SD order).

## Quick start

```bash
pip install -r requirements-cpu.txt      # laptop: tests, prompts, compat check, plots
python -m pytest -q tests                # everything below is exercised on CPU with --dry-run
python bench/compat_check.py             # Level 1 format table from the Hub, no GPU
```

On a GPU box (Colab T4 or better):

```bash
pip install -r requirements.txt
python bench/run.py --controls --prompts prompts/pilot.csv      # Level 2
python bench/run.py --sweep    --prompts prompts/pilot.csv      # Level 3
python bench/score.py --prompts prompts/pilot.csv               # adds quality columns
python bench/plot.py --metric pick_score                      # charts + tables
python bench/decider_eval.py --provider heuristic               # Level 4 baseline (offline)
TYPESAFE_API_KEY=... python bench/decider_eval.py               # Level 4 with Jev
```

Or open the notebooks: `notebooks/model_handoff_test.ipynb` (Level 1) and
`notebooks/colab_bench.ipynb` (Levels 2–4, results on Drive).

## The website (easiest way to try things)

```bash
python app/app.py            # on a machine with a GPU; opens http://127.0.0.1:7860
python app/app.py --demo     # any laptop: see the UI with fake images
```
On Colab, open `notebooks/colab_app.ipynb`, pick a T4 runtime, Run all, and click the public link it prints.
Prompt box, Method 1 / Method 2 picker, switch-point slider, a Sweep tab, and every run logged to `results/app/log.csv`.

## Layout

```
config.yaml                 every knob: models, steps, seeds, switch fractions, GPU price, scorers, decider
prompts/pilot.csv           60 prompts, 10 per group, with machine-checkable `expected`
prompts/full.csv            200 prompts (built by prompts/make_full.py; superset of pilot)
rerouter/schedule.py        (direction, switch_fraction, steps) -> which model runs which steps  [no torch]
rerouter/models.py          load SDXL + SSD-1B once; shared text encoders + VAE; two UNets on GPU
rerouter/handoff.py         generate(): UNet swap via callback_on_step_end (or two-pipeline route)
rerouter/decider.py         Jev typed questions -> profile -> (direction, switch_fraction); regex baseline
rerouter/utils.py           config, git hash, prompt files, `expected` format, append-only results CSV
bench/run.py                run controls / one setting / full sweep / router; resumable; errors logged as rows
bench/score.py              PickScore, HPSv2 (opt), CLIPScore, OCR match + CER, OWLv2 object count
bench/plot.py               cost-vs-quality curves, per-group winners, matched-cost test, success check
bench/decider_eval.py       answer key from the sweep, 70/30 split, learn policy, evaluate router by lookup
bench/compat_check.py       VAE/scheduler/weight-fingerprint table for model pairs
app/app.py                  Gradio website: type a prompt, pick a method and switch point, see images + GPU-seconds
notebooks/                  Level 1 handoff demo; Colab runner for Levels 2-4
scripts/                    level2_controls.sh, level3_sweep.sh, level4_router.sh
tests/                      CPU tests incl. a full dry-run of run -> score -> plot -> decider_eval
results/                    results.csv (append-only), images/, plots/   [git-ignored]
docs/                       compat_report.md, NOTES.md, make_notebooks.py
```

## Conventions that matter

- **`switch_fraction`** is the fraction of the *total* steps run by the **first** model.
  `s2l@0.4` with 30 steps: SSD-1B does steps 0–11, SDXL does 12–29. `s2l@0.0` equals
  large-alone and `s2l@1.0` equals small-alone, so the sweep endpoints double as a
  sanity check against the controls.
- **Equal cost without interpolation:** cost is linear in steps, so `s2l@f` and
  `l2s@(1-f)` run the same number of large and small steps. `bench/plot.py` reports this
  head-to-head per group (`order_head_to_head_*.csv`).
- **One row per image** in `results/results.csv`; the key is
  `(prompt_id, seed, method, direction, switch_fraction, steps)`. Re-running a command
  skips finished rows and retries error rows.
- **Timing** is `torch.cuda.synchronize()` wall clock around the whole pipeline call,
  after one warm-up run; peak memory is reset before each image.
- **`expected` column:** `count=3;object=red cup;...` for composition,
  `text=OPEN 24 HOURS` for text, empty otherwise.

## Level checklist

| level | gate | where |
|---|---|---|
| 1 | clean handoff image each way, timing table, pair map | `notebooks/model_handoff_test.ipynb` -> `results/level1_*.{csv,png}` |
| 2 | cost-vs-quality chart with the three controls | `scripts/level2_controls.sh` -> `results/plots/` |
| 3 | both method curves + controls, per-group winners, beats control 3? | `scripts/level3_sweep.sh` -> `winners_*.md`, `success_*.md`, `order_head_to_head_*.csv` |
| 4 | router point next to best fixed setting on held-out 30% | `scripts/level4_router.sh` -> `results/router_eval.md` |

## Memory on a 15 GB T4

Both UNets stay resident (SDXL 5.1 GB + SSD-1B 2.7 GB) plus encoders and VAE, about 9.6 GB
of weights. If you hit OOM at 1024², set `compute.attention_slicing: true` in
`config.yaml`, or add `bank.pipe.vae.enable_tiling()` after loading.
