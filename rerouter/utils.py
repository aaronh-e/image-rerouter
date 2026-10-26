"""Small helpers shared by every script: config, git hash, prompt files,
the `expected` column format, and append-only CSV logging.

Nothing in here imports torch, so the tests run on a laptop."""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Column order of results.csv. The first block is the schema from RESEARCH_PLAN.md
# section 6; the extras after `notes` make rows self-describing.
RESULT_COLUMNS = [
    "prompt_id", "group", "seed", "method", "direction", "switch_fraction", "steps",
    "gpu_type", "gpu_seconds", "peak_mem_gb", "usd_per_1000", "image_path",
    "pick_score", "hpsv2", "clip_score", "ocr_match", "count_match", "notes",
    # extras
    "ocr_text", "ocr_cer", "count_detected", "first_model", "second_model",
    "n_first_steps", "impl", "commit", "run_ts", "prompt",
]

# The columns that identify one run. Two rows with the same key are the same
# experiment, so run.py skips a job whose key already exists (resume support).
RUN_KEY = ["prompt_id", "seed", "method", "direction", "switch_fraction", "steps"]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit() -> str:
    """Short commit hash; outside a checkout (e.g. a zip uploaded to Colab), a
    fingerprint of the code instead, so rows from different code never look alike."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return source_fingerprint()


def source_fingerprint() -> str:
    """'src-' + 10 hex digits of the code and config that produce results. Line
    endings are normalised so a Windows-made zip and a Linux checkout agree."""
    h = hashlib.sha1()
    files = sorted(p for d in ("rerouter", "bench", "app") for p in (REPO_ROOT / d).rglob("*.py"))
    for p in files + [REPO_ROOT / "config.yaml"]:
        if p.exists():
            h.update(p.relative_to(REPO_ROOT).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return "src-" + h.hexdigest()[:10]


def repo_path(path: str | Path) -> Path:
    """A relative path that doesn't exist from the current folder is read from the
    repo root, so `prompts/pilot.csv` works whichever folder a script is run from."""
    p = Path(path)
    return p if p.exists() or p.is_absolute() else REPO_ROOT / p


# ---------------------------------------------------------------- prompts ----

def read_prompts(path: str | Path) -> list[dict]:
    path = repo_path(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"id", "group", "prompt", "expected"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return rows


def parse_expected(s: str | None) -> dict[str, Any]:
    """`expected` is a compact 'key=value;key=value' string so it survives CSV.

    Examples:
        "count=3;object=red cup"      -> {"count": 3, "object": "red cup"}
        "text=OPEN 24 HOURS"          -> {"text": "OPEN 24 HOURS"}
        ""                            -> {}
    """
    out: dict[str, Any] = {}
    if not s or not str(s).strip() or str(s).strip().lower() == "nan":
        return out
    for part in str(s).split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "count":
            out[k] = int(v)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------- settings ----

def setting_name(method: str, direction: str, switch_fraction: float, steps: int) -> str:
    """Folder name for images, e.g. 'handoff/s2l_sw0.40_st30' or 'control/large_st18'."""
    if method == "handoff":
        return f"{method}/{direction}_sw{switch_fraction:.2f}_st{steps}"
    return f"{method}/{direction}_st{steps}"


def image_path(images_dir: str | Path, method: str, direction: str,
               switch_fraction: float, steps: int, prompt_id: str, seed: int) -> Path:
    return Path(images_dir) / setting_name(method, direction, switch_fraction, steps) / f"{prompt_id}_s{seed}.png"


# ------------------------------------------------------------ results csv ----

def append_row(csv_path: str | Path, row: dict, columns: list[str] = RESULT_COLUMNS) -> None:
    """Append one row, writing the header first if the file is new.

    We flush after every row so a Colab disconnect loses at most one image."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})
        f.flush()
        os.fsync(f.fileno())


def run_key(row: dict) -> tuple:
    """Normalised key so '0.4' and 0.40 and '0.40' compare equal."""
    return (
        str(row["prompt_id"]), int(float(row["seed"])), str(row["method"]), str(row["direction"]),
        round(float(row["switch_fraction"]) if row.get("switch_fraction") not in ("", None) else -1.0, 3),
        int(float(row["steps"])),
    )


def existing_keys(csv_path: str | Path, include_errors: bool = False) -> set[tuple]:
    """Keys already in results.csv. Error rows are not counted by default, so a
    re-run retries them (and the old error row stays as a record)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    keys = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not include_errors and str(row.get("notes", "")).startswith("error"):
                continue
            keys.add(run_key(row))
    return keys
