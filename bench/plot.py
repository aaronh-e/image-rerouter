#!/usr/bin/env python
"""Cost-vs-quality curves, the per-group winner table, and the success check.

    python bench/plot.py                          # metric = pick_score
    python bench/plot.py --metric clip_score
    python bench/plot.py --metric ocr_match --groups text
    python bench/plot.py --metric count_match --groups composition

Outputs (results/plots/):
    cost_vs_quality_<metric>.png            overall curve: controls + s2l + l2s (+ router)
    cost_vs_quality_<metric>_by_group.png   2x3 small multiples, one per prompt group
    winners_<metric>.md / .csv              per group, the best method at each control-3 budget
    matched_cost_<metric>.csv               each handoff setting vs the nearest-cost large-only control
    order_head_to_head_<metric>.csv         s2l@f vs l2s@(1-f): same cost, paired comparison of the two orders
    success_<metric>.md                     which settings meet the 30%-cheaper + no-loss criterion

Conventions: x = mean GPU-seconds per image, y = mean metric, error bars = standard
error over images (prompt x seed). "large@N" = control #3 at N steps.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerouter import utils  # noqa: E402

# Validated categorical palette (dataviz skill reference instance). Fixed assignment:
# controls = blue, Method 1 (s2l) = orange, Method 2 (l2s) = aqua, router = violet.
COLORS = {"control": "#2a78d6", "s2l": "#eb6834", "l2s": "#1baf7a", "router": "#6250d6"}
GRID, INK, MUTED = "#e3e2de", "#0b0b0b", "#52514e"
GROUPS = ["composition", "text", "hands_faces", "product", "stylized", "dense"]
HIGHER_IS_BETTER = {"pick_score": True, "hpsv2": True, "clip_score": True, "ocr_match": True, "count_match": True, "ocr_cer": False}


# --------------------------------------------------------------- loading ----

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"prompt_id": str})
    df = df[~df["notes"].astype(str).str.startswith("error")].copy()
    for c in ("gpu_seconds", "switch_fraction", "steps", "seed", *HIGHER_IS_BETTER):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["setting"] = df.apply(setting_label, axis=1)
    df["family"] = np.where(df["method"] == "control", "control", np.where(df["method"] == "router", "router", df["direction"]))
    return df


def setting_label(r) -> str:
    if r["method"] == "control":
        return f"{r['direction']}@{int(r['steps'])}"
    if r["method"] == "router":
        return "router"
    return f"{r['direction']}@{r['switch_fraction']:.1f}"


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per setting: mean cost, mean quality, SEM, n images. The router is one
    setting even though it picks a different route per prompt."""
    d = df.dropna(subset=[metric, "gpu_seconds"]).copy()
    rt = d["family"] == "router"
    d.loc[rt, "direction"] = "router"
    d.loc[rt, ["switch_fraction", "steps"]] = np.nan
    g = d.groupby(["family", "setting", "method", "direction", "switch_fraction", "steps"], dropna=False)
    agg = g.agg(cost=("gpu_seconds", "mean"), q=(metric, "mean"), q_sem=(metric, "sem"), n=(metric, "size")).reset_index()
    return agg.sort_values("cost")


# -------------------------------------------------------------- plotting ----

def _style(ax, metric):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("GPU-seconds per image (lower is cheaper)", color=MUTED, fontsize=9)
    ax.set_ylabel(metric.replace("_", " "), color=MUTED, fontsize=9)


def draw(ax, agg: pd.DataFrame, metric: str, annotate: bool = True):
    # Control #3 line: large model at fewer steps = "the line to beat"
    ctl = agg[(agg.family == "control") & (agg.direction == "large")].sort_values("cost")
    if len(ctl):
        ax.errorbar(ctl.cost, ctl.q, yerr=ctl.q_sem, color=COLORS["control"], lw=2, marker="o", ms=7,
                    capsize=2, label="control: large only, 12-30 steps")
    small = agg[(agg.family == "control") & (agg.direction == "small")]
    if len(small):
        ax.errorbar(small.cost, small.q, yerr=small.q_sem, color=COLORS["control"], lw=0, elinewidth=1.5, marker="s", ms=8,
                    mfc="white", mew=2, capsize=2, label="control: small only")
    for fam, lab in (("s2l", "Method 1: small -> large"), ("l2s", "Method 2: large -> small")):
        m = agg[agg.family == fam].sort_values("cost")
        if len(m):
            ax.errorbar(m.cost, m.q, yerr=m.q_sem, color=COLORS[fam], lw=2, marker="o", ms=7, capsize=2, label=lab)
            if annotate:
                for _, r in m.iterrows():
                    ax.annotate(f"{r.switch_fraction:.1f}", (r.cost, r.q), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=7, color=MUTED)
    rt = agg[agg.family == "router"]
    if len(rt):
        ax.errorbar(rt.cost, rt.q, yerr=rt.q_sem, color=COLORS["router"], lw=0, elinewidth=1.5, marker="*", ms=14, capsize=2, label="Jev router")
    if annotate and len(ctl):
        for _, r in ctl.iterrows():
            ax.annotate(f"{int(r.steps)}", (r.cost, r.q), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7, color=MUTED)


def plot_overall(df, metric, out_dir, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = aggregate(df, metric)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    _style(ax, metric)
    draw(ax, agg, metric)
    ax.set_title(f"Cost vs quality{title_suffix}  (n = {int(agg.n.max()) if len(agg) else 0} images per point)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    p = Path(out_dir) / f"cost_vs_quality_{metric}.png"
    fig.savefig(p); plt.close(fig)
    return p, agg


def plot_by_group(df, metric, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = [g for g in GROUPS if g in set(df["group"])]
    if not groups:
        return None
    ncol = 3; nrow = math.ceil(len(groups) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), dpi=150, squeeze=False)
    for ax, g in zip(axes.flat, groups):
        _style(ax, metric)
        draw(ax, aggregate(df[df["group"] == g], metric), metric, annotate=False)
        ax.set_title(g, color=INK, fontsize=10, loc="left")
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=min(5, len(labels)))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = Path(out_dir) / f"cost_vs_quality_{metric}_by_group.png"
    fig.savefig(p); plt.close(fig)
    return p


# -------------------------------------------------------------- analysis ----

def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_delta(df: pd.DataFrame, metric: str, setting_a: str, setting_b: str) -> dict:
    """Mean (A - B) over images that exist for both settings, with a bootstrap CI
    and a sign test. Pairing by (prompt_id, seed) removes prompt difficulty."""
    a = df[df.setting == setting_a].set_index(["prompt_id", "seed"])[metric]
    b = df[df.setting == setting_b].set_index(["prompt_id", "seed"])[metric]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if j.empty:
        return dict(n=0, mean_delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_sign=np.nan)
    d = (j.a - j.b).to_numpy()
    rng = np.random.default_rng(0)
    boots = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    return dict(n=len(d), mean_delta=float(d.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), p_sign=sign_test_p(wins, losses), wins=wins, losses=losses)


def matched_cost_table(df: pd.DataFrame, metric: str, tolerance: float = 0.15) -> pd.DataFrame:
    """For every handoff/router setting, find the large-only control whose cost is
    closest (within +-tolerance) and report the paired quality delta. Positive
    delta = routing beats the 'line to beat' at about the same cost."""
    agg = aggregate(df, metric)
    ctl = agg[(agg.family == "control") & (agg.direction == "large")]
    rows = []
    for _, r in agg[agg.family.isin(["s2l", "l2s", "router"])].iterrows():
        if ctl.empty:
            break
        i = (ctl.cost - r.cost).abs().idxmin()
        c = ctl.loc[i]
        if abs(c.cost - r.cost) / max(c.cost, 1e-9) > tolerance:
            match = None
        else:
            match = c
        rec = dict(setting=r.setting, cost=r.cost, quality=r.q, control=(match.setting if match is not None else "none within tol"),
                   control_cost=(match.cost if match is not None else np.nan), control_quality=(match.q if match is not None else np.nan))
        if match is not None:
            rec.update(paired_delta(df, metric, r.setting, match.setting))
        rows.append(rec)
    return pd.DataFrame(rows)


def winners_table(df: pd.DataFrame, metric: str, slack: float = 0.05) -> pd.DataFrame:
    """Per group and per budget (each control-3 cost), the single best setting whose
    cost is <= budget*(1+slack). Shows which ORDER wins where."""
    rows = []
    hib = HIGHER_IS_BETTER.get(metric, True)
    for g in [x for x in GROUPS if x in set(df["group"])]:
        agg = aggregate(df[df["group"] == g], metric)
        budgets = agg[(agg.family == "control") & (agg.direction == "large")].sort_values("cost")
        for _, b in budgets.iterrows():
            cand = agg[agg.cost <= b.cost * (1 + slack)]
            if cand.empty:
                continue
            best = cand.loc[cand.q.idxmax() if hib else cand.q.idxmin()]
            rows.append(dict(group=g, budget_setting=b.setting, budget_cost=round(b.cost, 2), budget_quality=round(b.q, 3),
                             winner=best.setting, winner_cost=round(best.cost, 2), winner_quality=round(best.q, 3),
                             gain_over_control=round(best.q - b.q, 3)))
    return pd.DataFrame(rows)


def order_head_to_head(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Cost is linear in steps, so s2l@f and l2s@(1-f) run the SAME number of
    large and small steps and cost the same. That gives an exactly cost-matched
    comparison of the two orders with no interpolation: T-Stitch order vs Hybrid SD
    order, paired by image, overall and per group."""
    rows = []
    fr = sorted({round(x, 2) for x in df.loc[df.method == "handoff", "switch_fraction"].dropna()})
    groups = ["all"] + [g for g in GROUPS if g in set(df["group"])]
    for f in fr:
        a, b = f"s2l@{f:.1f}", f"l2s@{1 - f:.1f}"
        for g in groups:
            sub = df if g == "all" else df[df["group"] == g]
            d = paired_delta(sub, metric, a, b)
            if d["n"]:
                rows.append(dict(group=g, s2l=a, l2s=b, n=d["n"], mean_s2l_minus_l2s=round(d["mean_delta"], 4),
                                 ci_lo=round(d["ci_lo"], 4), ci_hi=round(d["ci_hi"], 4), p_sign=round(d["p_sign"], 4),
                                 winner=("s2l (small first)" if d["mean_delta"] > 0 else "l2s (large first)") if d["p_sign"] < 0.05 else "tie"))
    return pd.DataFrame(rows)


def success_report(df: pd.DataFrame, metric: str, matched: pd.DataFrame, min_saving: float = 0.30) -> str:
    agg = aggregate(df, metric)
    ref = agg[agg.setting == f"large@{int(df.steps.max())}"]
    if ref.empty:
        return "No full-step large control found; run the controls first.\n"
    ref = ref.iloc[0]
    lines = [f"# Success check ({metric})", "", f"Reference: {ref.setting} at {ref.cost:.2f} GPU-s, {metric} = {ref.q:.3f}", "",
             f"Criterion: >= {int(min_saving * 100)}% fewer GPU-seconds than the reference, quality delta vs reference whose 95% CI includes 0 or is positive, and beats the matched-cost large-only control.", "",
             "| setting | GPU-s | saving | delta vs ref [95% CI] | vs matched control | passes? |", "|---|---|---|---|---|---|"]
    for _, r in agg[agg.family.isin(["s2l", "l2s", "router"])].iterrows():
        saving = 1 - r.cost / ref.cost
        d = paired_delta(df, metric, r.setting, ref.setting)
        m = matched[matched.setting == r.setting]
        beats = (not m.empty) and pd.notna(m.iloc[0].get("mean_delta", np.nan)) and m.iloc[0]["mean_delta"] > 0
        no_loss = pd.notna(d["ci_hi"]) and d["ci_hi"] >= 0
        ok = saving >= min_saving and no_loss and beats
        mtxt = f"{m.iloc[0]['mean_delta']:+.3f} vs {m.iloc[0]['control']}" if not m.empty and pd.notna(m.iloc[0].get("mean_delta", np.nan)) else "n/a"
        lines.append(f"| {r.setting} | {r.cost:.2f} | {saving:+.0%} | {d['mean_delta']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] | {mtxt} | {'YES' if ok else 'no'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--metric", default="pick_score")
    ap.add_argument("--groups", nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = utils.load_config(args.config)
    out_dir = Path(args.out or cfg["paths"]["plots_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results or cfg["paths"]["results_csv"])
    if args.groups:
        df = df[df["group"].isin(args.groups)]
    if df[args.metric].notna().sum() == 0:
        sys.exit(f"no values for metric {args.metric}; run bench/score.py first")

    p1, agg = plot_overall(df, args.metric, out_dir, f" ({'+'.join(args.groups)})" if args.groups else "")
    p2 = plot_by_group(df, args.metric, out_dir)
    win = winners_table(df, args.metric)
    matched = matched_cost_table(df, args.metric)
    win.to_csv(out_dir / f"winners_{args.metric}.csv", index=False)
    (out_dir / f"winners_{args.metric}.md").write_text(win.to_markdown(index=False) if len(win) else "no data\n")
    matched.to_csv(out_dir / f"matched_cost_{args.metric}.csv", index=False)
    h2h = order_head_to_head(df, args.metric)
    h2h.to_csv(out_dir / f"order_head_to_head_{args.metric}.csv", index=False)
    report = success_report(df, args.metric, matched)
    (out_dir / f"success_{args.metric}.md").write_text(report)

    print(agg[["setting", "cost", "q", "q_sem", "n"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nWinners per group and budget:\n", win.to_string(index=False) if len(win) else "(none)")
    print("\nOrder head-to-head at equal cost (s2l@f vs l2s@1-f):\n", h2h.to_string(index=False) if len(h2h) else "(none)")
    print("\n" + report)
    print("Summary")
    print(f"  overall chart:   {p1}")
    print(f"  per-group chart: {p2}")
    print(f"  tables:          {out_dir}/winners_*.csv, matched_cost_*.csv, success_*.md")


if __name__ == "__main__":
    main()
