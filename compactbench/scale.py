"""Cross-scale analysis: read a scale-ladder sweep and ask where the findings hold.

The sweep writes one JSON per (task, rung) into runs/scale/. This module reads them
back and answers the question the ladder exists to answer: is the accuracy collapse
located at a fixed FRACTION of a model's own full KV budget, which is what the
error floor implies, or at a fixed ABSOLUTE bytes-per-token, which would make the
axis a hardware unit instead of an information one.

Models differ widely in their full-cache anchor. A GQA model with two KV heads and
a multi-head model of similar parameter count can sit an order of magnitude apart
in bytes per token, and that spread is the leverage the comparison needs.

  compactbench scale --runs runs/scale --out figures
"""
import glob
import json
import os
import re

# Findings are averaged over scorers, excluding the random control, which exists to
# floor the comparison and would drag the collapse point toward itself.
CONTROL = "Random"


def _rung(path):
    """runs/scale/frontier-qwen7b.json -> 'qwen7b'."""
    return re.sub(r"^[a-z]+-", "", os.path.basename(path)[:-len(".json")])


def load(runs="runs/scale"):
    """All rungs found under `runs`, as {tag: {task: parsed json}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(runs, "*.json"))):
        task = os.path.basename(path).split("-", 1)[0]
        out.setdefault(_rung(path), {})[task] = json.load(open(path))
    return out


def _interp_crossing(xs, ys, level):
    """First x (scanning high to low) where y falls below `level`, interpolated."""
    pts = sorted(zip(xs, ys), reverse=True)
    for (x1, y1), (x0, y0) in zip(pts, pts[1:]):
        if y1 >= level > y0:
            if y1 == y0:
                return x0
            t = (y1 - level) / (y1 - y0)
            return x1 + t * (x0 - x1)
    return None


def collapse_point(frontier, control=CONTROL):
    """Where method-averaged accuracy falls to half the full-cache baseline.

    Returned in both units: `frac` of the model's own full budget, and `bpt` bytes
    per token of history. Whichever of the two is stable across rungs is the unit
    the collapse actually lives in.
    """
    pts = frontier["points"]
    base = next((p["acc"] for p in pts if p["method"] == "baseline"), None)
    if not base:
        return None
    scored = [p for p in pts if p["method"] not in ("baseline", control)]
    by_frac = {}
    for p in scored:
        by_frac.setdefault(round(p["bpt_frac"], 6), []).append(p["acc"])
    fracs = sorted(by_frac)
    accs = [sum(by_frac[f]) / len(by_frac[f]) for f in fracs]
    half = 0.5 * base
    frac = _interp_crossing(fracs, accs, half)
    full = frontier["full_bpt_bytes_per_tok"]
    return {"baseline_acc": base, "half_level": half,
            "frac": frac, "bpt": None if frac is None else frac * full,
            "full_bpt": full}


def reversibility_crossover(rev):
    """Budget fraction where reversible retention overtakes the lossy summary."""
    pts = sorted(rev["points"], key=lambda p: p["budget_frac"])
    xs = [p["budget_frac"] for p in pts]
    d = [p["reversible_acc"] - p["irreversible_acc"] for p in pts]
    for (x0, d0), (x1, d1) in zip(zip(xs, d), zip(xs[1:], d[1:])):
        if d0 <= 0 < d1:
            return x0 + (0 - d0) / (d1 - d0) * (x1 - x0)
    return None


def attribution_split(attr, structural="StreamingLLM", scored="SnapKV"):
    """Overclaim rate of a positional policy vs a content-scored one, worst budget."""
    def worst(m):
        rows = [s for s in attr["summary"]
                if s["method"] == m and s["overclaim"] is not None]
        return max((s["overclaim"] for s in rows), default=None)
    return {"structural": worst(structural), "scored": worst(scored)}


def calibration_gap(conf):
    """Baseline ECE vs the worst compacted ECE, and whether stated confidence moved."""
    rows = conf["summary"]
    base = [s for s in rows if s["ratio"] == 0.0]
    comp = [s for s in rows if s["ratio"] > 0.0]
    if not base or not comp:
        return None
    b = max(base, key=lambda s: s["n"])
    w = max(comp, key=lambda s: s["ece"])
    return {"base_ece": b["ece"], "worst_ece": w["ece"],
            "ratio": (w["ece"] / b["ece"]) if b["ece"] else None,
            "base_conf": b["mean_conf"], "worst_conf": w["mean_conf"]}


def summarise(runs="runs/scale"):
    """One row per rung, with each finding reduced to the number that tests it."""
    rows = []
    for tag, tasks in load(runs).items():
        row = {"rung": tag}
        if "frontier" in tasks:
            f = tasks["frontier"]
            row["model"] = f.get("model")
            row["quant"] = f.get("quant", "fp16")
            row["collapse"] = collapse_point(f)
        if "reversibility" in tasks:
            row["crossover"] = reversibility_crossover(tasks["reversibility"])
        if "attribution" in tasks:
            row["attribution"] = attribution_split(tasks["attribution"])
        if "confidence" in tasks:
            row["calibration"] = calibration_gap(tasks["confidence"])
        rows.append(row)
    return rows


def _fmt(x, pct=False, nd=2, tex=False):
    if x is None:
        return "--"
    if pct:
        return f"{100*x:.0f}" + (r"\%" if tex else "%")
    return f"{x:.{nd}f}"


def latex_table(rows):
    """The replication table as it appears in the paper's cross-scale section."""
    out = [r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Model & Prec. & Full BPT & Collapse (frac) & Collapse (B/tok) & "
           r"Crossover & Overclaim (pos./scored) & ECE $\times$ \\", r"\midrule"]
    for r in sorted(rows, key=lambda r: r.get("collapse", {}).get("full_bpt") or 0):
        c = r.get("collapse") or {}
        a = r.get("attribution") or {}
        k = r.get("calibration") or {}
        name = (r.get("model") or r["rung"]).split("/")[-1].replace("_", r"\_")
        out.append(
            f"\\texttt{{{name}}} & {r.get('quant','--')} & "
            f"{c.get('full_bpt', 0):,.0f} & {_fmt(c.get('frac'), pct=True, tex=True)} & "
            f"{'--' if c.get('bpt') is None else format(c['bpt'], ',.0f')} & "
            f"{_fmt(r.get('crossover'), pct=True, tex=True)} & "
            f"{_fmt(a.get('structural'))} / {_fmt(a.get('scored'))} & "
            f"{_fmt(k.get('ratio'), nd=1)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def _rung_curves(data):
    """{label: (xs_fraction, ys_accuracy, full_bpt)} for every rung with a frontier run."""
    out = {}
    for tag, tasks in sorted(data.items()):
        f = tasks.get("frontier")
        if not f:
            continue
        by = {}
        for p in f["points"]:
            if p["method"] in ("baseline", CONTROL):
                continue
            by.setdefault(round(p["bpt_frac"], 6), []).append(p["acc"])
        xs = sorted(by)
        ys = [sum(by[x]) / len(by[x]) for x in xs]
        label = (f.get("model") or tag).split("/")[-1]
        if f.get("quant") == "nf4":
            label += " (nf4)"
        out[label] = (xs, ys, f["full_bpt_bytes_per_tok"])
    return out


NEW_GENERATION = ("Qwen3-", "phi-4", "OLMo-2", "Mistral-Small")
GEN_COLORS = {"old": "#2a78d6", "new": "#eb6834"}   # validated pair (CVD dE 24.7)


def _short(label):
    """Direct-label name: family and size only, '*' for 4-bit weights."""
    name = label.removesuffix(" (nf4)")
    for long, short in (("Mistral-Small-24B-Instruct-2501", "Mistral-24B"),
                        ("OLMo-2-1124-13B-Instruct", "OLMo-2-13B"), ("phi-4", "Phi-4")):
        name = name.replace(long, short)
    return name + ("*" if label.endswith(" (nf4)") else "")


def generation_fig(runs, out):
    """Main-text frontier figure coloured by model generation.

    One hue per generation, never one per model: with a dozen rungs a per-model
    palette would cycle. Older rungs are thin and share one legend entry; the
    newer ones are direct-labelled. Every rung is drawn, 4-bit runs included, so
    the figure shows exactly the settings the paper's numbers are computed over.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _rung_curves(load(runs))
    # Drawn at print size (0.55 textwidth ~ 3.0in) so fonts are not shrunk.
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    labelled = []
    for label, (xs, ys, _bpt) in curves.items():
        base = label.removesuffix(" (nf4)")
        gen = "new" if base.startswith(NEW_GENERATION) else "old"
        if gen == "old":
            ax.plot(xs, ys, color=GEN_COLORS["old"], linewidth=1.0, alpha=0.55)
        else:
            ax.plot(xs, ys, color=GEN_COLORS["new"], linewidth=2.0, marker="o",
                    markersize=3.5)
            labelled.append((label, xs, ys))
    # Direct labels at the 50% point, nudged apart so they do not collide.
    at = sorted(((ys[xs.index(0.5)], lab) for lab, xs, ys in labelled), reverse=True)
    last = None
    for y, lab in at:
        y = y if last is None else min(y, last - 0.075)
        ax.annotate(_short(lab), (0.5, y),
                    xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=6.5, color="#52514e", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        last = y
    ax.plot([], [], color=GEN_COLORS["old"], linewidth=1.0,
            label="released before Oct 2024")
    ax.plot([], [], color=GEN_COLORS["new"], linewidth=2.0, marker="o",
            markersize=3.5, label="released Nov 2024 onwards")
    ax.set_xlabel("budget (fraction of the model's full cache)", fontsize=7)
    ax.set_ylabel("scorer-averaged accuracy", fontsize=7)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "scale_gen.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def generation_bytes_fig(runs, out):
    """Appendix companion to `generation_fig`: every rung against budget fraction
    (left) and absolute bytes per token (right, log), coloured by generation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _rung_curves(load(runs))
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    for label, (xs, ys, bpt) in curves.items():
        base = label.removesuffix(" (nf4)")
        new = base.startswith(NEW_GENERATION)
        kw = dict(color=GEN_COLORS["new" if new else "old"],
                  linewidth=2.0 if new else 1.0, alpha=1.0 if new else 0.55,
                  marker="o" if new else None, markersize=3)
        axes[0].plot(xs, ys, **kw)
        axes[1].plot([x * bpt for x in xs], ys, **kw)
    axes[0].set_xlabel("budget (fraction of the model's full cache)")
    axes[1].set_xlabel("budget (bytes per token of history)")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[1].plot([], [], color=GEN_COLORS["old"], linewidth=1.0, label="released before Oct 2024")
    axes[1].plot([], [], color=GEN_COLORS["new"], linewidth=2.0, marker="o",
                 markersize=3, label="released Nov 2024 onwards")
    axes[1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "scale_gen_bytes.pdf")
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=200)
    plt.close(fig)
    return path


def scale_frontier_fig(runs, out, panels="both"):
    """Every rung's frontier, against budget fraction and/or absolute bytes.

    `panels`: "both" (default, the appendix version: fraction left, bytes right,
    read together to see which unit the collapse point lives in), or "fraction"
    (a single narrow panel, sized for a main-text figure where space is tight).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = _rung_curves(load(runs))
    os.makedirs(out, exist_ok=True)

    if panels == "both":
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
        for label, (xs, ys, full_bpt) in curves.items():
            axes[0].plot(xs, ys, marker="o", label=label)
            axes[1].plot([x * full_bpt for x in xs], ys, marker="o", label=label)
        axes[0].set_xlabel("budget (fraction of the model's full cache)")
        axes[1].set_xlabel("budget (bytes per token of history)")
        axes[1].set_xscale("log")
        for ax in axes:
            ax.set_ylabel("accuracy")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.3)
        axes[1].legend(fontsize=6)
        fname = "scale.pdf"
    else:
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        for label, (xs, ys, _full_bpt) in curves.items():
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, label=label)
        ax.set_xlabel("budget (fraction of the model's full cache)", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="upper left")
        fname = "scale_frac.pdf"

    fig.tight_layout()
    path = os.path.join(out, fname)
    fig.savefig(path)
    plt.close(fig)
    return path


def main(args):
    rows = summarise(args.runs)
    if not rows:
        print(f"no run JSONs under {args.runs}; run scripts/scale_sweep.sh first")
        return
    for r in rows:
        c = r.get("collapse") or {}
        k = r.get("calibration") or {}
        print(f"{r['rung']:>14} [{r.get('quant','?'):>4}] "
              f"full={c.get('full_bpt', 0):>8,.0f} B/tok  "
              f"collapse={_fmt(c.get('frac'), pct=True):>5} "
              f"({'--' if c.get('bpt') is None else format(c['bpt'], ',.0f')} B/tok)  "
              f"crossover={_fmt(r.get('crossover'), pct=True):>5}  "
              f"ECEx={_fmt(k.get('ratio'), nd=1)}")
    print()
    print(latex_table(rows))
    if not args.no_fig:
        print("\nwrote", scale_frontier_fig(args.runs, args.out, panels="both"))
        print("wrote", scale_frontier_fig(args.runs, args.out, panels="fraction"))
