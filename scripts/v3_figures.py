#!/usr/bin/env python
"""Figures for the revised paper.

    python scripts/v3_figures.py <runs_root> <out_dir>

  collapse_vs_size.pdf   collapse point against parameter count, one colour per
                         release group, size-matched pairs joined by a line
  mechanisms.pdf         every mechanism on the BPT axis, one panel per model
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from v3_report import MODELS, CONTENT, frontiers, collapse  # noqa: E402
import v3_report  # noqa: E402

GEN = {"earlier": "#2a78d6", "newer": "#eb6834"}
# (earlier tag, newer tag): same family, similar size, either side of the split
PAIRS = [("qwen0.5b", "qwen3-0.6b"), ("qwen1.5b", "qwen3-1.7b"),
         ("qwen3b-nf4", "qwen3-4b-nf4"), ("qwen7b-fp16", "qwen3-8b"),
         ("qwen14b-nf4", "qwen3-14b-nf4"), ("phi3medium-nf4", "phi4"),
         ("nemo12b-nf4", "mistral24b-nf4")]


def collapse_vs_size(out):
    fr = frontiers("fr")
    pts = {}
    for tag, f in fr.items():
        if tag not in MODELS:
            continue
        c = collapse(f, CONTENT + ["StreamingLLM"])
        if isinstance(c, str):
            c = 0.9
        if c is None:
            continue
        pts[tag] = (MODELS[tag][1], c, MODELS[tag][3])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for a, b in PAIRS:
        if a in pts and b in pts:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], color="#b9b7b1",
                    linewidth=0.9, zorder=1)
    for tag, (p, c, g) in pts.items():
        four_bit = "nf4" in tag or tag in ("qwen3b", "qwen7b", "phi3.5mini")
        ax.scatter([p], [c], s=22, color=GEN[g], zorder=3,
                   marker="o" if not four_bit else "D",
                   edgecolor="white", linewidth=0.5)
    ax.scatter([], [], color=GEN["earlier"], s=22, label="released before Oct 2024")
    ax.scatter([], [], color=GEN["newer"], s=22, label="released Nov 2024 onwards")
    ax.scatter([], [], color="#8a8983", s=18, marker="D", label="4-bit weights")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4, 8, 16])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16"])
    ax.set_xlabel("parameters (B, log scale)", fontsize=7)
    ax.set_ylabel("collapse point\n(fraction of own cache)", fontsize=7)
    ax.set_ylim(0.25, 0.95)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    path = os.path.join(out, "collapse_vs_size.pdf")
    fig.savefig(path); fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    return path


FAMILY_STYLE = {
    "eviction": dict(color="#2a78d6", marker="o"),
    "quantization": dict(color="#1b9e77", marker="s"),
    "evict+quant": dict(color="#7570b3", marker="^"),
    "prompt": dict(color="#d95f02", marker="D"),
    "summary": dict(color="#e7298a", marker="v"),
}


def mechanisms(out):
    files = sorted(glob.glob(f"{v3_report.ROOT}/v3/me-*.json"))
    if not files:
        return None
    fig, axes = plt.subplots(1, len(files), figsize=(2.75 * len(files), 2.0), sharey=True,
                             squeeze=False)
    for ax, p in zip(axes[0], files):
        d = json.load(open(p))
        tag = os.path.basename(p)[3:-5]
        pts = d["points"]
        # the best eviction method at each keep fraction, plus every other family
        best = {}
        for q in pts:
            if q["family"] == "eviction":
                k = q["setting"]
                if k not in best or q["acc"] > best[k]["acc"]:
                    best[k] = q
        for fam, st in FAMILY_STYLE.items():
            if fam == "eviction":
                xs = [best[k] for k in sorted(best, key=float)]
                lab = "eviction (best of six)"
            elif fam == "evict+quant":
                bestc = {}
                for q in pts:
                    if q["family"] == fam:
                        k = round(q["bpt_frac"], 4)
                        if k not in bestc or q["acc"] > bestc[k]["acc"]:
                            bestc[k] = q
                xs = [bestc[k] for k in sorted(bestc)]
                lab = "evict + quantize (best)"
            else:
                xs = sorted([q for q in pts if q["family"] == fam], key=lambda q: q["bpt_frac"])
                lab = {"quantization": "KV quantization", "prompt": "LLMLingua-2",
                       "summary": "model summary"}[fam]
            ax.plot([100 * q["bpt_frac"] for q in xs], [q["acc"] for q in xs],
                    linewidth=1.3, markersize=4, label=lab, **st)
        ax.set_title(MODELS.get(tag, (tag,))[0], fontsize=8)
        ax.set_xlabel("BPT (% of full cache)", fontsize=7)
        ax.set_xlim(0, 60)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy", fontsize=7)
    axes[0][0].set_ylim(-0.03, 1.03)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, frameon=False, loc="lower center", ncol=len(l),
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    path = os.path.join(out, "mechanisms.pdf")
    fig.savefig(path); fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    v3_report.ROOT = sys.argv[1]
    os.makedirs(sys.argv[2], exist_ok=True)
    for fn in (collapse_vs_size, mechanisms):
        print(fn(sys.argv[2]))
