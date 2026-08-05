"""Render figures from runs/*.json (whichever exist)."""
import json, os


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def frontier_fig(runs, out):
    plt = _mpl()
    d = json.load(open(os.path.join(runs, "frontier.json")))
    pts = d["points"]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for m in sorted({p["method"] for p in pts if p["method"] != "baseline"}):
        mp = sorted([p for p in pts if p["method"] == m], key=lambda x: x["bpt_frac"])
        ax.plot([p["bpt_frac"] * 100 for p in mp], [p["acc"] * 100 for p in mp],
                marker="o", ms=3, lw=1.2, label=m)
    base = [p for p in pts if p["method"] == "baseline"]
    if base:
        ax.scatter([100], [base[0]["acc"] * 100], marker="*", s=110, color="k",
                   zorder=5, label="full cache")
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 75, 100])
    ax.set_xticklabels([10, 25, 50, 75, 100])
    ax.set_xlabel("BPT budget (% of full cache)")
    ax.set_ylabel("needle accuracy (%)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "frontier.pdf"))
    print("wrote frontier.pdf")


def reversibility_fig(runs, out):
    plt = _mpl()
    d = json.load(open(os.path.join(runs, "reversibility.json")))
    pts = sorted(d["points"], key=lambda x: x["budget_frac"])
    x = [p["budget_frac"] * 100 for p in pts]
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    ax.plot(x, [p["reversible_acc"] * 100 for p in pts], marker="o", lw=1.5,
            color="#2471a3", label="reversible (raw, bounded)")
    ax.plot(x, [p["irreversible_acc"] * 100 for p in pts], marker="s", lw=1.5,
            color="#c0392b", label="irreversible (summary)")
    ax.set_xlabel("storage budget (% of full history)")
    ax.set_ylabel("fact recall (%)")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "reversibility.pdf"))
    print("wrote reversibility.pdf")


def attribution_fig(runs, out):
    plt = _mpl()
    d = json.load(open(os.path.join(runs, "attribution.json")))
    summ = [s for s in d["summary"] if s["auroc_selfreport"] is not None]
    methods = sorted({s["method"] for s in summ})
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for m in methods:
        ms = sorted([s for s in summ if s["method"] == m], key=lambda x: x["ratio"])
        ax.plot([s["ratio"] * 100 for s in ms], [s["auroc_selfreport"] for s in ms],
                marker="o", lw=1.4, label=m)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="no self-knowledge")
    ax.set_xlabel("compression ratio (%)")
    ax.set_ylabel("AUROC: self-report vs correctness")
    ax.set_ylim(0.3, 1.02); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "attribution.pdf"))
    print("wrote attribution.pdf")


def confidence_fig(runs, out):
    plt = _mpl()
    d = json.load(open(os.path.join(runs, "confidence.json")))
    summ = d["summary"]
    methods = sorted({s["method"] for s in summ if s["method"] != "baseline"})
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for m in methods:
        ms = sorted([s for s in summ if s["method"] == m], key=lambda x: x["ratio"])
        ax.plot([s["ratio"] * 100 for s in ms], [s["ece"] for s in ms],
                marker="o", lw=1.4, label=m)
    base = [s for s in summ if s["method"] == "baseline"]
    if base:
        ax.axhline(base[0]["ece"], color="k", ls=":", lw=1.2, label="full cache")
    ax.set_xlabel("compression ratio (%)")
    ax.set_ylabel("ECE (lower = better calibrated)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "confidence.pdf"))
    print("wrote confidence.pdf")


def render_all(runs="runs", out="figures"):
    os.makedirs(out, exist_ok=True)
    for name, fn in [("frontier.json", frontier_fig),
                     ("reversibility.json", reversibility_fig),
                     ("attribution.json", attribution_fig),
                     ("confidence.json", confidence_fig)]:
        if os.path.exists(os.path.join(runs, name)):
            fn(runs, out)
        else:
            print(f"({name} not found, skipped)")
