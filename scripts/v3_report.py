#!/usr/bin/env python
"""Summaries behind the revised paper, read from runs/scale and runs/v3.

    python scripts/v3_report.py [runs_root]

Sections:
  pairs       collapse point per model, with parameter count and release date
  audit       probe and stated confidence on full-present, full-absent, compacted
  survival    needle key/value survival split by probe answer, per method
  qa          question-agnostic vs question-aware frontier at matched budget
  multikey    single vs multikey collapse point
  mechanisms  every mechanism on the BPT axis
  dense       standard vs dense reversibility
"""
import glob, json, os, sys

from compactbench.scale import _interp_crossing

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
CONTENT = ["SnapKV", "TOVA", "Knorm", "ExpectedAttn"]

# tag -> (display name, params in B, release YYYY-MM, generation)
MODELS = {
    "qwen0.5b": ("Qwen2.5-0.5B", 0.5, "2024-09", "earlier"),
    "qwen1.5b": ("Qwen2.5-1.5B", 1.5, "2024-09", "earlier"),
    "qwen1.5b-nf4": ("Qwen2.5-1.5B*", 1.5, "2024-09", "earlier"),
    "qwen3b": ("Qwen2.5-3B*", 3.1, "2024-09", "earlier"),
    "qwen3b-nf4": ("Qwen2.5-3B*", 3.1, "2024-09", "earlier"),
    "qwen3b-fp16": ("Qwen2.5-3B", 3.1, "2024-09", "earlier"),
    "qwen7b": ("Qwen2.5-7B*", 7.6, "2024-09", "earlier"),
    "qwen7b-nf4": ("Qwen2.5-7B*", 7.6, "2024-09", "earlier"),
    "qwen7b-fp16": ("Qwen2.5-7B", 7.6, "2024-09", "earlier"),
    "qwen14b-nf4": ("Qwen2.5-14B*", 14.7, "2024-09", "earlier"),
    "phi3.5mini": ("Phi-3.5-mini*", 3.8, "2024-08", "earlier"),
    "phi3.5mini-nf4": ("Phi-3.5-mini*", 3.8, "2024-08", "earlier"),
    "phi3.5mini-bf16": ("Phi-3.5-mini", 3.8, "2024-08", "earlier"),
    "phi3medium-nf4": ("Phi-3-medium*", 14.0, "2024-05", "earlier"),
    "nemo12b-nf4": ("Mistral-Nemo-12B*", 12.2, "2024-07", "earlier"),
    "qwen3-0.6b": ("Qwen3-0.6B", 0.6, "2025-04", "newer"),
    "qwen3-1.7b": ("Qwen3-1.7B", 1.7, "2025-04", "newer"),
    "qwen3-4b-nf4": ("Qwen3-4B*", 4.0, "2025-04", "newer"),
    "qwen3-8b": ("Qwen3-8B", 8.2, "2025-04", "newer"),
    "qwen3-8b-nf4": ("Qwen3-8B*", 8.2, "2025-04", "newer"),
    "qwen3-14b-nf4": ("Qwen3-14B*", 14.8, "2025-04", "newer"),
    "smollm2-1.7b": ("SmolLM2-1.7B", 1.7, "2024-11", "newer"),
    "olmo2-1b": ("OLMo-2-1B", 1.5, "2025-04", "newer"),
    "olmo2-13b": ("OLMo-2-13B", 13.7, "2024-11", "newer"),
    "phi4": ("Phi-4", 14.7, "2024-12", "newer"),
    "mistral24b-nf4": ("Mistral-Small-24B*", 23.6, "2025-01", "newer"),
}


def frontiers(prefix):
    """{tag: frontier json} from runs/scale/frontier-* and runs/v3/<prefix>-*."""
    out = {}
    if prefix == "fr":
        for p in glob.glob(f"{ROOT}/scale/frontier-*.json"):
            out[os.path.basename(p)[len("frontier-"):-5]] = json.load(open(p))
    for p in glob.glob(f"{ROOT}/v3/{prefix}-*.json"):
        out[os.path.basename(p)[len(prefix) + 1:-5]] = json.load(open(p))
    return out


def curve(f, methods):
    by = {}
    for p in f["points"]:
        if p["method"] in methods:
            by.setdefault(round(1 - p["ratio"], 3), []).append(p["acc"])
    return {k: sum(v) / len(v) for k, v in sorted(by.items())}


def base(f):
    return next(p["acc"] for p in f["points"] if p["method"] == "baseline")


def collapse(f, methods):
    c = curve(f, methods)
    x = _interp_crossing(list(c), list(c.values()), 0.5 * base(f))
    if x is None and c and c[max(c)] < 0.5 * base(f):
        return ">0.90"
    return x


def pct(x):
    return x if isinstance(x, str) else ("--" if x is None else f"{100 * x:.0f}%")


def pairs():
    print("\n== collapse point by model (5 scorers / content only) ==")
    fr = frontiers("fr")
    rows = []
    for tag, f in fr.items():
        if tag not in MODELS:
            print("  (unlabelled)", tag); continue
        name, params, date, gen = MODELS[tag]
        c5 = collapse(f, CONTENT + ["StreamingLLM"]); c4 = collapse(f, CONTENT)
        best = max(CONTENT, key=lambda m: curve(f, [m]).get(0.5, 0))
        rows.append((gen, params, tag, name, date, base(f), c5, c4,
                     curve(f, CONTENT + ["StreamingLLM"]), best,
                     curve(f, [best])[0.5], curve(f, ["StreamingLLM"])[0.5],
                     curve(f, ["Random"])[0.9], [curve(f, [m])[0.9] for m in CONTENT]))
    for r in sorted(rows):
        gen, params, tag, name, date, b, c5, c4, cv, best, bacc, sacc, r90, c90 = r
        below = sum(a < r90 for a in c90)
        print(f"  {gen:7} {name:20} {params:5.1f}B {date}  base={b:.2f} "
              f"collapse={pct(c5):>6} content={pct(c4):>6}  "
              f"acc@.9={cv[0.9]:.2f} @.5={cv[0.5]:.2f}  best@.5={best}:{bacc:.2f} vs SLLM {sacc:.2f}"
              f"  content<random@.9: {below}/4")


def audit():
    print("\n== audit: probe yes-rate / mean confidence / accuracy ==")
    for p in sorted(glob.glob(f"{ROOT}/v3/au-*.json")):
        d = json.load(open(p)); tag = os.path.basename(p)[3:-5]
        print(f"  {MODELS.get(tag, (tag,))[0]}")
        for s in d["summary"]:
            f = lambda v: "  -- " if v is None else f"{v:5.2f}"
            print(f"    {s['method']:>13} r={s['ratio']:.1f} acc={f(s['acc'])} yes={f(s['probe_yes_rate'])} "
                  f"conf={f(s['mean_conf'])} ece={f(s['ece'])} aurocP={f(s['auroc_probe'])} "
                  f"aurocC={f(s['auroc_conf'])} over={f(s['overclaim'])} key={f(s['key_kept'])} val={f(s['value_kept'])}")


def survival():
    print("\n== needle survival on failures, by probe answer (pooled over models) ==")
    pooled = {}
    for p in sorted(glob.glob(f"{ROOT}/v3/au-*.json")):
        for r in json.load(open(p))["records"]:
            if r.get("key_kept") is None or r["correct"]:
                continue
            k = (r["method"], r["ratio"], r["probe_yes"])
            pooled.setdefault(k, []).append((r["key_kept"], r["value_kept"]))
    for k in sorted(pooled):
        v = pooled[k]
        print(f"  {k[0]:>13} r={k[1]} probe={'yes' if k[2] else 'no '} n={len(v):4d} "
              f"key_kept={sum(a for a, _ in v) / len(v):.2f} value_kept={sum(b for _, b in v) / len(v):.2f}")


def qa():
    print("\n== question-agnostic vs question-aware (accuracy at keep 0.9 / 0.5 / 0.25) ==")
    ag, aw = frontiers("fr"), frontiers("qa")
    for tag in sorted(aw):
        if tag not in ag:
            continue
        print(f"  {MODELS.get(tag, (tag,))[0]}")
        for m in CONTENT + ["StreamingLLM", "Random"]:
            a, w = curve(ag[tag], [m]), curve(aw[tag], [m])
            print(f"    {m:>13} " + "  ".join(f"{k}: {a[k]:.2f}->{w[k]:.2f}" for k in (0.9, 0.5, 0.25)))


def multikey():
    print("\n== single vs multikey collapse (5 scorers) ==")
    s, m = frontiers("fr"), frontiers("mk")
    for tag in sorted(m):
        name = MODELS.get(tag, (tag,))[0]
        print(f"  {name:20} single={pct(collapse(s[tag], CONTENT + ['StreamingLLM'])) if tag in s else '--':>6} "
              f"multikey={pct(collapse(m[tag], CONTENT + ['StreamingLLM'])):>6} base={base(m[tag]):.2f} "
              f"acc@.9 {curve(m[tag], CONTENT + ['StreamingLLM'])[0.9]:.2f}")


def mechanisms():
    print("\n== mechanisms on one BPT axis ==")
    for p in sorted(glob.glob(f"{ROOT}/v3/me-*.json")):
        d = json.load(open(p))
        print(f"  {d['model']}")
        for q in sorted(d["points"], key=lambda q: -q["bpt_frac"]):
            print(f"    {q['family']:>13} {q['method']:>20} {q['setting']:>5} "
                  f"BPT={100 * q['bpt_frac']:5.1f}% acc={q['acc']:.2f}")


def dense():
    print("\n== reversibility: standard vs dense (summary / raw) ==")
    for p in sorted(glob.glob(f"{ROOT}/v3/rd-*.json")):
        tag = os.path.basename(p)[3:-5]
        d = json.load(open(p))
        std = None
        for c in (f"{ROOT}/v3/rv-{tag}.json", f"{ROOT}/rev2/reversibility-{tag}.json",
                  f"{ROOT}/scale/reversibility-{tag}.json"):
            if os.path.exists(c):
                std = json.load(open(c)); break
        print(f"  {MODELS.get(tag, (tag,))[0]}  dense doc {d['avg_total_tokens']:.0f} tok")
        for i, q in enumerate(d["points"]):
            s = std["points"][i] if std else None
            ss = f"std {s['irreversible_acc']:.2f}/{s['reversible_acc']:.2f}  " if s else ""
            print(f"    B={q['budget_frac']:.2f} {ss}dense {q['irreversible_acc']:.2f}/{q['reversible_acc']:.2f}")


if __name__ == "__main__" and len(sys.argv) <= 2:
    for f in (pairs, audit, survival, qa, multikey, mechanisms, dense):
        f()


FULL_ORDER = ["qwen0.5b", "qwen1.5b", "qwen1.5b-nf4", "qwen3b-fp16", "qwen3b", "qwen7b-fp16",
              "qwen7b", "qwen14b-nf4", "phi3.5mini-bf16", "phi3.5mini",
              "qwen3-0.6b", "olmo2-1b", "qwen3-1.7b", "smollm2-1.7b", "qwen3-4b-nf4",
              "qwen3-8b", "qwen3-8b-nf4", "olmo2-13b", "phi4", "qwen3-14b-nf4",
              "mistral24b-nf4"]
PREC = {"qwen0.5b": "fp16", "qwen1.5b": "fp16", "qwen1.5b-nf4": "nf4", "qwen3b-fp16": "fp16",
        "qwen3b": "nf4", "qwen7b-fp16": "fp16", "qwen7b": "nf4", "qwen14b-nf4": "nf4",
        "phi3.5mini-bf16": "bf16", "phi3.5mini": "nf4", "qwen3-0.6b": "bf16", "olmo2-1b": "bf16",
        "qwen3-1.7b": "bf16", "smollm2-1.7b": "bf16", "qwen3-4b-nf4": "nf4", "qwen3-8b": "bf16",
        "qwen3-8b-nf4": "nf4", "olmo2-13b": "bf16", "phi4": "bf16", "qwen3-14b-nf4": "nf4",
        "mistral24b-nf4": "nf4"}
# reversibility files that use the corrected summary cap, by frontier tag
REV = {"qwen0.5b": "rev2/reversibility-qwen0.5b", "qwen1.5b": "rev2/reversibility-qwen1.5b",
       "qwen3b-fp16": "rev2/reversibility-qwen3b-fp16", "qwen7b-fp16": "rev2/reversibility-qwen7b-fp16",
       "phi3.5mini-bf16": "rev2/reversibility-phi3.5mini-bf16", "qwen14b-nf4": "v3/rv-qwen14b-nf4",
       "qwen3-0.6b": "v3/rv-qwen3-0.6b", "olmo2-1b": "v3/rv-olmo2-1b", "qwen3-1.7b": "v3/rv-qwen3-1.7b",
       "smollm2-1.7b": "v3/rv-smollm2-1.7b", "qwen3-4b-nf4": "v3/rv-qwen3-4b-nf4",
       "qwen3-8b": "scale/reversibility-qwen3-8b", "phi4": "scale/reversibility-phi4"}


def crossover(rev):
    pts = sorted(rev["points"], key=lambda p: p["budget_frac"])
    xs = [p["budget_frac"] for p in pts]
    d = [p["reversible_acc"] - p["irreversible_acc"] for p in pts]
    for (x0, d0), (x1, d1) in zip(zip(xs, d), zip(xs[1:], d[1:])):
        if d0 <= 0 < d1 and x1 < 1.0:
            return x0 + (0 - d0) / (d1 - d0) * (x1 - x0)
    return None


def latex_scale_table():
    fr, mk = frontiers("fr"), frontiers("mk")
    print("\n== LaTeX rows for the per-setting table ==")
    for tag in FULL_ORDER:
        if tag not in fr:
            print("% missing", tag); continue
        f = fr[tag]
        name, params, date, gen = MODELS[tag]
        name = name.rstrip("*")
        c5, c4 = collapse(f, CONTENT + ["StreamingLLM"]), collapse(f, CONTENT)
        mtag = {"qwen3b": "qwen3b-nf4"}.get(tag, tag)
        m = pct(collapse(mk[mtag], CONTENT + ["StreamingLLM"])) if mtag in mk else "--"
        rv = REV.get(tag)
        cx = "n/r"
        if rv and os.path.exists(f"{ROOT}/{rv}.json"):
            x = crossover(json.load(open(f"{ROOT}/{rv}.json")))
            cx = "--" if x is None else pct(x)
        t = lambda v: v.replace("%", "\\%").replace(">0.90", "$>$90\\%")
        print(f"{name} ({PREC[tag]}) & {date} & {f['full_bpt_bytes_per_tok']:,.0f} & "
              f"{t(pct(c5))} & {t(pct(c4))} & {t(m)} & {t(cx)} \\\\".replace(",", "{,}"))
        if tag == "phi3.5mini":
            print("\\midrule")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "table":
    latex_scale_table()
