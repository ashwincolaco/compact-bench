#!/usr/bin/env python
"""95% intervals behind the paper: collapse points, paired differences, overclaim.

    python scripts/intervals.py [runs_root]

Collapse points: parametric bootstrap with 2,000 replicates. The accuracy of
every (method, budget) cell and of the baseline is resampled binomially at its
observed rate and sample size, and the collapse point is recomputed with the
same interpolation as the paper. Cells are resampled independently, which
ignores the correlation from sharing needles across budgets.

Overclaim rates: Wilson score intervals on the probe's yes-rate among wrong
answers at compression ratio 0.5, per model and pooled over the models whose
probe passes both uncompacted controls.
"""
import json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import v3_report as V  # noqa: E402
from compactbench.scale import _interp_crossing  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "runs"
V.ROOT = ROOT
KEEP = [0.9, 0.75, 0.5, 0.25, 0.1]
SCORERS = V.CONTENT + ["StreamingLLM"]
PAIRS = [("qwen0.5b", "qwen3-0.6b"), ("qwen1.5b", "qwen3-1.7b"), ("qwen3b", "qwen3-4b-nf4"),
         ("qwen7b-fp16", "qwen3-8b"), ("qwen14b-nf4", "qwen3-14b-nf4")]
VALID_PROBE = ["qwen1.5b", "qwen3b-nf4", "qwen7b-nf4", "qwen14b-nf4", "qwen3-0.6b", "qwen3-4b-nf4"]


def collapse(acc, base):
    avg = {k: np.mean([acc[m][i] for m in SCORERS]) for i, k in enumerate(KEEP)}
    xs = sorted(avg)
    c = _interp_crossing(xs, [avg[x] for x in xs], 0.5 * base)
    if c is None and avg[0.9] < 0.5 * base:
        return 0.95  # already below half at the largest budget: censored above the grid
    return np.nan if c is None else c


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main(B=2000, n=48, seed=0):
    rng = np.random.default_rng(seed)
    fr = V.frontiers("fr")
    boots = {}
    print("== collapse point, 95% bootstrap interval ==")
    for tag in V.FULL_ORDER:
        if tag not in fr:
            continue
        f = fr[tag]
        acc = {m: [V.curve(f, [m])[k] for k in KEEP] for m in SCORERS}
        base = V.base(f)
        s = np.array([collapse({m: [rng.binomial(n, p) / n for p in acc[m]] for m in SCORERS},
                               rng.binomial(n, base) / n) for _ in range(B)])
        boots[tag] = s
        lo, hi = np.nanpercentile(s, [2.5, 97.5])
        print(f"  {V.MODELS[tag][0]:22} {collapse(acc, base):.3f}  [{lo:.3f}, {hi:.3f}]")
    print("== paired differences (earlier minus later) ==")
    for a, b in PAIRS:
        lo, hi = np.nanpercentile(boots[a] - boots[b], [2.5, 97.5])
        print(f"  {V.MODELS[a][0]} - {V.MODELS[b][0]}: [{lo:.3f}, {hi:.3f}]")
    early = np.vstack([boots[t] for t in boots if V.MODELS[t][3] == "earlier"])
    late = np.vstack([boots[t] for t in boots if V.MODELS[t][3] == "newer"])
    print(f"  groups disjoint in {np.mean(np.nanmin(early, 0) > np.nanmax(late, 0)):.3f} of replicates")

    print("== overclaim at ratio 0.5, Wilson 95% interval ==")
    pooled = {}
    for tag in VALID_PROBE:
        path = next((p for p in (f"{ROOT}/v3/au-{tag}.json",) if os.path.exists(p)), None)
        if path is None:
            continue
        R = json.load(open(path))["records"]
        for m in ("StreamingLLM", "SnapKV", "Random"):
            w = [x for x in R if x["method"] == m and x["ratio"] == 0.5 and not x["correct"]]
            k = sum(x["probe_yes"] for x in w)
            pooled.setdefault(m, [0, 0])
            pooled[m][0] += k
            pooled[m][1] += len(w)
    for m, (k, nn) in pooled.items():
        lo, hi = wilson(k, nn)
        print(f"  pooled {m:12} {k}/{nn} = {k / nn:.2f}  [{lo:.2f}, {hi:.2f}]")

    # Regression over models: one setting per model, 16-bit where available.
    ONE = ["qwen0.5b", "qwen1.5b", "qwen3b-fp16", "qwen7b-fp16", "qwen14b-nf4", "phi3.5mini-bf16",
           "qwen3-0.6b", "olmo2-1b", "qwen3-1.7b", "smollm2-1.7b", "qwen3-4b-nf4", "qwen3-8b",
           "olmo2-13b", "phi4", "qwen3-14b-nf4", "mistral24b-nf4"]
    tags = [t for t in ONE if t in fr]
    y = np.array([collapse({m: [V.curve(fr[t], [m])[k] for k in KEEP] for m in SCORERS}, V.base(fr[t]))
                  for t in tags])
    newer = np.array([V.MODELS[t][3] == "newer" for t in tags], float)
    X = np.column_stack([np.ones(len(y)), newer, np.log2([V.MODELS[t][1] for t in tags])])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    obs = y[newer == 0].mean() - y[newer == 1].mean()
    perm = [(lambda p: y[p == 0].mean() - y[p == 1].mean())(rng.permutation(newer)) for _ in range(20000)]
    print("== regression: collapse ~ release group + log2(params), one setting per model ==")
    print(f"  models={len(y)}  release group {100 * beta[1]:+.1f} pts  per doubling {100 * beta[2]:+.1f} pts  "
          f"permutation p={np.mean(np.array(perm) >= obs):.1e}")

    # Dose-response: probe answer on key survival, failures of the scattering methods.
    xs, ts = [], []
    for tag in VALID_PROBE:
        path = f"{ROOT}/v3/au-{tag}.json"
        if not os.path.exists(path):
            continue
        for r in json.load(open(path))["records"]:
            if r["method"] in ("SnapKV", "Random", "Knorm") and not r["correct"] and r.get("key_kept") is not None:
                xs.append(r["key_kept"]); ts.append(r["probe_yes"])
    x, t = np.array(xs), np.array(ts, float)
    Xd = np.column_stack([np.ones(len(x)), x]); w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xd @ w))
        H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, Xd.T @ (t - p))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    sig = lambda z: 1 / (1 + math.exp(-z))
    print("== dose-response: logit P(probe yes | failure) on share of key slots kept ==")
    print(f"  n={len(x)}  slope {w[1]:.2f} (SE {se[1]:.2f})  P(yes) at none kept {sig(w[0]):.2f}, "
          f"at all kept {sig(w[0] + w[1]):.2f}")


if __name__ == "__main__":
    main()
