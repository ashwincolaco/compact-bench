"""Task 1 — the accuracy-vs-BPT frontier.

Needle-in-a-haystack retrieval with natural filler, swept over KV-compaction
methods and compression ratios, every point placed on the shared
bytes-per-token-of-history axis.
"""
import json, os, random

from ..bpt import ModelDims, bpt_kv_eviction, bpt_full_cache
from ..data import (load_filler_pool, build_needle_context, build_multikey_context,
                    sample_needle, QUESTION)
from ..models import load_kvpress_pipeline, default_presses


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "nf4", "prequant"],
                   help="weight precision; nf4 fits larger models on small GPUs")
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000, 8000])
    p.add_argument("--positions", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=None,
                   help="subset of the default presses (default: all six)")
    p.add_argument("--query-aware", action="store_true",
                   help="compress with the question at the end of the context, so "
                        "query-scored presses can see it (default: question-agnostic)")
    p.add_argument("--variant", default="single", choices=["single", "multikey"],
                   help="multikey adds three same-format distractor needles")
    p.add_argument("--out", default="runs/frontier.json")


def main(args):
    import torch
    from ..models import DEFAULT_MODEL
    model_name = args.model or DEFAULT_MODEL
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pipe = load_kvpress_pipeline(model_name, quant=args.quant)
    tok = pipe.tokenizer
    dims = ModelDims.from_hf(pipe.model.config)
    full_bpt = bpt_full_cache(dims)
    filler = load_filler_pool()
    rng = random.Random(0)
    methods = default_presses()
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}

    results = {"baseline": {"0.0": []}}
    for m in methods:
        results[m] = {f"{r}": [] for r in args.ratios}

    total = 0
    for L in args.lengths:
        for pos in args.positions:
            for _ in range(args.trials):
                key, value = sample_needle(rng)
                if args.variant == "multikey":
                    text = build_multikey_context(tok, filler, L, pos, key, value, rng)
                else:
                    text = build_needle_context(tok, filler, L, pos, key, value)
                q = QUESTION.format(key=key)
                if args.query_aware:
                    # The question sits inside the compressed context and is asked
                    # again afterwards, so the only change is what the scorer sees.
                    text = text + "\n\nQuestion: " + q
                ans = pipe(text, question=q)["answer"]
                results["baseline"]["0.0"].append(int(value in ans)); total += 1
                for mname, mcls in methods.items():
                    for r in args.ratios:
                        ans = pipe(text, question=q,
                                   press=mcls(compression_ratio=r))["answer"]
                        results[mname][f"{r}"].append(int(value in ans)); total += 1
            print(f"  L={L} pos={pos} done ({total} generations)")

    agg = {"records": results, "model": model_name, "quant": args.quant,
           "full_bpt_bytes_per_tok": full_bpt,
           "config": vars(args), "points": []}
    base = results["baseline"]["0.0"]
    agg["points"].append({"method": "baseline", "ratio": 0.0, "bpt": full_bpt,
                          "bpt_frac": 1.0, "acc": sum(base) / max(1, len(base)),
                          "n": len(base)})
    for m in methods:
        for r in args.ratios:
            xs = results[m][f"{r}"]
            bpt = bpt_kv_eviction(dims, 1.0 - r)
            agg["points"].append({"method": m, "ratio": r, "bpt": bpt,
                                  "bpt_frac": bpt / full_bpt,
                                  "acc": sum(xs) / max(1, len(xs)), "n": len(xs)})
    json.dump(agg, open(args.out, "w"), indent=2)
    print(f"saved {args.out} | {total} generations "
          f"| max VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
