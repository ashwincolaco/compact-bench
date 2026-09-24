"""Task 3 — loss attribution: does the system KNOW what it dropped?

After compaction, the system is asked a yes/no answerability probe ("do your notes
contain X?") and then the question itself. The probe is the system's self-report of
what survived compaction; correctness of the actual answer is the ground truth. We
report, per method and budget:
  - AUROC of the self-report predicting correctness (1.0 = knows exactly what it
    dropped; 0.5 = no self-knowledge),
  - overclaim rate: P(probe says yes | answer wrong) -- silent loss,
  - underclaim rate: P(probe says no | answer right).

Caveat (documented): query-conditioned presses (e.g. SnapKV's observation window)
compress differently for the probe than for the question, since each call sees a
different trailing query. The probe therefore measures the system's self-knowledge
under its own compaction policy, not a fixed retained set.
"""
import json, os, random

from ..bpt import ModelDims, bpt_full_cache
from ..data import (load_filler_pool, build_needle_context, sample_needle,
                    QUESTION, PROBE)
from ..models import load_kvpress_pipeline, default_presses
from ..metrics import auroc


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "nf4"],
                   help="weight precision; nf4 fits larger models on small GPUs")
    p.add_argument("--methods", nargs="+", default=["SnapKV", "StreamingLLM", "Random"])
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000])
    p.add_argument("--positions", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", default="runs/attribution.json")


def _yes(text):
    t = text.strip().lower()
    return int(t.startswith("yes") or " yes" in t[:12])


def main(args):
    import torch
    from ..models import DEFAULT_MODEL
    model_name = args.model or DEFAULT_MODEL
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pipe = load_kvpress_pipeline(model_name, quant=args.quant)
    tok = pipe.tokenizer
    filler = load_filler_pool()
    rng = random.Random(0)
    presses = {k: v for k, v in default_presses().items() if k in args.methods}

    records = []   # one per (method, ratio, trial): probe_yes, correct
    total = 0
    for L in args.lengths:
        for pos in args.positions:
            for _ in range(args.trials):
                key, value = sample_needle(rng)
                text = build_needle_context(tok, filler, L, pos, key, value)
                q, probe = QUESTION.format(key=key), PROBE.format(key=key)
                for mname, mcls in presses.items():
                    for r in args.ratios:
                        press = mcls(compression_ratio=r)
                        say = _yes(pipe(text, question=probe, press=press)["answer"])
                        ans = pipe(text, question=q, press=mcls(compression_ratio=r))["answer"]
                        records.append({"method": mname, "ratio": r,
                                        "probe_yes": say, "correct": int(value in ans)})
                        total += 2
            print(f"  L={L} pos={pos} done ({total} generations)")

    dims = ModelDims.from_hf(pipe.model.config)
    out = {"model": model_name, "quant": args.quant,
           "full_bpt_bytes_per_tok": bpt_full_cache(dims),
           "config": vars(args), "records": records, "summary": []}
    for mname in presses:
        for r in args.ratios:
            xs = [x for x in records if x["method"] == mname and x["ratio"] == r]
            if not xs:
                continue
            says = [x["probe_yes"] for x in xs]
            cors = [x["correct"] for x in xs]
            wrong = [x for x in xs if x["correct"] == 0]
            right = [x for x in xs if x["correct"] == 1]
            out["summary"].append({
                "method": mname, "ratio": r, "n": len(xs),
                "acc": sum(cors) / len(xs),
                "probe_yes_rate": sum(says) / len(xs),
                "auroc_selfreport": auroc(says, cors),
                "overclaim": (sum(x["probe_yes"] for x in wrong) / len(wrong)) if wrong else None,
                "underclaim": (sum(1 - x["probe_yes"] for x in right) / len(right)) if right else None,
            })
    json.dump(out, open(args.out, "w"), indent=2)
    for s in out["summary"]:
        print(f"{s['method']:>13} r={s['ratio']}: acc={s['acc']:.2f} "
              f"auroc={s['auroc_selfreport'] if s['auroc_selfreport'] is None else round(s['auroc_selfreport'],2)} "
              f"overclaim={s['overclaim'] if s['overclaim'] is None else round(s['overclaim'],2)}")
    print("saved", args.out,
          "| max VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
