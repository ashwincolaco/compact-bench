"""Task 4 — calibrated compaction confidence.

After answering from a compacted context, the system states a confidence (0-100).
We measure, per method and budget: ECE (10 bins) and AUROC of confidence vs
correctness. The question is whether calibration DEGRADES with compression: a
well-behaved compaction operator should know when its memory no longer supports an
answer, so confidence should fall where accuracy falls. Verbal elicitation is the
baseline instrument; the protocol fixes the metrics, not the elicitation.
"""
import json, os, random, re

from ..bpt import ModelDims, bpt_full_cache
from ..data import load_filler_pool, build_needle_context, sample_needle, QUESTION
from ..models import load_kvpress_pipeline, default_presses
from ..metrics import auroc, ece


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--quant", default="fp16", choices=["fp16", "nf4"],
                   help="weight precision; nf4 fits larger models on small GPUs")
    p.add_argument("--methods", nargs="+", default=["SnapKV", "StreamingLLM", "Random"])
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000])
    p.add_argument("--positions", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.5, 0.75, 0.9])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--out", default="runs/confidence.json")


CONF_Q = ("{q} Then, on a new line, state how confident you are that your answer "
          "is correct as a number from 0 to 100.")


def _parse(answer):
    """Split an 'answer + confidence' completion into (answer_text, conf in [0,1])."""
    nums = re.findall(r"\b(\d{1,3})\b", answer)
    conf = None
    if nums:
        c = int(nums[-1])
        if 0 <= c <= 100:
            conf = c / 100.0
    return answer, (conf if conf is not None else 0.5)


def main(args):
    import torch
    from ..models import DEFAULT_MODEL
    model_name = args.model or DEFAULT_MODEL
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pipe = load_kvpress_pipeline(model_name, quant=args.quant)
    tok = pipe.tokenizer
    filler = load_filler_pool()
    rng = random.Random(0)
    presses = default_presses()

    records = []
    total = 0
    for L in args.lengths:
        for pos in args.positions:
            for _ in range(args.trials):
                key, value = sample_needle(rng)
                text = build_needle_context(tok, filler, L, pos, key, value)
                q = CONF_Q.format(q=QUESTION.format(key=key))
                if 0.0 in args.ratios:      # baseline once per trial, not per method
                    ans, conf = _parse(pipe(text, question=q)["answer"])
                    records.append({"method": "baseline", "ratio": 0.0, "conf": conf,
                                    "correct": int(value in ans)})
                    total += 1
                for mname in args.methods:
                    for r in [r for r in args.ratios if r > 0.0]:
                        press = presses[mname](compression_ratio=r)
                        ans, conf = _parse(pipe(text, question=q, press=press)["answer"])
                        records.append({"method": mname, "ratio": r, "conf": conf,
                                        "correct": int(value in ans)})
                        total += 1
            print(f"  L={L} pos={pos} done ({total} generations)")

    dims = ModelDims.from_hf(pipe.model.config)
    out = {"model": model_name, "quant": args.quant,
           "full_bpt_bytes_per_tok": bpt_full_cache(dims),
           "config": vars(args), "records": records, "summary": []}
    keys = sorted({(x["method"], x["ratio"]) for x in records})
    for mname, r in keys:
        xs = [x for x in records if x["method"] == mname and x["ratio"] == r]
        confs = [x["conf"] for x in xs]
        cors = [x["correct"] for x in xs]
        out["summary"].append({
            "method": mname, "ratio": r, "n": len(xs),
            "acc": sum(cors) / len(xs),
            "mean_conf": sum(confs) / len(confs),
            "ece": ece(confs, cors),
            "auroc_conf": auroc(confs, cors),
        })
    json.dump(out, open(args.out, "w"), indent=2)
    for s in out["summary"]:
        print(f"{s['method']:>13} r={s['ratio']}: acc={s['acc']:.2f} "
              f"conf={s['mean_conf']:.2f} ece={s['ece']:.2f} "
              f"auroc={'n/a' if s['auroc_conf'] is None else round(s['auroc_conf'],2)}")
    print("saved", args.out,
          "| max VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
