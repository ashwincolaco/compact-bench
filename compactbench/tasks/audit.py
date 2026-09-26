"""Task 5 — audit: self-report, stated confidence, and what actually survived.

One compressed cache answers two questions, the yes/no answerability probe and
the answer with a stated confidence, so both read exactly the same retained set
(this also lets the random press take part). For eviction presses the task logs
which share of the needle's key tokens and value tokens each layer and head kept.

Two uncompacted controls separate the effect of compaction from the instrument:
  full-present  the whole context, needle included (answerable),
  full-absent   the same filler with no needle (unanswerable without compaction).
If stated confidence stays high on full-absent, verbal confidence cannot report a
missing fact at all, compaction or not. If the probe says no there, the probe can.
"""
import json, os, random

from ..bpt import ModelDims, bpt_full_cache
from ..data import (load_filler_pool, build_needle_context, build_absent_context,
                    needle_token_positions, sample_needle, QUESTION, PROBE)
from ..models import load_kvpress_pipeline, default_presses
from ..presses import RecordingPress
from ..metrics import auroc, ece
from .attribution import _yes
from .confidence import CONF_Q

import re


def _parse(answer):
    """Stated confidence in [0, 1], or None when the completion states none.

    The last one-to-three-digit number is read as the confidence, as in the
    confidence task, but an unparsed completion is kept as missing instead of
    being scored as 0.5, so degenerate outputs do not enter the calibration.
    """
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", answer)]
    return nums[-1] / 100.0 if nums and nums[-1] <= 100 else None


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "nf4", "prequant"])
    p.add_argument("--methods", nargs="+", default=["SnapKV", "StreamingLLM", "Knorm", "Random"])
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000])
    p.add_argument("--positions", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.9])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--out", default="runs/audit.json")


def summarize(records):
    out = []
    for cond in sorted({(r["method"], r["ratio"]) for r in records}, key=str):
        xs = [r for r in records if (r["method"], r["ratio"]) == cond]
        cor = [r["correct"] for r in xs]
        say = [r["probe_yes"] for r in xs]
        parsed = [r for r in xs if r["conf"] is not None]
        conf = [r["conf"] for r in parsed]
        pcor = [r["correct"] for r in parsed]
        wrong = [r for r in xs if not r["correct"]]
        s = {"method": cond[0], "ratio": cond[1], "n": len(xs),
             "acc": sum(cor) / len(xs),
             "probe_yes_rate": sum(say) / len(xs),
             "conf_parsed": len(parsed) / len(xs),
             "mean_conf": sum(conf) / len(conf) if conf else None,
             "ece": ece(conf, pcor) if conf else None,
             "auroc_probe": auroc(say, cor),
             "auroc_conf": auroc(conf, pcor),
             "overclaim": sum(r["probe_yes"] for r in wrong) / len(wrong) if wrong else None,
             "conf_when_probe_no": (lambda z: sum(z) / len(z) if z else None)(
                 [r["conf"] for r in parsed if not r["probe_yes"]])}
        for part in ("key", "value"):
            v = [r[f"{part}_kept"] for r in xs if r.get(f"{part}_kept") is not None]
            s[f"{part}_kept"] = sum(v) / len(v) if v else None
        out.append(s)
    return out


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

    records, total = [], 0
    for L in args.lengths:
        for pos in args.positions:
            for t in range(args.trials):
                key, value = sample_needle(rng)
                text = build_needle_context(tok, filler, L, pos, key, value)
                qs = [PROBE.format(key=key), CONF_Q.format(q=QUESTION.format(key=key))]
                span = needle_token_positions(tok, text, key, value)
                base = {"length": L, "position": pos, "trial": t}

                def run(ctx, method, ratio, press=None, correct_if=True):
                    probe, ans = pipe(ctx, questions=qs, press=press,
                                      max_new_tokens=100)["answers"]
                    conf = _parse(ans.replace(value, ""))
                    r = {**base, "method": method, "ratio": ratio,
                         "probe_yes": _yes(probe), "conf": conf,
                         "correct": int(correct_if and value in ans), "answer": ans, "probe": probe[:40]}
                    if isinstance(press, RecordingPress):
                        r["key_kept"] = press.survival(span["key"])
                        r["value_kept"] = press.survival(span["value"])
                    records.append(r)

                run(text, "full-present", 0.0)
                run(build_absent_context(tok, filler, L), "full-absent", 0.0,
                    correct_if=False)
                for m in args.methods:
                    for r in args.ratios:
                        run(text, m, r, RecordingPress(press=presses[m](compression_ratio=r)))
                total += 2 + len(args.methods) * len(args.ratios)
            print(f"  L={L} pos={pos} done ({total} caches, 2 questions each)")

    dims = ModelDims.from_hf(pipe.model.config)
    out = {"model": model_name, "quant": args.quant,
           "full_bpt_bytes_per_tok": bpt_full_cache(dims),
           "config": vars(args), "summary": summarize(records), "records": records}
    json.dump(out, open(args.out, "w"), indent=2)
    for s in out["summary"]:
        f = lambda v: "  -- " if v is None else f"{v:5.2f}"
        print(f"{s['method']:>13} r={s['ratio']}: acc={f(s['acc'])} yes={f(s['probe_yes_rate'])} "
              f"conf={f(s['mean_conf'])} over={f(s['overclaim'])} "
              f"conf|no={f(s['conf_when_probe_no'])} key={f(s['key_kept'])} val={f(s['value_kept'])}")
    print("saved", args.out, "| max VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
