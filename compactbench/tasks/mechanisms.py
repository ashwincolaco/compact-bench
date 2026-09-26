"""Task 6 — mechanisms: methods from different literatures on one BPT axis.

The same needles are answered after five kinds of compaction, each priced in
bytes of retained state per token of history:
  eviction        KVPress scorer presses, keep fraction f       -> f
  quantization    b-bit KV cache, groups of 32, 1% fp16 outliers -> (b + 1.48) / 16
  evict+quant     keep f, then quantize the kept entries         -> f (b + 1.48) / 16
  prompt          LLMLingua-2 deletes tokens before prefill     -> m / n (measured)
  summary         the model rewrites the context to m tokens    -> m / n (measured)
All run question-agnostic: nothing sees the question before it compacts.
"""
import json, os, random

from ..bpt import ModelDims, bpt_full_cache
from ..data import load_filler_pool, build_needle_context, sample_needle, QUESTION
from ..models import load_kvpress_pipeline, default_presses, gen, ntok
from ..presses import QuantPress, EvictQuantPress, quant_bits_per_scalar

LINGUA = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--quant", default="fp16", choices=["fp16", "bf16", "nf4", "prequant"])
    p.add_argument("--lengths", type=int, nargs="+", default=[2000, 4000])
    p.add_argument("--positions", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--keep", type=float, nargs="+", default=[0.5, 0.25, 0.125])
    p.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3, 2])
    p.add_argument("--combo", nargs="+", default=["8x0.5", "4x0.5", "4x0.25"],
                   help="evict+quant points as BITSxKEEP")
    p.add_argument("--skip", nargs="*", default=[], choices=["prompt", "summary"])
    p.add_argument("--out", default="runs/mechanisms.json")


def _summary(tok, model, text, m):
    s = gen(tok, model, f"Summarize the following text in at most {max(20, int(m * 0.75))} "
            f"words, preserving every specific name and number:\n\n{text}", max_new=m + 40)
    return tok.decode(tok(s)["input_ids"][:m], skip_special_tokens=True)


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
    lingua = None
    if "prompt" not in args.skip:
        from llmlingua import PromptCompressor
        lingua = PromptCompressor(model_name=LINGUA, use_llmlingua2=True, device_map="cuda")

    # (family, method, setting) -> list of (correct, bpt_frac)
    res = {}
    def add(fam, meth, setting, ok, frac):
        res.setdefault((fam, meth, str(setting)), []).append((int(ok), frac))

    total = 0
    for L in args.lengths:
        for pos in args.positions:
            for _ in range(args.trials):
                key, value = sample_needle(rng)
                text = build_needle_context(tok, filler, L, pos, key, value)
                q = QUESTION.format(key=key)
                ask = lambda ctx, press=None: value in pipe(ctx, question=q, press=press)["answer"]
                add("full", "full", 1.0, ask(text), 1.0); total += 1
                for m, cls in presses.items():
                    for f in args.keep:
                        add("eviction", m, f, ask(text, cls(compression_ratio=1 - f)), f)
                        total += 1
                for b in args.bits:
                    add("quantization", "KV-quant", b, ask(text, QuantPress(bits=b)),
                        quant_bits_per_scalar(b) / 16); total += 1
                for c in args.combo:
                    b, f = c.split("x"); b, f = int(b), float(f)
                    for m in ("SnapKV", "StreamingLLM", "Knorm", "Random"):
                        press = EvictQuantPress(press=presses[m](compression_ratio=1 - f), bits=b)
                        add("evict+quant", f"{m}+{b}bit", f, ask(text, press),
                            f * quant_bits_per_scalar(b) / 16); total += 1
                n = ntok(tok, text)
                for f in args.keep:
                    if lingua is not None:
                        short = lingua.compress_prompt(text, rate=f, force_tokens=["\n", "."])["compressed_prompt"]
                        add("prompt", "LLMLingua-2", f, ask(short), ntok(tok, short) / n); total += 1
                    if "summary" not in args.skip:
                        short = _summary(tok, pipe.model, text, int(f * n))
                        add("summary", "summary", f, ask(short), ntok(tok, short) / n); total += 1
            print(f"  L={L} pos={pos} done ({total} generations)", flush=True)

    dims = ModelDims.from_hf(pipe.model.config)
    out = {"model": model_name, "quant": args.quant,
           "full_bpt_bytes_per_tok": bpt_full_cache(dims), "config": vars(args), "points": []}
    for (fam, meth, setting), xs in res.items():
        out["points"].append({"family": fam, "method": meth, "setting": setting, "n": len(xs),
                              "acc": sum(o for o, _ in xs) / len(xs),
                              "bpt_frac": sum(fr for _, fr in xs) / len(xs)})
    json.dump(out, open(args.out, "w"), indent=2)
    for p in sorted(out["points"], key=lambda p: -p["bpt_frac"]):
        print(f"{p['family']:>13} {p['method']:>20} {p['setting']:>5}  "
              f"BPT={p['bpt_frac']*100:5.1f}%  acc={p['acc']:.2f}")
    print("saved", args.out, "| max VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
