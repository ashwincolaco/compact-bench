"""Task 2 — recoverability at equal storage budget.

Both operators are bounded to the SAME budget of B tokens and compact only on
overflow. The irreversible operator summarizes its raw buffer to fit B (lossy: the
originals are gone). The reversible operator keeps raw chunks and evicts the
oldest to fit B (bounded too, not an archive), reading back kept chunks at query
time. Sweeping B isolates lossy-vs-recoverable storage at matched budget.

Reference finding (Qwen2.5-1.5B): the two trade places. Recoverable storage
converts budget into recall monotonically and leads once the budget is adequate
(>~50%); under tight budgets (<~25%) the lossy summary leads because it spans the
whole history while a bounded raw window only holds the recent fraction.
"""
import json, os, random

from ..data import build_fact_doc
from ..models import load_causal_lm, gen, ntok


def add_args(p):
    p.add_argument("--model", default=None)
    p.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--n_chunks", type=int, default=24)
    p.add_argument("--n_facts", type=int, default=12)
    p.add_argument("--out", default="runs/reversibility.json")


def _summarize_to_budget(tok, model, text, budget):
    words = max(20, int(budget * 0.75))
    s = gen(tok, model, f"Summarize the following notes in at most {words} words, "
            f"preserving every specific name, attribute, and number:\n\n{text}",
            max_new=min(380, int(budget * 1.4) + 40))
    return tok.decode(tok(s)["input_ids"][:budget], skip_special_tokens=True)


def _ask(tok, model, ctx, facts):
    correct = 0
    for ent, attr, val in facts:
        q = f"What is the {attr} of {ent}? Answer with just the value."
        if val in gen(tok, model, f"Notes:\n{ctx(ent)}\n\n{q}", max_new=20):
            correct += 1
    return correct / len(facts)


def run_irreversible(tok, model, chunks, facts, budget):
    buf = ""
    for ch in chunks:
        buf = (buf + " " + ch).strip()
        if ntok(tok, buf) > budget:          # compact only on overflow
            buf = _summarize_to_budget(tok, model, buf, budget)
    return _ask(tok, model, lambda ent: buf, facts)


def run_reversible(tok, model, chunks, facts, budget):
    store, used = [], 0
    for ch in chunks:
        store.append((ch, ntok(tok, ch))); used += store[-1][1]
        while used > budget and store:       # evict oldest -> bounded, unrecoverable
            used -= store.pop(0)[1]
    def ctx(ent):
        hits = [t for t, _ in store if ent in t]
        text = " ".join(hits) if hits else " ".join(t for t, _ in store)
        return tok.decode(tok(text)["input_ids"][:budget], skip_special_tokens=True)
    return _ask(tok, model, ctx, facts)


def main(args):
    import torch
    from ..models import DEFAULT_MODEL
    model_name = args.model or DEFAULT_MODEL
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok, model = load_causal_lm(model_name)
    rng = random.Random(0)

    docs = [build_fact_doc(rng, args.n_chunks, args.n_facts) for _ in range(args.trials)]
    total = sum(ntok(tok, " ".join(ch)) for ch, _ in docs) / len(docs)
    out = {"model": model_name, "avg_total_tokens": total,
           "config": vars(args), "points": []}
    for f in args.budgets:
        B = max(32, int(f * total))
        irr = [run_irreversible(tok, model, ch, fa, B) for ch, fa in docs]
        rev = [run_reversible(tok, model, ch, fa, B) for ch, fa in docs]
        pi, pr = sum(irr) / len(irr), sum(rev) / len(rev)
        out["points"].append({"budget_frac": f, "budget_tokens": B,
                              "irreversible_acc": pi, "reversible_acc": pr})
        print(f"B={f*100:4.0f}% ({B:5d} tok)  irreversible={pi:.2f}  reversible={pr:.2f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print("saved", args.out,
          "| max VRAM %.2f GiB" % (torch.cuda.max_memory_allocated() / 1024**3))
