"""Shared data builders: natural-filler needle contexts and fact documents."""
import random


def load_filler_pool(min_chars=400_000):
    """Natural English filler from wikitext-2 (varied, non-degenerate).

    Repetitive synthetic filler degrades attention scoring unfairly; natural text
    keeps the baseline honest (full-cache accuracy ~1.0 on the needle task).
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    buf, n = [], 0
    for r in ds:
        t = r["text"].strip()
        if len(t) > 40:
            buf.append(t); n += len(t)
            if n >= min_chars:
                break
    return " ".join(buf)


NEEDLE_KEYS = ["alpha", "bravo", "delta", "echo", "foxtrot", "sierra"]

QUESTION = "What is the special magic number for {key}? Answer with just the number."
PROBE = ("Do your notes contain the special magic number for {key}? "
         "Answer with just yes or no.")
CONFIDENCE = ("How confident are you that your answer is correct? "
              "Reply with just a number from 0 to 100.")


def build_needle_context(tok, filler, n_tokens, position, key, value):
    """Insert a key-value needle at a relative depth in natural filler."""
    needle = f" The special magic number for {key} is {value}. "
    ids = tok(filler)["input_ids"][:n_tokens]
    cut = int(len(ids) * position)
    return tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:])


def sample_needle(rng: random.Random):
    return rng.choice(NEEDLE_KEYS), str(rng.randint(100000, 999999))


# ---- fact documents for the reversibility task ----

ENTITIES = ["Zephyr Labs", "Marlowe Institute", "Calder Systems", "Tarn Foundation",
            "Vesper Group", "Othila Works", "Quill Dynamics", "Brenvik Co",
            "Halcyon Trust", "Yarrow Press", "Dunmore Bank", "Ferrand Studio"]
ATTRS = ["founding year", "headquarters city", "annual budget", "employee count",
         "flagship product", "registration code", "primary language", "server region",
         "audit grade", "license tier", "contact extension", "vault number"]


def build_fact_doc(rng: random.Random, n_chunks=24, n_facts=12):
    """A streaming document: filler chunks with (entity, attribute, value) facts."""
    filler = ("The committee reviewed the quarterly schedule and confirmed the "
              "regional rollout would proceed without changes to the timeline. ")
    facts, chunks = [], []
    pos = rng.sample(range(n_chunks), n_facts)
    fi = 0
    for c in range(n_chunks):
        body = filler * 3
        if c in pos:
            ent, attr = ENTITIES[fi], ATTRS[fi]
            val = str(rng.randint(1000, 9999))
            facts.append((ent, attr, val))
            body += f" Note: the {attr} of {ent} is {val}. "
            fi += 1
        chunks.append(body)
    return chunks, facts


# ---- variants added for the robustness studies ----

DISTRACTOR_KEYS = ["gamma", "kilo", "lima", "oscar", "tango", "victor", "zulu", "hotel"]


def build_multikey_context(tok, filler, n_tokens, position, key, value, rng, n_distractors=3):
    """The needle at `position`, plus same-format needles for other keys elsewhere.

    Distractors share the needle's template and a six-digit value, so a method has
    to keep the right key bound to the right number, not just any magic number.
    """
    ids = tok(filler)["input_ids"][:n_tokens]
    others = rng.sample(DISTRACTOR_KEYS, n_distractors)
    spots = [(position, f" The special magic number for {key} is {value}. ")]
    for k in others:
        d = rng.uniform(0.05, 0.95)
        while min(abs(d - s) for s, _ in spots) < 0.08:
            d = rng.uniform(0.05, 0.95)
        spots.append((d, f" The special magic number for {k} is {rng.randint(100000, 999999)}. "))
    spots.sort()
    out, prev = [], 0
    for d, s in spots:
        cut = int(len(ids) * d)
        out.append(tok.decode(ids[prev:cut])); out.append(s); prev = cut
    out.append(tok.decode(ids[prev:]))
    return "".join(out)


def build_absent_context(tok, filler, n_tokens):
    """The same filler with no needle: an uncompacted context that cannot answer."""
    return tok.decode(tok(filler)["input_ids"][:n_tokens])


def needle_token_positions(tok, context, key, value):
    """Token indices of the needle's key and value inside the pipeline's context ids.

    Reproduces KVPress's preprocessing (chat template, then split at the question
    separator) so the indices refer to the sequence the press actually scores.
    """
    if tok.chat_template is None:
        templated = (getattr(tok, "bos_token", "") or "") + context
    else:
        sep = "#" * (len(context) + 10)
        templated = tok.apply_chat_template(
            [{"role": "user", "content": context + sep}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False).split(sep)[0]
    enc = tok(templated, return_offsets_mapping=True, add_special_tokens=False)
    needle = f"The special magic number for {key} is {value}."
    start = templated.find(needle)
    k0 = start + needle.index(key); k1 = k0 + len(key)
    v0 = start + needle.index(value); v1 = v0 + len(value)
    s1 = start + len(needle)

    def span(a, b):
        return [i for i, (x, y) in enumerate(enc["offset_mapping"]) if x < b and y > a]
    return {"key": span(k0, k1), "value": span(v0, v1), "sentence": span(start, s1)}


def build_dense_fact_doc(rng: random.Random, n_chunks=24, facts_per_chunk=2):
    """A fact document with little filler, so that a summary has to drop facts.

    The standard document carries twelve facts in about 1,600 tokens of repeated
    filler, which a competent summarizer can keep verbatim at 10% of the budget.
    Here every chunk carries `facts_per_chunk` facts and one filler sentence.
    """
    filler = ("The committee reviewed the quarterly schedule and confirmed the "
              "regional rollout would proceed without changes to the timeline. ")
    pairs = [(e, a) for e in ENTITIES for a in ATTRS]
    rng.shuffle(pairs)
    facts, chunks, i = [], [], 0
    for c in range(n_chunks):
        body = filler
        for _ in range(facts_per_chunk):
            ent, attr = pairs[i]; i += 1
            val = str(rng.randint(1000, 9999))
            facts.append((ent, attr, val))
            body += f" Note: the {attr} of {ent} is {val}. "
        chunks.append(body)
    return chunks, facts
