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
