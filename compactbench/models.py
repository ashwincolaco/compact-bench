"""Model loading helpers. One place to change the default model."""
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUANT_CHOICES = ("fp16", "nf4")


def quant_kwargs(quant="fp16"):
    """Weight-precision kwargs shared by both loaders.

    fp16 is the reference precision. nf4 is 4-bit NF4 with double quantization and
    an fp16 compute dtype, which is what lets models above ~3B fit an 8 GB card.
    Weight precision is independent of the KV-cache budget the benchmark sweeps:
    BPT prices the cache, not the weights. Quantizing weights is still a confound
    for cross-model comparison, so it is recorded in every run's config and is
    meant to be controlled by running one model at both precisions.
    """
    if quant == "fp16":
        return {"dtype": torch.float16}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        return {"dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)}
    raise ValueError(f"unknown quant {quant!r}, expected one of {QUANT_CHOICES}")


def load_kvpress_pipeline(model_name=DEFAULT_MODEL, device="cuda", quant="fp16"):
    """KVPress text-generation pipeline (context compressed by a press per call)."""
    import kvpress  # noqa: F401  -- registers the "kv-press-text-generation" task
    from transformers import pipeline
    kw = quant_kwargs(quant)
    if quant == "fp16":
        return pipeline("kv-press-text-generation", model=model_name,
                        device=device, **kw)
    # A 4-bit model is already placed on the GPU by accelerate, so the pipeline
    # must not be handed a device as well.
    return pipeline("kv-press-text-generation", model=model_name,
                    model_kwargs=kw)


def load_causal_lm(model_name=DEFAULT_MODEL, quant="fp16"):
    """Plain causal LM + tokenizer for the agent-layer tasks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", **quant_kwargs(quant)).eval()
    return tok, model


def gen(tok, model, prompt, max_new=80):
    """Greedy chat-template generation, returns the completion text."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    n_in = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n_in:], skip_special_tokens=True).strip()


def ntok(tok, text):
    return len(tok(text)["input_ids"])


def default_presses():
    """The KV-compaction methods swept by the frontier/attribution/confidence tasks."""
    from kvpress import (SnapKVPress, StreamingLLMPress, TOVAPress, KnormPress,
                         ExpectedAttentionPress, RandomPress)
    return {"SnapKV": SnapKVPress, "StreamingLLM": StreamingLLMPress,
            "TOVA": TOVAPress, "Knorm": KnormPress,
            "ExpectedAttn": ExpectedAttentionPress, "Random": RandomPress}
