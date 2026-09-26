"""Model loading helpers. One place to change the default model."""
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QUANT_CHOICES = ("fp16", "bf16", "nf4", "prequant")


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
    if quant == "bf16":
        return {"dtype": torch.bfloat16}
    if quant == "prequant":
        # A checkpoint saved already quantized (e.g. bitsandbytes NF4): its config
        # carries the quantization, so only the compute dtype is set here.
        return {"dtype": torch.float16}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        return {"dtype": torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True)}
    raise ValueError(f"unknown quant {quant!r}, expected one of {QUANT_CHOICES}")


def _patch_kvpress_olmo2():
    """Give KVPress's query/key extraction OLMo-2's QK-norm.

    KVPress treats any attention module with q_proj/k_proj as Llama-like and only
    applies QK-norm for Qwen3 and Gemma3. OLMo-2 also normalizes queries and keys,
    over the flat projection before the per-head reshape, so without this the
    query-scored presses (SnapKV, ExpectedAttn) would score OLMo-2 with the wrong
    queries. Knorm needs nothing: it scores the cached keys, which are already
    normalized.
    """
    import importlib
    from kvpress import utils
    from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention
    if getattr(utils, "_olmo2_patched", False):
        return
    orig_q, orig_k = utils.get_prerope_query_states, utils.get_prerope_key_states

    def query_states(module, hidden_states):
        if not isinstance(module, Olmo2Attention):
            return orig_q(module, hidden_states)
        bsz, n, _ = hidden_states.shape
        q = module.q_norm(module.q_proj(hidden_states))
        return q.view(bsz, n, -1, module.head_dim).transpose(1, 2)

    def key_states(module, hidden_states):
        if not isinstance(module, Olmo2Attention):
            return orig_k(module, hidden_states)
        bsz, n, _ = hidden_states.shape
        k = module.k_norm(module.k_proj(hidden_states))
        return k.view(bsz, n, -1, module.head_dim).transpose(1, 2)

    # Presses bind these names at import time, so patch every module that took one.
    for name in ("utils", "presses.snapkv_press", "presses.expected_attention_press",
                 "presses.kvzip_press", "presses.non_causal_attention_press",
                 "presses.think_press", "presses.leverage_press"):
        mod = importlib.import_module(f"kvpress.{name}")
        if hasattr(mod, "get_prerope_query_states"):
            mod.get_prerope_query_states = query_states
        if hasattr(mod, "get_prerope_key_states"):
            mod.get_prerope_key_states = key_states

    # OLMo-2 keeps its rotary cos/sin in float32, which KVPress's scorers then
    # multiply against bf16 queries and keys. Cast them for scoring only. The hook
    # runs after the attention forward, so the model's own computation, and the
    # cache being compressed, are untouched.
    from kvpress.presses.base_press import BasePress
    from kvpress.presses.expected_attention_press import ExpectedAttentionPress
    orig_hook, orig_rope = BasePress.forward_hook, ExpectedAttentionPress.apply_avg_rope

    def forward_hook(self, module, input, kwargs, output):
        pe = kwargs.get("position_embeddings")
        if isinstance(module, Olmo2Attention) and pe is not None:
            dt = kwargs["hidden_states"].dtype
            kwargs = {**kwargs, "position_embeddings": tuple(t.to(dt) for t in pe)}
        return orig_hook(self, module, input, kwargs, output)

    def apply_avg_rope(self, module, mu, cov, q_len):
        if not isinstance(module, Olmo2Attention):
            return orig_rope(self, module, mu, cov, q_len)
        dt = mu.dtype
        mu, cov = orig_rope(self, module, mu.float(),
                            None if cov is None else cov.float(), q_len)
        return mu.to(dt), None if cov is None else cov.to(dt)

    BasePress.forward_hook = forward_hook
    ExpectedAttentionPress.apply_avg_rope = apply_avg_rope
    utils._olmo2_patched = True


def load_kvpress_pipeline(model_name=DEFAULT_MODEL, device="cuda", quant="fp16"):
    """KVPress text-generation pipeline (context compressed by a press per call)."""
    import kvpress  # noqa: F401  -- registers the "kv-press-text-generation" task
    _patch_kvpress_olmo2()
    from transformers import pipeline
    kw = quant_kwargs(quant)
    if quant in ("fp16", "bf16") and torch.cuda.device_count() <= 1:
        return pipeline("kv-press-text-generation", model=model_name,
                        device=device, **kw)
    # Several visible GPUs means a model too large for one card: let accelerate
    # shard it, which is also the path a 4-bit model needs.
    if quant in ("fp16", "bf16"):
        kw = {**kw, "device_map": "auto"}
    # A 4-bit model is already placed on the GPU by accelerate, so the pipeline
    # must not be handed a device as well.
    return pipeline("kv-press-text-generation", model=model_name,
                    model_kwargs=kw)


def load_causal_lm(model_name=DEFAULT_MODEL, quant="fp16"):
    """Plain causal LM + tokenizer for the agent-layer tasks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", **quant_kwargs(quant)).eval()
    return tok, model


def gen(tok, model, prompt, max_new=80):
    """Greedy chat-template generation, returns the completion text."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  # Hybrid-reasoning templates (Qwen3) think by default;
                                  # match KVPress, which disables it. Others ignore it.
                                  enable_thinking=False,
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
