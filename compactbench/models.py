"""Model loading helpers. One place to change the default model."""
import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_kvpress_pipeline(model_name=DEFAULT_MODEL, device="cuda"):
    """KVPress text-generation pipeline (context compressed by a press per call)."""
    import kvpress  # noqa: F401  -- registers the "kv-press-text-generation" task
    from transformers import pipeline
    return pipeline("kv-press-text-generation", model=model_name,
                    device=device, dtype=torch.float16)


def load_causal_lm(model_name=DEFAULT_MODEL):
    """Plain causal LM + tokenizer for the agent-layer tasks."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda").eval()
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
