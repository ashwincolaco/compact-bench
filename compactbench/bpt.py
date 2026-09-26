"""Bytes-per-token-of-history (BPT): the common budget axis of COMPACT-Bench.

BPT = (bytes of retained memory state) / (tokens of original history).

The retained state at inference is ultimately a KV cache, so every compaction
mechanism -- eviction, quantization, prompt compression, gisting, summarization --
is expressed in one currency: how many bytes of KV-equivalent memory survive per
token of the original context. This is what lets a KV evictor and a quantizer (or
an agent summarizer) be plotted on the same chart.
"""
from dataclasses import dataclass


@dataclass
class ModelDims:
    """Just enough of a model's shape to price its KV cache."""
    num_layers: int
    num_kv_heads: int      # key/value heads (GQA-aware)
    head_dim: int
    dtype_bytes: float = 2.0   # fp16 baseline

    def kv_bytes_per_token(self, dtype_bytes=None) -> float:
        """Bytes of KV cache for ONE token of context (K and V, all layers)."""
        b = self.dtype_bytes if dtype_bytes is None else dtype_bytes
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * b

    @classmethod
    def from_hf(cls, config):
        """Build from a HuggingFace model config."""
        kv = getattr(config, "num_key_value_heads", config.num_attention_heads)
        hd = getattr(config, "head_dim",
                     config.hidden_size // config.num_attention_heads)
        return cls(config.num_hidden_layers, kv, hd)


def bpt_kv_eviction(dims: ModelDims, keep_fraction: float) -> float:
    """Keep a fraction of token KV entries (H2O, SnapKV, StreamingLLM, ...)."""
    return keep_fraction * dims.kv_bytes_per_token()


def bpt_kv_quant(dims: ModelDims, bits: float, meta_bits: float = 0.0) -> float:
    """Quantize all KV entries to `bits` (KIVI, KVQuant, ...). fp16 baseline = 16 bits.

    `meta_bits` is what the quantizer stores per scalar beyond its payload: group
    scales and zero points, and any full-precision outliers. The quantizer in
    presses.py stores 1.48 extra bits (fp16 scale and zero per 32 values, plus 1%
    fp16 outliers with 32-bit indices); leaving it out prices a "4-bit" cache at
    25% of the full cache when it really costs 34%.
    """
    return dims.kv_bytes_per_token(dtype_bytes=(bits + meta_bits) / 8.0)


def bpt_prompt_compression(dims: ModelDims, n_history: int, n_kept_tokens: int) -> float:
    """Hard prompt compression / summarization: fewer NL tokens, each still KV-cached."""
    return (n_kept_tokens / n_history) * dims.kv_bytes_per_token()


def bpt_soft_tokens(dims: ModelDims, n_history: int, n_gist: int, gist_dim: int = None,
                    dtype_bytes: float = 2.0) -> float:
    """Soft compression to dense vectors (gisting, ICAE, xRAG)."""
    d = gist_dim if gist_dim is not None else dims.num_kv_heads * dims.head_dim
    return (n_gist * d * dtype_bytes) / n_history


def bpt_full_cache(dims: ModelDims) -> float:
    """The uncompacted baseline: one full KV entry per history token."""
    return dims.kv_bytes_per_token()


if __name__ == "__main__":
    # A quick sanity print for Qwen2.5-1.5B-ish dims. Counted naively, 75% eviction
    # and 4-bit quantization land at the same BPT; counted with the quantizer's
    # metadata, the 4-bit cache costs as much as keeping 34% of the tokens.
    d = ModelDims(num_layers=28, num_kv_heads=2, head_dim=128)
    full = bpt_full_cache(d)
    print("full-cache BPT (bytes/token):", full)
    for kf in (1.0, 0.5, 0.25, 0.1):
        print(f"  evict keep={kf:>4}: {bpt_kv_eviction(d, kf):8.0f}  ({bpt_kv_eviction(d,kf)/full:.0%})")
    for b in (16, 8, 4, 2):
        m = 0.0 if b == 16 else 1.48
        print(f"  quant {b:>2}-bit : {bpt_kv_quant(d, b, m):8.0f}  ({bpt_kv_quant(d, b, m)/full:.0%})")
    print("  prompt 8k->1k:", f"{bpt_prompt_compression(d, 8000, 1000):8.0f}")
