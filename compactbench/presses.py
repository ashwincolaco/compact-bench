"""Presses that KVPress does not ship: KV quantization, eviction followed by
quantization, and a recorder that logs which positions an eviction keeps.

Quantization is simulated (quantize, then dequantize in place), which is the
standard way KV-quantization accuracy is measured: the model reads exactly the
values a b-bit cache would hold. Keys are grouped per channel and values per
token (KIVI), and the largest 1% of entries in each layer are kept in fp16 as a
sparse outlier set (KVQuant's dense-and-sparse scheme). Without the outlier set,
4-bit keys break Qwen2.5 outright, because its first layer carries key channels
of magnitude ~300. BPT prices everything that is stored: b bits per scalar, one
fp16 scale and zero point per group, and 16 value bits plus a 32-bit index per
outlier, so a quantized point sits where its real storage would.
"""
from dataclasses import dataclass, field

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress

GROUP = 32          # KIVI's group size
META_BITS = 32      # one fp16 scale + one fp16 zero point per group
OUTLIER_FRAC = 0.01  # KVQuant's sparse fp16 outlier share
OUTLIER_BITS = 48    # fp16 value + 32-bit index per outlier


def fake_quant(x, bits, dim):
    """Asymmetric min-max quantization in groups of GROUP along `dim`."""
    if bits >= 16:
        return x
    x = x.transpose(dim, -1)
    shape = x.shape
    n = shape[-1]
    pad = (-n) % GROUP
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    g = xf.reshape(*shape[:-1], -1, GROUP)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-8) / levels
    q = ((g - lo) / scale).round().clamp(0, levels) * scale + lo
    q = q.reshape(*shape[:-1], -1)[..., :n].to(x.dtype)
    return q.transpose(dim, -1).contiguous()


def dense_sparse_quant(x, bits, dim):
    """Quantize all but the largest OUTLIER_FRAC of entries, which stay exact."""
    if bits >= 16:
        return x
    a = x.abs().float().flatten()
    k = max(1, int(OUTLIER_FRAC * a.numel()))
    thr = a.kthvalue(a.numel() - k).values
    mask = x.abs() > thr
    q = fake_quant(torch.where(mask, torch.zeros_like(x), x), bits, dim)
    return torch.where(mask, x, q)


def quant_bits_per_scalar(bits):
    """Stored bits per cached scalar: payload, group metadata, sparse outliers."""
    if bits >= 16:
        return 16
    return bits + META_BITS / GROUP + OUTLIER_FRAC * OUTLIER_BITS


def kivi_quant(keys, values, bits):
    """Keys grouped per channel along tokens, values per token along channels."""
    return dense_sparse_quant(keys, bits, dim=2), dense_sparse_quant(values, bits, dim=3)


@dataclass
class QuantPress(BasePress):
    """Quantize every cached key and value to `bits` (KIVI-style groups of 32)."""
    bits: int = 4

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        return kivi_quant(keys, values, self.bits)


@dataclass
class EvictQuantPress(BasePress):
    """Evict with a scorer press, then quantize what is kept."""
    press: ScorerPress = None
    bits: int = 8

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        keys, values = self.press.compress(module, hidden_states, keys, values,
                                           attentions, kwargs)
        return kivi_quant(keys, values, self.bits)


@dataclass
class RecordingPress(BasePress):
    """Run a scorer press and record the positions it keeps in every layer and head.

    `kept[layer]` is a (kv_heads, n_kept) LongTensor on the CPU. The eviction is
    computed exactly as ScorerPress.compress computes it, so the recorded set is
    the retained set the model then reads.
    """
    press: ScorerPress = None
    kept: dict = field(default_factory=dict)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        p = self.press
        if p.compression_ratio == 0:
            n = keys.shape[2]
            self.kept[module.layer_idx] = torch.arange(n).expand(keys.shape[1], n).clone()
            return keys, values
        scores = p.score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = int(keys.shape[2] * (1 - p.compression_ratio))
        idx = scores.topk(n_kept, dim=-1).indices
        self.kept[module.layer_idx] = idx[0].cpu()
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, idx).contiguous(), values.gather(2, idx).contiguous()

    def survival(self, positions):
        """Share of (layer, head) slots that keep each position, averaged over positions."""
        if not positions or not self.kept:
            return None
        pos = torch.tensor(positions)
        tot = 0.0
        for idx in self.kept.values():
            tot += torch.isin(idx, pos).sum().item() / (idx.shape[0] * len(positions))
        return tot / len(self.kept)
