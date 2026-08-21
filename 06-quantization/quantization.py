"""Quantization: fewer bytes per weight and per cached token.

Decode is bandwidth bound (see 05), so the two things worth shrinking are the two
things that get streamed out of memory every step.

  * Weights are read once per decode step regardless of batch size. Storing them
    at 8 or 4 bits cuts that traffic 2x or 4x.
  * The KV cache is read once per step per sequence and is what caps concurrency.
    Halving bytes per token doubles the blocks that fit, which doubles the batch,
    which is where the throughput actually comes from (see 03).

Both use affine quantization over small groups:

    scale = (max - min) / (2^bits - 1)
    zero  = round(-min / scale)
    q     = clamp(round(w / scale) + zero, 0, 2^bits - 1)
    w_hat = (q - zero) * scale

Group size is the whole tradeoff. One scale for a huge tensor is cheap to store
and inaccurate; one scale per 16 values is accurate and the scales themselves
start to cost real memory.

Honest caveat about speed: this dequantizes into fp32 and calls the normal matmul,
so on CPU it is *slower* than not quantizing. The win is bytes, and turning bytes
into latency needs a fused kernel that dequantizes inside the matmul loop. The
tables below therefore report memory and quality, not throughput, as the result.
"""

import copy
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- shared primitives, repeated so this folder stands on its own -------------


def reshape_to_heads(x, num_heads):
    """[batch, seq_len, num_heads * d] -> [batch, num_heads, seq_len, d]"""
    batch_size, seq_len, width = x.shape
    return x.view(batch_size, seq_len, num_heads, width // num_heads).transpose(1, 2)


def reshape_from_heads(x):
    """[batch, num_heads, seq_len, d] -> [batch, seq_len, num_heads * d]"""
    batch_size, num_heads, seq_len, d_k = x.shape
    return x.transpose(1, 2).reshape(batch_size, seq_len, num_heads * d_k)


def repeat_kv(x, n_rep):
    """Broadcast num_kv_heads up to num_heads for grouped-query attention."""
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, d_k = x.shape
    return (
        x[:, :, None]
        .expand(batch, num_kv_heads, n_rep, seq_len, d_k)
        .reshape(batch, num_kv_heads * n_rep, seq_len, d_k)
    )


def causal_mask(seq_len_q, seq_len_k, device):
    offset = seq_len_k - seq_len_q
    q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
    return k_pos <= q_pos


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------- quantization


def affine_quantize(x, bits, dim=-1):
    """Symmetric-range affine quantization with min/max taken along `dim`."""
    levels = 2 ** bits - 1
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(x / scale) + zero, 0, levels)
    return q.to(torch.uint8), scale, zero


def pack4(q):
    """Two 4-bit values per byte, so the stored size is genuinely halved."""
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous()


def unpack4(packed):
    return torch.stack([packed & 0xF, (packed >> 4) & 0xF], dim=-1).reshape(
        *packed.shape[:-1], -1
    )


class QuantizedLinear(nn.Module):
    """Weights held as low-bit integers with one scale per group of input
    channels, dequantized on the fly."""

    def __init__(self, linear, bits=8, group_size=64):
        super().__init__()
        weight = linear.weight.data
        out_features, in_features = weight.shape
        assert in_features % group_size == 0, "group_size must divide in_features"

        self.bits = bits
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size

        grouped = weight.view(out_features, in_features // group_size, group_size)
        q, scale, zero = affine_quantize(grouped, bits)

        self.register_buffer("q", pack4(q) if bits == 4 else q, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("zero", zero, persistent=False)
        self.bias = linear.bias

    def dequantize(self):
        q = unpack4(self.q) if self.bits == 4 else self.q
        weight = (q.float() - self.zero) * self.scale
        return weight.view(self.out_features, self.in_features)

    def forward(self, x):
        return F.linear(x, self.dequantize().to(x.dtype), self.bias)

    @property
    def nbytes(self):
        return (self.q.numel() * self.q.element_size()
                + self.scale.numel() * 4 + self.zero.numel() * 4)


def quantize_weights(model, bits, group_size):
    """Swap every nn.Linear for a quantized one. Embeddings and norms stay in
    fp32, which is what production recipes do."""
    quantized = copy.deepcopy(model)
    for module in quantized.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, bits, group_size))
    return quantized


def linear_bytes(model):
    total = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            total += module.nbytes
        elif isinstance(module, nn.Linear):
            total += module.weight.numel() * module.weight.element_size()
    return total


# --------------------------------------------------------------------- kv caches


class KVCache:
    bits = 32

    def __init__(self, max_seq_len, num_kv_heads, d_k, device):
        shape = (1, num_kv_heads, max_seq_len, d_k)
        self.K = torch.zeros(shape, device=device)
        self.V = torch.zeros(shape, device=device)
        self.num_kv_heads, self.d_k = num_kv_heads, d_k
        self.length = 0

    def update(self, K_new, V_new):
        end = self.length + K_new.shape[2]
        self.K[:, :, self.length:end] = K_new
        self.V[:, :, self.length:end] = V_new
        self.length = end
        return self.K[:, :, :end], self.V[:, :, :end]

    @property
    def bytes_per_token(self):
        return 2 * self.num_kv_heads * self.d_k * 4


class QuantizedKVCache:
    """K and V held as integers with an affine scale per (head, token).

    Quantizing on write is what real caches do -- a token's K/V is finished the
    moment it is produced, so its range is known and never revisited.

    Elements are stored one per byte here to keep the read path legible; the
    byte count below reflects packed storage, which is what actually ships.
    """

    def __init__(self, max_seq_len, num_kv_heads, d_k, device, bits=8):
        shape = (1, num_kv_heads, max_seq_len, d_k)
        meta = (1, num_kv_heads, max_seq_len, 1)
        self.bits = bits
        self.num_kv_heads, self.d_k = num_kv_heads, d_k
        self.qK = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.qV = torch.zeros(shape, dtype=torch.uint8, device=device)
        self.scale = torch.zeros((2, *meta), device=device)
        self.zero = torch.zeros((2, *meta), device=device)
        self.length = 0

    def update(self, K_new, V_new):
        end = self.length + K_new.shape[2]
        for index, (buffer, value) in enumerate(((self.qK, K_new), (self.qV, V_new))):
            q, scale, zero = affine_quantize(value.float(), self.bits)
            buffer[:, :, self.length:end] = q
            self.scale[index, :, :, self.length:end] = scale
            self.zero[index, :, :, self.length:end] = zero
        self.length = end

        return (self._read(0, self.qK, end, K_new.dtype),
                self._read(1, self.qV, end, V_new.dtype))

    def _read(self, index, buffer, end, dtype):
        q = buffer[:, :, :end].float()
        scale = self.scale[index, :, :, :end]
        zero = self.zero[index, :, :, :end]
        return ((q - zero) * scale).to(dtype)

    @property
    def bytes_per_token(self):
        data = 2 * self.num_kv_heads * self.d_k * self.bits / 8
        scales = 2 * self.num_kv_heads * 2 * 4  # scale and zero, fp32, per head
        return int(data + scales)


# -------------------------------------------------------------------------- model


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(num_heads * self.d_k, d_model, bias=False)

    def forward(self, x, cache=None):
        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V = reshape_to_heads(self.W_V(x), self.num_kv_heads)
        if cache is not None:
            K, V = cache.update(K, V)

        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)
        mask = causal_mask(Q.shape[2], K.shape[2], x.device)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
        return self.W_O(reshape_from_heads(output))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, num_kv_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, num_kv_heads, 4 * d_model) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def device(self):
        return self.token_emb.weight.device

    def make_caches(self, max_seq_len, bits=32):
        return [
            KVCache(max_seq_len, b.attn.num_kv_heads, b.attn.d_k, self.device)
            if bits == 32 else
            QuantizedKVCache(max_seq_len, b.attn.num_kv_heads, b.attn.d_k, self.device, bits)
            for b in self.blocks
        ]

    def forward(self, input_ids, caches=None):
        offset = caches[0].length if caches else 0
        positions = torch.arange(offset, offset + input_ids.shape[1], device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block, cache in zip(self.blocks, caches or [None] * len(self.blocks)):
            x = block(x, cache)
        return self.lm_head(self.ln_f(x))


# ------------------------------------------------------------- synthetic training


class PhraseLanguage:
    """Sequences made by concatenating short fixed phrases, so a small model can
    learn it in seconds and its logits become peaked enough for quantization
    error to actually show up."""

    def __init__(self, vocab_size=256, num_phrases=48, min_len=4, max_len=8, seed=3):
        rng = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.phrases = []
        for _ in range(num_phrases):
            length = int(torch.randint(min_len, max_len + 1, (1,), generator=rng))
            self.phrases.append(torch.randint(0, vocab_size, (length,), generator=rng).tolist())

    def sample(self, length, generator):
        out = []
        while len(out) < length:
            index = int(torch.randint(len(self.phrases), (1,), generator=generator))
            out.extend(self.phrases[index])
        return out[:length]

    def batch(self, batch_size, seq_len, generator):
        return torch.tensor([self.sample(seq_len, generator) for _ in range(batch_size)])


def train(model, language, steps=200, batch_size=16, seq_len=64, lr=3e-3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    loss = torch.zeros(())
    for _ in range(steps):
        batch = language.batch(batch_size, seq_len + 1, generator).to(model.device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    model.eval()
    return float(loss.detach())


# ---------------------------------------------------------------------- evaluation


@dataclass
class Quality:
    top1: float  # fraction of positions whose argmax still matches fp32
    kl: float  # mean KL(reference || quantized), in nats


@torch.no_grad()
def compare(reference_logits, logits):
    top1 = (reference_logits.argmax(-1) == logits.argmax(-1)).float().mean()
    log_p = F.log_softmax(reference_logits.float(), dim=-1)
    log_q = F.log_softmax(logits.float(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean()
    return Quality(float(top1), float(kl))


@torch.no_grad()
def decode_speed(model, tokens, steps=48, bits=32):
    caches = model.make_caches(len(tokens) + steps + 1, bits)
    fed = torch.tensor([tokens], device=model.device)
    start = time.perf_counter()
    for _ in range(steps):
        logits = model(fed, caches)[0, -1]
        fed = torch.tensor([[int(logits.argmax())]], device=model.device)
    return steps / (time.perf_counter() - start)


def main():
    torch.manual_seed(0)
    language = PhraseLanguage()

    model = TinyLM(language.vocab_size)
    start = time.perf_counter()
    loss = train(model, language)
    train_time = time.perf_counter() - start

    tokens = language.sample(192, torch.Generator().manual_seed(99))
    ids = torch.tensor([tokens])
    reference = model(ids)[0]
    fp32_weights = linear_bytes(model)

    print(f"4-layer d128 model, vocab {language.vocab_size}, "
          f"trained {train_time:.0f}s on a phrase language (loss {loss:.2f})")
    print(f"quality is measured against the fp32 model on the same {len(tokens)} tokens")
    print()

    print("weight-only quantization")
    print(f"  {'':<18}{'linear MiB':>12}{'vs fp32':>10}{'top-1 kept':>12}"
          f"{'KL':>11}{'tok/s':>9}")
    print(f"  {'fp32':<18}{fp32_weights / 2**20:>12.2f}{'1.0x':>10}{'100.0%':>12}"
          f"{0.0:>11.1e}{decode_speed(model, tokens[:64]):>9.1f}")

    for bits, group in ((8, 64), (4, 64), (4, 16)):
        quantized = quantize_weights(model, bits, group)
        quality = compare(reference, quantized(ids)[0])
        size = linear_bytes(quantized)
        print(f"  {f'int{bits}, group {group}':<18}{size / 2**20:>12.2f}"
              f"{f'{fp32_weights / size:.1f}x':>10}{quality.top1:>12.1%}"
              f"{quality.kl:>11.1e}{decode_speed(quantized, tokens[:64]):>9.1f}")
    print()

    print("KV cache quantization")
    print(f"  {'':<18}{'B/token':>12}{'vs fp32':>10}{'top-1 kept':>12}"
          f"{'KL':>11}{'in 16 MiB':>12}")
    fp32_caches = model.make_caches(len(tokens))
    fp32_cached = model(ids, fp32_caches)[0]
    fp32_per_token = fp32_caches[0].bytes_per_token * len(model.blocks)
    print(f"  {'fp32':<18}{fp32_per_token:>12}{'1.0x':>10}{'100.0%':>12}"
          f"{0.0:>11.1e}{2**24 // fp32_per_token:>12}")

    for bits in (8, 4):
        caches = model.make_caches(len(tokens), bits)
        quality = compare(fp32_cached, model(ids, caches)[0])
        per_token = caches[0].bytes_per_token * len(model.blocks)
        print(f"  {f'int{bits} per token':<18}{per_token:>12}"
              f"{f'{fp32_per_token / per_token:.1f}x':>10}{quality.top1:>12.1%}"
              f"{quality.kl:>11.1e}{2**24 // per_token:>12}")
    print()

    combined = quantize_weights(model, 8, 64)
    caches = combined.make_caches(len(tokens), 8)
    quality = compare(reference, combined(ids, caches)[0])
    print(f"int8 weights + int8 KV : top-1 {quality.top1:.1%}, KL {quality.kl:.1e}, "
          f"{fp32_weights / linear_bytes(combined):.1f}x weights, "
          f"{fp32_per_token / (caches[0].bytes_per_token * len(model.blocks)):.1f}x cache")
    print()
    print("int4 needs small groups to stay usable, and the scales then eat into the")
    print("saving. tok/s goes down because dequantizing into fp32 and calling the")
    print("normal matmul is strictly more work -- the latency win needs a fused kernel.")


if __name__ == "__main__":
    main()
