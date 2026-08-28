
"""KV caching for autoregressive transformer inference, shaped the way serving
engines actually do it.

Base design from
https://machinelearningmastery.com/kv-caching-in-llms-a-guide-for-developers/

The article grows the cache with torch.cat and decodes greedily for a fixed
count. Production stacks (vLLM, TensorRT-LLM, TGI) differ in five ways, all
implemented here:

1. The cache is allocated once and written in place. Concatenating a growing
   tensor reallocates and copies the entire cache every step, which is O(n^2)
   memory traffic on the hottest path in the system.
2. Grouped-query attention: queries keep num_heads, keys and values use a
   smaller num_kv_heads, shrinking the cache by num_heads / num_kv_heads.
3. F.scaled_dot_product_attention, which dispatches to fused FlashAttention or
   memory-efficient kernels instead of materialising the score matrix.
4. Temperature / top-p sampling with an EOS stop condition.
5. TTFT and TPOT are reported separately: prefill is compute bound, decode is
   memory bound, and they carry different service-level objectives.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class KVCache:
    """One layer's pre-allocated cache.

    `length` bounds every read, so clearing between requests is a counter
    assignment rather than a memset of the whole buffer.
    """

    def __init__(self, max_batch, max_seq_len, num_kv_heads, d_k, dtype, device):
        shape = (max_batch, num_kv_heads, max_seq_len, d_k)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def update(self, K_new, V_new):
        end = self.length + K_new.shape[2]
        self.K[:, :, self.length:end] = K_new
        self.V[:, :, self.length:end] = V_new
        self.length = end
        return self.K[:, :, :end], self.V[:, :, :end]

    def reset(self):
        self.length = 0

    @property
    def nbytes(self):
        return (self.K.numel() + self.V.numel()) * self.K.element_size()


class MultiHeadAttentionWithCache(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert num_heads % self.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.n_rep = num_heads // self.num_kv_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(num_heads * self.d_k, d_model, bias=False)

    def forward(self, x, cache=None):
        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        if cache is not None:
            K, V = cache.update(K, V)

        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        # A one-token query attends to the whole cache, so causal masking only
        # applies to multi-token passes, which here always start from an empty
        # cache. Resuming a partial prefill would need an explicit mask.
        output = F.scaled_dot_product_attention(Q, K, V, is_causal=Q.shape[2] > 1)
        return self.W_O(reshape_from_heads(output))


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


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttentionWithCache(d_model, num_heads, num_kv_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.ffn(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    """Minimal decoder-only model.

    Caches are owned by the caller rather than the module, so one set of weights
    can serve many independent sequences.
    """

    def __init__(self, vocab_size, d_model=256, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, num_kv_heads, 4 * d_model)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def allocate_cache(self, max_batch, max_seq_len):
        weight = self.lm_head.weight
        return [
            KVCache(max_batch, max_seq_len, block.attn.num_kv_heads, block.attn.d_k,
                    weight.dtype, weight.device)
            for block in self.blocks
        ]

    def forward(self, input_ids, caches=None):
        """Returns [batch, vocab] -- logits for the final position only."""
        offset = caches[0].length if caches else 0
        positions = torch.arange(
            offset, offset + input_ids.shape[1], device=input_ids.device
        )

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block, cache in zip(self.blocks, caches or [None] * len(self.blocks),
                                strict=True):
            x = block(x, cache)

        # Projecting only the last row skips a [seq_len, vocab] matmul per step.
        return self.lm_head(self.ln_f(x[:, -1]))


@dataclass
class SamplingParams:
    max_new_tokens: int = 192
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 0.95
    eos_id: int | None = None


def sample(logits, params, generator=None):
    """Greedy, or nucleus (top-p) sampling at the requested temperature."""
    if params.temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)
    # Keep the shortest prefix of the sorted tail whose mass reaches top_p.
    keep = (sorted_probs.cumsum(dim=-1) - sorted_probs) < params.top_p
    sorted_probs = sorted_probs * keep
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return sorted_ids.gather(-1, choice)


@dataclass
class GenerationResult:
    tokens: torch.Tensor
    num_generated: int
    ttft: float  # time to first token, dominated by prefill
    tpot: float  # mean time per output token during decode
    wall_time: float


@torch.no_grad()
def generate(model, input_ids, caches, params, generator=None):
    for cache in caches:
        cache.reset()

    start = time.perf_counter()
    logits = model(input_ids, caches)  # prefill: whole prompt, one parallel pass
    token = sample(logits, params, generator)
    ttft = time.perf_counter() - start

    generated = [token]
    decode_start = time.perf_counter()
    while len(generated) < params.max_new_tokens:
        if params.eos_id is not None and int(token) == params.eos_id:
            break
        logits = model(token, caches)  # decode: one token in, cache supplies the rest
        token = sample(logits, params, generator)
        generated.append(token)
    decode_time = time.perf_counter() - decode_start

    return GenerationResult(
        tokens=torch.cat([input_ids, *generated], dim=1),
        num_generated=len(generated),
        ttft=ttft,
        tpot=decode_time / max(len(generated) - 1, 1),
        wall_time=ttft + decode_time,
    )


@torch.no_grad()
def generate_without_kv_cache(model, input_ids, params, generator=None):
    """Baseline: the whole sequence is re-encoded from scratch at every step."""
    start = time.perf_counter()
    for _ in range(params.max_new_tokens):
        logits = model(input_ids)
        input_ids = torch.cat([input_ids, sample(logits, params, generator)], dim=1)
    return input_ids, time.perf_counter() - start


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vocab_size, prompt_len = 32000, 32
    params = SamplingParams(max_new_tokens=192, temperature=0.0)
    max_seq_len = prompt_len + params.max_new_tokens

    model = TinyLM(vocab_size).to(device=device, dtype=dtype).eval()
    prompt = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    caches = model.allocate_cache(max_batch=1, max_seq_len=max_seq_len)

    uncached_ids, uncached_time = generate_without_kv_cache(model, prompt, params)
    result = generate(model, prompt, caches, params)

    attn = model.blocks[0].attn
    elem = caches[0].K.element_size()
    per_token = 2 * len(caches) * attn.num_kv_heads * attn.d_k * elem
    mha_per_token = 2 * len(caches) * attn.num_heads * attn.d_k * elem

    print(f"prompt {prompt_len} tokens -> {result.num_generated} generated, "
          f"vocab {vocab_size}, {device}/{str(dtype).rsplit('.', 1)[-1]}")
    print()
    print(f"{'':<16}{'wall':>9}{'TTFT':>10}{'TPOT':>10}{'tokens/s':>11}")
    print(f"{'no KV cache':<16}{uncached_time:>8.2f}s{'-':>10}{'-':>10}"
          f"{params.max_new_tokens / uncached_time:>11.1f}")
    print(f"{'KV cache':<16}{result.wall_time:>8.2f}s{result.ttft * 1e3:>9.1f}ms"
          f"{result.tpot * 1e3:>9.2f}ms{result.num_generated / result.wall_time:>11.1f}")
    print()
    identical = torch.equal(uncached_ids, result.tokens)
    print(f"speedup          : {uncached_time / result.wall_time:.1f}x")
    print(f"identical output : {identical}")
    print(f"KV per token     : {per_token} B "
          f"({attn.num_heads} q-heads / {attn.num_kv_heads} kv-heads, "
          f"{mha_per_token / per_token:.0f}x smaller than MHA)")
    print(f"cache allocated  : {sum(c.nbytes for c in caches) / 1024:.1f} KiB "
          f"for {max_seq_len} tokens, written in place")

    if not identical:
        raise SystemExit("caching changed the output; that is a bug, not a tradeoff")


if __name__ == "__main__":
    main()