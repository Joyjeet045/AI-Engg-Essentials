"""Continuous (in-flight) batching, shaped the way inference servers do it.

Reference:
https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

Static batching runs a fixed group to completion, so every sequence waits for
the longest one in its group. Continuous batching admits and evicts at every
step. Production schedulers add two things on top of that:

  * a per-step token budget (vLLM's max_num_batched_tokens) so a step's cost is
    bounded whatever mix of prefill and decode it contains
  * chunked prefill, so a long prompt is spread across several steps instead of
    stalling every running decode for one huge forward pass

The cache here still hands each sequence a fixed slot sized for the worst case.
../03-paged-attention replaces that with on-demand block allocation.
"""

import time
from collections import deque
from dataclasses import dataclass, field

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


# -----------------------------------------------------------------------------


class SlotKVAttention(nn.Module):
    """Attention over a pre-allocated KV cache addressed by batch slot.

    A slot is a reusable cache row, so a sequence can be swapped in or out
    mid-flight without rebuilding the batch or the cache.
    """


    def __init__(self, d_model, num_heads, num_kv_heads, max_batch, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(num_heads * self.d_k, d_model, bias=False)

        shape = (max_batch, num_kv_heads, max_seq_len, self.d_k)
        self.register_buffer("cache_K", torch.zeros(shape), persistent=False)
        self.register_buffer("cache_V", torch.zeros(shape), persistent=False)

    def forward(self, x, slot_ids, positions, k_len):
        """x: [n, seq_len, d_model], slot_ids: [n], positions: [n, seq_len]."""
        n, seq_len, _ = x.shape

        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        rows = slot_ids.unsqueeze(1).expand(n, seq_len)
        self.cache_K[rows, :, positions] = K_new.transpose(1, 2)
        self.cache_V[rows, :, positions] = V_new.transpose(1, 2)

        K = repeat_kv(self.cache_K[slot_ids, :, :k_len], self.n_rep)
        V = repeat_kv(self.cache_V[slot_ids, :, :k_len], self.n_rep)

        # One per-row bound does two jobs: it enforces causality, and it hides
        # whatever a previous occupant of the slot left past this row's position.
        allowed = torch.arange(k_len, device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))


class SlotTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, max_batch, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = SlotKVAttention(d_model, num_heads, num_kv_heads, max_batch, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, slot_ids, positions, k_len):
        x = x + self.attn(self.ln_1(x), slot_ids, positions, k_len)
        x = x + self.ffn(self.ln_2(x))
        return x


class SlotCacheLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_batch=8, max_seq_len=1024):
        super().__init__()
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            SlotTransformerBlock(d_model, num_heads, num_kv_heads, 4 * d_model,
                                 max_batch, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def cache_nbytes(self):
        return sum(
            (b.attn.cache_K.numel() + b.attn.cache_V.numel()) * b.attn.cache_K.element_size()
            for b in self.blocks
        )

    def forward(self, input_ids, slot_ids, positions, k_len):
        """Returns [n, vocab] -- logits for each row's final position only."""
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, slot_ids, positions, k_len)
        return self.lm_head(self.ln_f(x[:, -1]))


@dataclass
class Request:
    id: int
    prompt: torch.Tensor
    max_new_tokens: int  # stands in for "decodes until it emits EOS"


@dataclass
class EngineConfig:
    label: str
    max_batch: int = 8
    max_batched_tokens: int = 128  # vLLM's max_num_batched_tokens
    chunked_prefill: bool = False  # split a prompt so the step fits the budget


@dataclass
class Sequence:
    """A request that has been admitted and given a slot."""

    request: Request
    slot: int
    num_cached: int = 0  # prompt tokens whose K/V are already in the slot
    last_token: int | None = None
    generated: list[int] = field(default_factory=list)
    t_first_token: float = 0.0
    t_last_token: float = 0.0
    max_gap: float = 0.0  # worst inter-token stall this sequence saw

    @property
    def prompt_len(self):
        return self.request.prompt.shape[0]

    @property
    def needs_prefill(self):
        return self.num_cached < self.prompt_len

    @property
    def done(self):
        return len(self.generated) >= self.request.max_new_tokens

    @property
    def next_position(self):
        """Absolute position of the token that will be fed on the next decode."""
        return self.prompt_len + len(self.generated) - 1

    def record_token(self, token, now):
        self.generated.append(token)
        self.last_token = token
        if self.t_first_token == 0.0:
            self.t_first_token = now
        else:
            self.max_gap = max(self.max_gap, now - self.t_last_token)
        self.t_last_token = now


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


@dataclass
class RunStats:
    label: str
    wall_time: float = 0.0
    steps: int = 0
    row_steps: int = 0  # decode rows the model actually computed
    useful_row_steps: int = 0  # of those, the ones a caller was still waiting on
    prefill_tokens: int = 0
    outputs: dict[int, list[int]] = field(default_factory=dict)
    ttft: list[float] = field(default_factory=list)
    tpot: list[float] = field(default_factory=list)
    stalls: list[float] = field(default_factory=list)

    def finish(self, seq, t0):
        self.outputs[seq.request.id] = seq.generated
        self.ttft.append(seq.t_first_token - t0)
        self.stalls.append(seq.max_gap)
        if len(seq.generated) > 1:
            span = seq.t_last_token - seq.t_first_token
            self.tpot.append(span / (len(seq.generated) - 1))

    @property
    def generated(self):
        return sum(len(o) for o in self.outputs.values())

    @property
    def utilization(self):
        return self.useful_row_steps / max(self.row_steps, 1)

    @property
    def throughput(self):
        return self.generated / self.wall_time


@torch.no_grad()
def run_prefill_chunk(model, seq, num_tokens):
    """Push the next num_tokens prompt tokens into the sequence's slot."""
    start, end = seq.num_cached, seq.num_cached + num_tokens
    ids = seq.request.prompt[start:end].unsqueeze(0)
    positions = torch.arange(start, end, device=model.device).unsqueeze(0)
    logits = model(ids, torch.tensor([seq.slot], device=model.device), positions, end)
    seq.num_cached = end
    return logits


@torch.no_grad()
def run_decode(model, seqs):
    """One token for every sequence, in a single pass, whatever their lengths."""
    device = model.device
    slot_ids = torch.tensor([s.slot for s in seqs], device=device)
    tokens = torch.tensor([s.last_token for s in seqs], device=device).unsqueeze(1)
    positions = torch.tensor([s.next_position for s in seqs], device=device).unsqueeze(1)
    return model(tokens, slot_ids, positions, int(positions.max()) + 1)


@torch.no_grad()
def run_static_batching(model, requests, cfg):
    """Fixed groups, each run to completion before the next group is admitted."""
    stats = RunStats("static")
    t0 = time.perf_counter()

    for start in range(0, len(requests), cfg.max_batch):
        group = [
            Sequence(request, slot)
            for slot, request in enumerate(requests[start:start + cfg.max_batch])
        ]

        for seq in group:
            logits = run_prefill_chunk(model, seq, seq.prompt_len)
            stats.prefill_tokens += seq.prompt_len
            stats.steps += 1
            seq.record_token(int(logits[0].argmax()), time.perf_counter())

        while any(not s.done for s in group):
            logits = run_decode(model, group)
            stats.steps += 1
            stats.row_steps += len(group)
            now = time.perf_counter()

            for row, seq in enumerate(group):
                # A finished row keeps riding along, burning compute, until the
                # whole group drains. That is the cost static batching pays.
                if seq.done:
                    continue
                stats.useful_row_steps += 1
                seq.record_token(int(logits[row].argmax()), now)

        for seq in group:
            stats.finish(seq, t0)

    stats.wall_time = time.perf_counter() - t0
    return stats


@torch.no_grad()
def run_continuous_batching(model, requests, cfg):
    """Admit, decode, evict, every step, under a fixed per-step token budget."""
    stats = RunStats(cfg.label)
    waiting = deque(requests)
    free_slots = deque(range(cfg.max_batch))
    running: list[Sequence] = []

    t0 = time.perf_counter()
    while waiting or running:
        while waiting and free_slots:
            running.append(Sequence(waiting.popleft(), free_slots.popleft()))

        budget = cfg.max_batched_tokens

        # Decode is scheduled first so running sequences keep a steady token rate.
        decoding = [s for s in running if not s.needs_prefill]
        if decoding:
            logits = run_decode(model, decoding)
            budget -= len(decoding)
            stats.steps += 1
            stats.row_steps += len(decoding)
            stats.useful_row_steps += len(decoding)
            now = time.perf_counter()
            for row, seq in enumerate(decoding):
                seq.record_token(int(logits[row].argmax()), now)

        # Whatever budget is left goes to prefill. Without chunking a prompt cannot
        # be split, so it overruns the budget and every running decode stalls for
        # the length of one full prefill pass.
        for seq in [s for s in running if s.needs_prefill]:
            if budget <= 0:
                break
            take = seq.prompt_len - seq.num_cached
            if cfg.chunked_prefill:
                take = min(take, budget)

            logits = run_prefill_chunk(model, seq, take)
            budget -= take
            stats.prefill_tokens += take
            stats.steps += 1
            if not seq.needs_prefill:
                seq.record_token(int(logits[0].argmax()), time.perf_counter())

        for seq in list(running):
            if seq.done:
                stats.finish(seq, t0)
                running.remove(seq)
                free_slots.append(seq.slot)  # freed now, not when the batch drains

    stats.wall_time = time.perf_counter() - t0
    return stats


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vocab_size, max_batch, num_requests = 32000, 8, 24
    max_seq_len = 1024

    model = SlotCacheLM(vocab_size, max_batch=max_batch, max_seq_len=max_seq_len)
    model = model.to(device=device, dtype=dtype).eval()

    rng = torch.Generator().manual_seed(7)
    requests = []
    for i in range(num_requests):
        # Long prompts plus a wide spread of completion lengths: the RAG-shaped
        # workload that static batching and unchunked prefill both handle badly.
        prompt_len = int(torch.randint(128, 512, (1,), generator=rng))
        max_new_tokens = int(torch.randint(8, 160, (1,), generator=rng))
        prompt = torch.randint(0, vocab_size, (prompt_len,), generator=rng).to(device)
        requests.append(Request(i, prompt, max_new_tokens))

    runs = [
        run_static_batching(model, requests, EngineConfig("static", max_batch)),
        run_continuous_batching(model, requests, EngineConfig("continuous", max_batch)),
        run_continuous_batching(
            model, requests,
            EngineConfig("continuous + chunked", max_batch, chunked_prefill=True),
        ),
    ]

    prompt_tokens = sum(r.prompt.shape[0] for r in requests)
    output_tokens = sum(r.max_new_tokens for r in requests)
    print(f"{num_requests} requests, batch {max_batch}, budget 128 tok/step, "
          f"{device}/{str(dtype).rsplit('.', 1)[-1]}")
    print(f"{prompt_tokens} prompt tokens, {output_tokens} output tokens, "
          f"completions {min(r.max_new_tokens for r in requests)}"
          f"-{max(r.max_new_tokens for r in requests)}")
    print()

    print(f"{'':<22}{'wall':>8}{'batch util':>12}{'tok/s':>9}"
          f"{'TTFT p50':>10}{'TTFT p99':>10}{'TPOT':>9}{'max stall':>11}")
    for stats in runs:
        print(f"{stats.label:<22}{stats.wall_time:>7.2f}s{stats.utilization:>12.1%}"
              f"{stats.throughput:>9.1f}{percentile(stats.ttft, 50):>9.2f}s"
              f"{percentile(stats.ttft, 99):>9.2f}s"
              f"{sum(stats.tpot) / len(stats.tpot) * 1e3:>8.1f}ms"
              f"{max(stats.stalls) * 1e3:>10.0f}ms")
    print()

    reference = runs[0].outputs
    identical = all(r.outputs == reference for r in runs[1:])
    print("speedup vs static: "
          + ", ".join(f"{r.label} {runs[0].wall_time / r.wall_time:.2f}x" for r in runs[1:]))
    print("wasted row-steps : "
          + ", ".join(f"{r.label} {r.row_steps - r.useful_row_steps}" for r in runs))
    print(f"identical output : {identical}")
    print(f"slot cache       : {model.cache_nbytes() / 2**20:.1f} MiB reserved for "
          f"{max_batch} x {max_seq_len} tokens, "
          f"{prompt_tokens + output_tokens} tokens ever live")

    if not identical:
        raise SystemExit("scheduling changed the output; the slot masking is wrong")


if __name__ == "__main__":
    main()
