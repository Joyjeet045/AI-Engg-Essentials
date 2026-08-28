"""PagedAttention: on-demand block allocation for the KV cache.

Reference: Kwon et al., "Efficient Memory Management for Large Language Model
Serving with PagedAttention" (vLLM), SOSP 2023.

../02-continuous-batching hands each sequence one contiguous slot sized for the
worst case, so a 300-token request still reserves max_seq_len. This file replaces
that with fixed-size blocks allocated as a sequence grows, addressed through a
per-sequence block table -- virtual memory applied to the KV cache. That buys
four things, all exercised by the demo at the bottom:

  * near-zero internal fragmentation: at most one partial block per sequence
  * concurrency bounded by live tokens instead of worst-case reservations
  * prefix sharing: requests with a common system prompt point at the same
    physical blocks, with copy-on-write when one of them writes into a shared
    partial block
  * preemption: when blocks run out, a sequence is evicted and recomputed later
    rather than the server going out of memory
"""

import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 16


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


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


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


@dataclass
class Request:
    id: int
    prompt: torch.Tensor
    max_new_tokens: int  # stands in for "decodes until it emits EOS"


# -----------------------------------------------------------------------------


class OutOfBlocks(Exception):
    """The cache is full; the scheduler has to free something before retrying."""


def blocks_needed(num_tokens):
    return (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE


class BlockAllocator:
    """Physical block pool with reference counts, so blocks can be shared."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self._refs = [0] * num_blocks

    @property
    def num_used(self):
        return self.num_blocks - len(self._free)

    def allocate(self):
        if not self._free:
            raise OutOfBlocks
        block = self._free.popleft()
        self._refs[block] = 1
        return block

    def share(self, block):
        self._refs[block] += 1
        return block

    def release(self, block):
        self._refs[block] -= 1
        if self._refs[block] == 0:
            self._free.append(block)

    def ref_count(self, block):
        return self._refs[block]


class PagedKVCache:
    """One layer's cache, stored as fixed-size blocks rather than one contiguous
    buffer per sequence."""

    def __init__(self, num_blocks, num_kv_heads, d_k, dtype, device):
        shape = (num_blocks, BLOCK_SIZE, num_kv_heads, d_k)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros(shape, dtype=dtype, device=device)

    def write(self, K_new, V_new, slot_mapping):
        """K_new/V_new are [num_tokens, num_kv_heads, d_k]; slot_mapping indexes
        the cache flattened to one row per (block, offset) pair."""
        self.K.view(-1, *self.K.shape[2:])[slot_mapping] = K_new
        self.V.view(-1, *self.V.shape[2:])[slot_mapping] = V_new

    def gather(self, block_tables):
        """block_tables: [n, num_blocks] -> K, V of [n, num_kv_heads, ctx, d_k].

        Scattered physical blocks are pulled back into logical order here. A real
        kernel fuses this into the attention itself instead of materialising it.
        """
        out = []
        for tensor in (self.K, self.V):
            gathered = tensor[block_tables]
            n, nb, bs, h, d = gathered.shape
            out.append(gathered.reshape(n, nb * bs, h, d).permute(0, 2, 1, 3))
        return out

    def copy_block(self, src, dst):
        self.K[dst] = self.K[src]
        self.V[dst] = self.V[src]

    @property
    def nbytes(self):
        return (self.K.numel() + self.V.numel()) * self.K.element_size()


class PagedAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
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

    def forward(self, x, cache, block_tables, slot_mapping, positions):
        n, seq_len, _ = x.shape

        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        flat = (n * seq_len, self.num_kv_heads, self.d_k)
        cache.write(
            K_new.permute(0, 2, 1, 3).reshape(flat),
            V_new.permute(0, 2, 1, 3).reshape(flat),
            slot_mapping,
        )

        K, V = cache.gather(block_tables)
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        # The gathered context is in logical order, so the same per-row position
        # bound as the slot cache works: causal, and blind to padded blocks.
        allowed = torch.arange(K.shape[2], device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))


class PagedTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = PagedAttention(d_model, num_heads, num_kv_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, cache, block_tables, slot_mapping, positions):
        x = x + self.attn(self.ln_1(x), cache, block_tables, slot_mapping, positions)
        x = x + self.ffn(self.ln_2(x))
        return x


class PagedLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            PagedTransformerBlock(d_model, num_heads, num_kv_heads, 4 * d_model)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def allocate_cache(self, num_blocks):
        weight = self.lm_head.weight
        return [
            PagedKVCache(num_blocks, block.attn.num_kv_heads, block.attn.d_k,
                         weight.dtype, weight.device)
            for block in self.blocks
        ]

    def forward(self, input_ids, caches, block_tables, slot_mapping, positions):
        """Returns [n, vocab] -- logits for each row's final position only."""
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block, cache in zip(self.blocks, caches, strict=True):
            x = block(x, cache, block_tables, slot_mapping, positions)
        return self.lm_head(self.ln_f(x[:, -1]))


@dataclass
class PagedSequence:
    request: Request
    tokens: list[int]  # prompt followed by everything generated so far
    num_cached: int = 0  # how many of those tokens have K/V in the cache
    block_table: list[int] = field(default_factory=list)
    num_shared: int = 0  # leading blocks borrowed from the prefix cache
    shared_tokens: int = 0
    preemptions: int = 0
    t_first_token: float = 0.0
    t_last_token: float = 0.0

    @property
    def prompt_len(self):
        return self.request.prompt.shape[0]

    @property
    def output(self):
        return self.tokens[self.prompt_len:]

    @property
    def num_uncached(self):
        return len(self.tokens) - self.num_cached

    @property
    def needs_prefill(self):
        # One uncached token is a decode step; more than one is a prefill chunk.
        return self.num_uncached > 1

    @property
    def done(self):
        return len(self.output) >= self.request.max_new_tokens


@dataclass
class PagedConfig:
    label: str
    num_blocks: int = 400
    max_batch: int = 8
    max_batched_tokens: int = 128
    chunked_prefill: bool = True
    share_prefix: bool = True
    watermark: float = 0.08  # keep this fraction free so preemption can converge


@dataclass
class PagedStats:
    label: str
    wall_time: float = 0.0
    steps: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    prefill_tokens: int = 0
    peak_blocks: int = 0
    addressed_at_peak: int = 0  # cached tokens reachable through the block tables
    max_concurrent: int = 0
    outputs: dict[int, list[int]] = field(default_factory=dict)
    ttft: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)

    @property
    def generated(self):
        return sum(len(o) for o in self.outputs.values())

    @property
    def throughput(self):
        return self.generated / self.wall_time

    @property
    def density(self):
        """Tokens addressed per physical slot.

        Just under 1.0 means allocation is tight and the only waste is the tail
        of each sequence's last block. Above 1.0 means shared blocks are being
        read by more than one sequence.
        """
        slots = self.peak_blocks * BLOCK_SIZE
        return self.addressed_at_peak / slots if slots else 0.0


class PagedEngine:
    """Continuous batching on top of a paged KV cache."""

    def __init__(self, model, cfg, shared_prefix=()):
        self.model = model
        self.cfg = cfg
        self.caches = model.allocate_cache(cfg.num_blocks)
        self.allocator = BlockAllocator(cfg.num_blocks)
        self.waiting: deque[PagedSequence] = deque()
        self.running: list[PagedSequence] = []
        self.prefix = list(shared_prefix) if cfg.share_prefix else []
        self.prefix_blocks = None
        self.admitting = True
        self.stats = PagedStats(cfg.label)
        self.t0 = 0.0

    # -- block management -------------------------------------------------

    def _allocate(self, protect):
        """Allocate a block, evicting somebody else's if the pool is empty."""
        while True:
            try:
                return self.allocator.allocate()
            except OutOfBlocks:
                self._preempt(self._pick_victim(protect))

    def _ensure(self, seq, num_tokens, protect):
        while len(seq.block_table) < blocks_needed(num_tokens):
            seq.block_table.append(self._allocate(protect))

    def _pick_victim(self, protect):
        # Newest first, matching vLLM: the sequence closest to its start loses
        # the least recomputed work. Shared prefix blocks are not worth taking.
        for seq in reversed(self.running):
            if seq is not protect and len(seq.block_table) > seq.num_shared:
                return seq
        raise RuntimeError("KV cache too small to run even one sequence")

    def _preempt(self, seq):
        """Recompute preemption: drop the blocks, re-prefill the tokens later."""
        for block in seq.block_table[seq.num_shared:]:
            self.allocator.release(block)
        del seq.block_table[seq.num_shared:]

        self.stats.recomputed_tokens += seq.num_cached - seq.shared_tokens
        self.stats.preemptions += 1
        seq.preemptions += 1
        seq.num_cached = seq.shared_tokens
        self.running.remove(seq)
        self.waiting.appendleft(seq)
        # Taking on new work under memory pressure just causes more preemption.
        self.admitting = False

    def _copy_on_write(self, seq, block_index, protect):
        """A block with more than one owner cannot be written to; fork it."""
        old = seq.block_table[block_index]
        if self.allocator.ref_count(old) == 1:
            return
        new = self._allocate(protect)
        for cache in self.caches:
            cache.copy_block(old, new)
        seq.block_table[block_index] = new
        self.allocator.release(old)

    def _free(self, seq):
        for block in seq.block_table:
            self.allocator.release(block)
        seq.block_table.clear()

    # -- scheduling -------------------------------------------------------

    def add_request(self, request):
        self.waiting.append(PagedSequence(request, request.prompt.tolist()))

    def _attach_prefix(self, seq):
        """Point a fresh sequence at the cached system prompt instead of
        recomputing it. Checked against the tokens, not assumed."""
        if self.prefix_blocks is None or seq.block_table:
            return
        if seq.tokens[:len(self.prefix)] != self.prefix:
            return
        seq.block_table = [self.allocator.share(b) for b in self.prefix_blocks]
        seq.num_shared = len(seq.block_table)
        seq.shared_tokens = len(self.prefix)
        seq.num_cached = len(self.prefix)

    def _register_prefix(self, seq):
        if self.prefix and self.prefix_blocks is None and seq.num_cached >= len(self.prefix):
            blocks = seq.block_table[:blocks_needed(len(self.prefix))]
            self.prefix_blocks = [self.allocator.share(b) for b in blocks]

    def _slot_mapping(self, seq, start, count):
        return torch.tensor(
            [seq.block_table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
             for p in range(start, start + count)],
            device=self.model.device,
        )

    def _block_tables(self, seqs):
        width = max(len(s.block_table) for s in seqs)
        # Padded columns repeat a live block; they sit past every row's position
        # and are masked out, so their contents never matter.
        return torch.tensor(
            [s.block_table + [s.block_table[-1]] * (width - len(s.block_table))
             for s in seqs],
            device=self.model.device,
        )

    def _step_decode(self):
        for seq in [s for s in self.running if not s.needs_prefill]:
            if seq not in self.running:  # an earlier allocation evicted it
                continue
            self._ensure(seq, seq.num_cached + 1, protect=seq)
            self._copy_on_write(seq, seq.num_cached // BLOCK_SIZE, protect=seq)

        decoding = [s for s in self.running if not s.needs_prefill]
        if not decoding:
            return

        device = self.model.device
        ids = torch.tensor([[s.tokens[s.num_cached]] for s in decoding], device=device)
        positions = torch.tensor([[s.num_cached] for s in decoding], device=device)
        slot_mapping = torch.cat([self._slot_mapping(s, s.num_cached, 1) for s in decoding])

        logits = self.model(ids, self.caches, self._block_tables(decoding),
                            slot_mapping, positions)
        self.stats.steps += 1
        now = time.perf_counter()

        for row, seq in enumerate(decoding):
            seq.num_cached += 1
            seq.tokens.append(int(logits[row].argmax()))
            seq.t_last_token = now

    def _step_prefill(self):
        budget = self.cfg.max_batched_tokens

        for seq in [s for s in self.running if s.needs_prefill]:
            if budget <= 0:
                break
            if seq not in self.running:
                continue

            self._attach_prefix(seq)
            take = seq.num_uncached
            if self.cfg.chunked_prefill:
                take = min(take, budget)

            start = seq.num_cached
            self._ensure(seq, start + take, protect=seq)
            for index in range(start // BLOCK_SIZE, (start + take - 1) // BLOCK_SIZE + 1):
                self._copy_on_write(seq, index, protect=seq)

            device = self.model.device
            ids = torch.tensor([seq.tokens[start:start + take]], device=device)
            positions = torch.arange(start, start + take, device=device).unsqueeze(0)
            logits = self.model(ids, self.caches, self._block_tables([seq]),
                                self._slot_mapping(seq, start, take), positions)

            seq.num_cached += take
            budget -= take
            self.stats.prefill_tokens += take
            self.stats.steps += 1
            self._register_prefix(seq)

            if seq.num_uncached == 0:
                seq.tokens.append(int(logits[0].argmax()))
                now = time.perf_counter()
                seq.t_last_token = now
                if seq.t_first_token == 0.0:
                    seq.t_first_token = now

    def _sample_usage(self):
        used = self.allocator.num_used
        addressed = sum(s.num_cached for s in self.running)
        if used > self.stats.peak_blocks:
            self.stats.peak_blocks = used
            self.stats.addressed_at_peak = addressed
        self.stats.max_concurrent = max(self.stats.max_concurrent, len(self.running))

    def _retire(self):
        for seq in list(self.running):
            if seq.done:
                self.stats.outputs[seq.request.id] = seq.output
                self.stats.ttft.append(seq.t_first_token - self.t0)
                self.stats.latency.append(seq.t_last_token - self.t0)
                self.running.remove(seq)
                self._free(seq)
                self.admitting = True  # memory freed, safe to take new work

    def run(self):
        self.t0 = time.perf_counter()
        headroom = max(1, int(self.cfg.num_blocks * self.cfg.watermark))

        while self.waiting or self.running:
            # Admitting right up to the last block would just get the new
            # sequence preempted again, so leave headroom.
            while self.admitting and self.waiting and len(self.running) < self.cfg.max_batch:
                if self.allocator.num_used > self.cfg.num_blocks - headroom:
                    break
                self.running.append(self.waiting.popleft())

            if not self.running:  # everything is preempted; take one back anyway
                self.running.append(self.waiting.popleft())

            self._step_decode()
            self._step_prefill()
            self._sample_usage()
            self._retire()

        self.stats.wall_time = time.perf_counter() - self.t0
        return self.stats


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    vocab_size, num_requests, max_batch = 32000, 12, 8
    max_seq_len = 1024

    model = PagedLM(vocab_size, max_seq_len=max_seq_len)
    model = model.to(device=device, dtype=dtype).eval()

    rng = torch.Generator().manual_seed(11)
    # A shared system prompt, deliberately not a multiple of BLOCK_SIZE so the
    # last shared block is partial and copy-on-write actually fires.
    system_prompt = torch.randint(0, vocab_size, (200,), generator=rng)

    requests = []
    for i in range(num_requests):
        suffix_len = int(torch.randint(32, 96, (1,), generator=rng))
        suffix = torch.randint(0, vocab_size, (suffix_len,), generator=rng)
        prompt = torch.cat([system_prompt, suffix]).to(device)
        max_new_tokens = int(torch.randint(16, 96, (1,), generator=rng))
        requests.append(Request(i, prompt, max_new_tokens))

    runs = []
    for cfg in (
        PagedConfig("roomy, no sharing", num_blocks=400, max_batch=max_batch,
                    share_prefix=False),
        PagedConfig("tight + prefix reuse", num_blocks=64, max_batch=max_batch,
                    share_prefix=True),
    ):
        engine = PagedEngine(model, cfg, shared_prefix=system_prompt.tolist())
        for request in requests:
            engine.add_request(request)
        runs.append((cfg, engine.run()))

    prompt_tokens = sum(r.prompt.shape[0] for r in requests)
    output_tokens = sum(r.max_new_tokens for r in requests)
    print(f"{num_requests} requests sharing a {len(system_prompt)}-token system "
          f"prompt, block size {BLOCK_SIZE}, {device}/{str(dtype).rsplit('.', 1)[-1]}")
    print(f"{prompt_tokens} prompt tokens, {output_tokens} output tokens, "
          f"batch {max_batch}")
    print()

    print(f"{'':<22}{'blocks':>8}{'wall':>8}{'tok/s':>8}{'peak':>7}{'density':>10}"
          f"{'preempt':>9}{'recomp':>8}{'TTFT p50':>11}")
    for cfg, stats in runs:
        print(f"{stats.label:<22}{cfg.num_blocks:>8}{stats.wall_time:>7.2f}s"
              f"{stats.throughput:>8.1f}{stats.peak_blocks:>7}"
              f"{stats.density:>9.2f}x{stats.preemptions:>9}"
              f"{stats.recomputed_tokens:>8}{percentile(stats.ttft, 50):>10.2f}s")
    print()

    baseline, tight = runs[0][1], runs[1][1]
    print(f"identical output   : {baseline.outputs == tight.outputs}")

    elem = model.blocks[0].attn.W_K.weight.element_size()
    kv_per_token = 2 * len(model.blocks) * model.blocks[0].attn.num_kv_heads \
        * model.blocks[0].attn.d_k * elem
    paged_bytes = tight.peak_blocks * BLOCK_SIZE * kv_per_token
    slot_bytes = max_batch * max_seq_len * kv_per_token

    print(f"KV per token       : {kv_per_token} B")
    print(f"paged peak         : {paged_bytes / 2**20:.1f} MiB in {tight.peak_blocks} "
          f"blocks ({tight.peak_blocks * BLOCK_SIZE} slots) addressing "
          f"{tight.addressed_at_peak} tokens")
    print(f"slot equivalent    : {slot_bytes / 2**20:.1f} MiB "
          f"({max_batch} x {max_seq_len} reserved) -> "
          f"{slot_bytes / paged_bytes:.1f}x more memory for the same work")
    print(f"same budget holds  : {tight.max_concurrent} sequences paged vs "
          f"{runs[1][0].num_blocks * BLOCK_SIZE // max_seq_len} with fixed slots")

    if baseline.outputs != tight.outputs:
        raise SystemExit("paging changed the output; eviction or copy-on-write is wrong")


if __name__ == "__main__":
    main()
