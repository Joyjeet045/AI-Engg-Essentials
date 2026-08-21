"""Automatic prefix caching: find shared prefixes by content, not by being told.

Project 03 shared a prefix that the engine was handed up front. Real traffic does
not work like that. Prefixes appear on their own: many users hit the same system
prompt, a RAG pipeline reuses the same retrieved passage, and above all a chat
turn is a strict extension of the turn before it.

vLLM's answer is to make blocks content-addressable. Each full block gets a hash
chained through its parent:

    hash(block_i) = H(hash(block_i-1), tokens in block_i)

so a single hash identifies the entire prefix up to that block, and two sequences
collide exactly when they truly share that prefix. A finished sequence does not
throw its blocks away: they drop to refcount zero and sit in an LRU pool, still
addressable, until the space is genuinely needed.

The consequence for a chat server is the whole point. Turn 5 of a conversation
re-sends the previous four turns, and every full block of them is a hit, so
prefill cost is proportional to the *new* tokens rather than the whole history.

One rule matters: a hit is only usable if every block before it also hit.
Attention needs an unbroken prefix, so a hit at block 7 behind a miss at block 3
buys nothing and the block is simply recomputed.
"""

import time
from collections import OrderedDict, deque
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


# ------------------------------------------------------------------ block manager


class OutOfBlocks(Exception):
    pass


def block_hashes(tokens, block_size, parent=None):
    """One chained hash per *full* block.

    A partial trailing block is deliberately not hashed: its contents will still
    change, so it must never be shared.
    """
    hashes = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        hashes.append(parent)
    return hashes


class PrefixAwareAllocator:
    """Block pool where a freed block keeps its contents and stays addressable.

    Three states a block can be in:
      * free      -- never used, or evicted; contents meaningless
      * in use    -- refcount > 0, owned by one or more sequences
      * cached    -- refcount 0 but still holds valid K/V, sitting in the LRU
    """

    def __init__(self, num_blocks, enable_prefix_caching=True):
        self.num_blocks = num_blocks
        self.enabled = enable_prefix_caching
        self.free = deque(range(num_blocks))
        self.refs = [0] * num_blocks
        self.hash_of = [None] * num_blocks
        self.by_hash = {}
        self.cached = OrderedDict()  # block -> None, coldest first
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def num_in_use(self):
        return sum(1 for r in self.refs if r > 0)

    def _take(self):
        if self.free:
            return self.free.popleft()
        if self.cached:
            block, _ = self.cached.popitem(last=False)  # coldest cached block
            self.by_hash.pop(self.hash_of[block], None)
            self.hash_of[block] = None
            self.evictions += 1
            return block
        raise OutOfBlocks

    def acquire(self, block_hash=None):
        """Take a block, reusing the one holding this content if it exists."""
        if self.enabled and block_hash is not None and block_hash in self.by_hash:
            block = self.by_hash[block_hash]
            self.cached.pop(block, None)
            self.refs[block] += 1
            self.hits += 1
            return block, True

        block = self._take()
        self.refs[block] = 1
        self.misses += 1
        return block, False

    def register(self, block, block_hash):
        """Publish a block once its contents are actually computed."""
        if not self.enabled or self.hash_of[block] is not None:
            return
        existing = self.by_hash.get(block_hash)
        if existing is not None and existing != block:
            return  # someone else already published this content
        self.hash_of[block] = block_hash
        self.by_hash[block_hash] = block

    def release(self, block):
        self.refs[block] -= 1
        if self.refs[block] > 0:
            return
        if self.hash_of[block] is not None:
            self.cached[block] = None  # retained, evictable, still a valid hit
        else:
            self.free.append(block)


# -------------------------------------------------------------------------- model


class PagedAttention(nn.Module):
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

    def forward(self, x, cache_K, cache_V, block_table, slot_mapping, positions):
        n, seq_len, _ = x.shape
        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        flat = (n * seq_len, self.num_kv_heads, self.d_k)
        cache_K.view(-1, self.num_kv_heads, self.d_k)[slot_mapping] = \
            K_new.permute(0, 2, 1, 3).reshape(flat)
        cache_V.view(-1, self.num_kv_heads, self.d_k)[slot_mapping] = \
            V_new.permute(0, 2, 1, 3).reshape(flat)

        K = repeat_kv(self._gather(cache_K, block_table), self.n_rep)
        V = repeat_kv(self._gather(cache_V, block_table), self.n_rep)

        allowed = torch.arange(K.shape[2], device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))

    @staticmethod
    def _gather(cache, block_table):
        blocks = cache[block_table]
        n, nb, bs, h, d = blocks.shape
        return blocks.reshape(n, nb * bs, h, d).permute(0, 2, 1, 3)


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = PagedAttention(d_model, num_heads, num_kv_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, cache_K, cache_V, block_table, slot_mapping, positions):
        x = x + self.attn(self.ln_1(x), cache_K, cache_V, block_table,
                          slot_mapping, positions)
        x = x + self.ffn(self.ln_2(x))
        return x


class PagedLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_seq_len=2048):
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
        return self.lm_head.weight.device

    def allocate_cache(self, num_blocks, block_size):
        attn = self.blocks[0].attn
        shape = (len(self.blocks), num_blocks, block_size, attn.num_kv_heads, attn.d_k)
        return torch.zeros(shape, device=self.device), torch.zeros(shape, device=self.device)

    def forward(self, input_ids, cache_K, cache_V, block_table, slot_mapping, positions):
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for index, block in enumerate(self.blocks):
            x = block(x, cache_K[index], cache_V[index], block_table,
                      slot_mapping, positions)
        return self.lm_head(self.ln_f(x[:, -1]))


# ------------------------------------------------------------------------- engine


@dataclass
class Config:
    block_size: int = 16
    num_blocks: int = 256
    enable_prefix_caching: bool = True
    label: str = ""


@dataclass
class Sequence:
    tokens: list[int]
    prompt_len: int
    max_new_tokens: int
    num_computed: int = 0
    blocks: list[int] = field(default_factory=list)
    hashes: list = field(default_factory=list)


@dataclass
class Stats:
    label: str
    wall_time: float = 0.0
    prompt_tokens: int = 0
    prefilled_tokens: int = 0
    outputs: list[list[int]] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)

    @property
    def skipped(self):
        return self.prompt_tokens - self.prefilled_tokens

    @property
    def hit_rate(self):
        return self.skipped / max(self.prompt_tokens, 1)


class Engine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.allocator = PrefixAwareAllocator(cfg.num_blocks, cfg.enable_prefix_caching)
        self.cache_K, self.cache_V = model.allocate_cache(cfg.num_blocks, cfg.block_size)
        self.stats = Stats(cfg.label)

    # -- block plumbing ------------------------------------------------

    def _slot_mapping(self, seq, start, count):
        bs = self.cfg.block_size
        return torch.tensor(
            [seq.blocks[p // bs] * bs + p % bs for p in range(start, start + count)],
            device=self.model.device,
        )

    def _block_table(self, seq):
        return torch.tensor([seq.blocks], device=self.model.device)

    def _publish_full_blocks(self, seq):
        """A block becomes shareable only once every token in it is computed."""
        bs = self.cfg.block_size
        for index in range(seq.num_computed // bs):
            if index < len(seq.hashes):
                self.allocator.register(seq.blocks[index], seq.hashes[index])

    def _grow(self, seq):
        bs = self.cfg.block_size
        while len(seq.blocks) * bs < len(seq.tokens):
            block, _ = self.allocator.acquire()
            seq.blocks.append(block)

    # -- request lifecycle ---------------------------------------------

    def _admit(self, seq):
        """Map the prompt onto cached blocks wherever the content already exists."""
        bs = self.cfg.block_size
        seq.hashes = block_hashes(seq.tokens[:seq.prompt_len], bs)

        prefix_hits = 0
        for index, block_hash in enumerate(seq.hashes):
            block, hit = self.allocator.acquire(block_hash)
            seq.blocks.append(block)
            # Attention needs an unbroken prefix, so a hit behind a miss is
            # taken but still recomputed.
            if hit and prefix_hits == index:
                prefix_hits += 1

        self._grow(seq)
        seq.num_computed = prefix_hits * bs

    def _release(self, seq):
        for block in seq.blocks:
            self.allocator.release(block)

    @torch.no_grad()
    def run(self, prompt, max_new_tokens):
        bs = self.cfg.block_size
        seq = Sequence(list(prompt), len(prompt), max_new_tokens)
        start = time.perf_counter()

        self._admit(seq)
        self.stats.prompt_tokens += seq.prompt_len

        count = seq.prompt_len - seq.num_computed
        ids = torch.tensor([seq.tokens[seq.num_computed:seq.prompt_len]], device=self.model.device)
        positions = torch.arange(seq.num_computed, seq.prompt_len,
                                 device=self.model.device).unsqueeze(0)
        logits = self.model(ids, self.cache_K, self.cache_V, self._block_table(seq),
                            self._slot_mapping(seq, seq.num_computed, count), positions)
        seq.num_computed = seq.prompt_len
        self.stats.prefilled_tokens += count
        self._publish_full_blocks(seq)

        seq.tokens.append(int(logits[0].argmax()))
        self.stats.ttft.append(time.perf_counter() - start)

        while len(seq.tokens) - seq.prompt_len < max_new_tokens:
            self._grow(seq)
            position = seq.num_computed
            ids = torch.tensor([[seq.tokens[position]]], device=self.model.device)
            logits = self.model(ids, self.cache_K, self.cache_V, self._block_table(seq),
                                self._slot_mapping(seq, position, 1),
                                torch.tensor([[position]], device=self.model.device))
            seq.num_computed += 1

            if seq.num_computed % bs == 0:
                # A block just filled up, so its content is final and hashable.
                parent = seq.hashes[-1] if seq.hashes else None
                start_token = seq.num_computed - bs
                seq.hashes.append(hash((parent, tuple(seq.tokens[start_token:seq.num_computed]))))
                self._publish_full_blocks(seq)

            seq.tokens.append(int(logits[0].argmax()))

        self._release(seq)
        self.stats.outputs.append(seq.tokens[seq.prompt_len:])
        return seq


# --------------------------------------------------------------------------- demo


def build_chat_workload(vocab_size, num_chats=3, turns=4, seed=5):
    """Multi-turn chat: every turn re-sends the whole conversation so far."""
    rng = torch.Generator().manual_seed(seed)

    def tokens(n):
        return torch.randint(0, vocab_size, (n,), generator=rng).tolist()

    system = tokens(192)
    conversations = [[tokens(48) for _ in range(turns)] for _ in range(num_chats)]

    # Interleave the chats, which is what a real server sees and what makes the
    # LRU pool do some work.
    requests = []
    for turn in range(turns):
        for chat in range(num_chats):
            requests.append((chat, turn, conversations[chat][turn]))
    return system, requests


def run_workload(model, cfg, system, requests, max_new_tokens=32):
    engine = Engine(model, cfg)
    histories = {}
    for chat, _turn, user in requests:
        history = histories.get(chat, list(system))
        prompt = history + user
        seq = engine.run(prompt, max_new_tokens)
        histories[chat] = seq.tokens  # the reply becomes context for the next turn

    return engine


def main():
    torch.manual_seed(0)
    vocab_size = 1024
    model = PagedLM(vocab_size).eval()
    system, requests = build_chat_workload(vocab_size)

    configs = [
        Config(label="no prefix caching", enable_prefix_caching=False),
        Config(label="prefix caching", enable_prefix_caching=True),
        Config(label="prefix caching, 64 blocks", num_blocks=64),
    ]

    results = []
    for cfg in configs:
        start = time.perf_counter()
        engine = run_workload(model, cfg, system, requests)
        engine.stats.wall_time = time.perf_counter() - start
        results.append(engine)

    print(f"{len(requests)} requests: 3 chats x 4 turns, 192-token system prompt, "
          f"48-token user turns, 32-token replies")
    print(f"block size {configs[0].block_size}, greedy")
    print()

    print(f"{'':<28}{'blocks':>8}{'prompt tok':>12}{'prefilled':>11}{'reused':>9}"
          f"{'hits':>7}{'evict':>7}{'TTFT p50':>11}{'wall':>8}")
    for engine in results:
        stats = engine.stats
        allocator = engine.allocator
        ttft = sorted(stats.ttft)[len(stats.ttft) // 2]
        print(f"{stats.label:<28}{engine.cfg.num_blocks:>8}{stats.prompt_tokens:>12}"
              f"{stats.prefilled_tokens:>11}{stats.hit_rate:>9.1%}"
              f"{allocator.hits:>7}{allocator.evictions:>7}"
              f"{ttft * 1e3:>10.1f}ms{stats.wall_time:>7.2f}s")
    print()

    reference = results[0].stats.outputs
    identical = all(e.stats.outputs == reference for e in results[1:])
    print(f"identical output : {identical}")
    print("  a cached block holds K/V for exactly the tokens its hash covers, so")
    print("  reusing it is arithmetically the same as recomputing it")

    if not identical:
        raise SystemExit("a cache hit returned different K/V; the hashing is wrong")


if __name__ == "__main__":
    main()
