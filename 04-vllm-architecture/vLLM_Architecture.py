"""The vLLM system architecture, end to end.

Reference: Kwon et al., "Efficient Memory Management for Large Language Model
Serving with PagedAttention" (vLLM), SOSP 2023 -- arXiv:2309.06180.

Every box in the paper's system diagram has a class here:

    Scheduler            -> Scheduler
      waiting / running / swapped queues, FCFS with preemption, and a
      SchedulerOutput that tells the workers exactly what to do this step.

    KV Cache Manager     -> BlockSpaceManager
      Block tables       -> BlockSpaceManager.tables (logical -> physical)
      GPU Block Allocator-> BlockSpaceManager.gpu
      CPU Block Allocator-> BlockSpaceManager.cpu   (swap space)

    Worker 0..N-1        -> Worker
      Cache Engine       -> CacheEngine  (owns this rank's KV shard, executes
                            the swap-in / swap-out / copy the manager ordered)
      Model Shard        -> ShardedAttention + ShardedMLP, Megatron-style
                            tensor parallelism over the attention heads

The KV cache manager never touches tensors; it only moves block numbers around
and hands the workers three dictionaries. That separation is the whole design.

Simulation notes, since this runs in one process on one device:
  * the workers are driven layer by layer and their partial outputs are summed,
    which stands in for the NCCL all-reduce real ranks perform
  * the "CPU" pool is a second tensor allocation, so swapping is a real copy
    between two distinct buffers even without a GPU
  * a prefill batch is executed one sequence at a time; vLLM flattens them into
    a single variable-length kernel launch
"""

import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

WAITING, RUNNING, SWAPPED, FINISHED = "waiting", "running", "swapped", "finished"


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


# -----------------------------------------------------------------------------


@dataclass
class EngineConfig:
    label: str = "vllm"
    vocab_size: int = 32000
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 4
    max_seq_len: int = 1024
    block_size: int = 16
    num_gpu_blocks: int = 64
    num_cpu_blocks: int = 128
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 1024
    tensor_parallel_size: int = 2
    preemption_mode: str = "recompute"  # or "swap"
    enable_prefix_caching: bool = True
    watermark: float = 0.02

    @property
    def head_dim(self):
        return self.d_model // self.num_heads


@dataclass
class Sequence:
    seq_id: int
    prompt: list[int]
    max_new_tokens: int
    tokens: list[int] = field(default_factory=list)
    num_computed: int = 0  # tokens whose K/V are in the cache
    status: str = WAITING
    num_shared: int = 0  # leading blocks borrowed from the prefix cache
    shared_tokens: int = 0
    recomputes: int = 0
    swaps: int = 0
    t_first_token: float = 0.0
    t_finished: float = 0.0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = list(self.prompt)

    @property
    def output(self):
        return self.tokens[len(self.prompt):]

    @property
    def num_uncomputed(self):
        return len(self.tokens) - self.num_computed

    @property
    def is_finished(self):
        return len(self.output) >= self.max_new_tokens


# ---------------------------------------------------------------- KV cache manager


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    """One pool of physical blocks. Reference counts let blocks be shared."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self._free = deque(range(num_blocks))
        self._refs = [0] * num_blocks

    @property
    def num_free(self):
        return len(self._free)

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


class BlockSpaceManager:
    """The KV cache manager: block tables plus the GPU and CPU allocators.

    Everything here is bookkeeping on integers. The actual bytes are moved by
    each worker's CacheEngine, using the mappings these methods return.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.gpu = BlockAllocator(cfg.num_gpu_blocks)
        self.cpu = BlockAllocator(cfg.num_cpu_blocks)
        self.tables: dict[int, list[int]] = {}
        self.watermark_blocks = max(1, int(cfg.watermark * cfg.num_gpu_blocks))
        self.prefix_tokens: list[int] = []
        self.prefix_blocks: list[int] | None = None
        self.prefix_hits = 0

    def _num_blocks(self, num_tokens):
        return (num_tokens + self.block_size - 1) // self.block_size

    # -- allocation ---------------------------------------------------

    def can_allocate(self, seq):
        borrowed = self._prefix_match(seq)
        # A preempted-and-recomputed sequence re-prefills what it already emitted,
        # so size against every known token, not just the prompt.
        need = self._num_blocks(len(seq.tokens)) - self._num_blocks(borrowed)
        return self.gpu.num_free - need - 1 >= self.watermark_blocks  # +1 for a fork

    def allocate(self, seq, blocks_to_copy):
        table: list[int] = []
        borrowed = self._prefix_match(seq)
        if borrowed:
            table = [self.gpu.share(b) for b in self.prefix_blocks]
            seq.num_shared = len(table)
            seq.shared_tokens = borrowed
            seq.num_computed = borrowed
            self.prefix_hits += 1

        while len(table) < self._num_blocks(len(seq.tokens)):
            table.append(self.gpu.allocate())
        self.tables[seq.seq_id] = table
        self._fork_range(seq, seq.num_computed, len(seq.tokens), blocks_to_copy)

    def _prefix_match(self, seq):
        if self.prefix_blocks is None:
            return 0
        n = len(self.prefix_tokens)
        return n if seq.tokens[:n] == self.prefix_tokens else 0

    def register_prefix(self, seq, prefix_tokens):
        """Cache a fully computed prompt prefix so later requests can borrow it."""
        if not self.cfg.enable_prefix_caching or self.prefix_blocks is not None:
            return
        if seq.num_computed < len(prefix_tokens):
            return
        table = self.tables[seq.seq_id]
        self.prefix_tokens = list(prefix_tokens)
        self.prefix_blocks = [
            self.gpu.share(b) for b in table[:self._num_blocks(len(prefix_tokens))]
        ]

    # -- growth and copy-on-write -------------------------------------

    def can_append_slot(self, seq):
        return self.gpu.num_free >= 1

    def append_slot(self, seq, blocks_to_copy):
        """Make room for one more token, forking the target block if it is shared."""
        table = self.tables[seq.seq_id]
        if len(table) < self._num_blocks(seq.num_computed + 1):
            table.append(self.gpu.allocate())
        self._fork_range(seq, seq.num_computed, seq.num_computed + 1, blocks_to_copy)

    def _fork_range(self, seq, start, end, blocks_to_copy):
        table = self.tables[seq.seq_id]
        for index in range(start // self.block_size, (end - 1) // self.block_size + 1):
            old = table[index]
            if self.gpu.ref_count(old) == 1:
                continue
            new = self.gpu.allocate()
            blocks_to_copy[old] = new  # the workers do the actual byte copy
            table[index] = new
            self.gpu.release(old)

    # -- swapping ------------------------------------------------------

    def _private(self, seq):
        """Blocks the sequence owns outright; the shared prefix stays resident."""
        return self.tables[seq.seq_id][seq.num_shared:]

    def can_swap_out(self, seq):
        return self.cpu.num_free >= len(self._private(seq))

    def swap_out(self, seq):
        table = self.tables[seq.seq_id]
        mapping = {}
        host = []
        for block in self._private(seq):
            destination = self.cpu.allocate()
            mapping[block] = destination
            host.append(destination)
            self.gpu.release(block)
        table[seq.num_shared:] = host  # now holds host block numbers
        return mapping

    def can_swap_in(self, seq):
        need = len(self._private(seq))
        return self.gpu.num_free - need >= self.watermark_blocks

    def swap_in(self, seq):
        table = self.tables[seq.seq_id]
        mapping = {}
        device = []
        for block in self._private(seq):
            destination = self.gpu.allocate()
            mapping[block] = destination
            device.append(destination)
            self.cpu.release(block)
        table[seq.num_shared:] = device
        return mapping

    def free(self, seq, on_host=False):
        table = self.tables.pop(seq.seq_id, [])
        for index, block in enumerate(table):
            if on_host and index >= seq.num_shared:
                self.cpu.release(block)
            else:
                self.gpu.release(block)

    # -- views for the model runner ------------------------------------

    def slot_mapping(self, seq, start, count, device):
        table = self.tables[seq.seq_id]
        return torch.tensor(
            [table[p // self.block_size] * self.block_size + p % self.block_size
             for p in range(start, start + count)],
            device=device,
        )

    def block_tables(self, seqs, device):
        width = max(len(self.tables[s.seq_id]) for s in seqs)
        rows = []
        for seq in seqs:
            table = self.tables[seq.seq_id]
            # Padded columns repeat a live block; they sit past every row's
            # position and are masked out, so their contents never matter.
            rows.append(table + [table[-1]] * (width - len(table)))
        return torch.tensor(rows, device=device)


# ---------------------------------------------------------------------- scheduler


@dataclass
class SchedulerOutput:
    """The plan the scheduler broadcasts to every worker for one step.

    This mirrors vLLM's execute_model signature: what to run, plus the three
    block movements the cache engines must perform first.
    """

    scheduled: list[Sequence] = field(default_factory=list)
    num_tokens: dict[int, int] = field(default_factory=dict)
    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)
    blocks_to_copy: dict[int, int] = field(default_factory=dict)
    is_prefill: bool = False

    @property
    def is_empty(self):
        return not self.scheduled


class Scheduler:
    """FCFS over three queues, with preemption when the GPU pool runs dry."""

    def __init__(self, cfg, block_manager):
        self.cfg = cfg
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.swapped: deque[Sequence] = deque()
        self.num_preempted = 0
        self.recomputed_tokens = 0
        self.swapped_out_blocks = 0
        self.swapped_in_blocks = 0

    @property
    def has_work(self):
        return bool(self.waiting or self.running or self.swapped)

    def add(self, seq):
        self.waiting.append(seq)

    def schedule(self):
        out = SchedulerOutput()

        # Prefill wins, but only when nothing is stranded on the host: bringing
        # swapped work back matters more than starting something new.
        if not self.swapped:
            budget = self.cfg.max_num_batched_tokens
            while self.waiting:
                seq = self.waiting[0]
                tokens = seq.num_uncomputed
                if out.scheduled and tokens > budget:
                    break
                if len(self.running) + len(out.scheduled) >= self.cfg.max_num_seqs:
                    break
                if not self.block_manager.can_allocate(seq):
                    break

                self.waiting.popleft()
                self.block_manager.allocate(seq, out.blocks_to_copy)
                seq.status = RUNNING
                out.scheduled.append(seq)
                out.num_tokens[seq.seq_id] = seq.num_uncomputed
                budget -= tokens

            if out.scheduled:
                out.is_prefill = True
                self.running.extend(out.scheduled)
                return out

        # Decode: every running sequence needs room for exactly one more token.
        preempted = []
        kept = []
        queue = deque(self.running)
        while queue:
            seq = queue.popleft()
            while not self.block_manager.can_append_slot(seq):
                victim = queue.pop() if queue else seq
                self._preempt(victim, out)
                preempted.append(victim)
                if victim is seq:
                    seq = None
                    break
            if seq is not None:
                self.block_manager.append_slot(seq, out.blocks_to_copy)
                kept.append(seq)
        self.running = kept

        # Only reclaim host-resident work once the pressure has eased.
        if not preempted:
            while self.swapped:
                seq = self.swapped[0]
                if len(self.running) >= self.cfg.max_num_seqs:
                    break
                if not self.block_manager.can_swap_in(seq):
                    break
                self.swapped.popleft()
                mapping = self.block_manager.swap_in(seq)
                out.blocks_to_swap_in.update(mapping)
                self.swapped_in_blocks += len(mapping)
                self.block_manager.append_slot(seq, out.blocks_to_copy)
                seq.status = RUNNING
                self.running.append(seq)

        out.scheduled = list(self.running)
        out.num_tokens = {s.seq_id: 1 for s in self.running}
        return out

    def _preempt(self, seq, out):
        self.num_preempted += 1
        self.running = [s for s in self.running if s is not seq]

        swap = self.cfg.preemption_mode == "swap" and self.block_manager.can_swap_out(seq)
        if swap:
            # The K/V survives on the host, so nothing has to be recomputed.
            mapping = self.block_manager.swap_out(seq)
            out.blocks_to_swap_out.update(mapping)
            self.swapped_out_blocks += len(mapping)
            seq.status = SWAPPED
            seq.swaps += 1
            self.swapped.append(seq)
        else:
            # Cheaper in memory, but the prompt has to be prefilled again.
            self.recomputed_tokens += seq.num_computed - seq.shared_tokens
            self.block_manager.free(seq)
            seq.num_computed = 0
            seq.num_shared = 0
            seq.shared_tokens = 0
            seq.status = WAITING
            seq.recomputes += 1
            self.waiting.appendleft(seq)

    def free_finished(self):
        done = [s for s in self.running if s.is_finished]
        for seq in done:
            seq.status = FINISHED
            self.block_manager.free(seq)
            self.running.remove(seq)
        return done


# ------------------------------------------------------------------------- worker


class CacheEngine:
    """One worker's KV cache: its head shard on device, plus host swap space.

    Layout is [layers, blocks, block_size, kv_heads_local, head_dim], so moving
    a block moves every layer at once. Real vLLM runs these copies on a separate
    CUDA stream that overlaps with the forward pass.
    """

    def __init__(self, cfg, num_kv_heads_local, device, dtype):
        tail = (cfg.block_size, num_kv_heads_local, cfg.head_dim)
        self.gpu_K = torch.zeros((cfg.num_layers, cfg.num_gpu_blocks, *tail),
                                 dtype=dtype, device=device)
        self.gpu_V = torch.zeros_like(self.gpu_K)
        self.cpu_K = torch.zeros((cfg.num_layers, cfg.num_cpu_blocks, *tail),
                                 dtype=dtype, device="cpu")
        self.cpu_V = torch.zeros_like(self.cpu_K)
        self.bytes_swapped = 0
        self.blocks_copied = 0

    def layer(self, index):
        return self.gpu_K[index], self.gpu_V[index]

    def _block_bytes(self):
        return self.gpu_K[0, 0].numel() * self.gpu_K.element_size() * 2

    def swap_in(self, mapping):
        for host, device in mapping.items():
            self.gpu_K[:, device] = self.cpu_K[:, host].to(self.gpu_K.device)
            self.gpu_V[:, device] = self.cpu_V[:, host].to(self.gpu_V.device)
        self.bytes_swapped += len(mapping) * self._block_bytes()

    def swap_out(self, mapping):
        for device, host in mapping.items():
            self.cpu_K[:, host] = self.gpu_K[:, device].cpu()
            self.cpu_V[:, host] = self.gpu_V[:, device].cpu()
        self.bytes_swapped += len(mapping) * self._block_bytes()

    def copy(self, mapping):
        for source, destination in mapping.items():
            self.gpu_K[:, destination] = self.gpu_K[:, source]
            self.gpu_V[:, destination] = self.gpu_V[:, source]
        self.blocks_copied += len(mapping)

    @property
    def nbytes(self):
        return (self.gpu_K.numel() + self.gpu_V.numel()) * self.gpu_K.element_size()


class ShardedAttention(nn.Module):
    """This rank's heads. Q/K/V are column-parallel, W_O is row-parallel, so the
    result is a partial sum that the all-reduce completes."""

    def __init__(self, cfg, tp_size):
        super().__init__()
        self.num_heads = cfg.num_heads // tp_size
        self.num_kv_heads = cfg.num_kv_heads // tp_size
        self.n_rep = self.num_heads // self.num_kv_heads
        self.d_k = cfg.head_dim

        self.W_Q = nn.Linear(cfg.d_model, self.num_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(cfg.d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(cfg.d_model, self.num_kv_heads * self.d_k, bias=False)
        self.W_O = nn.Linear(self.num_heads * self.d_k, cfg.d_model, bias=False)

    def forward(self, x, k_cache, v_cache, block_tables, slot_mapping, positions):
        n, seq_len, _ = x.shape

        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        flat = (n * seq_len, self.num_kv_heads, self.d_k)
        k_cache.view(-1, self.num_kv_heads, self.d_k)[slot_mapping] = \
            K_new.permute(0, 2, 1, 3).reshape(flat)
        v_cache.view(-1, self.num_kv_heads, self.d_k)[slot_mapping] = \
            V_new.permute(0, 2, 1, 3).reshape(flat)

        K = self._gather(k_cache, block_tables)
        V = self._gather(v_cache, block_tables)

        allowed = torch.arange(K.shape[2], device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))

    def _gather(self, cache, block_tables):
        blocks = cache[block_tables]
        n, nb, bs, h, d = blocks.shape
        return repeat_kv(blocks.reshape(n, nb * bs, h, d).permute(0, 2, 1, 3), self.n_rep)


class ShardedMLP(nn.Module):
    """Column-parallel up-projection, row-parallel down-projection."""

    def __init__(self, cfg, tp_size):
        super().__init__()
        d_ff = 4 * cfg.d_model // tp_size
        self.up = nn.Linear(cfg.d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class Worker(nn.Module):
    """One rank: a slice of every layer's weights, and a cache engine for the
    matching slice of the KV cache."""

    def __init__(self, cfg, rank, device, dtype):
        super().__init__()
        tp = cfg.tensor_parallel_size
        self.rank = rank
        self.attn = nn.ModuleList([ShardedAttention(cfg, tp) for _ in range(cfg.num_layers)])
        self.mlp = nn.ModuleList([ShardedMLP(cfg, tp) for _ in range(cfg.num_layers)])
        self.to(device=device, dtype=dtype)
        self.cache_engine = CacheEngine(cfg, cfg.num_kv_heads // tp, device, dtype)

    def execute_cache_ops(self, out):
        """Run before the forward pass, exactly as vLLM orders them."""
        self.cache_engine.swap_in(out.blocks_to_swap_in)
        self.cache_engine.swap_out(out.blocks_to_swap_out)
        self.cache_engine.copy(out.blocks_to_copy)


class Trunk(nn.Module):
    """Weights vLLM replicates on every rank: embeddings, norms and the head."""

    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.ln_1 = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_layers)])
        self.ln_2 = nn.ModuleList([nn.LayerNorm(cfg.d_model) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)


def shard_weights(cfg, trunk_seed, workers):
    """Split one set of full weights across the ranks, Megatron style.

    Column-parallel layers split the output dimension (rows of nn.Linear.weight);
    row-parallel layers split the input dimension (columns). Doing it from one
    seeded source is what lets tp=1 and tp=2 produce identical tokens.
    """
    torch.manual_seed(trunk_seed)
    tp = len(workers)
    d_k, d_ff = cfg.head_dim, 4 * cfg.d_model

    with torch.no_grad():
        for layer in range(cfg.num_layers):
            full = {
                "W_Q": torch.empty(cfg.num_heads * d_k, cfg.d_model),
                "W_K": torch.empty(cfg.num_kv_heads * d_k, cfg.d_model),
                "W_V": torch.empty(cfg.num_kv_heads * d_k, cfg.d_model),
                "W_O": torch.empty(cfg.d_model, cfg.num_heads * d_k),
                "up": torch.empty(d_ff, cfg.d_model),
                "down": torch.empty(cfg.d_model, d_ff),
            }
            for tensor in full.values():
                nn.init.normal_(tensor, std=0.02)

            for rank, worker in enumerate(workers):
                attn, mlp = worker.attn[layer], worker.mlp[layer]
                q = attn.num_heads * d_k
                kv = attn.num_kv_heads * d_k
                ff = d_ff // tp

                attn.W_Q.weight.copy_(full["W_Q"][rank * q:(rank + 1) * q])
                attn.W_K.weight.copy_(full["W_K"][rank * kv:(rank + 1) * kv])
                attn.W_V.weight.copy_(full["W_V"][rank * kv:(rank + 1) * kv])
                attn.W_O.weight.copy_(full["W_O"][:, rank * q:(rank + 1) * q])
                mlp.up.weight.copy_(full["up"][rank * ff:(rank + 1) * ff])
                mlp.down.weight.copy_(full["down"][:, rank * ff:(rank + 1) * ff])


def all_reduce(partials):
    """Stands in for the NCCL all-reduce each rank performs after a
    row-parallel matmul."""
    total = partials[0]
    for part in partials[1:]:
        total = total + part
    return total


# ------------------------------------------------------------------------- engine


@dataclass
class EngineStats:
    label: str
    wall_time: float = 0.0
    steps: int = 0
    prefill_steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    outputs: dict[int, list[int]] = field(default_factory=dict)
    ttft: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    peak_gpu_blocks: int = 0

    @property
    def throughput(self):
        return sum(len(o) for o in self.outputs.values()) / self.wall_time


class LLMEngine:
    """Scheduler + KV cache manager + workers, stepped one iteration at a time."""

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        assert cfg.num_heads % cfg.tensor_parallel_size == 0
        assert cfg.num_kv_heads % cfg.tensor_parallel_size == 0

        torch.manual_seed(seed)
        self.trunk = Trunk(cfg).to(device=self.device, dtype=self.dtype).eval()
        self.workers = [
            Worker(cfg, rank, self.device, self.dtype).eval()
            for rank in range(cfg.tensor_parallel_size)
        ]
        shard_weights(cfg, seed + 1, self.workers)

        self.block_manager = BlockSpaceManager(cfg)
        self.scheduler = Scheduler(cfg, self.block_manager)
        self.stats = EngineStats(cfg.label)
        self._next_id = 0
        self._prefix_tokens = None
        self.t0 = 0.0

    def add_request(self, prompt, max_new_tokens):
        blocks = (len(prompt) + max_new_tokens + self.cfg.block_size - 1) // self.cfg.block_size
        if blocks > self.cfg.num_gpu_blocks - self.block_manager.watermark_blocks:
            raise ValueError("request cannot fit in the GPU cache even when alone")
        seq = Sequence(self._next_id, list(prompt), max_new_tokens)
        self._next_id += 1
        self.scheduler.add(seq)
        return seq

    def set_prefix(self, tokens):
        """Prompt prefix worth caching, e.g. a shared system prompt."""
        self._prefix_tokens = list(tokens)

    # -- model execution ---------------------------------------------

    @torch.no_grad()
    def _run_layers(self, input_ids, positions, block_tables, slot_mapping):
        x = self.trunk.token_emb(input_ids) + self.trunk.pos_emb(positions)
        for layer in range(self.cfg.num_layers):
            h = self.trunk.ln_1[layer](x)
            x = x + all_reduce([
                w.attn[layer](h, *w.cache_engine.layer(layer),
                              block_tables, slot_mapping, positions)
                for w in self.workers
            ])
            h = self.trunk.ln_2[layer](x)
            x = x + all_reduce([w.mlp[layer](h) for w in self.workers])
        return self.trunk.lm_head(self.trunk.ln_f(x[:, -1]))

    def _run_prefill(self, out):
        for seq in out.scheduled:
            start = seq.num_computed
            count = out.num_tokens[seq.seq_id]
            ids = torch.tensor([seq.tokens[start:start + count]], device=self.device)
            positions = torch.arange(start, start + count, device=self.device).unsqueeze(0)

            logits = self._run_layers(
                ids, positions,
                self.block_manager.block_tables([seq], self.device),
                self.block_manager.slot_mapping(seq, start, count, self.device),
            )
            seq.num_computed += count
            self.stats.prefill_tokens += count

            if seq.num_uncomputed == 0:
                seq.tokens.append(int(logits[0].argmax()))
                if seq.t_first_token == 0.0:
                    seq.t_first_token = time.perf_counter()
            if self._prefix_tokens:
                self.block_manager.register_prefix(seq, self._prefix_tokens)

    def _run_decode(self, out):
        seqs = out.scheduled
        ids = torch.tensor([[s.tokens[s.num_computed]] for s in seqs], device=self.device)
        positions = torch.tensor([[s.num_computed] for s in seqs], device=self.device)
        slot_mapping = torch.cat([
            self.block_manager.slot_mapping(s, s.num_computed, 1, self.device) for s in seqs
        ])

        logits = self._run_layers(
            ids, positions, self.block_manager.block_tables(seqs, self.device), slot_mapping
        )
        for row, seq in enumerate(seqs):
            seq.num_computed += 1
            seq.tokens.append(int(logits[row].argmax()))
        self.stats.decode_tokens += len(seqs)

    def step(self):
        out = self.scheduler.schedule()
        if out.is_empty:
            return

        # The scheduler decided; the workers move the bytes.
        for worker in self.workers:
            worker.execute_cache_ops(out)

        if out.is_prefill:
            self._run_prefill(out)
            self.stats.prefill_steps += 1
        else:
            self._run_decode(out)
        self.stats.steps += 1

        self.stats.peak_gpu_blocks = max(self.stats.peak_gpu_blocks,
                                         self.block_manager.gpu.num_used)
        for seq in self.scheduler.free_finished():
            seq.t_finished = time.perf_counter()
            self.stats.outputs[seq.seq_id] = seq.output
            self.stats.ttft.append(seq.t_first_token - self.t0)
            self.stats.latency.append(seq.t_finished - self.t0)

    def run(self):
        self.t0 = time.perf_counter()
        while self.scheduler.has_work:
            self.step()
        self.stats.wall_time = time.perf_counter() - self.t0
        return self.stats


# --------------------------------------------------------------------------- demo


def build_workload(cfg, num_requests=12, seed=11):
    rng = torch.Generator().manual_seed(seed)
    # Not a multiple of block_size on purpose, so the last shared block is
    # partial and copy-on-write actually fires.
    system_prompt = torch.randint(0, cfg.vocab_size, (200,), generator=rng).tolist()

    requests = []
    for _ in range(num_requests):
        suffix_len = int(torch.randint(32, 96, (1,), generator=rng))
        suffix = torch.randint(0, cfg.vocab_size, (suffix_len,), generator=rng).tolist()
        max_new_tokens = int(torch.randint(16, 96, (1,), generator=rng))
        requests.append((system_prompt + suffix, max_new_tokens))
    return system_prompt, requests


def run_config(cfg, system_prompt, requests):
    engine = LLMEngine(cfg)
    engine.set_prefix(system_prompt)
    for prompt, max_new_tokens in requests:
        engine.add_request(prompt, max_new_tokens)
    stats = engine.run()
    return engine, stats


def main():
    base = EngineConfig()
    system_prompt, requests = build_workload(base)

    configs = [
        EngineConfig(label="tp=1, recompute", tensor_parallel_size=1,
                     preemption_mode="recompute"),
        EngineConfig(label="tp=2, recompute", tensor_parallel_size=2,
                     preemption_mode="recompute"),
        EngineConfig(label="tp=2, swap", tensor_parallel_size=2,
                     preemption_mode="swap"),
    ]

    results = [run_config(cfg, system_prompt, requests) for cfg in configs]

    prompt_tokens = sum(len(p) for p, _ in requests)
    output_tokens = sum(n for _, n in requests)
    print(f"{len(requests)} requests sharing a {len(system_prompt)}-token system prompt")
    print(f"{prompt_tokens} prompt tokens, {output_tokens} output tokens, "
          f"block size {base.block_size}, {base.num_gpu_blocks} GPU blocks / "
          f"{base.num_cpu_blocks} CPU blocks")
    print()

    print(f"{'':<18}{'wall':>8}{'tok/s':>8}{'steps':>7}{'peak':>6}{'preempt':>9}"
          f"{'recomp':>8}{'swap out':>10}{'swap in':>9}{'CoW':>5}{'hits':>6}")
    for (engine, stats), cfg in zip(results, configs):
        sched = engine.scheduler
        cow = engine.workers[0].cache_engine.blocks_copied
        print(f"{stats.label:<18}{stats.wall_time:>7.2f}s{stats.throughput:>8.1f}"
              f"{stats.steps:>7}{stats.peak_gpu_blocks:>6}{sched.num_preempted:>9}"
              f"{sched.recomputed_tokens:>8}{sched.swapped_out_blocks:>10}"
              f"{sched.swapped_in_blocks:>9}{cow:>5}"
              f"{engine.block_manager.prefix_hits:>6}")
    print()

    reference = results[0][1].outputs
    print(f"identical output    : {all(s.outputs == reference for _, s in results)}")
    print(f"  tp=1 vs tp=2 agree -> the head sharding and all-reduce are correct")
    print(f"  recompute vs swap  -> the block tables survive eviction either way")
    print()

    engine, stats = results[2]
    gpu_bytes = sum(w.cache_engine.nbytes for w in engine.workers)
    swapped_bytes = sum(w.cache_engine.bytes_swapped for w in engine.workers)
    per_worker = engine.workers[0].cache_engine.nbytes
    print(f"KV cache on device  : {gpu_bytes / 2**20:.1f} MiB across "
          f"{len(engine.workers)} workers ({per_worker / 2**20:.1f} MiB each, "
          f"{base.num_kv_heads // base.tensor_parallel_size} kv-head(s) per rank)")
    print(f"bytes swapped       : {swapped_bytes / 2**20:.1f} MiB moved between "
          f"device and host")
    print(f"TTFT p50 / p99      : {percentile(stats.ttft, 50):.2f}s / "
          f"{percentile(stats.ttft, 99):.2f}s")
    print(f"prefill / decode    : {stats.prefill_tokens} tokens prefilled, "
          f"{stats.decode_tokens} decoded over {stats.steps} steps")


if __name__ == "__main__":
    main()
