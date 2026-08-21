"""Disaggregated serving: put prefill and decode on different machines.

Prefill and decode are not the same workload. Prefill is compute bound, bursty,
and proportional to prompt length. Decode is memory-bandwidth bound, steady, and
one token wide. Running both on one device means they fight:

  * a long prefill occupies the device, and every running decode stalls for its
    whole duration -- the "max stall" column back in project 02
  * the batch size that keeps decode efficient is not the batch size that keeps
    prefill efficient
  * you cannot scale TTFT capacity without also buying decode capacity

Disaggregation separates them. Prefill workers compute the prompt's K/V and ship
it to decode workers, which never do anything but decode. TTFT and TPOT stop
being coupled: add prefill workers to fix time-to-first-token, add decode workers
to fix inter-token latency.

The bill is the KV transfer -- prompt_len * bytes_per_token, per request, across
the interconnect. That is why disaggregation only pays when prompts are long
enough for the interference to cost more than the copy.

Modelling note: both pools live in one process here, so the difference is made
explicit in the clock. Colocated charges t_prefill + t_decode because one device
does them in sequence; disaggregated charges max(t_prefill, t_decode) because two
devices run at once. Transfer is charged separately and in full; production
overlaps it with compute.
"""

import math
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


# --------------------------------------------------------------------------- kv


class SlotCache:
    """A worker's KV store. Kept outside the model so several workers can share
    one set of weights, which is what separate machines would each hold a copy of.
    """

    def __init__(self, num_layers, max_batch, num_kv_heads, max_seq_len, d_k):
        shape = (num_layers, max_batch, num_kv_heads, max_seq_len, d_k)
        self.K = torch.zeros(shape)
        self.V = torch.zeros(shape)
        self.free_slots = deque(range(max_batch))

    @property
    def bytes_per_token(self):
        layers, _, heads, _, d_k = self.K.shape
        return 2 * layers * heads * d_k * self.K.element_size()

    def transfer_from(self, source, src_slot, dst_slot, length):
        """Ship one sequence's K/V across. Returns the bytes moved."""
        self.K[:, dst_slot, :, :length] = source.K[:, src_slot, :, :length]
        self.V[:, dst_slot, :, :length] = source.V[:, src_slot, :, :length]
        return length * self.bytes_per_token


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

    def forward(self, x, K_buf, V_buf, slot_ids, positions, k_len):
        n, seq_len, _ = x.shape
        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        rows = slot_ids.unsqueeze(1).expand(n, seq_len)
        K_buf[rows, :, positions] = K_new.transpose(1, 2)
        V_buf[rows, :, positions] = V_new.transpose(1, 2)

        K = repeat_kv(K_buf[slot_ids, :, :k_len], self.n_rep)
        V = repeat_kv(V_buf[slot_ids, :, :k_len], self.n_rep)

        allowed = torch.arange(k_len, device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, num_kv_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, K_buf, V_buf, slot_ids, positions, k_len):
        x = x + self.attn(self.ln_1(x), K_buf, V_buf, slot_ids, positions, k_len)
        x = x + self.ffn(self.ln_2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_kv_heads=2,
                 num_layers=6, max_seq_len=2048):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, num_kv_heads, 4 * d_model) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def make_cache(self, max_batch):
        return SlotCache(self.num_layers, max_batch, self.num_kv_heads,
                         self.max_seq_len, self.d_k)

    def forward(self, input_ids, cache, slot_ids, positions, k_len):
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for index, block in enumerate(self.blocks):
            x = block(x, cache.K[index], cache.V[index], slot_ids, positions, k_len)
        return self.lm_head(self.ln_f(x[:, -1]))


# ------------------------------------------------------------------------ traffic


@dataclass
class Request:
    id: int
    arrival: float
    prompt: list[int]
    max_new_tokens: int

    first_token: float = 0.0
    finished: float = 0.0
    last_token_at: float = 0.0
    max_stall: float = 0.0
    output: list[int] = field(default_factory=list)

    @property
    def ttft(self):
        return self.first_token - self.arrival

    @property
    def tpot(self):
        return (self.finished - self.first_token) / max(len(self.output) - 1, 1)

    def record(self, token, now):
        if self.output:
            self.max_stall = max(self.max_stall, now - self.last_token_at)
        else:
            self.first_token = now
        self.output.append(token)
        self.last_token_at = now


def poisson_workload(vocab_size, num_requests, rate, seed=0):
    rng = torch.Generator().manual_seed(seed)
    requests, clock = [], 0.0
    for index in range(num_requests):
        clock += -math.log(1 - float(torch.rand((), generator=rng))) / rate
        prompt_len = int(torch.randint(768, 1536, (1,), generator=rng))
        requests.append(Request(
            id=index,
            arrival=clock,
            prompt=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=int(torch.randint(16, 40, (1,), generator=rng)),
        ))
    return requests


@dataclass
class Sequence:
    request: Request
    slot: int = -1
    prefill_slot: int = -1
    num_cached: int = 0
    last_token: int = 0

    @property
    def prompt_len(self):
        return len(self.request.prompt)

    @property
    def done(self):
        return len(self.request.output) >= self.request.max_new_tokens

    @property
    def next_position(self):
        return self.prompt_len + len(self.request.output) - 1


@dataclass
class Report:
    label: str
    clock: float = 0.0
    completed: list[Request] = field(default_factory=list)
    transferred_bytes: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def token_throughput(self):
        return sum(len(r.output) for r in self.completed) / self.clock


# ------------------------------------------------------------------------ engines


@torch.no_grad()
def prefill(model, cache, seq, slot):
    ids = torch.tensor([seq.request.prompt])
    positions = torch.arange(seq.prompt_len).unsqueeze(0)
    logits = model(ids, cache, torch.tensor([slot]), positions, seq.prompt_len)
    seq.num_cached = seq.prompt_len
    return int(logits[0].argmax())


@torch.no_grad()
def decode(model, cache, seqs):
    slot_ids = torch.tensor([s.slot for s in seqs])
    tokens = torch.tensor([[s.last_token] for s in seqs])
    positions = torch.tensor([[s.next_position] for s in seqs])
    logits = model(tokens, cache, slot_ids, positions, int(positions.max()) + 1)
    return [int(row.argmax()) for row in logits]


@torch.no_grad()
def run_colocated(model, requests, max_batch=4):
    """One device: a prefill and the decode batch take turns."""
    report = Report("colocated")
    cache = model.make_cache(max_batch)
    waiting = deque(requests)
    running: list[Sequence] = []
    clock = 0.0

    while waiting or running:
        if not running and waiting and waiting[0].arrival > clock:
            clock = waiting[0].arrival

        pending = None
        if waiting and cache.free_slots and waiting[0].arrival <= clock:
            pending = Sequence(waiting.popleft(), cache.free_slots.popleft())

        start = time.perf_counter()
        outputs = decode(model, cache, running) if running else []
        decode_seconds = time.perf_counter() - start
        report.decode_seconds += decode_seconds

        prefill_seconds = 0.0
        first = None
        if pending is not None:
            start = time.perf_counter()
            first = prefill(model, cache, pending, pending.slot)
            prefill_seconds = time.perf_counter() - start
            report.prefill_seconds += prefill_seconds

        # Serialised on one device, so a long prefill delays every running decode.
        clock += decode_seconds + prefill_seconds

        for seq, token in zip(running, outputs, strict=True):
            seq.last_token = token
            seq.request.record(token, clock)
        if pending is not None:
            pending.last_token = first
            pending.request.record(first, clock)
            running.append(pending)

        for seq in list(running):
            if seq.done:
                seq.request.finished = clock
                report.completed.append(seq.request)
                running.remove(seq)
                cache.free_slots.append(seq.slot)

    report.clock = clock
    return report


@torch.no_grad()
def run_disaggregated(model, requests, prefill_slots=1, decode_slots=4):
    """Two pools running at the same time, with the K/V shipped between them."""
    report = Report(f"disaggregated {prefill_slots}P/{decode_slots}D")
    prefill_cache = model.make_cache(prefill_slots)
    decode_cache = model.make_cache(decode_slots)
    waiting = deque(requests)
    running: list[Sequence] = []
    clock = 0.0

    while waiting or running:
        if not running and waiting and waiting[0].arrival > clock:
            clock = waiting[0].arrival

        # Every prefill worker takes one prompt, so the pool is a real lever on
        # time-to-first-token.
        pendings = []
        while (waiting and waiting[0].arrival <= clock and len(pendings) < prefill_slots
               and prefill_cache.free_slots and decode_cache.free_slots):
            seq = Sequence(waiting.popleft())
            seq.prefill_slot = prefill_cache.free_slots.popleft()
            seq.slot = decode_cache.free_slots.popleft()
            pendings.append(seq)

        start = time.perf_counter()
        outputs = decode(model, decode_cache, running) if running else []
        decode_seconds = time.perf_counter() - start
        report.decode_seconds += decode_seconds

        prefill_seconds = 0.0
        firsts = {}
        for seq in pendings:
            start = time.perf_counter()
            firsts[seq.request.id] = prefill(model, prefill_cache, seq, seq.prefill_slot)
            elapsed = time.perf_counter() - start
            report.prefill_seconds += elapsed
            prefill_seconds = max(prefill_seconds, elapsed)  # separate workers

        # Separate devices, so the two pools overlap instead of queueing.
        clock += max(decode_seconds, prefill_seconds)

        for seq, token in zip(running, outputs, strict=True):
            seq.last_token = token
            seq.request.record(token, clock)

        transfer_seconds = 0.0
        for seq in pendings:
            start = time.perf_counter()
            report.transferred_bytes += decode_cache.transfer_from(
                prefill_cache, seq.prefill_slot, seq.slot, seq.prompt_len
            )
            transfer_seconds = max(transfer_seconds, time.perf_counter() - start)
            prefill_cache.free_slots.append(seq.prefill_slot)
        clock += transfer_seconds

        for seq in pendings:
            seq.last_token = firsts[seq.request.id]
            seq.request.record(seq.last_token, clock)
            running.append(seq)

        for seq in list(running):
            if seq.done:
                seq.request.finished = clock
                report.completed.append(seq.request)
                running.remove(seq)
                decode_cache.free_slots.append(seq.slot)

    report.clock = clock
    return report


def main():
    torch.manual_seed(0)
    vocab_size, num_requests, rate = 1024, 24, 6.0

    model = TinyLM(vocab_size).eval()
    requests = poisson_workload(vocab_size, num_requests, rate)

    run_colocated(model, poisson_workload(vocab_size, 3, rate), max_batch=4)  # warm up

    builders = [
        ("colocated", lambda rs: run_colocated(model, rs, max_batch=4)),
        ("1P/4D", lambda rs: run_disaggregated(model, rs, 1, 4)),
        ("2P/4D", lambda rs: run_disaggregated(model, rs, 2, 4)),
    ]

    reports = []
    for _label, build in builders:
        # Wall-clock on a busy CPU is noisy and the virtual clock accumulates it,
        # so take the fastest of a couple of runs.
        best = None
        for _ in range(2):
            report = build(poisson_workload(vocab_size, num_requests, rate))
            if best is None or report.clock < best.clock:
                best = report
        reports.append(best)

    kv_per_token = model.make_cache(1).bytes_per_token
    prompt_tokens = sum(len(r.prompt) for r in requests)
    print(f"{num_requests} requests at {rate:.0f}/s, prompts 768-1536 tokens "
          f"({prompt_tokens} total), replies 16-40 tokens")
    print(f"6 layers, 2 kv-heads, {kv_per_token} B of KV per token")
    print()

    print(f"{'':<24}{'wall':>8}{'tok/s':>8}{'TTFT p50':>11}{'TTFT p99':>11}"
          f"{'TPOT p50':>11}{'max stall':>11}{'KV moved':>11}")
    for report in reports:
        ttft = [r.ttft for r in report.completed]
        tpot = [r.tpot for r in report.completed]
        stall = max(r.max_stall for r in report.completed)
        print(f"{report.label:<24}{report.clock:>7.2f}s{report.token_throughput:>8.0f}"
              f"{percentile(ttft, 50):>10.2f}s{percentile(ttft, 99):>10.2f}s"
              f"{percentile(tpot, 50) * 1e3:>10.1f}ms{stall * 1e3:>10.0f}ms"
              f"{report.transferred_bytes / 2**20:>10.1f}M")
    print()

    reference = {r.id: r.output for r in reports[0].completed}
    identical = all({r.id: r.output for r in rep.completed} == reference
                    for rep in reports[1:])
    print(f"identical output : {identical}")
    print("  the decode pool continues from transferred K/V, so a correct copy is")
    print("  indistinguishable from having prefilled locally")
    print()
    total = reports[0].prefill_seconds + reports[0].decode_seconds
    print(f"device time split : {reports[0].prefill_seconds / total:.0%} prefill, "
          f"{reports[0].decode_seconds / total:.0%} decode")
    print("with long prompts the prefill share is large, and every second of it is")
    print("a second the colocated decode batch spends frozen")

    if not identical:
        raise SystemExit("the decode pool diverged; the KV transfer is wrong")


if __name__ == "__main__":
    main()
