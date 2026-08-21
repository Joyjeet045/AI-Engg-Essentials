"""A serving benchmark: Poisson arrivals, percentiles, and goodput under an SLO.

Every project so far reported one number from one offline batch. That is not how
a server is judged. Requests arrive at some rate, and what matters is how many of
them are answered *well enough*, not how many tokens the box can emit.

So this measures the thing production teams actually optimise:

    goodput = completed requests per second that met BOTH
              a TTFT deadline and a TPOT deadline

The lesson is that throughput and goodput come apart. Push the arrival rate past
the knee and the server keeps accepting work and keeps emitting tokens -- raw
throughput even goes up -- while the fraction of requests that anyone would call
answered collapses. A benchmark that only prints tokens/sec cannot see this.

Timing method: arrivals are simulated against real service times. The engine runs
its real steps and each measured duration advances a virtual clock; when the
server is idle the clock jumps to the next arrival instead of sleeping. Service
costs are genuine, queueing is exact, and no wall-clock time is wasted waiting.
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


# -------------------------------------------------------------------------- model


class SlotAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, max_batch, max_seq_len):
        super().__init__()
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
        n, seq_len, _ = x.shape
        Q = reshape_to_heads(self.W_Q(x), self.num_heads)
        K_new = reshape_to_heads(self.W_K(x), self.num_kv_heads)
        V_new = reshape_to_heads(self.W_V(x), self.num_kv_heads)

        rows = slot_ids.unsqueeze(1).expand(n, seq_len)
        self.cache_K[rows, :, positions] = K_new.transpose(1, 2)
        self.cache_V[rows, :, positions] = V_new.transpose(1, 2)

        K = repeat_kv(self.cache_K[slot_ids, :, :k_len], self.n_rep)
        V = repeat_kv(self.cache_V[slot_ids, :, :k_len], self.n_rep)

        allowed = torch.arange(k_len, device=x.device) <= positions.unsqueeze(-1)
        output = F.scaled_dot_product_attention(Q, K, V, attn_mask=allowed.unsqueeze(1))
        return self.W_O(reshape_from_heads(output))


class Block(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, max_batch, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = SlotAttention(d_model, num_heads, num_kv_heads, max_batch, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x, slot_ids, positions, k_len):
        x = x + self.attn(self.ln_1(x), slot_ids, positions, k_len)
        x = x + self.ffn(self.ln_2(x))
        return x


class SlotLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_batch=8, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, num_kv_heads, 4 * d_model, max_batch, max_seq_len)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def forward(self, input_ids, slot_ids, positions, k_len):
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, slot_ids, positions, k_len)
        return self.lm_head(self.ln_f(x[:, -1]))


# ------------------------------------------------------------------ load generator


@dataclass
class Request:
    id: int
    arrival: float
    prompt: list[int]
    max_new_tokens: int

    first_token: float = 0.0
    finished: float = 0.0
    generated: int = 0

    @property
    def ttft(self):
        return self.first_token - self.arrival

    @property
    def tpot(self):
        return (self.finished - self.first_token) / max(self.generated - 1, 1)


def poisson_workload(vocab_size, num_requests, rate, seed=0):
    """Exponential inter-arrival gaps, which is what a Poisson process is."""
    rng = torch.Generator().manual_seed(seed)
    requests = []
    clock = 0.0
    for index in range(num_requests):
        gap = -math.log(1 - float(torch.rand((), generator=rng))) / rate
        clock += gap
        prompt_len = int(torch.randint(96, 160, (1,), generator=rng))
        requests.append(Request(
            id=index,
            arrival=clock,
            prompt=torch.randint(0, vocab_size, (prompt_len,), generator=rng).tolist(),
            max_new_tokens=int(torch.randint(32, 96, (1,), generator=rng)),
        ))
    return requests


# ------------------------------------------------------------------------- engine


@dataclass
class Sequence:
    request: Request
    slot: int
    num_cached: int = 0
    last_token: int = 0
    generated: list[int] = field(default_factory=list)

    @property
    def prompt_len(self):
        return len(self.request.prompt)

    @property
    def needs_prefill(self):
        return self.num_cached < self.prompt_len

    @property
    def done(self):
        return len(self.generated) >= self.request.max_new_tokens

    @property
    def next_position(self):
        return self.prompt_len + len(self.generated) - 1


@dataclass
class SLO:
    ttft: float = 1.0  # seconds to the first token
    tpot: float = 0.060  # seconds between subsequent tokens


@dataclass
class Report:
    rate: float
    makespan: float = 0.0
    completed: list[Request] = field(default_factory=list)

    def _met(self, slo):
        return [r for r in self.completed if r.ttft <= slo.ttft and r.tpot <= slo.tpot]

    def throughput(self):
        return len(self.completed) / self.makespan

    def token_throughput(self):
        return sum(r.generated for r in self.completed) / self.makespan

    def goodput(self, slo):
        return len(self._met(slo)) / self.makespan

    def met_fraction(self, slo):
        return len(self._met(slo)) / max(len(self.completed), 1)


@torch.no_grad()
def serve(model, requests, max_batch=8):
    """Continuous batching driven by a virtual clock.

    Real step durations advance the clock; idle time is skipped rather than slept
    through, so queueing is exact and the benchmark stays quick.
    """
    device = model.device
    waiting = deque(requests)
    free_slots = deque(range(max_batch))
    running: list[Sequence] = []
    clock = 0.0
    report = Report(rate=0.0)

    while waiting or running:
        if not running and waiting and waiting[0].arrival > clock:
            clock = waiting[0].arrival  # server is idle; skip ahead

        while waiting and free_slots and waiting[0].arrival <= clock:
            running.append(Sequence(waiting.popleft(), free_slots.popleft()))

        start = time.perf_counter()

        decoding = [s for s in running if not s.needs_prefill]
        logits = None
        if decoding:
            slot_ids = torch.tensor([s.slot for s in decoding], device=device)
            tokens = torch.tensor([[s.last_token] for s in decoding], device=device)
            positions = torch.tensor([[s.next_position] for s in decoding], device=device)
            logits = model(tokens, slot_ids, positions, int(positions.max()) + 1)

        prefilled = next((s for s in running if s.needs_prefill), None)
        prefill_logits = None
        if prefilled is not None:
            ids = torch.tensor([prefilled.request.prompt], device=device)
            positions = torch.arange(prefilled.prompt_len, device=device).unsqueeze(0)
            prefill_logits = model(ids, torch.tensor([prefilled.slot], device=device),
                                   positions, prefilled.prompt_len)
            prefilled.num_cached = prefilled.prompt_len

        clock += time.perf_counter() - start

        if logits is not None:
            for row, seq in enumerate(decoding):
                seq.last_token = int(logits[row].argmax())
                seq.generated.append(seq.last_token)
        if prefill_logits is not None:
            prefilled.last_token = int(prefill_logits[0].argmax())
            prefilled.generated.append(prefilled.last_token)
            prefilled.request.first_token = clock

        for seq in list(running):
            if seq.done:
                seq.request.finished = clock
                seq.request.generated = len(seq.generated)
                report.completed.append(seq.request)
                running.remove(seq)
                free_slots.append(seq.slot)

    report.makespan = clock
    return report


def bar(value, peak, width=12):
    filled = 0 if peak <= 0 else int(round(width * value / peak))
    return "#" * filled + "." * (width - filled)


def main():
    torch.manual_seed(0)
    vocab_size, max_batch, num_requests = 1024, 4, 40
    slo = SLO(ttft=0.5, tpot=0.040)

    model = SlotLM(vocab_size, d_model=256, num_layers=6,
                   max_batch=max_batch, max_seq_len=512).eval()
    rates = [2, 4, 6, 10, 20]

    reports = []
    for rate in rates:
        requests = poisson_workload(vocab_size, num_requests, rate, seed=rate)
        report = serve(model, requests, max_batch)
        report.rate = rate
        reports.append(report)

    peak = max(r.goodput(slo) for r in reports)

    print(f"{num_requests} requests per point, Poisson arrivals, batch {max_batch}, "
          f"96-160 token prompts, 32-96 token replies")
    print(f"SLO: TTFT <= {slo.ttft:.1f}s and TPOT <= {slo.tpot * 1e3:.0f}ms")
    print()
    print(f"{'offered':>9}{'done':>8}{'tok/s':>9}{'TTFT p50':>11}{'TTFT p99':>11}"
          f"{'TPOT p99':>11}{'met SLO':>10}{'goodput':>10}  ")
    for report in reports:
        ttft = [r.ttft for r in report.completed]
        tpot = [r.tpot for r in report.completed]
        print(f"{report.rate:>7}/s{report.throughput():>7.1f}/s"
              f"{report.token_throughput():>9.0f}"
              f"{percentile(ttft, 50):>10.2f}s{percentile(ttft, 99):>10.2f}s"
              f"{percentile(tpot, 99) * 1e3:>10.1f}ms"
              f"{report.met_fraction(slo):>10.0%}"
              f"{report.goodput(slo):>8.1f}/s  "
              f"{bar(report.goodput(slo), peak)}")
    print()

    best = max(reports, key=lambda r: r.goodput(slo))
    busiest = max(reports, key=lambda r: r.token_throughput())
    print(f"peak goodput      : {best.goodput(slo):.1f} req/s at {best.rate}/s offered")
    print(f"peak raw tok/s    : {busiest.token_throughput():.0f} at "
          f"{busiest.rate}/s offered, where {busiest.met_fraction(slo):.0%} of "
          f"requests met the SLO")
    print()
    print("Past the knee the queue absorbs the extra load: tokens/sec flattens out")
    print("because the batch is already full, so the only thing the extra arrivals")
    print("buy is waiting. A benchmark that prints tokens/sec alone cannot see it.")


if __name__ == "__main__":
    main()
