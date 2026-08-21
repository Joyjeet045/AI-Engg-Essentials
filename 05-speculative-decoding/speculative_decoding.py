"""Speculative decoding: many tokens verified per weight load.

Decode is memory-bandwidth bound. Every step streams the whole weight matrix out
of HBM to produce one token per sequence, so with P parameters at 2 bytes and
batch B the arithmetic intensity is roughly

    2 * P * B FLOPs / 2 * P bytes = 2B FLOPs per byte

against hardware that needs a few hundred to saturate. Batching raises B, which
is what 02 and 03 were for, but memory runs out long before the ratio does.

Speculative decoding attacks the numerator instead. A cheap drafter proposes k
tokens; the target verifies all k in one forward pass, because the candidates are
already known and can be scored in parallel exactly like a prefill. One weight
load now yields up to k+1 accepted tokens.

Crucially it is not an approximation:

  * greedy   -- accept a proposal only if it equals the target's own argmax, so
                the emitted sequence is byte-identical to plain greedy decoding
  * sampling -- accept with probability min(1, p_target/p_draft), else resample
                from the normalised residual max(0, p_target - p_draft). The
                accepted stream is drawn from exactly the target's distribution.

Two drafters are compared: a small model trained alongside the target, and prompt
lookup, which uses no model at all.

Acceptance rate is the whole game, and it only means something if the draft and
the target genuinely agree, so both are trained here for a few seconds on a
synthetic phrase language -- sequences built by concatenating short fixed phrases.
Inside a phrase the next token is determined and both models get it right; at a
phrase boundary it is a near-uniform choice and they disagree. That is the shape
of real text, most tokens easy and a few not, and it is the reason the technique
pays off at all.
"""

import time
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


def causal_mask(seq_len_q, seq_len_k, device):
    """Queries are the last seq_len_q positions of a seq_len_k history.

    Verification feeds several tokens against a cache that is already populated,
    so SDPA's is_causal flag cannot be used: it aligns the mask to the top left,
    which is only correct when the two lengths are equal.
    """
    offset = seq_len_k - seq_len_q
    q_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(seq_len_k, device=device).unsqueeze(0)
    return k_pos <= q_pos


class KVCache:
    """Pre-allocated cache that can also give tokens back.

    Rollback is what makes speculation possible: rejected proposals leave K/V
    behind that must disappear before the next round. Because `length` bounds
    every read, discarding them is a subtraction, not a memset.
    """

    def __init__(self, max_seq_len, num_kv_heads, d_k, dtype, device):
        shape = (1, num_kv_heads, max_seq_len, d_k)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0

    def update(self, K_new, V_new):
        end = self.length + K_new.shape[2]
        self.K[:, :, self.length:end] = K_new
        self.V[:, :, self.length:end] = V_new
        self.length = end
        return self.K[:, :, :end], self.V[:, :, :end]

    def rollback(self, length):
        self.length = min(self.length, length)

    def reset(self):
        self.length = 0


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
    """Verification needs a distribution at every fed position, so unlike the
    other projects this forward returns all of them. Passing no caches gives a
    plain batched forward, which is what training uses."""

    def __init__(self, vocab_size, d_model=128, num_heads=8, num_kv_heads=2,
                 num_layers=4, max_seq_len=1024):
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

    def make_caches(self, max_seq_len):
        weight = self.lm_head.weight
        return [
            KVCache(max_seq_len, b.attn.num_kv_heads, b.attn.d_k, weight.dtype, weight.device)
            for b in self.blocks
        ]

    def forward(self, input_ids, caches=None):
        """Returns [batch, seq_len, vocab] -- logits at every fed position."""
        offset = caches[0].length if caches else 0
        positions = torch.arange(offset, offset + input_ids.shape[1], device=input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        for block, cache in zip(self.blocks, caches or [None] * len(self.blocks)):
            x = block(x, cache)
        return self.lm_head(self.ln_f(x))


class PhraseLanguage:
    """Sequences made by concatenating short fixed phrases.

    Inside a phrase the next token is determined; at a boundary it is a near
    uniform choice among phrases. Easy tokens and hard tokens, in roughly the
    proportion that makes speculation worth doing.
    """

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


def train(model, language, steps, batch_size=16, seq_len=64, lr=3e-3, seed=0):
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


# ------------------------------------------------------------------------ sampling


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 0.95


def to_probs(logits, params):
    if params.temperature == 0.0:
        return None
    probs = torch.softmax(logits.float() / params.temperature, dim=-1)
    ordered, ids = torch.sort(probs, descending=True, dim=-1)
    keep = (ordered.cumsum(dim=-1) - ordered) < params.top_p
    ordered = ordered * keep
    return torch.zeros_like(probs).scatter_(-1, ids, ordered / ordered.sum(-1, keepdim=True))


def draw(probs, generator=None):
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


# ------------------------------------------------------------------------ drafters


class ModelDrafter:
    """A small model trained on the same data as the target.

    It runs gamma times per round, so it only pays off if it is far cheaper than
    the target and still agrees with it often.
    """

    label = "draft model"

    def __init__(self, model, max_seq_len, params):
        self.model = model
        self.params = params
        self.caches = model.make_caches(max_seq_len)
        self.forwards = 0

    def reset(self):
        for cache in self.caches:
            cache.reset()

    def rollback(self, length):
        for cache in self.caches:
            cache.rollback(length)

    def propose(self, tokens, gamma, generator=None):
        proposals, probs = [], []
        pending = tokens[self.caches[0].length:]  # tokens the draft has not cached yet
        for _ in range(gamma):
            fed = torch.tensor([pending], device=self.model.device)
            logits = self.model(fed, self.caches)[0, -1]
            self.forwards += 1

            p = to_probs(logits, self.params)
            token = int(logits.argmax()) if p is None else draw(p, generator)
            proposals.append(token)
            probs.append(p)
            pending = [token]
        return proposals, probs


class NgramDrafter:
    """Prompt lookup: no model at all.

    Find the most recent earlier occurrence of the current n-gram and propose
    whatever followed it. Free, and surprisingly effective whenever the output
    quotes the input -- summarisation, code editing, retrieval answers.
    """

    label = "prompt lookup"

    def __init__(self, n=3):
        self.n = n
        self.forwards = 0

    def reset(self):
        pass

    def rollback(self, length):
        pass

    def propose(self, tokens, gamma, generator=None):
        pattern = tokens[-self.n:]
        for start in range(len(tokens) - self.n - 1, -1, -1):
            if tokens[start:start + self.n] == pattern:
                found = tokens[start + self.n:start + self.n + gamma]
                if found:
                    # q is a point mass, so acceptance reduces to p_target(x).
                    return found, [None] * len(found)
        return [], []


# ---------------------------------------------------------------------- generation


@dataclass
class RunStats:
    label: str
    tokens: list[int] = field(default_factory=list)
    wall_time: float = 0.0
    target_forwards: int = 0
    drafter_forwards: int = 0
    proposed: int = 0
    accepted: int = 0

    @property
    def generated(self):
        return len(self.tokens)

    @property
    def acceptance(self):
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def per_forward(self):
        return self.generated / max(self.target_forwards, 1)

    @property
    def tpot_ms(self):
        return self.wall_time / max(self.generated, 1) * 1e3


@torch.no_grad()
def generate_baseline(model, prompt, params, generator=None):
    stats = RunStats("baseline")
    caches = model.make_caches(len(prompt) + params.max_new_tokens + 1)
    tokens = list(prompt)

    start = time.perf_counter()
    fed = torch.tensor([tokens], device=model.device)
    while len(tokens) - len(prompt) < params.max_new_tokens:
        logits = model(fed, caches)[0, -1]
        stats.target_forwards += 1
        p = to_probs(logits, params)
        token = int(logits.argmax()) if p is None else draw(p, generator)
        tokens.append(token)
        stats.tokens.append(token)
        fed = torch.tensor([[token]], device=model.device)

    stats.wall_time = time.perf_counter() - start
    return stats


@torch.no_grad()
def generate_speculative(model, drafter, prompt, params, gamma=4, generator=None):
    """Invariant, held at the top of every round: each cache covers all tokens
    except the last one, so the last token is always available to feed."""
    stats = RunStats(drafter.label)
    budget = len(prompt) + params.max_new_tokens + gamma + 2
    caches = model.make_caches(budget)
    drafter.reset()
    tokens = list(prompt)

    start = time.perf_counter()
    while len(tokens) - len(prompt) < params.max_new_tokens:
        proposals, draft_probs = drafter.propose(tokens, gamma, generator)
        stats.proposed += len(proposals)

        # One pass scores the last accepted token plus every proposal, giving
        # len(proposals) + 1 distributions for the price of a single weight load.
        fed = tokens[caches[0].length:] + proposals
        logits = model(torch.tensor([fed], device=model.device), caches)[0]
        stats.target_forwards += 1
        checked = logits[-(len(proposals) + 1):]

        accepted = 0
        correction = None
        for index, proposal in enumerate(proposals):
            target_p = to_probs(checked[index], params)
            if target_p is None:
                if int(checked[index].argmax()) == proposal:
                    accepted += 1
                    continue
                correction = int(checked[index].argmax())
                break

            draft_p = draft_probs[index]
            ratio = 1.0 if draft_p is None else float(
                target_p[proposal] / draft_p[proposal].clamp(min=1e-9)
            )
            if torch.rand((), generator=generator).item() < min(1.0, ratio):
                accepted += 1
                continue

            residual = target_p if draft_p is None else (target_p - draft_p).clamp(min=0)
            correction = draw(residual / residual.sum(), generator)
            break

        stats.accepted += accepted
        emitted = proposals[:accepted]
        if correction is None:
            # Everything was accepted, so the trailing distribution is a free token.
            bonus = to_probs(checked[len(proposals)], params)
            emitted.append(int(checked[len(proposals)].argmax()) if bonus is None
                           else draw(bonus, generator))
        else:
            emitted.append(correction)

        room = params.max_new_tokens - (len(tokens) - len(prompt))
        emitted = emitted[:room]
        tokens.extend(emitted)
        stats.tokens.extend(emitted)

        # Rejected proposals left K/V behind in both caches; drop it.
        for cache in caches:
            cache.rollback(len(tokens) - 1)
        drafter.rollback(len(tokens) - 1)

    stats.wall_time = time.perf_counter() - start
    stats.drafter_forwards = drafter.forwards
    return stats


def timed(make_run, repeats=2):
    """Wall-clock on a busy CPU is noisy; keep the fastest of a few runs."""
    best = None
    for _ in range(repeats):
        stats = make_run()
        if best is None or stats.wall_time < best.wall_time:
            best = stats
    return best


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    language = PhraseLanguage()
    params = SamplingParams(max_new_tokens=128, temperature=0.0)
    prompt_len = 64

    target = TinyLM(language.vocab_size, d_model=128, num_layers=6)
    # The draft has to be cheap enough that running it k times still costs less
    # than the target forwards it saves. Here it is roughly 1/24 of the target.
    draft = TinyLM(language.vocab_size, d_model=64, num_heads=4, num_kv_heads=1,
                   num_layers=1)
    target, draft = target.to(device), draft.to(device)

    start = time.perf_counter()
    target_loss = train(target, language, steps=200)
    draft_loss = train(draft, language, steps=200, seed=1)
    train_time = time.perf_counter() - start

    prompt = language.sample(prompt_len, torch.Generator().manual_seed(99))
    generate_baseline(target, prompt, SamplingParams(max_new_tokens=8))  # warm up

    baseline = timed(lambda: generate_baseline(target, prompt, params))

    runs = [baseline]
    budget = prompt_len + params.max_new_tokens + 16
    for gamma in (2, 4, 8):
        stats = timed(lambda k=gamma: generate_speculative(
            target, ModelDrafter(draft, budget, params), prompt, params, k
        ))
        stats.label = f"draft model, k={gamma}"
        runs.append(stats)

    for gamma in (4, 8):
        stats = timed(lambda k=gamma: generate_speculative(
            target, NgramDrafter(n=3), prompt, params, k
        ))
        stats.label = f"prompt lookup, k={gamma}"
        runs.append(stats)

    print(f"{prompt_len}-token prompt -> {params.max_new_tokens} tokens, greedy, {device}")
    print(f"6-layer d128 target / 1-layer d64 draft, trained {train_time:.0f}s on a "
          f"phrase language (loss {target_loss:.2f} / {draft_loss:.2f})")
    print()
    print(f"{'':<22}{'wall':>8}{'TPOT':>10}{'target fwd':>12}{'draft fwd':>11}"
          f"{'tok/fwd':>9}{'accept':>9}{'speedup':>9}")
    for stats in runs:
        print(f"{stats.label:<22}{stats.wall_time:>7.2f}s{stats.tpot_ms:>9.2f}ms"
              f"{stats.target_forwards:>12}{stats.drafter_forwards:>11}"
              f"{stats.per_forward:>9.2f}{stats.acceptance:>9.1%}"
              f"{baseline.wall_time / stats.wall_time:>9.2f}x")
    print()

    identical = all(r.tokens == baseline.tokens for r in runs[1:])
    print(f"identical output : {identical}")
    print("  greedy verification only accepts a proposal equal to the target's own")
    print("  argmax, so speculation changes the cost and never the answer")


if __name__ == "__main__":
    main()
