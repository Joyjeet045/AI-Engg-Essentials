"""Paged attention without materialising the gather.

Projects 03, 04 and 07 all read the cache the same way: follow the block table,
pull every block into one contiguous [n, heads, ctx, d] tensor, then run softmax
attention over it. That is correct and it is what makes the code readable, but it
allocates the whole context twice (K and V) plus an [n, heads, s_q, ctx] score
matrix, every layer, every step.

A real kernel never does that. It walks the block table *inside* the attention
loop, processing a tile of keys at a time and folding each tile into a running
result with the online-softmax recurrence:

    m_new = max(m, rowmax(s))            running maximum
    alpha = exp(m - m_new)               rescale what came before
    l     = alpha * l + rowsum(p)        running denominator
    acc   = alpha * acc + p @ v          running numerator
    out   = acc / l

Peak memory then depends on the *tile*, not on the context length, which is the
whole trick behind FlashAttention and behind vLLM's paged kernel.

Tile size is the knob. One block per tile minimises memory and maximises loop
overhead; a tile as large as the context is exactly the gather baseline. Real
kernels sit in between -- FlashAttention tiles are 64 to 128 keys regardless of
the 16-token page size, because the tile is a compute-efficiency choice and the
page is a memory-management one.

Honest caveat: the loop here is Python, so small tiles are slower in wall-clock
even though they are strictly better in memory. The algorithm is right; the
language is wrong. In Triton or CUDA the loop lives in registers and SRAM, which
is where the speed comes from.
"""

import math
import time

import torch


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


BLOCK_SIZE = 16


def make_cache(num_blocks, num_kv_heads, d_k, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (num_blocks, BLOCK_SIZE, num_kv_heads, d_k)
    return (torch.randn(shape, generator=generator),
            torch.randn(shape, generator=generator))


def load_tile(cache, block_ids, n_rep):
    """[n, tile] block numbers -> [n, heads, tile * BLOCK_SIZE, d_k]"""
    blocks = cache[block_ids]
    n, tile, bs, h, d = blocks.shape
    return repeat_kv(blocks.reshape(n, tile * bs, h, d).permute(0, 2, 1, 3), n_rep)


def gather_attention(Q, cache_K, cache_V, block_table, positions, n_rep):
    """What 03/04/07 do: pull the whole context in, then attend over it."""
    K = load_tile(cache_K, block_table, n_rep)
    V = load_tile(cache_V, block_table, n_rep)

    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    key_pos = torch.arange(K.shape[2], device=Q.device)
    allowed = key_pos <= positions.unsqueeze(-1)
    scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), V)


def flash_paged_attention(Q, cache_K, cache_V, block_table, positions, n_rep,
                          tile_blocks=4):
    """Stream the block table, folding each tile in with an online softmax.

    Nothing the size of the context is ever allocated.
    """
    n, heads, seq_q, d_k = Q.shape
    device = Q.device
    scale = 1.0 / math.sqrt(d_k)

    # A finite floor rather than -inf keeps the rescale well defined on the first
    # tile and on tiles that are entirely masked out.
    running_max = torch.full((n, heads, seq_q, 1), -1e30, device=device)
    running_sum = torch.zeros((n, heads, seq_q, 1), device=device)
    accumulator = torch.zeros((n, heads, seq_q, d_k), device=device)

    num_blocks = block_table.shape[1]
    for start in range(0, num_blocks, tile_blocks):
        block_ids = block_table[:, start:start + tile_blocks]
        K = load_tile(cache_K, block_ids, n_rep)
        V = load_tile(cache_V, block_ids, n_rep)

        key_pos = torch.arange(start * BLOCK_SIZE, start * BLOCK_SIZE + K.shape[2],
                               device=device)
        allowed = key_pos <= positions.unsqueeze(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))

        tile_max = torch.maximum(running_max, scores.amax(dim=-1, keepdim=True))
        rescale = torch.exp(running_max - tile_max)
        weights = torch.exp(scores - tile_max)

        running_sum = rescale * running_sum + weights.sum(dim=-1, keepdim=True)
        accumulator = rescale * accumulator + torch.matmul(weights, V)
        running_max = tile_max

    return accumulator / running_sum.clamp(min=1e-20)


def peak_intermediate_bytes(n, heads, seq_q, tile_keys, d_k, element=4):
    """Largest set of temporaries alive at once, from shapes alone."""
    kv_tile = 2 * n * heads * tile_keys * d_k * element
    scores = n * heads * seq_q * tile_keys * element
    running = n * heads * seq_q * (d_k + 2) * element
    return kv_tile + scores + running


def timed(call, repeats=3):
    call()  # warm up
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def run_case(n, heads, num_kv_heads, d_k, seq_q, context, tiles):
    n_rep = heads // num_kv_heads
    num_blocks = context // BLOCK_SIZE
    cache_K, cache_V = make_cache(n * num_blocks, num_kv_heads, d_k)

    generator = torch.Generator().manual_seed(1)
    Q = torch.randn((n, heads, seq_q, d_k), generator=generator)
    # Each sequence owns a disjoint, deliberately shuffled set of blocks, so the
    # gather is doing real scattered reads rather than a contiguous slice.
    block_table = torch.randperm(n * num_blocks, generator=generator).view(n, num_blocks)
    positions = torch.arange(context - seq_q, context).unsqueeze(0).expand(n, seq_q)

    reference = gather_attention(Q, cache_K, cache_V, block_table, positions, n_rep)
    baseline_peak = peak_intermediate_bytes(n, heads, seq_q, context, d_k)
    rows = [(
        "gather (materialised)",
        0.0,
        baseline_peak,
        1.0,
        timed(lambda: gather_attention(Q, cache_K, cache_V, block_table, positions, n_rep)),
    )]

    for tile in tiles:
        out = flash_paged_attention(Q, cache_K, cache_V, block_table, positions,
                                    n_rep, tile)
        error = float((out - reference).abs().max())
        peak = peak_intermediate_bytes(n, heads, seq_q, tile * BLOCK_SIZE, d_k)
        rows.append((
            f"streaming, tile {tile * BLOCK_SIZE:>4} keys",
            error,
            peak,
            baseline_peak / peak,
            timed(lambda t=tile: flash_paged_attention(
                Q, cache_K, cache_V, block_table, positions, n_rep, t)),
        ))
    return rows


def main():
    torch.manual_seed(0)
    n, heads, num_kv_heads, d_k = 4, 8, 2, 64
    tiles = (1, 4, 16, 64)

    try:
        import triton  # noqa: F401
        backend = "triton is available; the same loop belongs in a @triton.jit kernel"
    except ImportError:
        backend = "triton not installed; the tile loop runs in Python here"

    print(f"paged attention, {n} sequences, {heads} q-heads / {num_kv_heads} kv-heads, "
          f"d_k {d_k}, page {BLOCK_SIZE}")
    print(backend)

    for seq_q, context in ((1, 4096), (128, 4096), (1, 16384)):
        phase = "decode" if seq_q == 1 else "prefill chunk"
        print()
        print(f"{phase}: {seq_q} query token(s) over {context} cached keys")
        print(f"  {'':<26}{'max |err|':>12}{'peak temp':>12}{'smaller':>10}{'time':>11}")
        for label, error, peak, ratio, seconds in run_case(n, heads, num_kv_heads, d_k,
                                                           seq_q, context, tiles):
            print(f"  {label:<26}{error:>12.2e}{peak / 2**20:>11.2f}M"
                  f"{ratio:>9.0f}x{seconds * 1e3:>9.2f}ms")

    print()
    print("Error is at fp32 rounding, so the recurrence is exact -- the running")
    print("rescale is algebraically identical to one big softmax.")
    print("Memory follows the tile and ignores the context, which is the point.")
    print("Small tiles are slower only because the loop is Python; a Triton or CUDA")
    print("kernel keeps the tile in SRAM and never writes it out at all.")


if __name__ == "__main__":
    main()
