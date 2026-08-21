# LLM Inference Optimisation, From Scratch

Four self-contained PyTorch projects that build up to the vLLM architecture. Each
one is a single runnable file with a `main()` that prints a comparison table and a
correctness check, plus a README explaining the principle.

They are meant to be read in order. Each project ends where the next one begins.

| | Project | Fixes | Headline |
|---|---|---|---|
| 01 | [KV Cache](01-kv-cache) | $O(n^2)$ recomputation | 4× smaller cache with GQA, identical output |
| 02 | [Continuous Batching](02-continuous-batching) | idle batch slots | 1.6× throughput, 43% waste removed |
| 03 | [PagedAttention](03-paged-attention) | reserved-but-unused memory | 8 concurrent sequences instead of 1 |
| 04 | [vLLM Architecture](04-vllm-architecture) | everything, assembled | scheduler + cache manager + sharded workers |

## The progression

**01** caches K and V so each decode step costs a constant amount of work instead
of a growing one. That makes single-stream generation fast, but a server has many
streams.

**02** batches those streams. Static batching wastes nearly half its compute on
sequences that already finished, so the batch is re-formed every step instead.
That fills the batch — but every sequence now reserves a worst-case slot, so
memory caps concurrency.

**03** replaces those slots with paged allocation. Blocks are handed out on demand
and addressed through a per-sequence block table, which also makes prefix sharing
and preemption possible.

**04** assembles the real system: a scheduler with waiting / running / swapped
queues, a KV cache manager over GPU and CPU block pools, and tensor-parallel
workers each holding a shard of the weights and the matching shard of the cache.

## Running

```powershell
cd 01-kv-cache;            python KV_Cache.py
cd ..\02-continuous-batching; python InFlight_Batching.py
cd ..\03-paged-attention;  python PagedAttention.py
cd ..\04-vllm-architecture; python vLLM_Architecture.py
```

Requires PyTorch (tested on 2.12, CPU). Every folder is standalone — a few small
tensor helpers are repeated on purpose so you can lift a folder out and it still
runs. All demos pick up CUDA and bfloat16 automatically if a GPU is present.

## Reading the output

Every demo prints an `identical output` line. Each optimisation here changes the
*cost* of generation, never the answer, so that line is the real test — if a mask,
a block table or an all-reduce is wrong, the tokens diverge and it prints `False`.
