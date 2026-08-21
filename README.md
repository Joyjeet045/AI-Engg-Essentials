# LLM Inference Optimisation, From Scratch

Ten self-contained PyTorch projects that build from a bare KV cache up to a full
serving stack. Each is a single runnable file with a `main()` that prints a
comparison table and a correctness check, plus a README explaining the principle.

They're meant to be read in order. Each project ends where the next one begins.

| | Project | Fixes | Headline |
|---|---|---|---|
| 01 | [KV Cache](01-kv-cache) | $O(n^2)$ recomputation | 4× smaller cache with GQA, identical output |
| 02 | [Continuous Batching](02-continuous-batching) | idle batch slots | 1.6× throughput, 43% waste removed |
| 03 | [PagedAttention](03-paged-attention) | reserved-but-unused memory | 8 concurrent sequences instead of 1 |
| 04 | [vLLM Architecture](04-vllm-architecture) | everything, assembled | scheduler + cache manager + sharded workers |
| 05 | [Speculative Decoding](05-speculative-decoding) | one token per weight load | 2.4 tokens per target forward |
| 06 | [Quantization](06-quantization) | bytes per weight and per token | 3.6× weights, 4× cache, argmax intact |
| 07 | [Prefix Caching](07-prefix-caching) | recomputing shared prompts | 79% of prompt tokens skipped |
| 08 | [Benchmark Harness](08-benchmark-harness) | single-number benchmarks | goodput collapses while tok/s rises |
| 09 | [Fused Paged Attention](09-fused-paged-attention) | materialising the whole context | 992× less peak memory, exact |
| 10 | [Disaggregated Serving](10-disaggregated-serving) | prefill blocking decode | 2.6× better TTFT p99 |

## The arc

**01–04 build a server.** Cache K and V so each decode step costs a constant
amount of work. Batch the streams and re-form the batch every step so no slot
idles. Replace fixed slots with paged allocation so memory stops capping
concurrency. Then assemble the real thing — scheduler, KV cache manager over GPU
and CPU pools, tensor-parallel workers.

**05–07 make it cheaper.** Verify many tokens per weight load instead of one.
Store weights and cached tokens in fewer bits. Stop recomputing prefixes that
have already been through the model.

**08 measures it properly.** Poisson arrivals, percentiles, and goodput under an
SLO — the instrument every other project should be judged with.

**09–10 remove the last two compromises.** Stream the block table inside the
attention loop instead of materialising the context. Split prefill and decode
onto separate workers so they stop fighting.

## Running

```powershell
cd 01-kv-cache;               python KV_Cache.py
cd ..\02-continuous-batching; python InFlight_Batching.py
cd ..\03-paged-attention;     python PagedAttention.py
cd ..\04-vllm-architecture;   python vLLM_Architecture.py
cd ..\05-speculative-decoding;    python speculative_decoding.py
cd ..\06-quantization;        python quantization.py
cd ..\07-prefix-caching;      python prefix_caching.py
cd ..\08-benchmark-harness;   python benchmark.py
cd ..\09-fused-paged-attention;   python fused_paged_attention.py
cd ..\10-disaggregated-serving;   python disaggregated.py
```

Requires PyTorch (tested on 2.12, CPU). Every folder is standalone — a few small
tensor helpers are repeated on purpose so you can lift a folder out and it still
runs. CUDA and bfloat16 are picked up automatically when a GPU is present.

Projects 05 and 06 train a small model for ~20 seconds first, on a synthetic
phrase language. That isn't decoration: acceptance rate and quantization error are
both meaningless on random weights, because untrained models agree at chance and
their distributions are too flat for rounding to change anything.

## Reading the output

Most demos print an `identical output` line. Nearly every optimisation here
changes the *cost* of generation and not the answer, so that line is the real
test — if a mask, a block table, a rollback or an all-reduce is wrong, the tokens
diverge and it prints `False`.

Project 06 is the deliberate exception. Quantization changes the answer by
construction, so it reports **top-1 agreement** and **KL divergence** against fp32
instead. Project 09 reports max absolute error, which sits at fp32 rounding
because the online-softmax recurrence is exact.
