# Optimizations

How SimpleLLM got faster, in the order the changes landed. Every gain is end-to-end unless marked as a micro-benchmark.

## Starting point

The original SimpleLLM used a 44-file, 5,762-line copy of OpenAI's `triton_kernels` library for its MoE, plus separate kernels for RMSNorm and RoPE. At 1,000 output tokens it ran 262 tok/s (20b, batch 1) and 5,369 tok/s (20b, batch 64), against tuned vLLM's 327 and 8,864. Its profile showed slow attention (1.11 ms vs vLLM's 0.53 at batch 1) and large host overhead at batch 64 (1.75 ms per step).

## What worked

| # | change | measured effect |
|---|---|---|
| 1 | **MXFP4 MoE on tensor cores at any batch size.** At load time each expert's 4-bit weights are reordered into the order tensor-core fragments expect, and the bits of 4 consecutive values are interleaved so turning them into bf16 takes a few bit operations and one multiply per 2 values. One grouped kernel handles any number of tokens. | 1.33x on the MoE at batch 1, 1.10x at batch 64 |
| 2 | **One decode-attention kernel.** Split-KV (flash-decoding) attention that also handles GQA, the sliding window, appending the new K/V and gpt-oss's attention sinks, replacing four separate ops. | +33% at batch 1, +36% at batch 64 |
| 3 | **RoPE as one fused kernel at every batch size.** | +8% at batch 64 |
| 4 | **Greedy sampling inside the CUDA graph.** The argmax and the next step's input ids and positions are written inside the captured graph, so a decode step never waits on Python. | +20% at batch 64 |
| 5 | **Fixed decode tile shapes.** Triton's autotuner tuned the MoE kernel on the CUDA-graph warmup batch, where every row routes to the same few experts, and picked different tiles in different processes. Pinning the measured best tile for decode removed the variance. | +3.4% at 120b batch 64 |
| 6 | **Routing tile table per expert.** The routing kernel built its table of (expert, rows) tiles by searching a [tiles x experts] grid (256 x 128 at 120b batch 64). It now writes each expert's tiles directly. | 24.8 → 3.1 µs per call; router step 1.63 → 0.44 ms; +9.6% at 120b batch 64 |
| 7 | **Compact sliding-window KV cache.** Half of gpt-oss's layers only attend to the last 128 tokens, but every layer stored the full length. Sliding layers now keep a small ring buffer. | KV per 131k sequence 9.66 → 5.00 GB (120b), 6.44 → 3.33 GB (20b); sequences that fit at 131k: 1 → 2 (120b), 9 → 16+ (20b) |
| 8 | **CUDA graphs for any set of active slots.** Graphs used to replay only when the busy slots were exactly 0..B−1. Under continuous batching, slots free up out of order, so only 5 to 20% of serving steps used a graph; eager steps cost 16.6 to 19.1 ms vs 2.7 to 6.5 ms for graphs. Slot indices are now a graph input, and every graph is captured at startup. | serving 1.56x (20b, all at once); TPOT p50 at 2 req/s 27.4 → 5.8 ms |
| 9 | **Paged KV cache for full-attention layers.** The engine reserved a worst-case 9,216-row cache per request (509 MB at 120b), so only 19 to 20 requests fit next to 61 GB of weights. Full-attention layers now draw 256-row pages from a shared pool, and a request reserves pages for its prompt plus `max_tokens`. | 120b serving 1.47x (64 concurrent requests instead of 19) |
| 10 | **MoE token tile follows tokens per expert.** The MoE always used 16-token tiles, sized for decode, so during 120b prefill (about 128 tokens per expert) each weight tile was unpacked about 8 times. The tile now grows to 32 or 64 for prefill; decode keeps 16. | 4,096-token prefill MoE 236 → 151 ms; 120b serving +12.6%, 20b serving +18% |

Item 8 also fixed a latent bug: capturing a graph during serving appends K/V at row `max_seq_len − 4`, which on the ring-buffer layers can land inside a live request's window. Capturing all graphs at startup, before any request holds a slot, avoids it.

Matching the original engine's bf16 rounding points (in RMSNorm, RoPE, the attention sinks and the MoE output) made no difference to speed.

## What did not work

- **Per-token MXFP4 GEMV** (lookup tables, an fp16 bit trick, an fp32 denormal trick): at best 1.10x over the original MoE at one token, too small to keep.
- **Tensor-core MXFP4 without reordering the weights at load time** (inline PTX + `tl.dot`, and `tl.dot_scaled`): slower, because the data had to be rearranged inside the kernel on every call.
- **Tuning decode attention at 131k:** about 60 tile, split, warp and stage variants and a restructured loop gained under 4%. The kernel already reads the KV cache at 2.5 to 2.8 TB/s, close to the H100's limit.

## Things we learned that are not widely known

- **Over HTTP, vLLM's serving path costs about 2x throughput at 1,000 concurrent streams.** The same vLLM engine does 22,482 output tok/s in-process but 9,633 to 10,244 tok/s behind `vllm serve` + `vllm bench serve`. The commonly quoted "~7.8k tok/s on H100" is in this HTTP setting, not the engine's limit.
- **The best attention backend changes with context length.** At 1,000 tokens vLLM's Triton attention is fastest; at 131k, FlashInfer and FlashAttention are 2.1x faster than it. A config tuned at short context is badly tuned for long context.
- **Benchmark in separate processes.** Triton autotuning picks kernels per process, so repeated runs inside one process understate the real run-to-run spread.
- **The progress bar mattered.** The original engine's per-step `tqdm` update cost 0.38 ms per step, about 10% at batch 1.
