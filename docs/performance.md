# Performance

Every SimpleLLM vs vLLM number in the README, with the method behind it and every individual run.

All numbers are from one NVIDIA H100 80GB HBM3 (driver 580.95.05). SimpleLLM runs on torch 2.9.1, Triton 3.5.1 and flash-attn 2.8.3; vLLM 0.31.0 runs in its own environment. Both engines load the official `openai/gpt-oss-20b` and `openai/gpt-oss-120b` MXFP4 checkpoints, unmodified.

"Original SimpleLLM" is this repository before the single-file kernel rewrite: the same engine design, but using a copy of OpenAI's `triton_kernels` library (44 files, 5,762 lines) in place of `kernels.py`.

## How the decode benchmark works

- **One harness for every engine.** It uses 20 fixed prompts (97 to 471 tokens) in the Harmony chat template, greedy decoding, `ignore_eos` (every sequence produces exactly the requested number of tokens), a warmup run and CUDA graphs. Each engine runs in its own process.
- **Batch 1** runs 5 prompts one after another. **Batch 64** sends 64 requests at once, cycling through the prompts.
- **Decode tok/s = B x (N − 1) / (t_end − t_first)**: B sequences of N tokens each, timed from the first generated token to the last. The first token is excluded because it comes from prefill; time to first token (TTFT) is reported separately. The same formula and the same clock are used for every engine.
- **3 runs in separate processes**, engines interleaved run by run (SimpleLLM, vLLM, original, repeat). We report the median. Separate processes matter: Triton picks kernel configs per process, so repeated runs inside one process understate the real spread.
- **Noise.** We measured run-to-run spread before comparing anything: 0.65% (batch 1) and 0.97% (batch 64) for 1,000-token runs, about 2.2% for short 256-token screens. A difference is called a tie when it is within max(both engines' spreads, 2%).
- **Same output.** Every run produced exactly the requested tokens, and repeated runs of SimpleLLM produced identical tokens.

### How vLLM was tuned

For every scenario we searched vLLM's options and used the fastest configuration as the comparison:

- every MoE backend: `triton_kernels`, Triton unfused, Marlin, Humming, FlashInfer CUTLASS (FP8, TRT-LLM and DeepGEMM backends do not support this setup; batched Marlin is multi-GPU only)
- every attention backend: FlashAttention 3, Triton, FlashInfer
- 14 scheduler and graph options, including CUDA graph modes and sizes, `max_num_seqs`, `gpu_memory_utilization` and async scheduling (turning async scheduling off costs 30%)

The best settings change with the workload:

| scenario | best vLLM configuration |
|---|---|
| 20b, batch 1 | Marlin MoE + Triton attention |
| 20b, batch 64 | Triton MoE + Triton attention |
| 120b, batch 1 | Marlin MoE + Triton attention, `gpu_memory_utilization` 0.95 |
| 120b, batch 64 | Triton MoE + Triton attention |
| 131k context | FlashInfer or FlashAttention (2.1x faster than Triton attention at this length) |
| serving, 20b / 120b | Triton MoE + Triton attention / Triton MoE + FlashAttention, `gpu_memory_utilization` 0.95 |

The original SimpleLLM was tuned too: `max_num_seqs`, `max_seq_len` and allocator settings (all within noise), plus `TQDM_DISABLE=1`, because its per-step progress-bar update cost 0.38 ms per step.

## Decode, 1,000 output tokens

| model | batch | SimpleLLM (3 runs, spread) | tuned vLLM (3 runs, spread) | original SimpleLLM | vs vLLM | vs original |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 1 | **381.3** (380.8, 381.4, 381.3; 0.15%) | 330.9 (330.9, 331.4, 330.8; 0.20%) | 264.8 | **+15.2%** | +44.0% |
| gpt-oss-20b | 64 | **9,843.5** (9,842.8, 9,843.5, 9,852.0; 0.09%) | 8,880.7 (8,864.1, 8,880.7, 8,882.0; 0.20%) | 5,346.3 | **+10.8%** | +84.1% |
| gpt-oss-120b | 1 | **267.7** (267.2, 267.7, 267.7; 0.19%) | 232.9 (231.2, 232.9, 233.0; 0.76%) | 183.6 | **+14.9%** | +45.8% |
| gpt-oss-120b | 64 | **4,780.4** (4,780.4, 4,778.8, 4,782.5; 0.08%) | 4,269.5 (4,269.5, 4,269.2, 4,271.7; 0.06%) | 3,037.8 | **+12.0%** | +57.4% |

Output tokens per second. All four points are wins, each far outside the 2% noise band.

### Where the time goes in one decode step

Milliseconds per step, from profiles of each engine.

**gpt-oss-20b**

| section | original, b1 | vLLM, b1 | SimpleLLM, b1 | original, b64 | vLLM, b64 | SimpleLLM, b64 |
|---|---|---|---|---|---|---|
| RMSNorm + residual | 0.10 | 0.12 | 0.10 | 0.11 | 0.11 | 0.11 |
| QKV projection + RoPE | 0.42 | 0.42 | 0.36 | 1.05 | 0.44 | 0.39 |
| attention (core, sinks, output projection) | 1.11 | 0.53 | 0.46 | 3.78 | 1.12 | 1.06 |
| router + top-k | 0.29 | 0.33 | 0.22 | 0.32 | 0.32 | 0.27 |
| MoE gate/up + SwiGLU | 0.55 | 0.58 | 0.52 | 2.66 | 2.72 | 2.59 |
| MoE down + combine | 0.59 | 0.42 | 0.39 | 1.77 | 1.87 | 1.50 |
| logits + sampling | 0.42 | 0.46 | 0.42 | 0.57 | 0.53 | 0.47 |
| launch / host gaps | 0.39 | 0.17 | 0.17 | 1.75 | 0.12 | 0.21 |
| **step** | 3.85 | 3.04 | **2.65** | 12.02 | 7.22 | **6.59** |

**gpt-oss-120b** (before the routing fix described in [optimizations.md](optimizations.md), which cut router + top-k at batch 64 from 1.63 to 0.44 ms and the step from 14.44 to 13.18 ms)

| section | vLLM, b1 | SimpleLLM, b1 | vLLM, b64 | SimpleLLM, b64 |
|---|---|---|---|---|
| RMSNorm + residual | 0.177 | 0.144 | 0.174 | 0.159 |
| QKV projection + RoPE | 0.630 | 0.505 | 0.651 | 0.575 |
| attention | 0.794 | 0.648 | 1.674 | 1.548 |
| router + top-k | 0.469 | 0.349 | 0.523 | 1.626 |
| MoE gate/up + SwiGLU | 0.883 | 0.739 | 6.706 | 6.293 |
| MoE down + combine | 0.630 | 0.565 | 4.578 | 3.533 |
| logits + sampling | 0.440 | 0.380 | 0.536 | 0.446 |
| launch / host gaps | 0.280 | 0.434 | 0.245 | 0.259 |
| **step (unprofiled)** | 4.300 | **3.765** | 15.079 | 14.438 |

SimpleLLM is not faster because of one trick: almost every section is a little faster, and the original engine's large host overhead at batch 64 (1.75 ms per step) is gone.

## Maximum context: 131,072-token window

8 fixed coding prompts of about 122,000 tokens each, 8,192 output tokens, greedy, `ignore_eos`, 3 runs after warmup. SimpleLLM prefills long prompts in 4,096-token chunks. vLLM uses `max_model_len` 131072 with every MoE and attention backend screened at this length.

| model | batch | SimpleLLM tok/s | vLLM tok/s | difference | SimpleLLM TTFT | vLLM TTFT |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 1 | **265.7** (spread 0.03%) | 228.6 (3.66%) | **+16.2%** | 9.6 s | 6.1 s |
| gpt-oss-20b | 14 | **708.6** (0.02%) | 671.2 (0.75%) | **+5.6%** | 133 s | 70 s |
| gpt-oss-120b | 1 | **187.5** (0.2%) | 164.0 (0.1%) | **+14.3%** | 14.6 s | 9.0 s |
| gpt-oss-120b | 2 | **268.4** (0.02%) | 236.7 (0.09%) | **+13.4%** | 29.3 s | 14.2 s |

- **Batch sizes are the largest that both engines fit.** At 131k, 20b fits 16 or more sequences in SimpleLLM and about 14.8 in vLLM; 120b fits 2 in SimpleLLM and about 2.07 in vLLM. The original SimpleLLM cannot run at this length: it prefills the whole 122k-token prompt in one pass and runs out of memory.
- **Decode is attention-bound at this length.** SimpleLLM's decode-attention kernel reads the KV cache at 2.54 TB/s (batch 1) and 2.82 TB/s (batch 8), close to the H100's memory bandwidth. About 60 tile, split, warp and stage variants gained under 4%.
- **Time to first token is SimpleLLM's weak point here.** It prefills each long prompt alone, one 4,096-token chunk at a time; vLLM is 1.6x faster to the first token at batch 1 and about 2x at the larger batches.
- These numbers were measured on the kernel version just before the paged KV cache and the larger MoE prefill tile were added. On short decode screens those two changes measured within 0.5%; the long-context runs have not been repeated on the current code.

## Online serving

200 requests from a fixed trace: 40% chat prompts (97 to 471 tokens) and 60% coding prompts (1,000 to 8,000 tokens cut from real source files, no shared prefixes). 386,871 prompt tokens and 111,422 output tokens in total, with per-request output budgets of 100 to 1,000 tokens. Greedy, `ignore_eos`, requests arrive by a Poisson process at the listed rate, or all at once ("inf"). Both engines are driven in-process by the same client (SimpleLLM has no HTTP server), with 64 concurrent sequences and `max_seq_len` 9216. vLLM uses its streaming `AsyncLLM`, prefix caching off. Every run of both engines produced exactly 111,422 output tokens, and the two engines produced the same tokens at every rate.

TPOT (time per output token) = (t_end − t_first) / (n_out − 1) per request. E2E is measured from the scheduled arrival time.

**gpt-oss-20b**

| rate | engine | out tok/s (3 runs) | TTFT p50 / p90 / p99 | TPOT p50 / p90 / p99 | E2E p50 / p90 / p99 | verdict |
|---|---|---|---|---|---|---|
| all at once | SimpleLLM | **3,665.1** (3,665.1, 3,659.5, 3,669.7) | 11.4 / 22.0 / 24.6 s | 15.4 / 17.9 / 20.7 ms | 19.6 / 28.0 / 30.3 s | **1.149x, win** |
| all at once | vLLM | 3,189.7 (3,217.5, 3,189.7, 3,143.5) | 13.2 / 24.9 / 28.1 s | 17.0 / 19.5 / 22.9 ms | 22.3 / 32.1 / 34.8 s | |
| 2 req/s | SimpleLLM | 1,154.8 (1,154.9, 1,154.8, 1,154.8) | **44 / 173 / 262 ms** | **5.3 / 6.2 / 7.3 ms** | **2.9 / 4.9 / 6.2 s** | 1.009x, tie |
| 2 req/s | vLLM | 1,144.1 (1,144.2, 1,144.0, 1,144.1) | 105 / 297 / 410 ms | 7.1 / 8.0 / 9.1 ms | 3.9 / 6.6 / 7.8 s | |

At 2 req/s throughput is set by the arrival rate, so both engines tie on it; SimpleLLM is ahead on every latency percentile.

**gpt-oss-120b**

| rate | engine | out tok/s (3 runs) | TTFT p50 / p90 / p99 | TPOT p50 / p90 / p99 | E2E p50 / p90 / p99 | verdict |
|---|---|---|---|---|---|---|
| all at once | SimpleLLM | 1,797.0 (1,797.0, 1,795.9, 1,798.4) | 22.2 / 44.2 / 50.0 s | 32.2 / 35.8 / 41.5 ms | 39.1 / 57.8 / 61.8 s | 0.946x, loss |
| all at once | vLLM | **1,899.4** (1,900.7, 1,899.4, 1,898.6) | 20.2 / 40.0 / 45.5 s | 29.1 / 31.7 / 36.2 ms | 35.5 / 53.4 / 58.5 s | |
| 2.5 req/s | SimpleLLM | 1,326.9 | 108 / 354 / 525 ms | 25.3 / 29.4 / 32.3 ms | 12.8 / 22.4 / 28.3 s | 1.028x, tie |
| 2.5 req/s | vLLM | 1,290.7 (spread 3.2%) | 174 / 796 / 2,489 ms | 26.5 / 31.3 / 40.6 ms | 13.4 / 24.4 / 28.2 s | |
| 2 req/s | SimpleLLM | **1,111.8** | 93 / 333 / 481 ms | 18.0 / 20.6 / 22.5 ms | 9.0 / 16.0 / 19.7 s | **1.021x, win** |
| 2 req/s | vLLM | 1,089.4 | 138 / 318 / 437 ms | 19.1 / 21.5 / 22.6 ms | 9.8 / 17.1 / 20.4 s | |
| 1.5 req/s | SimpleLLM | **856.9** | 82 / 300 / 460 ms | 11.7 / 13.6 / 16.4 ms | 6.2 / 10.6 / 13.2 s | **1.021x, win** |
| 1.5 req/s | vLLM | 839.3 | 98 / 282 / 381 ms | 14.9 / 15.8 / 17.2 ms | 7.9 / 13.3 / 15.3 s | |
| 1 req/s | SimpleLLM | 582.3 | 74 / 280 / 428 ms | 7.7 / 9.4 / 12.1 ms | 4.2 / 7.0 / 8.6 s | 1.010x, tie |
| 1 req/s | vLLM | 576.5 | 85 / 260 / 340 ms | 11.1 / 13.9 / 15.5 ms | 6.0 / 10.3 / 13.8 s | |

- **Below saturation**, SimpleLLM has lower median TTFT, TPOT and end-to-end time at every rate. Its TTFT p90/p99 is higher at 1 to 2 req/s because a long prompt is prefilled in one go while decoding waits.
- **At saturation SimpleLLM loses by 5.4%.** Its 62.0 s run is 18.5 s of prefill that blocks decoding plus 43.5 s of decode (2,304 steps, mean batch 48.3). vLLM finishes in 58.6 s because its chunked prefill runs prompt chunks inside decode steps. Overlapping about 3.4 s of SimpleLLM's prefill would reach parity.
- **Prefill alone** (1 output token per request) takes 25.1 s in SimpleLLM vs 15.1 s in vLLM on 120b (15.4k vs 25.6k prompt tok/s), before the larger MoE prefill tile brought SimpleLLM's in-serving prefill to 18.5 s.

## 1,000 concurrent requests, and the numbers found online

We were asked whether our numbers agree with figures quoted online for vLLM serving gpt-oss-20b on an H100: 7,817 output tok/s with a mean TPOT of 91.0 ms at 1,000 requests, and "31.27 tok/s with 459.72 ms TTFT" for one 2,048-token prompt with 64 output tokens. The source gives no configuration, version or command, so we ran both workloads ourselves.

**Workload A:** 1,000 requests, each a random 1,000-token prompt (no chat template, like `vllm bench serve --dataset-name random`) with exactly 1,000 output tokens, all sent at once.

| engine | out tok/s (2 runs) | TTFT mean | TPOT mean | concurrency |
|---|---|---|---|---|
| figure found online (vLLM, unverified) | 7,817 | 4.9 s | 91.0 ms | unknown |
| vLLM defaults, over HTTP (`vllm serve` + `vllm bench serve`) | 10,244 / 9,633 | 10.1 / 9.0 s | 85.8 / 90.3 ms | up to 1,000 |
| vLLM defaults, in-process | 22,452 / 22,513 | 3.1 / 2.8 s | 41.5 / 41.6 ms | 1,000 |
| vLLM with the batch-64 tuned settings, in-process | 5,916 / 5,929 | 79 / 79 s | 10.4 / 10.3 ms | 64 |
| SimpleLLM, 256 slots | 12,224 / 13,141 | 35 / 31 s | 18.3 / 17.3 ms | 256 |
| SimpleLLM, 64 slots | 7,299 / 7,319 | 64 / 64 s | 7.9 / 7.9 ms | 64 |

**Workload B:** one request, 2,048-token prompt, 64 output tokens, 3 runs.

| engine | TTFT | time per output token | tok/s per request |
|---|---|---|---|
| figure found online (unverified) | 459.72 ms | "31.27" | |
| vLLM defaults | 28.5 to 30.8 ms | 3.2 ms | 275 to 278 |
| SimpleLLM | 44.9 to 46.4 ms | **2.7 ms** | **296 to 299** |

What this shows:

1. **The online 7,817 tok/s is an HTTP number, and it agrees with ours.** vLLM behind `vllm serve` gives 9,633 to 10,244 tok/s with a mean TPOT of 86 to 90 ms (online: 91.0 ms). The same engine driven in-process gives 22.5k tok/s, 2.2x more: with 1,000 streams, the API server and benchmark client (streaming, detokenization, request handling on the CPU) are the limit. We did not separate the server's share from the client's. All engine-vs-engine numbers in this document use the same in-process client for both engines.
2. **The "31.27" single-stream figure does not match this setup.** Both engines take about 3 ms per token, roughly 10x faster; that figure probably comes from a different, slower configuration.
3. **At equal concurrency SimpleLLM is faster:** 7,309 vs 5,923 tok/s at 64 sequences (1.23x), TPOT 7.9 vs 10.4 ms.
4. **At 1,000-way concurrency SimpleLLM loses:** 12.2k to 13.1k tok/s at 256 slots vs vLLM's 22.5k (0.54x to 0.58x). The causes are concurrency (256 vs about 1,000 sequences decoding at once; 256 / 13.9 ms per step caps SimpleLLM at 18.4k tok/s) and prefill blocking decode (20 to 26 s of each 76 to 82 s run). SimpleLLM's per-step speed is not the cause.
5. **Why only 256 slots:** a 1,024-slot engine runs out of memory at startup while capturing CUDA graphs. The engine keeps a fixed 4 GiB reserve for activations and graphs, but captures one graph per batch size up to `max_num_seqs`, so the reserve is too small for large slot counts. See What's Next in the README.

## Raw data

The benchmark harness, frozen prompt sets, serving trace and raw per-run results were produced alongside this work but are not included in this repository.
