# Correctness

How we checked that SimpleLLM computes the same model as OpenAI's reference, and not just a fast approximation of it.

## No re-quantization

Both SimpleLLM and vLLM load the original OpenAI checkpoints. `config.json` declares `quant_method: mxfp4`: the MoE expert weights ship in MXFP4, while attention, the router, embeddings and the output layer are bf16. `model/model.py` reads the published `*_blocks` and `*_scales` tensors and only reorders their layout in memory; `kernels.py` turns MXFP4 into bf16 inside the matmul. There is no FP8, INT8 or other re-quantization anywhere in the code.

After a generation, the loaded Python modules include no `vllm` and no `triton_kernels`; `flash_attn` is the only GPU library used besides PyTorch and Triton (for prefill attention).

## Kernel tests

121 checks cover every kernel at gpt-oss-20b and real gpt-oss-120b layer-0 shapes: token counts up to 1,024, imbalanced expert routing (including batch 64), ring-buffer and paged decode. Each kernel is compared against a PyTorch float32 reference and against the original `triton_kernels` implementation.

Negative controls: deliberately broken versions of each kernel must fail these tests, and all of them do.

## Teacher-forced greedy match

20 prompts x 1,000 steps. At every step the model is fed the reference's previous token (teacher forcing), and we check whether its own top choice matches the reference's. The reference is the original SimpleLLM in bf16. Two engines in bf16 that add numbers in a different order will disagree occasionally when two tokens are almost tied, so we set limits from the reference engine's own variation (single vs batched) before running the test:

| model | SimpleLLM, batch 1 | SimpleLLM, batch 20 | limit | result |
|---|---|---|---|---|
| gpt-oss-20b | 0.875% of steps | 0.870% | 0.895% | pass |
| gpt-oss-120b | 1.465% | 1.655% | 3.27% | pass |

For scale, the original engine disagrees with its own frozen output on 0.78% to 0.87% of steps depending on mode.

A second, stricter rule requires the reference's top-2 margin at every disagreement to be small (a near-tie). 120b passes (every margin ≤ 1.75, limit 4.0). 20b fails it at one step out of 20,000, which we diagnosed with a float32 Hugging Face gpt-oss reference:

- **Batch 1, prompt 12, step 503:** a precision tie. In fp32 the reference token wins by 0.061 logits; in bf16 (both Hugging Face's and ours) the two tokens are exactly tied.
- **Batch 20, prompt 19, step 896:** a real one-step deviation. fp32 and Hugging Face bf16 prefer the reference token by 1.00 logit, but SimpleLLM's batch-20 run puts the other token 0.75 above it. This is the one open correctness item; it has been present since an early kernel version.

## Accuracy against OpenAI's published scores

OpenAI's own eval harness (`gpt_oss/evals` from github.com/openai/gpt-oss): its question sets, answer permutations, prompt templates and graders, temperature 1.0, 8 repeats per question, low reasoning effort. Both engines got byte-identical prompt tokens and were called in-process with no retries. Output was capped at 32,768 tokens (20b) and 16,384 (120b); no reply came close to the cap. Intervals are 95% bootstrap intervals over questions. Published values are from the gpt-oss model card (arXiv 2508.10925), Table 3, without tools.

| model | benchmark | SimpleLLM | vLLM 0.31.0 | published | SimpleLLM − vLLM, paired |
|---|---|---|---|---|---|
| 20b | GPQA Diamond, 198 questions x 8 | 56.3% [51.5, 61.1] | 56.8% [51.9, 61.5] | 56.8 | −0.4 pt [−3.0, +2.0] |
| 20b | AIME 2025, 30 questions x 8 | 36.3% [23.3, 49.2] | 37.9% [25.0, 50.8] | 37.1 | −1.7 pt [−9.6, +5.4] |
| 120b | GPQA Diamond, 198 questions x 8 | 66.8% [61.6, 71.8] | 65.5% [60.4, 70.7] | 67.1 | +1.3 pt [−0.9, +3.4] |
| 120b | AIME 2025, 30 questions x 8 | 50.8% [36.3, 65.0] | 51.7% [37.5, 65.4] | 50.4 | −0.8 pt [−7.5, +5.0] |

All four match: the published value is inside both engines' intervals, and no difference between the engines is significant. Medium and high reasoning effort were not run.

**Controls.** A deliberately broken kernel (MXFP4 scale exponent off by one) scores 22% [12, 34] on 50 GPQA questions, which is chance (25%): its replies are gibberish, so the pipeline does detect a numerics bug.

**Replies without a final answer:** SimpleLLM 3 of 1,584 GPQA and 7 of 240 AIME (20b), vLLM 4 and 0. Four per engine are attempted tool calls (the model asks for Python, but no tools are offered). Six of SimpleLLM's come from its stop rule, which ends a reply at the second `<|end|>` token; a reply with two reasoning segments gets cut off, while vLLM keeps going until `<|return|>`. All six were already incoherent and the effect (at most 2.5 points on AIME) is inside the interval, but it is a real behaviour difference (see What's Next in the README).

## Changes that alter numerics

Some optimizations change the order in which numbers are added. Each was checked before being adopted:

- **Paged KV cache.** On the serving trace, 198 of 200 requests are token-identical to the previous engine. The other two are prompts just over 4,096 tokens, whose tiny second prefill chunk goes through flash-attn's paged path. flash-attn picks how to split the work based on the cache length it is given, so partial sums are added in a different order (max logit difference 0.56, same top token at that step). Against fp32 both versions are equally accurate (per-layer max error 0.062 to 0.072 vs 0.059 to 0.062), so this is rounding, not a bug.
- **CUDA graphs for any slot set.** Replaying graphs on scattered, unsorted slots at batch 1 to 20, with contexts up to 6,605 tokens (wrapping the ring buffer), is bit-identical to eager decode. Replaying with stale slot indices is detected.
- **Compact sliding-window cache.** flash-attn's paged ring vs a dense cache on a 13.5k-token prefill: output and log-sum-exp bit-identical. 131k decode produces the same tokens as before.
- **Routing tile table and MoE prefill tile.** Generated tokens identical to the previous kernels on both models.

Batch size itself changes numerics (it changes kernel tiling), in SimpleLLM and in vLLM alike, so the same prompt can produce different text at batch 20 than at batch 64.
