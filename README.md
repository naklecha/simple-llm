# SimpleLLM

SimpleLLM is an inference engine for OpenAI's gpt-oss-20b and gpt-oss-120b on a single NVIDIA H100. It is 1,351 lines of Python: `llm.py` (649 lines) is the engine, `model/` (405) is the model and tokenizer, and `kernels.py` (297) holds every GPU kernel, written in Triton.

It loads the official MXFP4 checkpoints without re-quantizing them, and its GPQA-Diamond and AIME 2025 scores match OpenAI's published numbers.

Compared with vLLM 0.31.0 tuned for each workload, SimpleLLM decodes 11-15% faster at batch 1 and batch 64 on both models, and 6-16% faster at 131k context. vLLM is faster at time to first token on long prompts, at serving 120b when all requests arrive at once, and at 1,000 concurrent requests; see [What's Next](#whats-next). Method and every run: [docs/performance.md](docs/performance.md).

<p align="center">
  <img src="docs/images/performance.svg" width="900"
       alt="SimpleLLM vs tuned vLLM 0.31.0 on one H100. Decode, 1,000 tokens: 20b batch 1 381 vs 331 tok/s (+15.2%), 20b batch 64 9,844 vs 8,881 (+10.8%), 120b batch 1 268 vs 233 (+14.9%), 120b batch 64 4,780 vs 4,270 (+12.0%). Max context: faster decode at every batch, 1.6x slower time to first token. Serving: 20b +14.9%, 120b -5.4% when all requests arrive at once, tie to +2.1% at 1-2.5 req/s. 1,000 requests: +23.4% at equal concurrency, -43.6% at each engine's max. Accuracy matches OpenAI's published GPQA and AIME scores.">
</p>

## Quickstart

Requirements: one NVIDIA H100 (80 GB), CUDA 12.8+, Python 3.12. The only GPU dependencies are PyTorch, Triton and the prebuilt flash-attn wheel (used for prefill attention).

```bash
./setup.sh ./gpt-oss-120b          # Python 3.12 venv, torch 2.9.1 (cu128), flash-attn 2.8.3, downloads gpt-oss-120b
source ./.venv/bin/activate
hf download openai/gpt-oss-20b --local-dir ./gpt-oss-20b --exclude "original/*" --exclude "metal/*"   # optional: the 20b model
```

```python
from llm import LLM

engine = LLM("./gpt-oss-20b")      # or "./gpt-oss-120b"
outputs = engine.generate(["What is the meaning of life?"], max_tokens=100).result()
print(outputs[0].text)
engine.stop()
```

The full API (constructor options, `chat`, sampling settings, output fields) is in [docs/api.md](docs/api.md); more examples are in `cookbook/`.

## Where the speedup comes from

Each change below was measured end to end when it landed. Details, and the ideas that did not work: [docs/optimizations.md](docs/optimizations.md).

| change | gain |
|---|---|
| MXFP4 MoE on tensor cores at any batch size (weights reordered once at load time) | MoE 1.33x at batch 1, 1.10x at batch 64 |
| one decode-attention kernel for sinks, sliding window, GQA and the KV append (replaces four ops) | +33% at batch 1, +36% at batch 64 |
| greedy argmax and next-step inputs computed inside the CUDA graph | +20% at batch 64 |
| RoPE as one fused kernel at every batch size | +8% at batch 64 |
| routing writes its tile table per expert instead of searching a grid | 24.8 → 3.1 µs per call, +9.6% at 120b batch 64 |
| fixed decode tile shapes instead of per-process autotuning | +3.4% at 120b batch 64, and no run-to-run variance |
| CUDA graphs for any set of active slots, all captured at startup | serving 1.56x |
| paged KV cache for full-attention layers | 120b serves 64 requests instead of 19, serving 1.47x |
| ring buffer for sliding-window layers | KV per 131k sequence halved, 2x the sequences at 131k |
| MoE token tile grows with tokens per expert during prefill | 4k-token prefill MoE 236 → 151 ms, 120b serving +12.6% |

## How we measured

- **Same conditions for both engines:** the same prompts and chat template, greedy decoding, `ignore_eos` (every sequence produces exactly the requested tokens), warmup, CUDA graphs, each engine in its own process.
- **Same metric:** decode tok/s = sequences x (tokens − 1) / (time of last token − time of first token). Time to first token is reported separately.
- **vLLM tuned for every case:** every MoE backend, every attention backend and 14 scheduler and graph options; the fastest configuration is the comparison.
- **3 runs per point in separate processes**, engines interleaved run by run, median reported. Anything within 2% (or within either engine's run-to-run spread, if larger) is called a tie.
- **Correctness:** every kernel is tested against a float32 reference, and deliberately broken kernels must fail. Accuracy is measured with OpenAI's own eval harness. Details: [docs/correctness.md](docs/correctness.md).

## What's Next

- [ ] Chunked prefill mixed with decode steps (prefill blocks decoding today; why vLLM wins 120b serving at full load by 5.4%)
- [ ] Faster time to first token on long prompts (9.6 s vs vLLM's 6.1 s at 122k tokens)
- [ ] More than 256 concurrent requests (graph memory reserve is fixed at 4 GiB; why vLLM is 1.8x faster at 1,000 requests)
- [ ] Per-request temperature, top-p and top-k, sampled inside the CUDA graph (today a batch shares one temperature)
- [ ] Stop on `<|return|>` instead of the second `<|end|>` (cuts off some replies with two reasoning segments)
- [ ] Match OpenAI's reference exactly: no sink clamp in decode, 128-token sliding window (today 129)
- [ ] Fix the one batched 20b step that differs from a float32 reference ([details](docs/correctness.md#teacher-forced-greedy-match))
- [ ] Re-run the 131k-context benchmarks and accuracy evals on the latest code
- [ ] `OpenAI/gpt-oss-120b` tensor parallelism on 8x H100s
- [ ] Support for other GPUs, other MoE models and other architectures

## License

Apache 2.0. The MXFP4 weight layout follows the Hopper path of OpenAI's open-source MoE kernels (MIT); see the comment in `kernels.py`.
