# API

`LLM(model_path, max_num_seqs=64, max_seq_len=4096)` loads the model, sizes the KV cache and captures the CUDA graphs. `max_num_seqs` is how many sequences can decode at once. `max_seq_len` is the longest prompt plus output one sequence may have (up to 131,072).

| method | what it does |
|---|---|
| `generate(prompts, max_tokens=100, temperature=0.0, ignore_eos=False, reasoning_effort="medium")` | runs a list of user prompts, each wrapped in the Harmony chat template |
| `chat(messages, ...)` | runs one conversation, given as `[{"role": "system" \| "developer" \| "user" \| "assistant", "content": ...}]` |
| `stop()` | waits for active work to finish and stops the background loop |

`generate` and `chat` return a `concurrent.futures.Future` right away, and you can call them from many threads at once. All requests share one queue and are batched together. `.result()` gives one `GenerationOutput` per prompt:

| field | contents |
|---|---|
| `text` | the final answer |
| `reasoning` | the model's analysis channel (its chain of thought), when present |
| `raw_text` | the full output including Harmony special tokens |
| `token_ids` | the generated token ids |

## Example: chat with reasoning

```python
from llm import LLM

engine = LLM("./gpt-oss-120b", max_num_seqs=10, max_seq_len=10000)
messages = [{"role": "user", "content": "Write a haiku about the ocean"}]
out = engine.chat(messages, max_tokens=1000, reasoning_effort="low").result()[0]
print(out.reasoning)   # chain of thought
print(out.text)        # final answer
engine.stop()
```

More examples are in `cookbook/`: `simple.py` (batch generation) and `chat.py` (a terminal chat).
