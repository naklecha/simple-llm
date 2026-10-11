#!/usr/bin/env python3
"""SimpleLLM: gpt-oss-120b inference engine with continuous batching and async request queue."""
import os, time, threading, queue, concurrent.futures, bisect
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import torch
import torch.nn.functional as F
from tqdm import tqdm
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from model import GptOssConfig, GptOssForCausalLM, Tokenizer


@dataclass
class GenerationOutput:
    """Result of a single prompt generation, with optional reasoning extraction for o1-style models."""
    text: str                              # Final response text (analysis channel filtered out)
    token_ids: list[int]                   # Raw generated token IDs before parsing
    reasoning: Optional[str] = None        # Extracted analysis/reasoning if model uses Harmony protocol
    raw_text: Optional[str] = None         # Unprocessed model output including all special tokens


@dataclass
class _Request:
    """Internal request tracking for the inference loop's continuous batching system.
    
    A single _Request can contain multiple prompts (batch submission). The pending_indices
    tracks which prompts haven't been assigned to slots yet, allowing partial scheduling
    when fewer slots are available than prompts submitted.
    """
    token_ids: list[list[int]]             # Tokenized prompts (one list per prompt in batch)
    max_tokens: int                        # Generation limit per prompt
    temperature: float                     # Sampling temperature (0 = greedy/argmax)
    ignore_eos: bool                       # If True, generate until max_tokens even after EOS
    future: concurrent.futures.Future      # Caller blocks on this; set when all prompts complete
    results: list = field(default_factory=list)          # GenerationOutput per prompt (None until done)
    pending_indices: list = field(default_factory=list)  # Prompt indices not yet assigned to slots


class LLM:
    """
    High-throughput LLM inference engine with continuous batching and CUDA graph acceleration.
    
    You can submit requests from any thread via generate() or chat(), and results come back
    through a Future. The engine runs a background inference loop that dynamically batches
    concurrent requests to maximize GPU utilization.
    """
    def __init__(self, model_path: str, max_num_seqs: int = 64, max_seq_len: int = 4096, dtype=torch.bfloat16):
        self.device = torch.device("cuda")
        self.dtype, self.max_seq_len = dtype, max_seq_len
        self.config = GptOssConfig.from_json(os.path.join(model_path, "config.json"))

        # Load model weights and initialize tokenizer. QKV fusion combines 3 projections into 1,
        # reducing kernel launch overhead during decode. KV cache is pre-allocated for all slots.
        self.tokenizer = Tokenizer(model_path)
        self.model = GptOssForCausalLM.from_pretrained(model_path, self.config, self.device, dtype)
        self.model.eval()
        self.model.fuse_qkv()

        # KV memory (4 GiB kept for activations and CUDA graphs): full-attention layers draw 256-row pages from one shared pool,
        # sliding layers keep a 256-row ring per slot (window 128 + the new token) plus one scratch ring for long prefills.
        # If the pool can hold every slot at max_seq_len, slot s owns fixed pages; otherwise a request reserves the pages for
        # prompt + max_tokens when it is admitted and returns them when it finishes (requests wait in order until pages are free).
        layers, row = self.model.model.layers, 2 * self.config.num_key_value_heads * self.config.head_dim * 2
        n_full = sum(not l.attn.is_sliding for l in layers)
        ring_bytes = (len(layers) - n_full) * row * (max_num_seqs * 256 + (18 * 256 if max_seq_len > 4096 else 0))
        free = torch.cuda.mem_get_info()[0] - 4 * 1024**3 - ring_bytes
        self.max_num_seqs, self._seq_pages = max_num_seqs, -(-max_seq_len // 256)
        self._num_pages = min(int(free // (n_full * row * 256)), max_num_seqs * self._seq_pages)
        assert self._num_pages >= self._seq_pages, "not enough memory for one max_seq_len sequence"
        self._static_pages = self._num_pages == max_num_seqs * self._seq_pages
        self.model.init_kv_cache(self.max_num_seqs, max_seq_len, self._num_pages, self.device, dtype)
        self._reset_pages()

        # CUDA graph storage: one captured graph per batch size. Graphs require static tensor addresses,
        # so we pre-allocate separate input/output buffers for each batch size we capture.
        # The graph pool allows memory sharing between graphs of different batch sizes.
        self._cuda_graphs, self._graph_input_ids, self._graph_positions = {}, {}, {}
        self._graph_slot_indices, self._graph_outputs, self._graph_tokens = {}, {}, {}
        self._fed = None  # (batch_size, slots) whose next inputs the last graph replay already wrote on the GPU
        self._graph_pool = torch.cuda.graph_pool_handle()

        # Pre-allocated decode step tensors: reused every decode iteration to avoid allocation overhead.
        # _slot_remap_buffer handles non-contiguous slot indices when CUDA graphs can't be used.
        self._decode_input_ids = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)
        self._decode_positions = torch.zeros(self.max_num_seqs, 1, dtype=torch.long, device=self.device)
        self._slot_remap_buffer = torch.zeros(self.max_num_seqs, dtype=torch.long, device=self.device)

        # Async request submission: callers push to queue, background thread processes.
        # This decouples request submission from GPU execution for better concurrency.
        self._request_queue: queue.Queue[_Request] = queue.Queue()
        self._loop_running, self._loop_thread = False, None

        # Warmup pass: first CUDA kernel invocations trigger JIT compilation which is slow.
        # Running a dummy decode ensures all kernels are compiled before real inference.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self.model.decode(dummy, dummy, torch.zeros(1, dtype=torch.long, device=self.device))
        # Capture every decode graph now, before any request holds a slot: capture appends K/V at row max_seq_len-4, which
        # on ring (sliding) layers can fall inside a live window, so capturing during serving is unsafe
        for b in range(1, self.max_num_seqs + 1): self._capture_cuda_graph(b)
        self.model.clear_all_slots()
        tqdm.write(f"✓ Engine ready | batch_size={self.max_num_seqs} seq={max_seq_len}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # CUDA GRAPH CAPTURE: eliminates CPU overhead by replaying pre-recorded GPU operations
    # ═══════════════════════════════════════════════════════════════════════════════

    def _reset_pages(self):
        """All slots empty: fixed pages per slot, or every page back in the free pool"""
        if self._static_pages: self.model.block_table.copy_(torch.arange(self._num_pages, dtype=torch.int32).view(self.max_num_seqs, -1))
        self._free_pages, self._slot_pages = list(range(self._num_pages)), {}

    def _alloc_pages(self, slot: int, tokens: int) -> bool:
        """Reserve pages for a sequence of up to `tokens` tokens in this slot (no-op with fixed pages); False if the pool is short"""
        if self._static_pages: return True
        need = -(-min(tokens, self.max_seq_len) // 256)
        if need > len(self._free_pages): return False
        self._slot_pages[slot] = pages = [self._free_pages.pop() for _ in range(need)]
        self.model.block_table[slot, :need] = torch.tensor(pages, dtype=torch.int32)
        return True

    def _release(self, slot: int):
        """Sequence finished: reset its length and return its pages to the pool"""
        self.model.clear_slot(slot)
        self._free_pages += self._slot_pages.pop(slot, [])

    def _capture_cuda_graph(self, batch_size: int):
        """
        Capture a CUDA graph for the decode step at a specific batch size.
        
        CUDA graphs record a sequence of GPU operations and replay them with minimal CPU involvement.
        This is critical for decode performance where each step is a small matmul (batch x 1 token).
        Without graphs, kernel launch overhead dominates the actual compute time.
        
        A few constraints to be aware of: graphs require static memory addresses, so we allocate
        dedicated input/output tensors for each batch size. They also assume contiguous slot indices
        like [0, 1, 2, ...N-1], so non-contiguous slots fall back to the slower direct path (this
        happens when some slots free mid-batch from completed sequences). Finally, cache sequence
        lengths get modified during capture, so we save and restore them afterward.
        """
        if batch_size in self._cuda_graphs: return

        # Save cache state: graph capture modifies cache_seqlens, which we restore after
        saved_cache_lengths = [layer.attn._cache_seqlens.clone() for layer in self.model.model.layers]

        # Allocate static buffers: CUDA graphs require tensor memory addresses to be constant.
        # Position 10 is arbitrary but > 0 to avoid any edge cases in attention masking.
        self._graph_input_ids[batch_size] = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        self._graph_positions[batch_size] = torch.full((batch_size, 1), 10, dtype=torch.long, device=self.device)
        self._graph_slot_indices[batch_size] = torch.arange(batch_size, dtype=torch.long, device=self.device)

        # Warmup on separate stream: CUDA requires kernels to be "warmed up" before graph capture.
        # Running on a separate stream allows this to happen without blocking the main stream.
        # Three iterations ensures all code paths are exercised and any lazy initialization completes.
        # Warmup/capture append K/V at the cache tail, never over live prompt positions of slots 0..B-1
        for layer in self.model.model.layers: layer.attn._cache_seqlens[:batch_size] = self.max_seq_len - 4
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.model.decode(self._graph_input_ids[batch_size], self._graph_positions[batch_size], self._graph_slot_indices[batch_size])
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # Actual graph capture: all GPU operations within this context are recorded, not executed.
        # The pool= argument enables memory sharing between graphs of different batch sizes.
        for layer in self.model.model.layers: layer.attn._cache_seqlens[:batch_size] = self.max_seq_len - 4
        self._cuda_graphs[batch_size] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graphs[batch_size], pool=self._graph_pool):
            logits = self.model.decode(self._graph_input_ids[batch_size], self._graph_positions[batch_size], self._graph_slot_indices[batch_size])
            # Greedy next token and next inputs stay on the GPU: a replay can follow a replay with no host writes
            tokens = logits[:, 0].argmax(-1)
            self._graph_input_ids[batch_size][:, 0].copy_(tokens)
            self._graph_positions[batch_size].add_(1)
            self._graph_outputs[batch_size], self._graph_tokens[batch_size] = logits, tokens

        # Restore cache state: capture modified cache_seqlens, but we want to preserve real values
        for layer, saved in zip(self.model.model.layers, saved_cache_lengths): layer.attn._cache_seqlens.copy_(saved)

    def _ensure_graph_captured(self, batch_size: int):
        """Lazily capture CUDA graph on first use of each batch size. Captures happen during inference."""
        if batch_size not in self._cuda_graphs and batch_size <= self.max_num_seqs:
            self._capture_cuda_graph(batch_size)

    # ═══════════════════════════════════════════════════════════════════════════════
    # CORE INFERENCE: prefill processes prompts in parallel, decode generates one token at a time
    # ═══════════════════════════════════════════════════════════════════════════════

    def _prefill_long(self, seq: list[int], slot: int, chunk: int = 4096) -> torch.Tensor:
        """Prefill one long prompt in chunks: each chunk appends its K/V to the slot and attends causally to everything before it."""
        for start in range(0, len(seq), chunk):
            toks = torch.tensor(seq[start:start + chunk], dtype=torch.long, device=self.device)
            n, positions = len(toks), torch.arange(start, start + len(toks), device=self.device)
            slot_t = torch.tensor([slot], dtype=torch.int32, device=self.device)
            hidden, residual = self.model.model.embed_tokens(toks), None
            for layer in self.model.model.layers:
                attn = layer.attn
                if residual is None: residual, hidden = hidden, layer.input_layernorm(hidden.unsqueeze(0)).squeeze(0)
                else: hidden, residual = (x.squeeze(0) for x in layer.input_layernorm(hidden.unsqueeze(0), residual.unsqueeze(0)))
                qkv = attn._qkv_proj(hidden)
                q = qkv[..., :attn._q_size].view(1, n, attn.num_heads, attn.head_dim)
                k = qkv[..., attn._q_size:attn._q_size + attn._kv_size].view(1, n, attn.num_kv_heads, attn.head_dim)
                v = qkv[..., attn._q_size + attn._kv_size:].view(1, n, attn.num_kv_heads, attn.head_dim)
                q, k = attn.rotary_emb(positions.unsqueeze(0), q, k)
                attn._cache_seqlens[slot] = start
                window = (attn.sliding_window, 0) if attn.is_sliding else (-1, -1)
                if attn._ring:  # sliding: the shared scratch ring (256-row pages, logical page b -> b % 18) holds this chunk + window
                    table = (torch.arange((start + n + 255) // 256, device=self.device) % 18).to(torch.int32).unsqueeze(0)
                    out, lse = flash_attn_with_kvcache(q, attn._scratch[0].view(-1, 256, attn.num_kv_heads, attn.head_dim), attn._scratch[1].view(-1, 256, attn.num_kv_heads, attn.head_dim),
                                                       k=k, v=v, cache_seqlens=attn._cache_seqlens[slot:slot + 1], block_table=table,
                                                       softmax_scale=attn.scaling, causal=True, window_size=window, return_softmax_lse=True)
                else:  # full attention: the slot's pages in the shared pool
                    out, lse = flash_attn_with_kvcache(q, attn._kv_cache, attn._v_cache, k=k, v=v, cache_seqlens=attn._cache_seqlens[slot:slot + 1],
                                                       block_table=attn._table[slot:slot + 1, :(start + n + 255) // 256], softmax_scale=attn.scaling,
                                                       causal=True, window_size=window, return_softmax_lse=True)
                attn._cache_seqlens[slot] = start + n
                sink = torch.sigmoid(lse - attn.sinks.view(1, attn.num_heads, 1)).transpose(1, 2).unsqueeze(-1).to(out.dtype)  # prefill: no clamp
                hidden = attn.o_proj((out * sink).reshape(n, -1))
                hidden, residual = (x.squeeze(0) for x in layer.post_attention_layernorm(hidden.unsqueeze(0), residual.unsqueeze(0)))
                hidden = layer.mlp(hidden.unsqueeze(0)).squeeze(0)
        p = torch.arange(max(0, len(seq) - 256), len(seq), device=self.device)  # sliding layers: keep the last 256 positions in the slot's ring
        for layer in self.model.model.layers:
            if layer.attn._ring:
                for cache, scratch in zip((layer.attn._kv_cache, layer.attn._v_cache), layer.attn._scratch): cache[slot, p % 256] = scratch[p % (18 * 256)]
        hidden, _ = self.model.model.norm(hidden.unsqueeze(0), residual.unsqueeze(0))
        return self.model.lm_head(hidden.squeeze(0)[-1:])

    def _prefill(self, sequences: list[list[int]], slot_indices: list[int]) -> torch.Tensor:
        """
        Process prompt tokens for multiple sequences in a single forward pass, populating KV cache.
        
        Uses flash_attn_varlen_func which handles variable-length sequences packed into a single
        tensor. This is more efficient than padding because we don't waste compute on pad tokens.
        
        Args:
            sequences: List of token ID lists, one per prompt (can have different lengths)
            slot_indices: Which KV cache slots to populate (one per sequence)
            
        Returns:
            Logits tensor [num_sequences, vocab_size] for the last token of each sequence
        """
        if len(sequences) == 1 and len(sequences[0]) > 4096: return self._prefill_long(sequences[0], slot_indices[0])
        num_seqs, seq_lens = len(sequences), [len(s) for s in sequences]
        total_tokens, max_seq_len = sum(seq_lens), max(seq_lens)

        # Pack all tokens into a single flat tensor. Flash attention uses cu_seqlens (cumulative
        # sequence lengths) to know where each sequence starts/ends within the packed tensor.
        flat_tokens = torch.tensor([t for s in sequences for t in s], dtype=torch.long, device=self.device)
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int32, device=self.device)
        cu_seqlens[1:] = seq_lens_t.cumsum(0)

        # Cache write indices. Full-attention layers: position P of slot S goes to row P % 256 of page block_table[S, P // 256] in the
        # flattened pool. Sliding layers: only the last 256 positions are kept, position P -> row S * 256 + P % 256 of the rings.
        positions = torch.cat([torch.arange(l, device=self.device) for l in seq_lens])
        slot_indices_t = torch.tensor(slot_indices, dtype=torch.long, device=self.device)
        slot_rep = torch.repeat_interleave(slot_indices_t, seq_lens_t.long())
        page_idx = self.model.block_table[slot_rep, positions // 256].long() * 256 + positions % 256
        keep = positions >= torch.repeat_interleave(seq_lens_t.long(), seq_lens_t.long()) - 256
        ring_idx, ring_src = slot_rep[keep] * 256 + positions[keep] % 256, keep.nonzero().squeeze(1)

        # Forward through transformer: embedding lookup, then N layers of attention + MLP
        hidden = self.model.model.embed_tokens(flat_tokens)
        residual = None

        for layer in self.model.model.layers:
            attn = layer.attn

            # Fused RMSNorm + residual: combines normalization with residual add in single kernel.
            # First layer has no residual, so we just normalize. Subsequent layers fuse the add.
            if residual is None:
                residual, hidden = hidden, layer.input_layernorm(hidden.unsqueeze(0)).squeeze(0)
            else:
                hidden, residual = (x.squeeze(0) for x in layer.input_layernorm(hidden.unsqueeze(0), residual.unsqueeze(0)))

            # Fused QKV projection: single matmul produces Q, K, V concatenated, then we split.
            # Reshape to [total_tokens, num_heads, head_dim] for flash attention.
            qkv = attn._qkv_proj(hidden)
            q = qkv[..., :attn._q_size].view(total_tokens, attn.num_heads, attn.head_dim)
            k = qkv[..., attn._q_size:attn._q_size + attn._kv_size].view(total_tokens, attn.num_kv_heads, attn.head_dim)
            v = qkv[..., attn._q_size + attn._kv_size:].view(total_tokens, attn.num_kv_heads, attn.head_dim)

            # Rotary position embeddings: encode position information directly into Q and K vectors
            q, k = (x.squeeze(0) for x in attn.rotary_emb(positions.unsqueeze(0), q.unsqueeze(0), k.unsqueeze(0)))

            # Populate KV cache: scatter keys and values to their cache rows (flattened [rows, num_kv_heads, head_dim])
            kv_flat = attn._kv_cache.view(-1, attn.num_kv_heads, attn.head_dim)
            v_flat = attn._v_cache.view(-1, attn.num_kv_heads, attn.head_dim)
            if attn._ring: kv_flat.index_copy_(0, ring_idx, k[ring_src]); v_flat.index_copy_(0, ring_idx, v[ring_src])
            else: kv_flat.index_copy_(0, page_idx, k); v_flat.index_copy_(0, page_idx, v)
            attn._cache_seqlens.index_copy_(0, slot_indices_t, seq_lens_t)

            # Flash attention with sink correction and window_size for sliding attention layers
            # window_size=(left, right): (128, 0) = attend to previous 128 tokens only
            window = (attn.sliding_window, 0) if attn.is_sliding else (-1, -1)
            attn_out, lse, _ = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seq_len, max_seq_len,
                                                       causal=True, softmax_scale=attn.scaling, window_size=window, return_attn_probs=True)
            # Sink correction: learned per-head bias modulates attention output
            sink = torch.sigmoid(lse - attn.sinks.view(attn.num_heads, 1)).transpose(0, 1).unsqueeze(-1).to(attn_out.dtype)
            hidden = attn.o_proj((attn_out * sink).reshape(total_tokens, -1))

            # Post-attention: fused norm + residual, then MoE/MLP layer
            hidden, residual = (x.squeeze(0) for x in layer.post_attention_layernorm(hidden.unsqueeze(0), residual.unsqueeze(0)))
            hidden = layer.mlp(hidden.unsqueeze(0)).squeeze(0)

        # Final norm and project to vocabulary. Only need logits for last token of each sequence
        # (the position where we'll sample the next token), so we index with cu_seqlens[1:] - 1.
        hidden, _ = self.model.model.norm(hidden.unsqueeze(0), residual.unsqueeze(0))
        return self.model.lm_head(hidden.squeeze(0)[cu_seqlens[1:] - 1])

    def _decode_step(self, input_ids: torch.Tensor, positions: torch.Tensor, slot_indices: list[int]) -> torch.Tensor:
        """
        Generate logits for one token per sequence using cached KV values from prefill or prior decode.
        
        Two execution paths: fast path uses CUDA graphs when slot indices are contiguous [0,1,2...N-1]
        which eliminates CPU kernel launch overhead. Slow path handles non-contiguous slots (e.g. [0,2,5]
        after some sequences completed) via direct model.decode() with explicit slot remapping.
        """
        batch_size = len(slot_indices)

        # Fast path: CUDA graph for any slot set (slot indices are a graph input)
        if batch_size in self._cuda_graphs:
            self._graph_input_ids[batch_size].copy_(input_ids)
            self._graph_positions[batch_size].copy_(positions)
            self._graph_slot_indices[batch_size].copy_(torch.tensor(slot_indices), non_blocking=True)
            self._cuda_graphs[batch_size].replay()
            return self._graph_outputs[batch_size]

        # Slow path: non-contiguous slots after some sequences completed, incurs kernel launch overhead
        return self.model.decode(input_ids, positions, torch.tensor(slot_indices, device=self.device, dtype=torch.long))

    def _sample_tokens(self, logits: torch.Tensor, temperature: float) -> list[int]:
        """Sample next tokens from logits. Temperature 0 = greedy (argmax), >0 = stochastic."""
        if temperature == 0: return logits.argmax(dim=-1).tolist()
        return torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1).tolist()

    # ═══════════════════════════════════════════════════════════════════════════════
    # PUBLIC API: thread-safe request submission, returns Future for async result retrieval
    # ═══════════════════════════════════════════════════════════════════════════════

    def generate(self, prompts: list[str], max_tokens: int = 100, temperature: float = 0.0,
                 ignore_eos: bool = False, reasoning_effort: str = "medium") -> concurrent.futures.Future:
        """
        Submit one or more prompts for parallel generation.
        
        This is the primary API for batch inference. Prompts are formatted using the Harmony
        chat template and queued for the background inference loop. All prompts share the same
        generation parameters (temperature, max_tokens).
        
        Args:
            prompts: List of user message strings to generate responses for
            max_tokens: Maximum tokens to generate per prompt (stops early on EOS unless ignore_eos)
            temperature: Sampling temperature (0 = greedy/deterministic, higher = more random)
            ignore_eos: If True, continue generating until max_tokens even after EOS token
            reasoning_effort: Controls model's reasoning verbosity ("low", "medium", "high")
            
        Returns:
            Future that resolves to list[GenerationOutput] when all prompts complete.
            Use future.result() to block and get results, or future.add_done_callback() for async.
        """
        if not self._loop_running:
            self._loop_running, self._loop_thread = True, threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()

        future = concurrent.futures.Future()
        token_ids = [self.tokenizer.apply_chat_template([{"role": "user", "content": p}], reasoning_effort=reasoning_effort) for p in prompts]
        request = _Request(token_ids=token_ids, max_tokens=max_tokens, temperature=temperature, ignore_eos=ignore_eos,
                           future=future, results=[None] * len(prompts), pending_indices=list(range(len(prompts))))
        self._request_queue.put(request)
        return future

    def chat(self, messages: list[dict], max_tokens: int = 100, temperature: float = 0.0,
             ignore_eos: bool = False, reasoning_effort: str = "medium") -> concurrent.futures.Future:
        """
        Multi-turn chat generation with full conversation history.
        
        Unlike generate() which takes simple prompts, this accepts a full message list including
        system messages, prior turns, etc. The entire conversation is encoded using Harmony
        protocol formatting and passed to the model.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                      Roles: "system"/"developer" (instructions), "user", "assistant" (prior turns)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            ignore_eos: Continue generating past EOS token
            reasoning_effort: Model reasoning verbosity control
            
        Returns:
            Future that resolves to list[GenerationOutput] with a single result.
        """
        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()

        future = concurrent.futures.Future()
        token_ids = [self.tokenizer.apply_chat_template(messages, reasoning_effort=reasoning_effort)]
        request = _Request(token_ids=token_ids, max_tokens=max_tokens, temperature=temperature, ignore_eos=ignore_eos,
                           future=future, results=[None], pending_indices=[0])
        self._request_queue.put(request)
        return future

    def stop(self, timeout: float = 30.0):
        """Gracefully stop the background inference loop. Waits for active generations to complete."""
        if not self._loop_running: return
        self._loop_running = False
        if self._loop_thread: self._loop_thread.join(timeout=timeout); self._loop_thread = None

    # ═══════════════════════════════════════════════════════════════════════════════
    # BACKGROUND INFERENCE LOOP: implements continuous batching with dynamic slot management
    # ═══════════════════════════════════════════════════════════════════════════════

    def _drain_queue(self, pending_requests: list, pbar, total_prompts: int):
        """
        Phase 1: Drain incoming requests from the thread-safe queue.
        
        Multiple requests may arrive between decode steps; we batch them all together.
        This non-blocking drain ensures we capture all pending work without stalling
        the inference loop waiting for new requests.
        """
        while True:
            try: req = self._request_queue.get_nowait()
            except queue.Empty: break

            pending_requests.append(req)
            total_prompts += len(req.token_ids)
            if pbar is None:
                pbar = tqdm(total=total_prompts, desc="Generating", unit="req", ncols=100,
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
            else:
                pbar.total = total_prompts
                pbar.refresh()
        return pbar, total_prompts

    def _assign_slots_to_pending(self, pending_requests: list, free_slots: list):
        """
        Phase 2: Assign pending prompts to free slots.
        
        One prompt gets one slot. Requests with multiple prompts may be partially
        scheduled if fewer slots are available than prompts submitted. This enables
        continuous batching where new work joins mid-generation as slots free up.
        
        Returns:
            List of (request, prompt_idx, slot) tuples for newly assigned work.
        """
        new_work = []
        for req in list(pending_requests):
            while req.pending_indices and free_slots:
                if not self._alloc_pages(free_slots[0], len(req.token_ids[req.pending_indices[0]]) + req.max_tokens): return new_work  # wait for pages
                new_work.append((req, req.pending_indices.pop(0), free_slots.pop(0)))
            if not req.pending_indices:
                pending_requests.remove(req)
        return new_work

    def _handle_prefill_completion(self, req: _Request, prompt_idx: int, tok: int, slot: int,
                                   free_slots: list, active_generations: dict, end_token_id: int, pbar) -> bool:
        """
        Handle a single prefill result, checking for immediate completion.
        
        Sequences complete immediately when max_tokens=1 or the first sampled token is EOS.
        Otherwise, the sequence enters active_generations for continued decode steps.
        
        Returns:
            True if sequence completed immediately, False if it needs more decode steps.
        """
        # Check for immediate completion (max_tokens=1 or first token is EOS)
        if 1 >= req.max_tokens or (not req.ignore_eos and tok in self.tokenizer.eos_token_ids):
            reasoning, final, raw = self.tokenizer.parse_harmony_output([tok])
            req.results[prompt_idx] = GenerationOutput(final or self.tokenizer.decode([tok]), [tok], reasoning, raw)
            self._release(slot)
            # Keep free_slots sorted so pop(0) returns lowest slot. CUDA graphs require
            # contiguous slots [0,1,2..N-1], so reusing low slots maximizes graph hits.
            bisect.insort(free_slots, slot)
            if pbar: pbar.update(1)
            # When all prompts in a request complete, resolve the future
            if all(r is not None for r in req.results):
                req.future.set_result(req.results)
            return True

        # Track end_token occurrences: Harmony format uses <|end|> to delimit channels.
        # Two <|end|> tokens indicates end of response (analysis + final channels).
        end_count = 1 if tok == end_token_id else 0
        active_generations[slot] = (req, prompt_idx, [tok], len(req.token_ids[prompt_idx]), end_count)
        return False

    def _process_generated_token(self, slot: int, tok: int, active_generations: dict,
                                  free_slots: list, end_token_id: int, pbar) -> bool:
        """
        Process a generated token and check completion conditions.
        
        Sequences complete when: max_tokens reached, EOS token generated, or two <|end|>
        tokens seen (Harmony format channel delimiter). Completed slots are immediately
        cleared and returned to free_slots for reuse by pending requests.
        
        Returns:
            True if sequence completed, False if generation continues.
        """
        req, prompt_idx, tokens, pos, end_count = active_generations[slot]
        tokens.append(tok)
        pos += 1
        if tok == end_token_id: end_count += 1

        # Stop conditions: max length, EOS token, two end tokens, or KV cache limit reached
        should_stop = (len(tokens) >= req.max_tokens or pos >= self.max_seq_len or
                       (not req.ignore_eos and (tok in self.tokenizer.eos_token_ids or end_count >= 2)))
        if not should_stop:
            active_generations[slot] = (req, prompt_idx, tokens, pos, end_count)
            return False

        # Sequence complete: parse output, store result, release slot
        reasoning, final, raw = self.tokenizer.parse_harmony_output(tokens)
        req.results[prompt_idx] = GenerationOutput(final or self.tokenizer.decode(tokens), tokens, reasoning, raw)
        del active_generations[slot]
        self._release(slot)
        # Keep free_slots sorted so pop(0) returns lowest slot. CUDA graphs require
        # contiguous slots [0,1,2..N-1], so reusing low slots maximizes graph hits.
        bisect.insort(free_slots, slot)
        if pbar: pbar.update(1)

        # When all prompts in a request complete, resolve the future
        if all(r is not None for r in req.results):
            req.future.set_result(req.results)
        return True

    @torch.inference_mode()
    def _inference_loop(self):
        """
        Main inference loop implementing continuous batching.
        
        Unlike static batching where all requests start and end together, continuous batching
        lets new requests join mid-generation and completed sequences exit immediately (freeing
        their slot for the next request). Different sequences in the same batch can be at
        completely different generation steps.
        
        The loop manages three collections:
          - pending_requests: requests waiting for free slots (may have multiple prompts each)
          - active_generations: sequences currently generating (one per occupied slot)
          - free_slots: available KV cache slots ready for new work
        
        Each iteration: drain queue → assign slots → prefill new → decode active → check completions.
        """
        end_token_id = self.tokenizer.encode(self.tokenizer.END, add_special_tokens=False)[0]

        # Slot management: each slot is a fixed region in KV cache that holds one sequence.
        # Free slots are recycled immediately when sequences complete.
        free_slots = list(range(self.max_num_seqs))
        active_generations = {}  # slot -> (request, prompt_idx, tokens, position, end_count)
        pending_requests = []    # requests with prompts not yet assigned to slots

        # Throughput tracking: rolling window of recent decode timings for tok/s display
        total_prompts = 0
        decode_times = deque(maxlen=16)
        pbar = None
        self.model.clear_all_slots(); self._reset_pages()

        while self._loop_running or pending_requests or active_generations:
            # Phase 1 & 2: Drain incoming requests and assign pending prompts to free slots
            pbar, total_prompts = self._drain_queue(pending_requests, pbar, total_prompts)

            # Idle state: no work to do, sleep briefly to avoid busy-waiting
            if not pending_requests and not active_generations:
                if not self._loop_running: break
                time.sleep(0.001)
                continue

            new_work = self._assign_slots_to_pending(pending_requests, free_slots)

            # Phase 3: Prefill newly assigned prompts (populates KV cache for each)
            if new_work: self._run_prefill(new_work, free_slots, active_generations, end_token_id, pbar)
            if not active_generations: continue

            # Phase 4-6: Decode step for all active sequences, check completions, release slots
            self._run_decode_step(active_generations, free_slots, decode_times, end_token_id, pbar)

        if pbar: pbar.close()

    def _run_prefill(self, new_work: list, free_slots: list, active_generations: dict, end_token_id: int, pbar):
        """
        Phase 3: Prefill newly assigned prompts with chunking to avoid OOM.
        
        Limits total tokens per prefill batch to MAX_PREFILL_TOKENS to prevent memory exhaustion
        on MoE intermediate buffers. Without chunking, prefilling 64 long sequences at once would
        allocate ~250MB for intermediates and OOM. Processes chunks sequentially.
        """
        MAX_PREFILL_TOKENS = 4096  # Conservative limit; MoE intermediates scale with token count

        # Build chunks that fit within token budget
        chunks, chunk, tokens = [], [], 0
        for w in new_work:
            seq_len = len(w[0].token_ids[w[1]])
            if tokens + seq_len > MAX_PREFILL_TOKENS and chunk: chunks.append(chunk); chunk, tokens = [], 0
            chunk.append(w); tokens += seq_len
        if chunk: chunks.append(chunk)

        # Process each chunk: prefill, sample first token, check for immediate completion
        self._fed = None
        for chunk in chunks:
            seqs, slots = [r.token_ids[i] for r, i, _ in chunk], [s for _, _, s in chunk]
            sampled = self._sample_tokens(self._prefill(seqs, slots), chunk[0][0].temperature)
            for i, (req, idx, slot) in enumerate(chunk):
                self._handle_prefill_completion(req, idx, sampled[i], slot, free_slots, active_generations, end_token_id, pbar)

    def _run_decode_step(self, active_generations: dict, free_slots: list, decode_times: deque, end_token_id: int, pbar):
        """
        Phase 4-6: Decode step for all active sequences.
        
        All sequences generate their next token in parallel using cached KV values.
        This is where the bulk of generation time is spent, running one decode step
        per token across all active sequences simultaneously.
        """
        slots = list(active_generations.keys())
        batch_size = len(slots)
        self._ensure_graph_captured(batch_size)
        temperature = active_generations[slots[0]][0].temperature
        graph = batch_size in self._cuda_graphs  # any slot set: slot indices are a graph input (copied below when they change)

        t0 = time.perf_counter()
        if graph and temperature == 0:
            # Greedy graph path: inputs come from the previous replay unless the batch changed (one H2D copy each)
            if self._fed != (batch_size, tuple(slots)):
                last = [active_generations[s] for s in slots]
                self._graph_input_ids[batch_size].copy_(torch.tensor([[g[2][-1]] for g in last]), non_blocking=True)
                self._graph_positions[batch_size].copy_(torch.tensor([[g[3]] for g in last]), non_blocking=True)
                self._graph_slot_indices[batch_size].copy_(torch.tensor(slots), non_blocking=True)
            self._cuda_graphs[batch_size].replay()
            sampled = self._graph_tokens[batch_size].tolist()
        else:
            # Prepare decode inputs: last generated token and current position for each sequence
            input_ids = self._decode_input_ids[:batch_size]
            positions = self._decode_positions[:batch_size]
            for i, slot in enumerate(slots):
                req, prompt_idx, tokens, pos, _ = active_generations[slot]
                input_ids[i, 0] = tokens[-1]
                positions[i, 0] = pos
            logits = self._decode_step(input_ids, positions, slots)
            sampled = self._sample_tokens(logits[:, 0], temperature)
        decode_times.append((batch_size, time.perf_counter() - t0))
        self._fed = (batch_size, tuple(slots)) if graph and temperature == 0 else None

        # Update progress bar with rolling average throughput
        if pbar and decode_times:
            tok_per_sec = sum(n for n, _ in decode_times) / max(sum(t for _, t in decode_times), 1e-9)
            pbar.set_postfix_str(f"batch_size={batch_size} tok/s={tok_per_sec:.0f}")

        # Process each generated token: update state, check completion, release finished slots
        for i, slot in enumerate(slots):
            if self._process_generated_token(slot, sampled[i], active_generations, free_slots, end_token_id, pbar): self._fed = None
