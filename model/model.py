"""gpt-oss-120b Model optimized for H100 with Flash Attention, MXFP4 quantized MoE, and fused Triton kernels"""
import gc, json, math, re
from dataclasses import dataclass, field
from pathlib import Path
import torch, torch.nn as nn
from tqdm import tqdm
from flash_attn import flash_attn_with_kvcache
from safetensors import safe_open
from kernels import rms_norm, fused_add_rms_norm, fused_rope, prep_moe, moe_grouped, decode_attention


@dataclass
class GptOssConfig:
  vocab_size: int = 201088; hidden_size: int = 2880; intermediate_size: int = 2880
  num_hidden_layers: int = 36; num_attention_heads: int = 64; num_key_value_heads: int = 8; head_dim: int = 64
  max_position_embeddings: int = 131072; num_local_experts: int = 128; num_experts_per_tok: int = 4
  rope_theta: float = 150000.0; rope_scaling_factor: float = 32.0; original_max_position_embeddings: int = 4096
  beta_fast: float = 32.0; beta_slow: float = 1.0; sliding_window: int = 128
  layer_types: list = field(default_factory=list); quantization_config: dict = field(default_factory=lambda: {"quant_method": "mxfp4"})
  rms_norm_eps: float = 1e-5; tie_word_embeddings: bool = False; eos_token_id: int = 200002; pad_token_id: int = 199999

  @classmethod
  def from_json(cls, path: str):
    data = json.load(open(path))
    # Flatten nested rope_scaling config into top-level fields
    rope = data.pop("rope_scaling", {})
    if rope: data.update(rope_scaling_factor=rope.get("factor",32.0), original_max_position_embeddings=rope.get("original_max_position_embeddings",4096), beta_fast=rope.get("beta_fast",32.0), beta_slow=rope.get("beta_slow",1.0))
    return cls(**{k:v for k,v in data.items() if k in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
  """RMSNorm using Triton kernels; fuses add+norm into single pass when residual is provided"""
  def __init__(self, size: int, eps: float = 1e-5):
    super().__init__(); self.weight, self.eps = nn.Parameter(torch.ones(size)), eps
  
  def forward(self, x, residual=None):
    # Fused path combines residual addition and normalization, reducing memory reads/writes
    if residual is not None and residual.is_contiguous():
      return fused_add_rms_norm(x, residual, self.weight, self.eps)
    return rms_norm(x, self.weight, self.eps)


class RotaryEmbedding(nn.Module):
  """Rotary Position Embeddings with YaRN scaling to support 32x longer sequences than training"""
  def __init__(self, dim: int, max_pos: int, base: float = 10000.0, scale: float = 1.0, orig_max: int = 4096, beta_fast: float = 32.0, beta_slow: float = 1.0):
    super().__init__()
    self.dim, self.max_pos, self.base, self.scale, self.orig_max = dim, max_pos, base, scale, orig_max
    self.beta_fast, self.beta_slow = beta_fast, beta_slow
    # mscale adjusts attention logits to compensate for longer sequences (YaRN paper formula)
    self.mscale = 0.1 * math.log(scale) + 1.0 if scale > 1 else 1.0
    self._cos, self._sin = None, None

  def _build_cache(self, device, dtype):
    """Precompute sin/cos embeddings for all positions up to max_pos, cached for reuse"""
    if self._cos is not None and self._cos.device == device: return
    
    # YaRN: low frequencies (slow-changing) are interpolated, high frequencies kept original
    # Preserves short-range position info while extending long-range. Don't modify without understanding YaRN paper.
    def find_dim(rot): return (self.dim * math.log(self.orig_max / (rot * 2 * math.pi))) / (2 * math.log(self.base))
    low, high = max(find_dim(self.beta_fast), 0), min(find_dim(self.beta_slow), self.dim // 2 - 1)
    if low == high: high += 0.001
    
    pos_freqs = self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
    ramp = torch.clamp((torch.arange(self.dim // 2, device=device, dtype=torch.float32) - low) / (high - low), 0, 1)
    inv_freq = (1.0 / (self.scale * pos_freqs)) * ramp + (1.0 / pos_freqs) * (1 - ramp)
    
    emb = torch.cat([torch.outer(torch.arange(self.max_pos, device=device, dtype=torch.float32), inv_freq)] * 2, -1)
    self._cos, self._sin = (emb.cos() * self.mscale).to(dtype), (emb.sin() * self.mscale).to(dtype)

  def forward(self, positions, query, key):
    self._build_cache(query.device, query.dtype)
    
    # Decode path (seq_len=1, any batch): fused Triton kernel rotates Q and K (strided views into QKV) in one pass
    if query.shape[1] == 1:
      return fused_rope(positions.unsqueeze(0) if positions.dim() == 1 else positions, query, key, self._cos, self._sin)
    
    # Prefill path: standard PyTorch rotation using cached cos/sin tables
    cos, sin = self._cos[positions], self._sin[positions]
    if positions.dim() == 1: cos, sin = cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2)
    else: cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    half = query.shape[-1] // 2
    return query * cos + torch.cat((-query[..., half:], query[..., :half]), -1) * sin, key * cos + torch.cat((-key[..., half:], key[..., :half]), -1) * sin


class Attention(nn.Module):
  """Multi-head attention with GQA (8 KV heads shared across 64 query heads) and slot-based KV caching"""
  def __init__(self, config: GptOssConfig, layer_idx: int):
    super().__init__()
    self.num_heads, self.num_kv_heads, self.head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    self.scaling = self.head_dim ** -0.5
    self._q_size, self._kv_size = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
    
    # Track if this is a sliding window attention layer (alternates with full attention)
    layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
    self.is_sliding = layer_type == "sliding_attention"
    self.sliding_window = config.sliding_window if self.is_sliding else None
    
    # Separate Q/K/V projections during loading, fused into single linear after load
    self._qkv_proj = None
    self.q_proj = nn.Linear(config.hidden_size, self._q_size, bias=True)
    self.k_proj = nn.Linear(config.hidden_size, self._kv_size, bias=True)
    self.v_proj = nn.Linear(config.hidden_size, self._kv_size, bias=True)
    self.o_proj = nn.Linear(self._q_size, config.hidden_size, bias=True)
    
    self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta, config.rope_scaling_factor, config.original_max_position_embeddings, config.beta_fast, config.beta_slow)
    
    # Learned attention sinks: per-head bias via sigmoid that prevents attention collapse on certain patterns
    self.sinks = nn.Parameter(torch.zeros(self.num_heads))
    
    # Slot-based KV cache [num_slots, rows, num_kv_heads, head_dim] (rows = max_seq_len, or a ring for sliding layers)
    self._kv_cache = self._v_cache = self._cache_seqlens = None
    self._ring = 0  # > 0: sliding layer stores position p in cache row p % _ring
    self._table = self._scratch = None  # full-attention layers: shared block table into a page pool; sliding: scratch ring for long prefills

  def fuse_qkv(self):
    """Combine Q, K, V into single fused projection, reducing 3 matmuls to 1"""
    if self._qkv_proj is not None: return
    
    device, dtype = self.q_proj.weight.device, self.q_proj.weight.dtype
    self._qkv_proj = nn.Linear(self.q_proj.in_features, self._q_size + 2*self._kv_size, bias=True, device=device, dtype=dtype)
    
    # Concatenate weights [Q | K | V] and biases
    with torch.no_grad():
      self._qkv_proj.weight[:self._q_size] = self.q_proj.weight
      self._qkv_proj.weight[self._q_size:self._q_size+self._kv_size] = self.k_proj.weight
      self._qkv_proj.weight[self._q_size+self._kv_size:] = self.v_proj.weight
      self._qkv_proj.bias[:self._q_size] = self.q_proj.bias
      self._qkv_proj.bias[self._q_size:self._q_size+self._kv_size] = self.k_proj.bias
      self._qkv_proj.bias[self._q_size+self._kv_size:] = self.v_proj.bias
    
    # Delete original projections to free memory
    del self.q_proj, self.k_proj, self.v_proj
    self.q_proj = self.k_proj = self.v_proj = None


class FusedMoE(nn.Module):
  """Mixture of Experts with 128 experts, top-4 routing, MXFP4 quantized weights via custom Triton kernels"""
  def __init__(self, config: GptOssConfig, layer_idx: int):
    super().__init__()
    self.hidden_size, self.intermediate_size, self.top_k = config.hidden_size, config.intermediate_size, config.num_experts_per_tok
    self.router = nn.Linear(self.hidden_size, config.num_local_experts, bias=True)
    
    # MXFP4 weights in tensor-core fragment order (set during from_pretrained by kernels.prep_moe)
    self._w1 = self._w2 = self._w1_bias = self._w2_bias = None

  def forward(self, hidden):
    batch, seq_len, hidden_dim = hidden.shape
    flat = hidden.view(batch * seq_len, hidden_dim)
    return moe_grouped(flat, *self._w1, self._w1_bias, *self._w2, self._w2_bias, self.router(flat), self.top_k).view(batch, seq_len, hidden_dim)


class TransformerBlock(nn.Module):
  """Single transformer layer: pre-norm attention then pre-norm MoE with fused residual connections"""
  def __init__(self, config: GptOssConfig, layer_idx: int):
    super().__init__()
    self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.attn = Attention(config, layer_idx)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    self.mlp = FusedMoE(config, layer_idx)

  def forward(self, hidden, residual=None):
    # Returns (normalized_hidden, new_residual); caller handles attention/MLP
    if residual is None: return self.input_layernorm(hidden), hidden
    return self.input_layernorm(hidden, residual)


class GptOssModel(nn.Module):
  """GPT-OSS Transformer backbone: embedding + 36 transformer layers + final RMSNorm"""
  def __init__(self, config: GptOssConfig):
    super().__init__()
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_hidden_layers)])
    self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class GptOssForCausalLM(nn.Module):
  """Complete GPT-OSS model with LM head for next-token prediction"""
  def __init__(self, config: GptOssConfig):
    super().__init__()
    self.config, self.model = config, GptOssModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

  def decode(self, input_ids, positions, slot_indices):
    """Single-token decode: embedding lookup, all layers with KV cache, project to logits"""
    batch_size = input_ids.shape[0]
    hidden, residual = self.model.embed_tokens(input_ids), None
    
    for layer in self.model.layers:
      hidden, residual = layer(hidden, residual)
      attn = layer.attn
      
      # Fused QKV projection + RoPE for position encoding
      qkv = attn._qkv_proj(hidden)
      query = qkv[..., :attn._q_size].view(batch_size, 1, attn.num_heads, attn.head_dim)
      key = qkv[..., attn._q_size:attn._q_size+attn._kv_size].view(batch_size, 1, attn.num_kv_heads, attn.head_dim)
      value = qkv[..., attn._q_size+attn._kv_size:].view(batch_size, 1, attn.num_kv_heads, attn.head_dim)
      query, key = attn.rotary_emb(positions, query, key)
      
      # Attends to cached KV (sliding layers: previous 128 tokens), appends K/V, applies the sinks (kernels.decode_attention)
      attn_out = decode_attention(query, key, value, attn._kv_cache, attn._v_cache, attn._cache_seqlens, slot_indices, attn.sinks,
                                  attn.scaling, attn.sliding_window if attn.is_sliding else 0, attn._ring, attn._table)
      hidden = attn.o_proj(attn_out)
      
      hidden, residual = layer.post_attention_layernorm(hidden, residual)
      hidden = layer.mlp(hidden)
    
    hidden, _ = self.model.norm(hidden, residual)
    return self.lm_head(hidden)

  def init_kv_cache(self, num_slots: int, max_seq_len: int, num_pages: int, device, dtype, page: int = 256, ring: int = 256):
    """Full-attention layers share a pool of `page`-row pages addressed through one per-slot block table (filled by llm.py);
    sliding layers keep a `ring`-row ring per slot (position p -> row p % ring; ring > window), plus one shared scratch ring
    (one 4096-token prefill chunk + window, 256-row pages) used while a long prompt is prefilled"""
    self.block_table = torch.zeros(num_slots, -(-max_seq_len // page), dtype=torch.int32, device=device)
    for layer in self.model.layers:
      attn, hk, d = layer.attn, layer.attn.num_kv_heads, layer.attn.head_dim
      if attn.is_sliding:
        assert ring > attn.sliding_window
        attn._ring, attn._table, shape = ring, None, (num_slots, ring, hk, d)
        if max_seq_len > 4096: attn._scratch = [torch.zeros(18 * 256, hk, d, device=device, dtype=dtype) for _ in range(2)]
      else: attn._ring, attn._table, shape = 0, self.block_table, (num_pages, page, hk, d)
      attn._kv_cache = torch.zeros(shape, device=device, dtype=dtype)
      attn._v_cache = torch.zeros(shape, device=device, dtype=dtype)
      attn._cache_seqlens = torch.zeros(num_slots, dtype=torch.int32, device=device)

  def clear_slot(self, idx: int):
    """Reset cache length for a slot (content remains but will be overwritten on next prefill)"""
    for layer in self.model.layers: layer.attn._cache_seqlens[idx] = 0

  def clear_all_slots(self):
    """Reset all cache lengths (used between independent generation batches)"""
    for layer in self.model.layers: layer.attn._cache_seqlens.zero_()

  def fuse_qkv(self):
    """Apply QKV fusion to all layers after weight loading"""
    for layer in self.model.layers: layer.attn.fuse_qkv()

  @classmethod
  def from_pretrained(cls, path: str, config: GptOssConfig, device, dtype):
    """Load from safetensors with memory-efficient incremental MoE weight processing"""
    path = Path(path)
    files = sorted(path.glob("*.safetensors"))
    if not files: raise FileNotFoundError(f"No safetensors in {path}")
    
    # Create on meta device (no memory allocation); weights materialized during loading
    with torch.device('meta'): model = cls(config)
    
    # First pass: separate MoE expert weights (need swizzling) from regular weights
    non_moe_weights, moe_layers = {}, {}
    for file in tqdm(files, desc="Loading weights", unit="file", ncols=80):
      with safe_open(file, framework="pt", device=str(device)) as sf:
        for key in sf.keys():
          if '.mlp.experts.' in key:
            match = re.search(r'layers\.(\d+)\.mlp\.experts\.', key)
            if match:
              layer_idx = int(match.group(1))
              if layer_idx not in moe_layers: moe_layers[layer_idx] = {}
              moe_layers[layer_idx][key.split('.experts.')[-1]] = (file, key)
          else:
            tensor = sf.get_tensor(key)
            non_moe_weights[key] = tensor.to(dtype) if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16) else tensor
    
    # Materialize non-MoE weights from meta device, mapping HuggingFace key names
    mapped = {k.replace('.self_attn.','.attn.'):v for k,v in non_moe_weights.items()}
    def materialize(module, prefix=""):
      for name, child in module.named_children(): materialize(child, f"{prefix}{name}.")
      for name, param in module.named_parameters(recurse=False):
        full_name = f"{prefix}{name}"
        if full_name in mapped: module._parameters[name] = nn.Parameter(mapped[full_name], requires_grad=False)
        elif param.device.type == 'meta': module._parameters[name] = nn.Parameter(torch.empty(param.shape, device=device, dtype=dtype).normal_(std=0.02), requires_grad=False)
    materialize(model)
    del non_moe_weights; gc.collect(); torch.cuda.empty_cache()
    
    # Process MoE layers one at a time to avoid OOM; swizzle MXFP4 weights for Triton kernels
    file_cache = {}
    for layer_idx in tqdm(sorted(moe_layers.keys()), desc="Processing MoE", unit="layer", ncols=80):
      moe = model.model.layers[layer_idx].mlp
      weights = {}
      
      # Load all expert weights for this layer
      for short_key, (file, full_key) in moe_layers[layer_idx].items():
        if file not in file_cache: file_cache[file] = safe_open(file, framework="pt", device=str(device))
        tensor = file_cache[file].get_tensor(full_key)
        weights[short_key] = tensor.to(dtype) if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16) else tensor
      
      # Swizzle MXFP4 blocks into memory layout required by Triton kernel (critical for performance)
      if all(k in weights for k in ['gate_up_proj_blocks','gate_up_proj_scales','down_proj_blocks','down_proj_scales']):
        num_experts, intermediate, hidden = weights['gate_up_proj_blocks'].shape[0], config.intermediate_size, config.hidden_size
        w1_blocks = weights['gate_up_proj_blocks'].view(num_experts, 2*intermediate, -1).contiguous()
        w1_scales = weights['gate_up_proj_scales'].view(num_experts, 2*intermediate, -1).contiguous()
        w2_blocks = weights['down_proj_blocks'].view(num_experts, hidden, -1).contiguous()
        w2_scales = weights['down_proj_scales'].view(num_experts, hidden, -1).contiguous()
        
        # Rearrange once at load time into the layout the grouped MoE kernel streams (kernels.prep_moe)
        moe._w1, moe._w2 = prep_moe(w1_blocks, w1_scales), prep_moe(w2_blocks, w2_scales)
        if 'gate_up_proj_bias' in weights: moe._w1_bias = weights['gate_up_proj_bias'].to(torch.float32)
        if 'down_proj_bias' in weights: moe._w2_bias = weights['down_proj_bias'].to(torch.float32)
      
      # Router weights are standard float, no special processing
      if 'router.weight' in weights: moe.router.weight.data = weights['router.weight']
      if 'router.bias' in weights: moe.router.bias.data = weights['router.bias']
      
      # Free layer weights after processing to minimize peak memory
      del weights; gc.collect(); torch.cuda.empty_cache()
    
    file_cache.clear()
    return model

