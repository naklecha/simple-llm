"""All GPU kernels for SimpleLLM (gpt-oss): RMSNorm, RoPE, MXFP4 MoE. Triton, no external kernel libraries."""
import torch, triton, triton.language as tl

# ── RMSNorm ──────────────────────────────────────────────────────────────────────
@triton.jit
def _rms_norm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, x_stride, res_stride, out_stride, res_out_stride, n, eps,
                     HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
  row, cols = tl.program_id(0), tl.arange(0, BLOCK)
  mask = cols < n
  x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
  if HAS_RES:  # residual add in fp32, stored back as bf16 (same rounding point as the reference model)
    x += tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(res_out_ptr + row * res_out_stride + cols, x.to(res_out_ptr.dtype.element_ty), mask=mask)
  w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
  y = x * tl.rsqrt(tl.sum(x * x, axis=0) / n + eps) * w
  tl.store(out_ptr + row * out_stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)

def _norm(x, residual, weight, eps):
  x2 = x.reshape(-1, x.shape[-1])
  out = torch.empty_like(x2)
  res2 = residual.reshape(-1, x.shape[-1]) if residual is not None else x2
  res_out = torch.empty_like(x2) if residual is not None else out
  _rms_norm_kernel[(x2.shape[0],)](x2, res2, weight, out, res_out, x2.stride(0), res2.stride(0), out.stride(0), res_out.stride(0),
                                   x2.shape[1], eps, HAS_RES=residual is not None, BLOCK=triton.next_power_of_2(x2.shape[1]))
  return out.view(x.shape), res_out.view(x.shape)

def rms_norm(x, weight, eps=1e-5): return _norm(x, None, weight, eps)[0]

def fused_add_rms_norm(x, residual, weight, eps=1e-5):
  """Returns (rmsnorm(x + residual) * weight, x + residual)."""
  return _norm(x, residual, weight, eps)

# ── RoPE (NeoX rotate-half, cos/sin tables already carry the YaRN mscale) ──────────
@triton.jit
def _rope_kernel(q_ptr, k_ptr, q_out_ptr, k_out_ptr, pos_ptr, cos_ptr, sin_ptr, q_tok, q_head, k_tok, k_head, qo_tok, qo_head, ko_tok, ko_head,
                 cs_stride, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr):
  t = tl.program_id(0)
  pos = tl.load(pos_ptr + t)
  d = tl.arange(0, D)
  rot = tl.where(d < D // 2, d + D // 2, d - D // 2)  # rotate_half partner index
  cos = tl.load(cos_ptr + pos * cs_stride + d).to(tl.float32)[None, :]
  sin = tl.load(sin_ptr + pos * cs_stride + d).to(tl.float32)[None, :]
  # x*cos + rotate_half(x)*sin with the original kernel's rounding: bf16(rot*sin) first, then one rounding of x*cos + that
  hq = tl.arange(0, HQ)[:, None]
  q = tl.load(q_ptr + t * q_tok + hq * q_head + d[None, :])
  qr = tl.load(q_ptr + t * q_tok + hq * q_head + rot[None, :])
  qr = tl.where(d[None, :] < D // 2, -qr, qr)
  tl.store(q_out_ptr + t * qo_tok + hq * qo_head + d[None, :], (q.to(tl.float32) * cos + (qr * sin).to(tl.bfloat16).to(tl.float32)).to(q.dtype))
  hk = tl.arange(0, HK)[:, None]
  k = tl.load(k_ptr + t * k_tok + hk * k_head + d[None, :])
  kr = tl.load(k_ptr + t * k_tok + hk * k_head + rot[None, :])
  kr = tl.where(d[None, :] < D // 2, -kr, kr)
  tl.store(k_out_ptr + t * ko_tok + hk * ko_head + d[None, :], (k.to(tl.float32) * cos + (kr * sin).to(tl.bfloat16).to(tl.float32)).to(k.dtype))

def fused_rope(positions, query, key, cos_cache, sin_cache):
  """query [..., Hq, D], key [..., Hk, D] (strided views allowed, last dim contiguous); positions broadcast over the leading dims.
  Any number of tokens (decode at any batch, or prefill). Returns new (query, key)."""
  hq, hk, d = query.shape[-2], key.shape[-2], query.shape[-1]
  pos = positions.expand(query.shape[:-2]).reshape(-1)
  q, k = query.reshape(-1, hq, d) if query.stride(-1) == 1 else query.contiguous().view(-1, hq, d), key.reshape(-1, hk, d) if key.stride(-1) == 1 else key.contiguous().view(-1, hk, d)
  q_out, k_out = torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(k.shape, dtype=k.dtype, device=k.device)
  _rope_kernel[(q.shape[0],)](q, k, q_out, k_out, pos, cos_cache, sin_cache, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                              q_out.stride(0), q_out.stride(1), k_out.stride(0), k_out.stride(1), cos_cache.stride(0), HQ=hq, HK=hk, D=d)
  return q_out.view(query.shape), k_out.view(key.shape)

# ── Decode attention: GQA split-KV (flash-decoding) with sliding window, KV append and gpt-oss sinks ─────────────────
@triton.jit
def _attn_decode_kernel(q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, len_ptr, slot_ptr, m_ptr, l_ptr, o_ptr, bt_ptr, q_tok, k_tok, v_tok, c_slot, c_pos, bt_slot,
                        scale, WINDOW: tl.constexpr, RING: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
  # program (token, kv head, split): the G query heads sharing this kv head attend over one slice of keys [lo, L]
  b, hk, sp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
  slot = tl.load(slot_ptr + b).to(tl.int64)
  L = tl.load(len_ptr + slot)  # tokens already cached; the new token sits at position L
  lo = tl.maximum(L - WINDOW, 0) if WINDOW > 0 else 0  # flash-attn window (W, 0): keys j >= L - W
  chunk = tl.cdiv(tl.cdiv(L + 1 - lo, NSPLIT), BLOCK) * BLOCK
  start, end = lo + sp * chunk, tl.minimum(lo + sp * chunk + chunk, L + 1)
  g, d, n = tl.arange(0, 16), tl.arange(0, D), tl.arange(0, BLOCK)
  q = tl.load(q_ptr + b * q_tok + (hk * G + g[:, None]) * D + d[None, :], mask=(g < G)[:, None], other=0.0)  # 16 rows for the MMA
  kn = tl.load(k_ptr + b * k_tok + hk * D + d)
  vn = tl.load(v_ptr + b * v_tok + hk * D + d)
  if sp == 0:  # append the new token's K/V (other splits read it from registers, never from the cache)
    row = L % RING if RING > 0 else L  # sliding layers keep a ring of RING rows: position p lives in row p % RING
    if PAGE > 0: row = tl.load(bt_ptr + slot * bt_slot + L // PAGE).to(tl.int64) * PAGE + L % PAGE  # paged: block table -> pool row
    tl.store(kc_ptr + slot * c_slot + row * c_pos + hk * D + d, kn)
    tl.store(vc_ptr + slot * c_slot + row * c_pos + hk * D + d, vn)
  m, l, acc = tl.full([16], float("-inf"), tl.float32), tl.zeros([16], tl.float32), tl.zeros([16, D], tl.float32)
  for s0 in range(start, end, BLOCK):
    pos = s0 + n
    old = (pos < end) & (pos != L)
    rows = (pos % RING) if RING > 0 else pos
    if PAGE > 0: rows = tl.load(bt_ptr + slot * bt_slot + pos // PAGE, mask=old, other=0).to(tl.int64) * PAGE + pos % PAGE
    off = slot * c_slot + rows[:, None] * c_pos + hk * D + d[None, :]
    k = tl.where((pos == L)[:, None], kn[None, :], tl.load(kc_ptr + off, mask=old[:, None], other=0.0))
    v = tl.where((pos == L)[:, None], vn[None, :], tl.load(vc_ptr + off, mask=old[:, None], other=0.0))
    sc = tl.dot(q, tl.trans(k)) * scale  # scale includes log2(e)
    sc = tl.where((pos < end)[None, :], sc, float("-inf"))
    mn = tl.maximum(m, tl.max(sc, 1))
    p = tl.exp2(sc - mn[:, None])
    al = tl.exp2(m - mn)
    l = l * al + tl.sum(p, 1)
    acc = acc * al[:, None] + tl.dot(p.to(v.dtype), v)
    m = mn
  hq = hk * G + g
  idx = (b * (G * tl.num_programs(1)) + hq) * NSPLIT + sp
  tl.store(m_ptr + idx, m, mask=g < G)
  tl.store(l_ptr + idx, l, mask=g < G)
  tl.store(o_ptr + idx[:, None] * D + d[None, :], acc, mask=(g < G)[:, None])

@triton.jit
def _attn_combine_kernel(m_ptr, l_ptr, o_ptr, sink_ptr, out_ptr, len_ptr, slot_ptr, H, NSPLIT: tl.constexpr, D: tl.constexpr):
  # merge the splits, then the gpt-oss sink: out * sigmoid(clamp(lse - sink, -5, 5)) (decode clamp as in the reference model)
  b, h = tl.program_id(0), tl.program_id(1)
  sp, d = tl.arange(0, NSPLIT), tl.arange(0, D)
  base = (b * H + h) * NSPLIT
  m = tl.load(m_ptr + base + sp)
  w = tl.exp2(m - tl.max(m, 0))
  lsum = tl.sum(tl.load(l_ptr + base + sp) * w, 0)
  o = tl.sum(tl.load(o_ptr + (base + sp)[:, None] * D + d[None, :]) * w[:, None], 0) / lsum
  lse = (tl.max(m, 0) + tl.log2(lsum)) * 0.6931471805599453
  f = tl.sigmoid(tl.minimum(tl.maximum(lse - tl.load(sink_ptr + h).to(tl.float32), -5.0), 5.0))
  tl.store(out_ptr + (b * H + h) * D + d, (o.to(tl.bfloat16).to(tl.float32) * f.to(tl.bfloat16).to(tl.float32)).to(out_ptr.dtype.element_ty))  # original's bf16 rounding
  if h == 0:  # every split has read the old length: advance it
    slot = tl.load(slot_ptr + b)
    tl.store(len_ptr + slot, tl.load(len_ptr + slot) + 1)

def decode_attention(q, k, v, k_cache, v_cache, seqlens, slots, sinks, scale, window=0, ring=0, table=None):
  """One new token per sequence. q [B,1,H,D], k/v [B,1,Hk,D] (strided views ok); caches [S, T, Hk, D]; seqlens [S] int32 is
  read, the new K/V appended at that position and the length advanced; ring > 0: position p is cache row p % ring (window < ring).
  table [S, pages] int32: paged cache, caches are a pool [P, page, Hk, D] and position p of slot s is row p % page of page table[s, p // page].
  Returns [B, 1, H*D] = sink-scaled attention output."""
  page = k_cache.shape[1] if table is not None else 0
  b, _, h, d = q.shape
  hk = k.shape[2]
  nsplit = max(1, min(64, triton.next_power_of_2(triton.cdiv(264, b * hk))))
  m = torch.empty(b * h * nsplit, dtype=torch.float32, device=q.device)
  l, o = torch.empty_like(m), torch.empty(b * h * nsplit, d, dtype=torch.float32, device=q.device)
  _attn_decode_kernel[(b, hk, nsplit)](q, k, v, k_cache, v_cache, seqlens, slots, m, l, o, seqlens if table is None else table, q.stride(0), k.stride(0),
                                       v.stride(0), 0 if page else k_cache.stride(0), k_cache.stride(1), table.stride(0) if page else 0,
                                       scale * 1.4426950408889634, WINDOW=window or 0, RING=ring, PAGE=page, NSPLIT=nsplit,
                                       G=h // hk, D=d, BLOCK=64, num_warps=4)
  out = torch.empty(b, 1, h * d, dtype=q.dtype, device=q.device)
  _attn_combine_kernel[(b, h)](m, l, o, sinks, out, seqlens, slots, h, NSPLIT=nsplit, D=d, num_warps=1)
  return out

# ── MoE grouped (any token count): (token, expert) pairs sorted by expert, tensor-core tiles on fragment-ordered MXFP4 ──
# Weight layout + upcast follow the Hopper path of OpenAI's triton_kernels (MIT): at load time each expert's [N, K/2] bytes
# are permuted into wgmma fragment order (16-byte loads per thread) and the bits of 4 consecutive values are interleaved so
# the bf16 upcast is and/shift + one mul.bf16x2 per 2 values; weights are the left tl.dot operand (no layout conversion).
@triton.jit
def _topk_kernel(logit_ptr, ids_ptr, gam_ptr, E, stride, TOPK: tl.constexpr, BE: tl.constexpr):
  # gpt-oss routing: top-k on the router logits, softmax over the k selected (lowest index wins ties)
  t, e = tl.program_id(0), tl.arange(0, BE)
  l = tl.load(logit_ptr + t * stride + e, mask=e < E, other=float("-inf")).to(tl.float32)
  j, v = tl.arange(0, TOPK), tl.zeros([TOPK], tl.float32)
  for i in tl.static_range(TOPK):
    m = tl.max(l, 0)
    idx = tl.min(tl.where(l == m, e, BE), 0)
    tl.store(ids_ptr + t * TOPK + i, idx)
    v = tl.where(j == i, m, v)
    l = tl.where(e == idx, float("-inf"), l)
  p = tl.exp(v - tl.max(v, 0))
  tl.store(gam_ptr + t * TOPK + j, (p / tl.sum(p, 0)).to(tl.bfloat16).to(tl.float32))  # routing weights in bf16 like the original

@triton.jit
def _combine_kernel(part_ptr, out_ptr, N, TOPK: tl.constexpr, BLOCK: tl.constexpr):
  t, c = tl.program_id(0), tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
  acc = tl.zeros([BLOCK], tl.float32)
  for j in tl.static_range(TOPK): acc += tl.load(part_ptr + (t * TOPK + j) * N + c, mask=c < N, other=0.0)
  tl.store(out_ptr + t * N + c, acc.to(out_ptr.dtype.element_ty), mask=c < N)

def prep_moe(blocks, scales):
  """checkpoint blocks [E, N, K/32, 16] + scales [E, N, K/32] -> (fragment-ordered uint8 [E, N/4, 2K], tiled scales [E, N/32, K])
  (kernel uses num_warps=4: the scale tiling matches 4 warps)."""
  e, n = blocks.shape[:2]
  if e > 8:  # experts are independent: 8 at a time keeps the int32 temporaries small (128-expert layers)
    parts = [prep_moe(blocks[i:i + 8], scales[i:i + 8]) for i in range(0, e, 8)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
  w = blocks.reshape(e, n, -1)
  w = torch.nn.functional.pad(w, (0, -w.shape[2] % 32, 0, -n % 16))
  np_, kb = w.shape[1], w.shape[2]
  w = w.view(e, np_ // 16, 1, 2, 4, 2, 1, kb // 32, 4, 2, 4, 1, 1).permute(0, 1, 4, 7, 11, 5, 10, 8, 2, 9, 3, 6, 12).reshape(e, np_ // 4, kb * 4 // 4, 4).to(torch.int32)
  lo, hi = w & 15, w >> 4
  pair = lambda f, i: f(lo[..., i]) | (f(hi[..., i]) << 16)
  first, second, third = (pair(lambda v: ((v & 8) << 12) | ((v & 7) << 6), i) for i in range(3))
  fourth = pair(lambda v: ((v & 8) << 11) | ((v & 6) << 9) | ((v & 1) << 13), 3)
  srl = lambda x, s: (x >> s) & ((1 << (32 - s)) - 1)
  w = (first | srl(second, 3) | srl(third, 6) | fourth).contiguous().view(torch.uint8).reshape(e, np_ // 4, kb * 4)
  s = torch.nn.functional.pad(scales.reshape(e, n, -1), (0, scales.shape[2] % 2, 0, -n % 128))  # scales: 32-row x 2-group tiles
  ns, ks = s.shape[1], s.shape[2]
  s = s.reshape(e, ns // 128, 2, 4, 2, 8, ks // 2, 2).permute(0, 1, 3, 6, 2, 5, 7, 4).reshape(e, ns // 32, ks * 32).contiguous()
  return w, s

@triton.jit
def _upcast_mxfp4(w, s, BN: tl.constexpr, BK: tl.constexpr):
  # w uint8 [BN/4, 2*BK] fragment-ordered, s uint8 [BN, BK/32] E8M0 -> bf16 [BN, BK] = e2m1 * 2^(s-127)
  r0, r1 = tl.inline_asm_elementwise(r"""{
    .reg .b32 b, c, d<7>, bias;
    mov.b32 bias, 0x7e807e80;
    and.b32 $0, $4, 0b10000001110000001000000111000000; mul.bf16x2 $0, $0, bias;
    shl.b32 b, $4, 3; and.b32 $1, b, 0b10000001110000001000000111000000; mul.bf16x2 $1, $1, bias;
    shl.b32 c, $4, 6; and.b32 $2, c, 0b10000001110000001000000111000000; mul.bf16x2 $2, $2, bias;
    shl.b32 d0, $4, 1; and.b32 d1, d0, 0b10000000000000001000000000000000;
    shr.b32 d2, $4, 3; and.b32 d3, d2, 0b00000001100000000000000110000000; or.b32 d4, d1, d3;
    shr.b32 d5, $4, 7; and.b32 d6, d5, 0b00000000010000000000000001000000; or.b32 $3, d4, d6; mul.bf16x2 $3, $3, bias;
    }""", constraints="=r,=r,=r,=r,r", args=[w], dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
  x = tl.join(r0, r1).reshape(BN // 4, BK // 2, 4, 2).trans(0, 1, 3, 2).reshape(BN // 4, 4 * BK)
  x = x.reshape(BN // 16, 4, BK // 64, 2, 4, 8, 2, 2).trans(0, 6, 1, 3, 2, 5, 4, 7).reshape(BN, BK)  # undo the fragment order
  s = s.reshape(BN // 32 // 4, 4, BK // 64, 2, 8, 2, 2).trans(0, 3, 1, 6, 4, 2, 5).reshape(BN, BK // 32)  # undo the scale tiling
  sc = (s.to(tl.int16) << 7).to(tl.bfloat16, bitcast=True)  # E8M0 -> bf16 2^(s-127)
  return x * sc.expand_dims(2).broadcast_to(BN, BK // 32, 32).reshape(BN, BK)

@triton.jit
def _sort_pairs_kernel(ids_ptr, sorted_ptr, tile_e_ptr, tile_s_ptr, tile_n_ptr, P, BP: tl.constexpr, BE: tl.constexpr,
                       BM: tl.constexpr, MAXT: tl.constexpr, MAXJ: tl.constexpr):
  # one program: stable sort of (token, expert) pairs by expert, then a tile table (expert, first sorted row, rows)
  p = tl.arange(0, BP)
  ids = tl.load(ids_ptr + p, mask=p < P, other=0)
  key = tl.sort(tl.where(p < P, ids * BP + p, BE * BP + p))
  tl.store(sorted_ptr + p, key % BP, mask=p < P)
  cnt = tl.histogram(ids, BE) - tl.where(tl.arange(0, BE) == 0, BP - P, 0)
  off = tl.cumsum(cnt, 0) - cnt
  tiles = (cnt + BM - 1) // BM
  t0 = tl.cumsum(tiles, 0) - tiles
  tl.store(tile_e_ptr + tl.arange(0, MAXT), -1)  # unused tiles; written per expert below (a [tiles, experts] search was slow at 128 experts)
  tl.debug_barrier()
  j = tl.arange(0, MAXJ)[None, :]
  m = j < tiles[:, None]
  tl.store(tile_e_ptr + t0[:, None] + j, tl.arange(0, BE)[:, None] + 0 * j, mask=m)
  tl.store(tile_s_ptr + t0[:, None] + j, off[:, None] + j * BM, mask=m)
  tl.store(tile_n_ptr + t0[:, None] + j, tl.minimum(cnt[:, None] - j * BM, BM), mask=m)

def _decode_cfg(configs, args, **kw):  # decode batches use one fixed tile: graph warmup routes every row to the same experts, so tuning there picks badly
  fixed = [c for c in configs if (c.kwargs["BN"], c.kwargs["BK"], c.num_stages, c.num_warps) == (128, 128, 4, 4)]
  return fixed if args["PB"] <= 1024 and fixed else configs

@triton.autotune(configs=[triton.Config(dict(BN=bn, BK=bk), num_warps=w, num_stages=st) for bn in (128, 256) for bk in (64, 128, 256)
                          for w in (4, 8) for st in (3, 4)], key=["N", "K", "PB", "BM"], prune_configs_by=dict(early_config_prune=_decode_cfg))
@triton.jit
def _moe_grouped_kernel(x_ptr, w_ptr, s_ptr, b_ptr, sorted_ptr, tile_e_ptr, tile_s_ptr, tile_n_ptr, gam_ptr, out_ptr, N, K, PB,
                        x_stride, out_stride, TOPK: tl.constexpr, SWIGLU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
  tile, pid_n = tl.program_id(0), tl.program_id(1)
  e = tl.load(tile_e_ptr + tile)
  if e < 0: return
  e = e.to(tl.int64)
  m = tl.arange(0, BM)
  mmask = m < tl.load(tile_n_ptr + tile)
  pair = tl.load(sorted_ptr + tl.load(tile_s_ptr + tile) + m, mask=mmask, other=0)
  xrow = pair // TOPK if SWIGLU else pair  # gate/up reads token rows, down reads the pair's intermediate row
  rows = pid_n * BN + tl.arange(0, BN)
  r4 = pid_n * (BN // 4) + tl.arange(0, BN // 4)
  kw, kk = tl.arange(0, 2 * BK), tl.arange(0, BK)
  acc = tl.zeros([BN, BM], tl.float32)  # acc^T = W x^T
  for k0 in range(0, K, BK):
    w = tl.load(w_ptr + e * ((N + 15) // 16 * 4) * (2 * K) + r4[:, None] * (2 * K) + 2 * k0 + kw[None, :],
                mask=(r4 < (N + 15) // 16 * 4)[:, None] & (2 * k0 + kw < 2 * K)[None, :], other=0)
    r32, kc = pid_n * (BN // 32) + tl.arange(0, BN // 32), tl.arange(0, BK)
    s = tl.load(s_ptr + e * ((N + 127) // 128 * 4) * (K // 32 * 32) + r32[:, None] * (K // 32 * 32) + k0 + kc[None, :],
                mask=(r32 < (N + 127) // 128 * 4)[:, None] & (k0 + kc < K)[None, :], other=0)
    xt = tl.load(x_ptr + xrow[None, :] * x_stride + k0 + kk[:, None], mask=mmask[None, :] & (k0 + kk < K)[:, None], other=0.0)
    acc = tl.dot(_upcast_mxfp4(w, s, BN, BK), xt, acc)
  y = acc + tl.load(b_ptr + e * N + rows, mask=rows < N, other=0.0)[:, None]
  if SWIGLU:  # interleaved rows: 2i = gate, 2i+1 = up; gpt-oss clamp 7, alpha 1.702, (up + 1)
    gt, up = tl.split(tl.reshape(tl.trans(y), [BM, BN // 2, 2]))
    gt, up = tl.minimum(gt, 7.0), tl.minimum(tl.maximum(up, -7.0), 7.0)
    cols = pid_n * (BN // 2) + tl.arange(0, BN // 2)
    tl.store(out_ptr + pair[:, None] * out_stride + cols[None, :], (gt * tl.sigmoid(1.702 * gt) * (up + 1)).to(out_ptr.dtype.element_ty),
             mask=mmask[:, None] & (cols < N // 2)[None, :])
  else:
    g = tl.load(gam_ptr + pair, mask=mmask, other=0.0)
    tl.store(out_ptr + pair[None, :] * out_stride + rows[:, None], (y * g[None, :]).to(tl.bfloat16).to(tl.float32), mask=mmask[None, :] & (rows < N)[:, None])  # bf16 per-expert output like the original

def moe_grouped(x, w1, s1, b1, w2, s2, b2, router_logits, top_k=4, BM=None):
  """x [T, H] bf16; w*/s* from prep_moe; b* fp32 [E, N]; router_logits [T, E]. Returns [T, H] bf16.
  BM = token rows per tile: 16 for decode; prefill uses bigger tiles so each dequantized weight tile serves more tokens."""
  t, h = x.shape
  e, n1, i = router_logits.shape[1], b1.shape[1], w2.shape[2] // 2
  p = t * top_k
  BM = BM or min(64, max(16, triton.next_power_of_2(max(1, p // e))))  # decode: 16; 1k-token prefill at 128 experts: 32; 4k: 64
  ids = torch.empty(p, dtype=torch.int32, device=x.device)
  gam = torch.empty(p, dtype=torch.float32, device=x.device)
  be = triton.next_power_of_2(e)
  _topk_kernel[(t,)](router_logits, ids, gam, e, router_logits.stride(0), TOPK=top_k, BE=be)
  grid_t = triton.cdiv(p, BM) + min(e, p)  # upper bound on non-empty tiles
  maxt = triton.next_power_of_2(grid_t)
  srt = torch.empty(p, dtype=torch.int32, device=x.device)
  te, ts, tn = (torch.empty(maxt, dtype=torch.int32, device=x.device) for _ in range(3))
  _sort_pairs_kernel[(1,)](ids, srt, te, ts, tn, p, BP=max(triton.next_power_of_2(p), 16), BE=be, BM=BM, MAXT=maxt,
                           MAXJ=triton.next_power_of_2(triton.cdiv(p, BM)), num_warps=8)
  inter = torch.empty(p, n1 // 2, dtype=x.dtype, device=x.device)
  _moe_grouped_kernel[lambda m: (grid_t, triton.cdiv(n1, m["BN"]))](x, w1, s1, b1, srt, te, ts, tn, gam, inter, n1, h, triton.next_power_of_2(p), x.stride(0), inter.stride(0),
                                                                  TOPK=top_k, SWIGLU=True, BM=BM)
  part = torch.empty(p, h, dtype=torch.float32, device=x.device)
  _moe_grouped_kernel[lambda m: (grid_t, triton.cdiv(h, m["BN"]))](inter, w2, s2, b2, srt, te, ts, tn, gam, part, h, i, triton.next_power_of_2(p), inter.stride(0), part.stride(0),
                                                                 TOPK=top_k, SWIGLU=False, BM=BM)
  out = torch.empty(t, h, dtype=x.dtype, device=x.device)
  _combine_kernel[(t, triton.cdiv(h, 1024))](part, out, h, TOPK=top_k, BLOCK=1024)
  return out
