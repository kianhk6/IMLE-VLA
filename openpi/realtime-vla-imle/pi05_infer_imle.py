import torch
import triton
import triton.language as tl
import numpy as np
import torch.nn as nn
import os
import sentencepiece
from pi0_infer_imle import (
    vision_encoder,
    layer_norm_matmul_n256_1152_2048_bias,
    rms_matmul_n_2048_2560_qkv_rope,
    matmul_n_2048_2048_res,
    matmul_n_16384_2048_res,
    rms_matmul_n_2048_16384_gate,
    matmul_small_bias,
    matmul_small_bias_silu,
    matmul_small_gate_up_hybrid,
    matmul_small_gate_dispatch,
    matmul_k8_n_256,
    matmul_abT_scale,
    matmul_abT_scale_dispatch,
    matmul_small_res_gate_hybrid,
    # ── Round-2 throughput optimisation imports ──
    rms_norm_kernel,
    matmul_res_fused_rms,
    matmul_qkv_rope_only,
    USE_GATED_CONCAT,
    rope_split_qkv_kernel,
    _get_qkv_tmp,
)

USE_HYBRID_RES_GATE = os.getenv("USE_HYBRID_RES_GATE", "1") == "1"
USE_FUSED_ADD_RMS_NORM = os.getenv("USE_FUSED_ADD_RMS_NORM", "1") == "1"
USE_CUBLAS_ROPE_QKV    = os.getenv("USE_CUBLAS_ROPE_QKV", "1") == "1"
USE_CUBLAS_MATMUL_ROPE_QKV = os.getenv("USE_CUBLAS_MATMUL_ROPE_QKV", "1") == "1"  # cuBLAS for matmul_rope_qkv kernel
USE_FUSED_ADARMS_QKV   = os.getenv("USE_FUSED_ADARMS_QKV", "1") == "1"  # Fuse adarms_norm + QKV + RoPE


# ============================================================================
# Custom FlashAttention (forward-only) for encoder, supporting headdim=256.
# Based on Dao-AILab's Triton implementation, simplified and adapted for
# (batch=1, nheads=8, headdim=256) and valid_len masking.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=1),
        # Extra configs for wider search on different SMs / seq lengths
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16}, num_warps=2, num_stages=2),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _flash_attn_forward_kernel(
    Q,
    K,
    V,
    Out,
    valid_len_ptr,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    valid_len = tl.load(valid_len_ptr).to(tl.int32)

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= softmax_scale

        # Mask out keys beyond valid_len
        mask = (start_n + offs_n)[None, :] < valid_len
        qk = tl.where(mask, qk, float("-inf"))

        # Numerically stable online softmax (relative to m_i_new)
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        l_i_new = alpha * l_i + l_ij

        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o = acc_o * alpha[:, None] + tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new

    acc_o = acc_o / l_i[:, None]

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o.to(tl.bfloat16))
        else:
            tl.store(out_ptrs, acc_o.to(tl.bfloat16), mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o.to(tl.bfloat16), mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs,
                acc_o.to(tl.bfloat16),
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            )


def flash_attention_forward(q, k, v, valid_len, softmax_scale=None, causal: bool = False):
    # q, k, v: (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
    if q.ndim == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        batch = 1
        unbatched = True
    else:
        batch = q.shape[0]
        unbatched = False

    seqlen_q, nheads, headdim = q.shape[1], q.shape[2], q.shape[3]
    seqlen_k = k.shape[1]

    # Ensure last dim is contiguous
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

    if softmax_scale is None:
        softmax_scale = 1.0 / (headdim**0.5)

    seqlen_q_rounded = ((seqlen_q + 127) // 128) * 128

    o = torch.empty_like(q)

    BLOCK_HEADDIM = triton.next_power_of_2(headdim)
    # Our model uses headdim=256; support up to 256 cleanly.
    if BLOCK_HEADDIM > 256:
        BLOCK_HEADDIM = 256

    CACHE_KEY_SEQLEN_Q = seqlen_q // 32
    CACHE_KEY_SEQLEN_K = seqlen_k // 32

    def get_grid(META):
        return (
            triton.cdiv(seqlen_q, META["BLOCK_M"]),
            batch * nheads,
        )

    _flash_attn_forward_kernel[get_grid](
        q,
        k,
        v,
        o,
        valid_len,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    if unbatched:
        o = o.squeeze(0)

    # LSE is not used in inference, return None for API compatibility.
    return o, None

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        # Extra configs to cover more seq_len/hidden shapes
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=1),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def matmul_small_res_gate(inp_ptr, weight_ptr, out_ptr, res_ptr, gate_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        matmul_acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            matmul_acc = tl.dot(x, w, matmul_acc)
        
        gate = tl.load(
            gate_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)

        acc += matmul_acc * gate
        
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

def matmul_k_32_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )

@triton.jit
def _store_gate_kernel(style_ptr, gate_ptr, seq_len: tl.constexpr, features: tl.constexpr):
    """Simple kernel to store gate from style buffer."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    cols = tl.arange(0, features)
    s_gate = tl.load(style_ptr + 2 * features + cols, mask=cols < features, other=0.0).to(tl.bfloat16)
    for i in range(pid, seq_len, psize):
        tl.store(gate_ptr + i * features + cols, s_gate, mask=cols < features)

@triton.jit
def adarms_norm_kernel(
    x_ptr,
    style_ptr,
    normed_x_ptr, 
    gate_ptr, 
    seq_len: tl.constexpr, 
    features: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr = 1024  # Set to max features for single-pass
):
    """Adaptive RMS normalization with style modulation. Single-pass optimized."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        # Load entire row at once (features=1024 fits in BLOCK_SIZE=1024)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < features
        
        # Single-pass: load x, compute sum_sq, then normalize
        x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Compute RMS normalization factor
        sum_sq = tl.sum(x_val * x_val)
        rms_factor = tl.rsqrt(sum_sq / features + 1e-6)
        
        # Normalize and apply style modulation in one go
        x_norm = x_val * rms_factor
        s_scale = tl.load(style_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s_shift = tl.load(style_ptr + features + cols, mask=mask, other=0.0).to(tl.float32)
        s_gate = tl.load(style_ptr + 2 * features + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Apply style: output = x_norm * (1 + scale) + shift
        output_val = x_norm * (1.0 + s_scale) + s_shift
        
        # Store both outputs - SINGLE PASS!
        tl.store(normed_x_ptr + row_x_offset + cols, output_val.to(tl.bfloat16), mask=mask)
        tl.store(gate_ptr + row_x_offset + cols, s_gate.to(tl.bfloat16), mask=mask)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "head_dim", "num_heads"],
)
@triton.jit
def adarms_norm_qkv_rope_kernel(
    x_ptr, style_ptr, gate_ptr,
    weight_qkv_ptr, rope_weights_ptr,
    q_ptr, k_ptr, v_ptr,
    seq_len: tl.constexpr, features: tl.constexpr,
    head_dim: tl.constexpr, num_heads: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 64, BLOCK_SIZE_N: tl.constexpr = 32, BLOCK_SIZE_K: tl.constexpr = 64,
):
    """Fused: adarms_norm + QKV matmul + RoPE in a single kernel.
    Eliminates intermediate buffer write/read and reduces kernel launches."""
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    
    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)
    
    assert head_dim % BLOCK_SIZE_N == 0
    
    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]
        
        # Phase 1: Compute RMS normalization per row
        sum_sq_per_row = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < features
            x = tl.load(
                x_ptr + offs_i * features + offs_k[None, :],
                mask=(offs_i < seq_len) & mask_k[None, :],
                other=0.0
            ).to(tl.float32)
            # Accumulate sum of squares per row
            sum_sq_per_row += tl.sum(x * x, axis=1)
        rms_factors = tl.rsqrt(sum_sq_per_row / features + 1e-6)
        
        # Phase 2: Matmul with normalized + styled input
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            mask_k = offs_k < features
            
            # Load x, normalize, and apply style
            x = tl.load(
                x_ptr + offs_i * features + offs_k[None, :],
                mask=(offs_i < seq_len) & mask_k[None, :],
                other=0.0
            ).to(tl.float32)
            
            # Normalize: x_norm = x * rms_factor (per row)
            x_norm = x * rms_factors[:, None]
            
            # Apply style modulation
            s_scale = tl.load(style_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            s_shift = tl.load(style_ptr + features + offs_k, mask=mask_k, other=0.0).to(tl.float32)
            x_mod = x_norm * (1.0 + s_scale[None, :]) + s_shift[None, :]
            
            # Load weight and accumulate
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask=mask_k[:, None] & (offs_j < (num_heads + 2) * head_dim),
                other=0.0
            )
            accumulator = tl.dot(x_mod, w, accumulator)
        
        # Phase 3: Apply RoPE to Q and K, then split QKV
        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(
                rope_weights_ptr + offs_i * head_dim + offs_j % head_dim,
                mask=offs_i < seq_len, other=0.0
            )
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)
        
        accumulator = accumulator.to(tl.bfloat16)
        
        # Phase 4: Route to Q/K/V buffers
        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask=(offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        
        pid += psize

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        # Extra configs: trade off BLOCK_M/BLOCK_N and pipeline depth
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def adarms_norm_matmul_bias_kernel(
    x_ptr,           # [seq_len, features], bf16
    style_ptr,       # [3 * features], bf16
    weight_ptr,      # [features, hidden], bf16
    bias_ptr,        # [hidden], bf16
    out_ptr,         # [seq_len, hidden], bf16
    seq_len: tl.constexpr,
    features: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < seq_len
    mask_n = offs_n < hidden

    # 1) Compute RMS per row across features
    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, features, BLOCK_K):
        k_idx = k + offs_k
        mask_k = k_idx < features
        x = tl.load(
            x_ptr + offs_m[:, None] * features + k_idx[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=1)
    rms = tl.rsqrt(sum_sq / features + 1e-6)

    # 2) Matmul over normalized, styled x
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, features, BLOCK_K):
        k_idx = k + offs_k
        mask_k = k_idx < features

        x = tl.load(
            x_ptr + offs_m[:, None] * features + k_idx[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        x_norm = x * rms[:, None]

        s_scale = tl.load(style_ptr + k_idx, mask=mask_k, other=0.0).to(tl.float32)
        s_shift = tl.load(style_ptr + features + k_idx, mask=mask_k, other=0.0).to(tl.float32)
        # Gate lives in style_ptr + 2 * features, but is not used in this fused path.

        x_mod = x_norm * (1.0 + s_scale[None, :]) + s_shift[None, :]

        w = tl.load(
            weight_ptr + k_idx[:, None] * hidden + offs_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x_mod, w)

    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    tl.store(
        out_ptr + offs_m[:, None] * hidden + offs_n[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )

def matmul_1_1024_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        # Extra configs for different matmul tilings
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
    ],
    key=["seq_len", "features"],
)
@triton.jit
def matmul_rope_qkv(
    inp_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    weight_qkv_ptr, rope_weights_ptr, q_ptr, k_ptr, v_ptr,
    BLOCK_SIZE_M : tl.constexpr = 64, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    grid_m = triton.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = triton.cdiv((num_heads + 2) * head_dim, BLOCK_SIZE_N)

    assert head_dim % BLOCK_SIZE_N == 0, f"head_dim {head_dim} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_N}"

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = (offs_i < seq_len) & (offs_k[None, :] < features),
                other = 0,
            )
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = (offs_k[:, None] < features) & (offs_j < (num_heads + 2) * head_dim),
                other = 0
            )
            accumulator = tl.dot(x, w, accumulator)

        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(accumulator.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(rope_weights_ptr + offs_i * head_dim + offs_j % head_dim, mask = offs_i < seq_len, other = 0)
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            accumulator = tl.interleave(x0_, x1_)

        accumulator = accumulator.to(tl.bfloat16)

        if start_j < num_heads * head_dim:
            out_ptr = q_ptr
            out_stride = num_heads * head_dim
        elif start_j < (num_heads + 1) * head_dim:
            out_ptr = k_ptr
            out_stride = head_dim
        else:
            out_ptr = v_ptr
            out_stride = head_dim
        tl.store(
            out_ptr + offs_i * out_stride + offs_j % out_stride,
            accumulator,
            mask = (offs_i < seq_len) & (offs_j < (num_heads + 2) * head_dim)
        )
        pid += psize

def matmul_rope_qkv_dispatch(inp, weight_qkv, rope_weight, Q, K, V, seq_len, features, head_dim, num_heads):
    """Dispatch function for matmul_rope_qkv: cuBLAS GEMM + Triton RoPE epilogue or full Triton kernel."""
    if USE_CUBLAS_MATMUL_ROPE_QKV:
        # cuBLAS GEMM + Triton RoPE epilogue (replaces fused Triton GEMM+RoPE)
        qkv_tmp = _get_qkv_tmp(inp, (num_heads + 2) * head_dim)
        torch.matmul(inp.view(seq_len, features), weight_qkv, out=qkv_tmp)
        rope_split_qkv_kernel[(128,)](qkv_tmp, rope_weight, Q, K, V, seq_len, head_dim, num_heads)
    else:
        # Original fused Triton kernel
        matmul_rope_qkv[(128,)](
            inp, seq_len, features, head_dim, num_heads,
            weight_qkv, rope_weight, Q, K, V,
        )

def matmul_k_1024_2560_qkv_rope(x_normed, weight_qkv, rope_weight, Q, K, V):
    seq_len = x_normed.shape[0]
    if USE_CUBLAS_ROPE_QKV:
        # cuBLAS GEMM + Triton RoPE epilogue (replaces fused Triton GEMM+RoPE)
        qkv_tmp = _get_qkv_tmp(x_normed, 2560)
        torch.matmul(x_normed.view(seq_len, 1024), weight_qkv, out=qkv_tmp)
        rope_split_qkv_kernel[(128,)](qkv_tmp, rope_weight, Q, K, V,
                                       seq_len, 256, 8)
    else:
        # Use dispatch function which can use cuBLAS if flag is enabled
        matmul_rope_qkv_dispatch(x_normed, weight_qkv, rope_weight, Q, K, V, seq_len, 1024, 256, 8)

def adarms_norm_style_proj(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    """Normalize x with adaptive RMS and style modulation.
    Note: This function is only called when USE_FUSED_ADARMS_QKV=False.
    When USE_FUSED_ADARMS_QKV=True, adarms_norm_qkv_rope_fused is called instead."""
    seq_len = x.shape[0]
    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 1024  # Increased for single-pass optimization
    )

def adarms_norm_qkv_rope_fused(x, style, gate, weight_qkv, rope_weight, Q, K, V):
    """Fused: adarms_norm + QKV matmul + RoPE in a single kernel.
    Eliminates intermediate x_normed buffer write/read and reduces kernel launches."""
    seq_len = x.shape[0]
    if USE_CUBLAS_ROPE_QKV:
        # Hybrid: cuBLAS GEMM + fused norm+RoPE epilogue
        # Note: We still need to do norm first to get normalized input for cuBLAS
        # But we can't fully fuse norm with cuBLAS since cuBLAS needs the normalized input
        # So we do: norm kernel → cuBLAS GEMM → RoPE epilogue (3 operations, but norm is unavoidable)
        # Use a temporary buffer for normalized x (we need it for cuBLAS matmul)
        x_normed_tmp = torch.empty_like(x)
        adarms_norm_kernel[(seq_len,)](x, style, x_normed_tmp, gate, seq_len=seq_len, features=1024, BLOCK_SIZE=1024)
        qkv_tmp = _get_qkv_tmp(x_normed_tmp, 2560)
        torch.matmul(x_normed_tmp.view(seq_len, 1024), weight_qkv, out=qkv_tmp)
        rope_split_qkv_kernel[(128,)](qkv_tmp, rope_weight, Q, K, V, seq_len, 256, 8)
    else:
        # Fully fused Triton kernel: norm + matmul + RoPE all in one (1 kernel launch)
        grid_m = triton.cdiv(seq_len, 64)
        grid_n = triton.cdiv(2560, 32)  # (8+2)*256 = 2560
        adarms_norm_qkv_rope_kernel[(grid_m * grid_n,)](
            x, style, gate, weight_qkv, rope_weight, Q, K, V,
            seq_len=seq_len, features=1024, head_dim=256, num_heads=8
        )
        # Store gate separately using a simple kernel (if needed)
        if gate is not None:
            _store_gate_kernel[(seq_len,)](style, gate, seq_len, 1024)
def adarms_norm_style_proj_final(x, time_emb, mod_w, mod_b, x_normed, gate, style):
    seq_len = x.shape[0]

    adarms_norm_kernel[(seq_len,)](
        x, 
        style,
        x_normed,
        gate,
        seq_len = seq_len, 
        features = 1024, 
        BLOCK_SIZE = 1024  # Increased for single-pass optimization
    )

def adarms_matmul_k_1024_32_bias(
    x,
    time_emb,
    mod_w,
    mod_b,
    x_normed,
    gate,
    style,
    weight,
    bias,
    out,
):
    """IMLE action output: fused adarms_norm → matmul → bias.

    Directly outputs actions (not velocity). This is used for IMLE mode where the model
    predicts actions in a single forward pass at time=1.0, without any velocity computation
    or integration step (unlike flow matching which computes velocity and integrates).
    """
    seq_len = x.shape[0]
    # Fused AdaRMSNorm + projection kernel (features=1024, hidden=32).
    grid_m = triton.cdiv(seq_len, 64)
    grid_n = triton.cdiv(32, 32)
    adarms_norm_matmul_bias_kernel[(grid_m, grid_n)](
        x,
        style,
        weight,
        bias,
        out,
        seq_len=seq_len,
        features=1024,
        hidden=32,
    )

def matmul_k_2048_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    if USE_HYBRID_RES_GATE:
        matmul_small_res_gate_hybrid(x, weight, out, gate)
        return
    matmul_small_res_gate[(128,)](
        x,
        weight,
        out,
        out, 
        gate,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024,
        # BLOCK_SIZE_* are selected by Triton autotune.
    )

def matmul_k_4096_1024_gate(x, weight, out, gate):
    seq_len = x.shape[0]
    if USE_HYBRID_RES_GATE:
        matmul_small_res_gate_hybrid(x, weight, out, gate)
        return
    matmul_small_res_gate[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        gate,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024,
        # BLOCK_SIZE_* are selected by Triton autotune.
    )

@triton.jit
def softmax_kernel_masklen(
    inp_ptr,
    queries: tl.constexpr,
    keys: tl.constexpr,
    valid_keys_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    assert BLOCK_SIZE >= keys, f"BLOCK_SIZE must be >= keys, got {BLOCK_SIZE} < {keys}"

    valid_keys_len = tl.load(valid_keys_len_ptr).to(tl.int32)
    valid_keys_len = tl.maximum(0, tl.minimum(valid_keys_len, keys))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        attn_mask = (offs_i < queries) & (offs_j < keys) & (offs_j < valid_keys_len)
        vals = tl.load(inp_ptr + offs_i * keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True).to(tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * keys + offs_j, vals.to(tl.bfloat16),
                 mask=(offs_i < queries) & (offs_j < keys))

def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers["vision_x"],
        weights["vision_final_norm_w"],
        weights["vision_final_norm_b"],
        weights["encoder_multi_modal_projector_w"],
        weights["encoder_multi_modal_projector_b"],
        buffers["encoder_x"],
        buffers["vision_x_norm"],
    )
    if USE_FUSED_ADD_RMS_NORM:
        # ── Restructured encoder loop: fuse add_inplace + rms_norm ──
        seq_len = encoder_seq_len
        scale = 1.0 / (256**0.5)
        seqlen_q = encoder_seq_len
        nheads = 8
        headdim = 256
        # First iteration: standalone rms_norm + QKV (no preceding residual add)
        rms_norm_kernel[(seq_len,)](buffers["encoder_x"], buffers["encoder_x_norm"], seq_len, 2048)
        matmul_qkv_rope_only(
            buffers["encoder_x_norm"], weights["encoder_attn_qkv_w"][0],
            buffers["encoder_rope_weights"], buffers["encoder_Q"],
            buffers["encoder_K"][0, :encoder_seq_len], buffers["encoder_V"][0, :encoder_seq_len],
        )
        for i in range(17):  # i = 0..16
            Q_reshaped = buffers["encoder_Q"].view(seqlen_q, nheads, headdim).contiguous()
            K_slice = buffers["encoder_K"][i, :encoder_seq_len]
            V_slice = buffers["encoder_V"][i, :encoder_seq_len]
            K_reshaped = K_slice.unsqueeze(1).expand(-1, nheads, -1)
            V_reshaped = V_slice.unsqueeze(1).expand(-1, nheads, -1)
            out_flash, _ = flash_attention_forward(
                Q_reshaped, K_reshaped, V_reshaped,
                buffers["valid_encoder_len"], softmax_scale=scale, causal=False,
            )
            buffers["encoder_ctx_buf"].copy_(out_flash.view(-1, headdim))
            # Fused: encoder_x += ctx @ attn_o_w; encoder_x_norm = rms_norm(encoder_x)
            matmul_res_fused_rms(
                buffers["encoder_ctx_buf"].view(-1, 2048),
                weights["encoder_attn_o_w"][i], buffers["encoder_x"], buffers["encoder_x_norm"],
            )
            # FFN gate (rms_norm already done in fused kernel)
            if USE_GATED_CONCAT:
                matmul_small_gate_up_hybrid(
                    buffers["encoder_x_norm"], weights.get("encoder_ffn_gate_up_w", [None]*18)[i],
                    buffers["encoder_hidden"]
                )
            else:
                matmul_small_gate_dispatch(
                    buffers["encoder_x_norm"], weights["encoder_ffn_gate_w"][i],
                    weights["encoder_ffn_up_w"][i], buffers["encoder_hidden"], seq_len, 2048, 16384,
                )
            # Fused: encoder_x += hidden @ ffn_down_w; encoder_x_norm = rms_norm(encoder_x)
            matmul_res_fused_rms(
                buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i],
                buffers["encoder_x"], buffers["encoder_x_norm"],
            )
            # QKV + RoPE for next iteration (rms_norm already done)
            matmul_qkv_rope_only(
                buffers["encoder_x_norm"], weights["encoder_attn_qkv_w"][i + 1],
                buffers["encoder_rope_weights"], buffers["encoder_Q"],
                buffers["encoder_K"][i + 1, :encoder_seq_len], buffers["encoder_V"][i + 1, :encoder_seq_len],
            )
    else:
        # ── Original encoder loop ──
        for i in range(18):
            rms_matmul_n_2048_2560_qkv_rope(
                buffers["encoder_x"],
                weights["encoder_attn_qkv_w"][i],
                buffers["encoder_rope_weights"],
                buffers["encoder_Q"],
                buffers["encoder_K"][i, :encoder_seq_len],
                buffers["encoder_V"][i, :encoder_seq_len],
                buffers["encoder_x_norm"],
            )
            if i != 17:
                scale = 1.0 / (256**0.5)
                seqlen_q = encoder_seq_len
                nheads = 8
                headdim = 256
                Q_reshaped = buffers["encoder_Q"].view(seqlen_q, nheads, headdim).contiguous()
                K_slice = buffers["encoder_K"][i, :encoder_seq_len]
                V_slice = buffers["encoder_V"][i, :encoder_seq_len]
                K_reshaped = K_slice.unsqueeze(1).expand(-1, nheads, -1)
                V_reshaped = V_slice.unsqueeze(1).expand(-1, nheads, -1)
                out_flash, _ = flash_attention_forward(
                    Q_reshaped, K_reshaped, V_reshaped,
                    buffers["valid_encoder_len"], softmax_scale=scale, causal=False,
                )
                buffers["encoder_ctx_buf"].copy_(out_flash.view(-1, headdim))
                matmul_n_2048_2048_res(
                    buffers["encoder_ctx_buf"].view(-1, 2048),
                    weights["encoder_attn_o_w"][i], buffers["encoder_x"],
                )
                rms_matmul_n_2048_16384_gate(
                    buffers["encoder_x"], 
                    weights.get("encoder_ffn_gate_w", [None]*18)[i],
                    weights.get("encoder_ffn_up_w", [None]*18)[i],
                    weights.get("encoder_ffn_gate_up_w", [None]*18)[i],
                    buffers["encoder_hidden"], buffers["encoder_x_norm"],
                )
                matmul_n_16384_2048_res(
                    buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i], buffers["encoder_x"],
                )

@triton.jit
def softmax_kernel_prefix_suffix(
    inp_ptr,
    queries: tl.constexpr,
    keys_prefix: tl.constexpr,
    keys_suffix: tl.constexpr,
    valid_prefix_len_ptr,
    out_ptr,
    BLOCK_SIZE_M: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    big_neg = -2.3819763e38
    total_keys: tl.constexpr = keys_prefix + keys_suffix
    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    valid_prefix_len = tl.load(valid_prefix_len_ptr).to(tl.int32)
    valid_prefix_len = tl.maximum(0, tl.minimum(valid_prefix_len, keys_prefix))

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        in_bounds = (offs_i < queries) & (offs_j < total_keys)
        is_prefix = offs_j < keys_prefix
        prefix_ok = is_prefix & (offs_j < valid_prefix_len)
        suffix_ok = (~is_prefix)
        attn_mask = in_bounds & (prefix_ok | suffix_ok)

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=attn_mask, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True).to(tl.float32)
        vals = vals / vsum
        tl.store(out_ptr + offs_i * total_keys + offs_j, vals.to(tl.bfloat16), mask=in_bounds)

def transformer_decoder(weights, buffers, encoder_seq_len, num_steps=1):
    """Transformer decoder for action prediction.
    
    For IMLE (num_steps=1):
        - Takes noise as input, processes through transformer, outputs actions directly
        - NO velocity computation, NO integration step, NO residual addition
        - Single forward pass at time=1.0
    
    For Flow Matching (num_steps>1):
        - Computes velocity at each step and integrates: x_t + dt * v_t
        - Multiple denoising steps from time=1.0 to time=0.0
    """
    for step in range(num_steps):
        matmul_k_32_1024_bias(
            buffers['diffusion_noise'],
            weights['decoder_action_in_proj_w'],
            weights['decoder_action_in_proj_b'],
            buffers['decoder_x']
        )
        seq_len = buffers['decoder_x'].shape[0]
        for i in range(18):
            if USE_FUSED_ADARMS_QKV:
                # Fused: adarms_norm + QKV matmul + RoPE in one kernel
                adarms_norm_qkv_rope_fused(
                    buffers['decoder_x'],
                    buffers['decoder_style_attn'][step, i],
                    buffers['gate_buf'],
                    weights['decoder_attn_qkv_w'][i],
                    buffers['decoder_rope_weights'],
                    buffers['decoder_q_buf'],
                    buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                    buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                )
            else:
                adarms_norm_style_proj(
                    buffers['decoder_x'],
                    buffers['decoder_time_emb'][step],
                    weights['decoder_pre_attn_norm_mod_w'][i],
                    weights['decoder_pre_attn_norm_mod_b'][i],
                    buffers['x_normed_buf'],
                    buffers['gate_buf'],
                    buffers['decoder_style_attn'][step, i]
                )
                matmul_k_1024_2560_qkv_rope(
                    buffers['x_normed_buf'],
                    weights['decoder_attn_qkv_w'][i],
                    buffers['decoder_rope_weights'],
                    buffers['decoder_q_buf'],
                    buffers['encoder_K'][i, encoder_seq_len:encoder_seq_len + seq_len],
                    buffers['encoder_V'][i, encoder_seq_len:encoder_seq_len + seq_len],
                )
            total_queries = buffers['decoder_q_buf'].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale_dispatch(
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i, :encoder_seq_len + seq_len],
                buffers['decoder_logits_buf'],
                256 ** -0.5,
            )

            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers['decoder_logits_buf'],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers['valid_encoder_len'],
                buffers['decoder_attn_buf'],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )

            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i, :encoder_seq_len + seq_len],
                buffers['decoder_q_buf'],
            )
            matmul_k_2048_1024_gate(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )
            adarms_norm_style_proj(
                buffers['decoder_x'],
                buffers['decoder_time_emb'][step],
                weights['decoder_pre_ffn_norm_mod_w'][i],
                weights['decoder_pre_ffn_norm_mod_b'][i],
                buffers['x_normed_buf'],
                buffers['gate_buf'],
                buffers['decoder_style_ffn'][step, i]
            )
            seq_len = buffers['decoder_x'].shape[0]
            if USE_GATED_CONCAT:
                matmul_small_gate_up_hybrid(
                    buffers['x_normed_buf'],
                    weights.get('decoder_ffn_gate_up_w', [None]*18)[i],
                    buffers['decoder_hidden'],
                )
            else:
                matmul_small_gate_dispatch(
                    buffers['x_normed_buf'],
                    weights['decoder_ffn_gate_w'][i],
                    weights['decoder_ffn_up_w'][i],
                    buffers['decoder_hidden'],
                    seq_len,
                    1024,
                    4096,
                )
            matmul_k_4096_1024_gate(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x'],
                buffers['gate_buf']
            )

        # IMLE: direct action prediction — outputs actions directly, NOT velocity
        # Unlike flow matching which computes velocity v_t and integrates (x_t + dt * v_t),
        # IMLE directly predicts actions from the noise input in a single forward pass.
        # No velocity computation, no integration step, no residual addition.
        adarms_matmul_k_1024_32_bias(
            buffers['decoder_x'],
            buffers['decoder_time_emb'][step],
            weights['decoder_final_norm_mod_w'],
            weights['decoder_final_norm_mod_b'],
            buffers['x_normed_buf'],
            buffers['gate_buf'],
            buffers['decoder_style_final'][step],
            weights['decoder_action_out_proj_w'],
            weights['decoder_action_out_proj_b'],
            buffers['action_output'],  # Direct action output, not velocity
        )

def pi05_model(weights, buffers, num_views, encoder_seq_len, num_steps=1):
    """Run the full pi05 model pipeline.
    
    Args:
        weights: Model weights dictionary
        buffers: Model buffers dictionary
        num_views: Number of camera views
        encoder_seq_len: Sequence length for encoder
        num_steps: Number of denoising steps (1 for IMLE, 10+ for flow matching)
    """
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len, num_steps)

class Pi05InferenceIMLE:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
        is_imle: bool = True,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        self.is_imle = is_imle  # Store IMLE flag for verification
        if discrete_state_input:
            # Use the SAME SentencePiece tokenizer as JAX/openpi for consistency
            # Reference: openpi/src/openpi/models/tokenizer.py:18-20
            sp_tokenizer_path = self._download_paligemma_tokenizer()
            self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=sp_tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        print(f"max_prompt_len: {self.max_prompt_len}, max_tokenize_len: {self.max_tokenize_len}")
        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),

            "decoder_time_embeds":                torch.zeros(10, 1024,                  dtype=torch.bfloat16, device="cuda"),
            "decoder_time_mlp_in_w":              torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_b":              torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_w":             torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_b":             torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_w":           torch.empty(32, 1024,                      dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                          dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_w":        torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_attn_norm_mod_b":        torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_w":         torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_ffn_norm_mod_b":         torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_w":          torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_b":          torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),
            "decoder_final_norm_mod_w":           torch.empty(1024, 3 * 1024,         dtype=torch.bfloat16, device="cuda"), 
            "decoder_final_norm_mod_b":           torch.empty(3 * 1024,               dtype=torch.bfloat16, device="cuda"), 
            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),

            
        }

        if USE_GATED_CONCAT:
            self.weights.update({
                "encoder_ffn_gate_up_w":                 torch.empty(18, 2048, 32768,        dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_gate_up_w":                 torch.empty(18, 1024, 8192,         dtype = torch.bfloat16, device = "cuda"),
            })
        else:
            self.weights.update({
                "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
                "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            })

        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size 

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'diffusion_noise':                    torch.empty(chunk_size, 32,                  dtype = torch.bfloat16, device = "cuda"),
            'vision_x':                           torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_norm':                      torch.empty(num_views, 256, 1152,            dtype = torch.bfloat16, device = "cuda"),
            'vision_QKV':                         torch.empty(num_views, 256, 3 * 1152,        dtype = torch.bfloat16, device = "cuda"),
            'vision_hidden':                      torch.empty(num_views, 256, 4304,            dtype = torch.bfloat16, device = "cuda"),
            'vision_x_split_k_buf':               torch.empty((num_views * 256 * 1152 * 4,),   dtype = torch.float32, device = "cuda"),
            'encoder_rope_weights':               torch.empty(encoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'encoder_x':                          torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_x_norm':                     torch.empty(encoder_seq_len, 2048,           dtype = torch.bfloat16, device = "cuda"),
            'encoder_K':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_V':                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256,   dtype = torch.bfloat16, device = "cuda"),
            'encoder_Q':                          torch.empty(encoder_seq_len * 8, 256,        dtype = torch.bfloat16, device = "cuda"),
            'encoder_hidden':                     torch.empty(encoder_seq_len, 16384,          dtype = torch.bfloat16, device = "cuda"),
            'valid_encoder_len':                  torch.empty((1,),                           dtype = torch.int32, device = "cuda"),
            'encoder_logits_buf':                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.float32,  device="cuda"),
            'encoder_attn_buf':                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype=torch.bfloat16, device="cuda"),
            'encoder_ctx_buf':                    torch.empty((encoder_seq_len * 8, 256),     dtype=torch.bfloat16, device="cuda"),
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((decoder_seq_len, 1024),         dtype=torch.bfloat16,  device = "cuda"),
            'action_output':                      torch.empty((decoder_seq_len, 32),           dtype = torch.bfloat16, device = "cuda"),
            'decoder_time_emb':                   torch.empty((10, decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_style_attn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_ffn':                      torch.empty((10, 18, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"), 
            'decoder_style_final':                      torch.empty((10, decoder_seq_len, 1024 * 3),         dtype = torch.bfloat16, device = "cuda"),            
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_logits_buf':                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype=torch.float32, device="cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
            'x_normed_buf':                       torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'gate_buf':                           torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers['encoder_rope_weights'].copy_(self._rope_table[:prefix_alloc])

        self.buffers['valid_encoder_len'].fill_(self.num_views * 256 + 1)
        
        for k, v in checkpoint.items():
            if k == "embedding_weight":
                continue
            if USE_GATED_CONCAT:
                if k == "encoder_ffn_gate_w":
                    self.weights["encoder_ffn_gate_up_w"][..., :16384].copy_(v)
                elif k == "encoder_ffn_up_w":
                    self.weights["encoder_ffn_gate_up_w"][..., 16384:].copy_(v)
                elif k == "decoder_ffn_gate_w":
                    self.weights["decoder_ffn_gate_up_w"][..., :4096].copy_(v)
                elif k == "decoder_ffn_up_w":
                    self.weights["decoder_ffn_gate_up_w"][..., 4096:].copy_(v)
                else:
                    self.weights[k].copy_(v)
            else:
                self.weights[k].copy_(v)
        num_steps = 1
        
        # Weight scaling for flow matching only.
        # Flow matching computes: x_0 = x_t + dt * v_t, where dt = -1/num_steps.
        # Baking dt into the output projection weights avoids an explicit residual step.
        # IMLE directly predicts actions (not velocity), so no scaling is needed.
        if not is_imle:
            self.weights['decoder_action_out_proj_w'] *= -1.0 / num_steps
            self.weights['decoder_action_out_proj_b'] *= -1.0 / num_steps

        for step in range(num_steps):
            matmul_1_1024_1024_bias_silu(
                self.weights['decoder_time_embeds'][step].view(1, -1),
                self.weights['decoder_time_mlp_in_w'],
                self.weights['decoder_time_mlp_in_b'],
                self.buffers['decoder_x_buf']
            )
            matmul_1_1024_1024_bias_silu(
                self.buffers['decoder_x_buf'],
                self.weights['decoder_time_mlp_out_w'],
                self.weights['decoder_time_mlp_out_b'],
                self.buffers['decoder_time_emb'][step]
            )
            for i in range(18):
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_attn_norm_mod_w'][i],
                    self.buffers['decoder_style_attn'][step, i],
                    self.weights['decoder_pre_attn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
                matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                    self.buffers['decoder_time_emb'][step],
                    self.weights['decoder_pre_ffn_norm_mod_w'][i],
                    self.buffers['decoder_style_ffn'][step, i],
                    self.weights['decoder_pre_ffn_norm_mod_b'][i],
                    seq_len = decoder_seq_len,
                    features = 1024, 
                    hidden = 3072,
                    BLOCK_SIZE_N = 32,
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_K = 32
                )
            matmul_small_bias[((decoder_seq_len + 31) // 32) * (3072 // 32),](
                self.buffers['decoder_time_emb'][step],
                self.weights['decoder_final_norm_mod_w'],
                self.buffers['decoder_style_final'][step],
                self.weights['decoder_final_norm_mod_b'],
                seq_len = decoder_seq_len,
                features = 1024, 
                hidden = 3072,
                BLOCK_SIZE_N = 32,
                BLOCK_SIZE_M = 32,
                BLOCK_SIZE_K = 32
                )
            
        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w_t = torch.from_numpy(emb_w)
            else:
                emb_w_t = emb_w
            emb_w_t = emb_w_t.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w_t.shape[0],
                embedding_dim=emb_w_t.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w_t)
            self._prompt_embed_scale = float(emb_w_t.shape[1] ** 0.5)
        self.encoder_seq_len = encoder_seq_len

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()
        
        # Create separate CUDA graphs for profiling each section
        self.vision_graph = None
        self.encoder_graph = None
        self.decoder_graph = None

    def _download_paligemma_tokenizer(self) -> str:
        """Find the PaliGemma tokenizer used by JAX/openpi.
        
        Checks common cache locations:
        1. $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model (default: ~/.cache/openpi)
        2. Downloads it there if not found
        
        Reference: openpi/src/openpi/models/tokenizer.py:18-20
        """
        # Same cache layout as openpi.shared.download: $OPENPI_DATA_HOME (default ~/.cache/openpi)
        data_home = os.environ.get("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))
        cache_dir = os.path.join(data_home, "big_vision")
        openpi_cache_path = os.path.join(cache_dir, "paligemma_tokenizer.model")
        if os.path.exists(openpi_cache_path):
            print(f"Using existing tokenizer from: {openpi_cache_path}")
            return openpi_cache_path

        # Not cached yet: download into the openpi cache
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer_path = os.path.join(cache_dir, "paligemma_tokenizer.model")
        
        if not os.path.exists(tokenizer_path):
            print("Downloading PaliGemma tokenizer (same as JAX/openpi)...")
            import urllib.request
            url = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
            urllib.request.urlretrieve(url, tokenizer_path)
            print(f"Tokenizer downloaded to: {tokenizer_path}")
        
        return tokenizer_path

    def estimate_max_prompt_len(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        """Estimate max prompt length using SentencePiece tokenizer (same as JAX)."""
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        # Use SentencePiece encode with add_bos=True (same as JAX)
        # Reference: openpi/src/openpi/models/tokenizer.py:26-29
        token_ids = tokenizer.encode(full_prompt, add_bos=True)
        return min(len(token_ids), int(max_tokenize_len))

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        """Build prompt embeddings using SentencePiece tokenizer (same as JAX).
        
        Reference: openpi/src/openpi/models/tokenizer.py:22-48
        """
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        # Use SentencePiece encode with add_bos=True (same as JAX)
        # Reference: openpi/src/openpi/models/tokenizer.py:26-29
        token_ids = self.tokenizer.encode(full_prompt, add_bos=True)
        # Truncate if needed (same as JAX tokenizer.py:45-46)
        if len(token_ids) > self.max_tokenize_len:
            token_ids = token_ids[:self.max_tokenize_len]
        token_ids = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def record_run(self):
        pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def record_infer_graph(self):
        for _ in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()
    
    def _record_vision_graph(self):
        """Record CUDA graph for vision encoder only."""
        for _ in range(3):
            vision_encoder(self.weights, self.buffers, self.num_views)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.vision_graph = torch.cuda.CUDAGraph()
            self.vision_graph.capture_begin()
            vision_encoder(self.weights, self.buffers, self.num_views)
            self.vision_graph.capture_end()
    
    def _record_encoder_graph(self):
        """Record CUDA graph for transformer encoder only."""
        for _ in range(3):
            transformer_encoder(self.weights, self.buffers, self.encoder_seq_len)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.encoder_graph = torch.cuda.CUDAGraph()
            self.encoder_graph.capture_begin()
            transformer_encoder(self.weights, self.buffers, self.encoder_seq_len)
            self.encoder_graph.capture_end()
    
    def _record_decoder_graph(self, num_steps=1):
        """Record CUDA graph for transformer decoder only."""
        for _ in range(3):
            transformer_decoder(self.weights, self.buffers, self.encoder_seq_len, num_steps=num_steps)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.decoder_graph = torch.cuda.CUDAGraph()
            self.decoder_graph.capture_begin()
            transformer_decoder(self.weights, self.buffers, self.encoder_seq_len, num_steps=num_steps)
            self.decoder_graph.capture_end()

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Run forward pass through the model.
        
        Args:
            observation_images_normalized: Normalized observation images
            diffusion_noise: Noise for diffusion/IMLE
            task_prompt: Task description text
            state_tokens: Discretized state tokens
            num_steps: Number of denoising steps (1 for IMLE, 10+ for flow matching)
        
        Returns:
            Action predictions
        """
        # DEBUG: Print to confirm IMLE mode is being used
        if num_steps == 1 and self.is_imle:
            print(f"[DEBUG] Triton forward() called with num_steps=1 and is_imle=True - using IMLE mode (single forward pass)")
        elif num_steps == 1 and not self.is_imle:
            print(f"[DEBUG] Triton forward() called with num_steps=1 but is_imle=False - weights may be scaled for flow matching!")
        elif num_steps > 1:
            print(f"[DEBUG] Triton forward() called with num_steps={num_steps} - using flow matching mode")
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        
        # For num_steps=1 (IMLE), use the pre-recorded graph. For other values, run directly.
        if num_steps == 1:
            self.infer_graph.replay()
        else:
            # Run without CUDA graph for non-standard num_steps
            pi05_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len, num_steps)
        return self.buffers['action_output']
    
    def forward_with_profiling(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
        num_steps: int = 1,
        use_cuda_graphs=True,
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass with detailed profiling of each section.
        
        Args:
            num_steps: Number of denoising steps (default: 1 for IMLE)
            use_cuda_graphs: If True, uses separate CUDA graphs for each section.
                            If False, calls sections individually (slower due to overhead).
        
        Returns:
            actions: torch.Tensor, the output actions
            timings: dict with keys 'vision_encoder', 'transformer_encoder', 'transformer_decoder', 'total'
                    values are in milliseconds
        """
        # Prepare inputs (same as forward)
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(
                task_prompt=task_prompt,
                state_tokens=state_tokens
            )
        else:
            prompt_embeds = self.weights['language_embeds']
            prompt_len = self.weights['language_embeds'].shape[0]
        start = self.num_views * 256
        self.buffers['encoder_x'][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers['valid_encoder_len'].fill_(start + prompt_len)
        self.buffers['decoder_rope_weights'].copy_(self.get_decoder_rope_weights(prompt_len))
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        
        # Initialize graphs if needed and use_cuda_graphs is True
        if use_cuda_graphs:
            if self.vision_graph is None:
                self._record_vision_graph()
            if self.encoder_graph is None:
                self._record_encoder_graph()
            # Decoder graph needs to be recreated if num_steps changes
            if self.decoder_graph is None:
                self._record_decoder_graph(num_steps=num_steps)
        
        timings = {}
        torch.cuda.synchronize()
        
        # Time vision encoder (line 527)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if use_cuda_graphs:
            self.vision_graph.replay()
        else:
            vision_encoder(self.weights, self.buffers, self.num_views)
        end_event.record()
        torch.cuda.synchronize()
        timings['vision_encoder'] = start_event.elapsed_time(end_event)
        
        # Time transformer encoder (line 528)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if use_cuda_graphs:
            self.encoder_graph.replay()
        else:
            transformer_encoder(self.weights, self.buffers, self.encoder_seq_len)
        end_event.record()
        torch.cuda.synchronize()
        timings['transformer_encoder'] = start_event.elapsed_time(end_event)
        
        # Time transformer decoder (line 529) - includes all num_steps
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if use_cuda_graphs:
            # Check if we need to recreate decoder graph for different num_steps
            if self.decoder_graph is None:
                self._record_decoder_graph(num_steps=num_steps)
            self.decoder_graph.replay()
        else:
            transformer_decoder(self.weights, self.buffers, self.encoder_seq_len, num_steps=num_steps)
        end_event.record()
        torch.cuda.synchronize()
        timings['transformer_decoder'] = start_event.elapsed_time(end_event)
        
        timings['total'] = timings['vision_encoder'] + timings['transformer_encoder'] + timings['transformer_decoder']
        
        return self.buffers['action_output'], timings
