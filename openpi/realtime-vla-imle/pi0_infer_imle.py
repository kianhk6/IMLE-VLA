import os
import torch
import triton
import triton.language as tl

USE_CUBLASLT_GATE = os.getenv("USE_CUBLASLT_GATE", "0") == "1"
USE_HYBRID_GATE = os.getenv("USE_HYBRID_GATE", "1") == "1"
USE_HYBRID_RES = os.getenv("USE_HYBRID_RES", "1") == "1"
USE_CUBLAS_TWOPART_RES = os.getenv("USE_CUBLAS_TWOPART_RES", "1") == "1"
USE_CUBLAS_BIAS_RES_MOD = os.getenv("USE_CUBLAS_BIAS_RES_MOD", "1") == "1"
USE_CUBLAS_4304_1152_RES = os.getenv("USE_CUBLAS_4304_1152_RES", "1") == "1"
MERGE_BLOCK_SIZE = int(os.getenv("MERGE_BLOCK_SIZE", "4096"))  # 512 for original, 4096 for optimized

# ── Round-2 throughput optimisation flags ──────────────────────────────────
USE_SCALED_MM          = os.getenv("USE_SCALED_MM", "1") == "1"         # fuse scale into Q@K^T GEMM
USE_ADDMM_BIAS_RES    = os.getenv("USE_ADDMM_BIAS_RES", "1") == "1"   # fuse bias into ViT FFN down GEMM
USE_FUSED_BIAS_ADD     = os.getenv("USE_FUSED_BIAS_ADD", "1") == "1"   # fuse bias into encoder twopart GEMM
USE_FAST_LAYER_NORM    = os.getenv("USE_FAST_LAYER_NORM", "1") == "1"  # improved layer_norm kernel
USE_FUSED_ADD_RMS_NORM = os.getenv("USE_FUSED_ADD_RMS_NORM", "1") == "1"  # fuse add_inplace + rms_norm
USE_CUBLAS_ROPE_QKV    = os.getenv("USE_CUBLAS_ROPE_QKV", "1") == "1"  # cuBLAS GEMM + Triton RoPE epilogue
USE_GATED_CONCAT       = os.getenv("USE_GATED_CONCAT", "1") == "1"     # concat gate & up weights

_CUBLASLT_GATE_TMP = {}
_CUBLASLT_RES_TMP = {}
_GATE_UP_TMP = {}

def _get_gate_up_tmp(out: torch.Tensor) -> torch.Tensor:
    shape = list(out.shape)
    shape[-1] = shape[-1] * 2
    key = (tuple(shape), out.device, out.dtype)
    tmp = _GATE_UP_TMP.get(key)
    if tmp is None or tmp.shape != tuple(shape) or tmp.device != out.device or tmp.dtype != out.dtype:
        tmp = torch.empty(*shape, dtype=out.dtype, device=out.device)
        _GATE_UP_TMP[key] = tmp
    return tmp

def _get_gate_tmp(out: torch.Tensor) -> torch.Tensor:
    key = (out.shape, out.device, out.dtype)
    tmp = _CUBLASLT_GATE_TMP.get(key)
    if tmp is None or tmp.shape != out.shape or tmp.device != out.device or tmp.dtype != out.dtype:
        tmp = torch.empty_like(out)
        _CUBLASLT_GATE_TMP[key] = tmp
    return tmp

def _get_res_tmp(out: torch.Tensor) -> torch.Tensor:
    key = (out.shape, out.device, out.dtype)
    tmp = _CUBLASLT_RES_TMP.get(key)
    if tmp is None or tmp.shape != out.shape or tmp.device != out.device or tmp.dtype != out.dtype:
        tmp = torch.empty_like(out)
        _CUBLASLT_RES_TMP[key] = tmp
    return tmp

# =============================================================================
# Fused epilogue kernels for hybrid cuBLAS + Triton approach
# =============================================================================

@triton.jit
def gelu_mul_kernel(gate_ptr, up_ptr, out_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr = 1024):
    """Fused GELU(gate) * up -> out. Avoids bf16->float32->bf16 round-trips."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # GELU approximation: x * sigmoid(1.702 * x * (1 + 0.044715 * x^2))
        gelu = gate * tl.sigmoid(1.5957691216057308 * gate * (1.0 + 0.044715 * gate * gate))
        result = (gelu * up).to(tl.bfloat16)
        tl.store(out_ptr + offs, result, mask=mask)

@triton.jit
def gelu_inplace_kernel(ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr = 1024):
    """In-place GELU (tanh approx) in bf16."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        x = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = x * tl.sigmoid(1.5957691216057308 * x * (1.0 + 0.044715 * x * x))
        tl.store(ptr + offs, result.to(tl.bfloat16), mask=mask)

@triton.jit
def add_inplace_kernel(src_ptr, dst_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr = 4096):
    """Fused dst += src in bf16. Optimized with larger block size for better memory throughput."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        src = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dst = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = (src + dst).to(tl.bfloat16)
        tl.store(dst_ptr + offs, result, mask=mask)

@triton.jit
def add_bias_inplace_kernel(src_ptr, bias_ptr, dst_ptr, numel: tl.constexpr, hidden: tl.constexpr, BLOCK_SIZE: tl.constexpr = 4096):
    """Fused dst += src + bias[i % hidden] in bf16. Optimized with efficient modulo computation."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        src = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dst = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Optimize modulo: use bitwise AND if hidden is power of 2, otherwise use efficient modulo
        # For non-power-of-2, we can tile the bias loading to reduce modulo operations
        bias_offsets = offs % hidden
        b = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
        result = (dst + src + b).to(tl.bfloat16)
        tl.store(dst_ptr + offs, result, mask=mask)

@triton.jit
def gate_add_inplace_kernel(src_ptr, gate_ptr, dst_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr = 2048):
    """Fused dst += src * gate in bf16. Optimized with larger block size and fused operations."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        # Load all three inputs
        src = tl.load(src_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dst = tl.load(dst_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Fused multiply-add: dst + src * gate
        result = (dst + src * gate).to(tl.bfloat16)
        tl.store(dst_ptr + offs, result, mask=mask)

@triton.jit
def fused_add_rms_norm_kernel(
    src_ptr, residual_ptr, out_ptr,
    seq_len: tl.constexpr, features: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 2048,  # Set to max features to enable single-pass
):
    """Fused: residual[i] += src[i]; out[i] = rms_norm(residual[i]).
    True single-pass: load once, compute sum, normalize, and store both outputs."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    
    for i in range(pid, seq_len, psize):
        # Load entire row at once (features=2048 fits in BLOCK_SIZE=2048)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < features
        
        # Load both inputs in one go
        s = tl.load(src_ptr + i * features + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + i * features + offs, mask=mask, other=0.0).to(tl.float32)
        
        # Compute sum: x = r + s
        x = r + s
        
        # Compute sum of squares and normalization factor
        sum_sq = tl.sum(x * x)
        factor = tl.rsqrt(sum_sq / features + 1e-6)
        
        # Normalize
        x_norm = x * factor
        
        # Store both residual (updated) and output (normalized) - SINGLE PASS!
        tl.store(residual_ptr + i * features + offs, x.to(tl.bfloat16), mask=mask)
        tl.store(out_ptr + i * features + offs, x_norm.to(tl.bfloat16), mask=mask)

@triton.jit
def rope_split_qkv_kernel(
    inp_ptr, rope_ptr, q_ptr, k_ptr, v_ptr,
    seq_len: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr = 64, BLOCK_SIZE_N: tl.constexpr = 32,
):
    """Apply RoPE to Q/K columns and route QKV from a fused matmul output.
    inp has shape [seq_len, (num_heads+2)*head_dim]."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    total_cols: tl.constexpr = (num_heads + 2) * head_dim
    grid_m = tl.cdiv(seq_len, BLOCK_SIZE_M)
    grid_n = tl.cdiv(total_cols, BLOCK_SIZE_N)

    for p in range(pid, grid_m * grid_n, psize):
        pid_m = p // grid_n
        pid_n = p % grid_n
        start_i = pid_m * BLOCK_SIZE_M
        start_j = pid_n * BLOCK_SIZE_N
        offs_i = start_i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_SIZE_N)[None, :]

        data = tl.load(
            inp_ptr + offs_i * total_cols + offs_j,
            mask=(offs_i < seq_len) & (offs_j < total_cols), other=0.0,
        )
        # Apply RoPE to Q and K columns (first (num_heads+1)*head_dim cols)
        if start_j < (num_heads + 1) * head_dim:
            x0, x1 = tl.split(data.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x_cossin = tl.load(
                rope_ptr + offs_i * head_dim + offs_j % head_dim,
                mask=offs_i < seq_len, other=0.0,
            )
            x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
            x0_ = x0 * x_cos - x1 * x_sin
            x1_ = x1 * x_cos + x0 * x_sin
            data = tl.interleave(x0_, x1_)

        data = data.to(tl.bfloat16)

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
            data,
            mask=(offs_i < seq_len) & (offs_j < total_cols),
        )

@triton.jit
def layer_norm_fast_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr,
    seq_len: tl.constexpr, features: tl.constexpr,
    eps: tl.constexpr = 1e-5,
):
    """Improved LayerNorm: rsqrt, pre-loaded weights/bias, fused w*x+b."""
    MAX_LEN: tl.constexpr = 2048
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    offs = tl.arange(0, MAX_LEN)
    mask = offs < features
    # Pre-load weights and bias into registers (reused across all rows)
    w = tl.load(norm_w_ptr + offs, mask=mask, other=0.0)
    b = tl.load(norm_b_ptr + offs, mask=mask, other=0.0)
    for i in range(pid, seq_len, psize):
        x = tl.load(x_ptr + i * features + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x) * (1.0 / features)
        x = x - mean
        var = tl.sum(tl.where(mask, x * x, 0.0)) * (1.0 / features)
        x = x * tl.rsqrt(var + eps)
        x = x * w + b
        tl.store(out_ptr + i * features + offs, x.to(tl.bfloat16), mask=mask)

# ── Temp-buffer cache for cuBLAS QKV outputs ────────────────────────────
_QKV_TMP = {}
def _get_qkv_tmp(x, out_dim):
    """Get/create a reusable temp buffer for QKV matmul output."""
    key = (x.shape[0], out_dim, x.device)
    if key not in _QKV_TMP:
        _QKV_TMP[key] = torch.empty(x.shape[0], out_dim, dtype=torch.bfloat16, device=x.device)
    return _QKV_TMP[key]

def matmul_small_res_gate_hybrid(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, gate: torch.Tensor) -> None:
    """Hybrid: cuBLAS GEMM + fused Triton gated residual add.
    out += (x @ weight) * gate  (out is also the residual)
    """
    tmp = _get_res_tmp(out)
    torch.matmul(x, weight, out=tmp)    # cuBLAS GEMM
    # Fused epilogue: out += tmp * gate
    numel = out.numel()
    grid = ((numel + 2047) // 2048,)  # Updated to match new BLOCK_SIZE=2048
    gate_add_inplace_kernel[grid](tmp, gate, out, numel)

def matmul_4304_1152_bias_res_cublas(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
    """cuBLAS GEMM + fused Triton epilogue for bias + residual add.
    Replaces split-k matmul + merge_split_k_bias_res entirely.
    out += x @ weight + bias  (out is also the residual)
    """
    tmp = _get_res_tmp(out)
    x_2d = x.reshape(-1, x.shape[-1])
    out_2d = out.reshape(-1, out.shape[-1])
    tmp_2d = tmp.reshape(-1, tmp.shape[-1])
    if USE_ADDMM_BIAS_RES:
        # Fuse bias into GEMM via cuBLASLt: tmp = bias + x @ weight
        torch.addmm(bias, x_2d, weight, out=tmp_2d)
        # Simple residual add: out += tmp
        numel = out_2d.numel()
        grid = ((numel + 4095) // 4096,)  # Updated to match new BLOCK_SIZE=4096
        add_inplace_kernel[grid](tmp_2d, out_2d, numel)
    else:
        torch.matmul(x_2d, weight, out=tmp_2d)
        numel = out_2d.numel()
        hidden = out_2d.shape[-1]
        grid = ((numel + 4095) // 4096,)
        add_bias_inplace_kernel[grid](tmp_2d, bias, out_2d, numel, hidden)

def matmul_small_gate_cublas(x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, out: torch.Tensor) -> None:
    # Two GEMMs via cuBLASLt, then GELU(gate) * up (unfused - slow)
    tmp = _get_gate_tmp(out)
    torch.matmul(x, weight1, out=tmp)
    torch.matmul(x, weight2, out=out)
    tmp_f = tmp.float()
    tmp_f = tmp_f * torch.sigmoid(1.5957691216057308 * tmp_f * (1 + 0.044715 * tmp_f * tmp_f))
    out.mul_(tmp_f.to(out.dtype))

def matmul_small_gate_hybrid(x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor, out: torch.Tensor) -> None:
    """Hybrid: cuBLAS GEMMs + fused Triton epilogue for GELU*mul."""
    tmp = _get_gate_tmp(out)
    torch.matmul(x, weight1, out=tmp)   # cuBLAS GEMM 1: gate
    torch.matmul(x, weight2, out=out)   # cuBLAS GEMM 2: up
    # Fused epilogue: GELU(tmp) * out -> out
    numel = out.numel()
    grid = ((numel + 1023) // 1024,)  # gelu_mul_kernel uses BLOCK_SIZE=1024
    gelu_mul_kernel[grid](tmp, out, out, numel)

@triton.jit
def gelu_mul_concat_kernel(concat_ptr, out_ptr, seq_len: tl.constexpr, hidden: tl.constexpr, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    numel = seq_len * hidden
    for i in range(pid * BLOCK_SIZE, numel, psize * BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        
        row = offs // hidden
        col = offs % hidden
        
        gate_idx = row * (hidden * 2) + col
        up_idx = gate_idx + hidden
        
        gate = tl.load(concat_ptr + gate_idx, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(concat_ptr + up_idx, mask=mask, other=0.0).to(tl.float32)
        
        gelu = gate * tl.sigmoid(1.5957691216057308 * gate * (1.0 + 0.044715 * gate * gate))
        result = (gelu * up).to(tl.bfloat16)
        tl.store(out_ptr + offs, result, mask=mask)

def matmul_small_gate_up_hybrid(x: torch.Tensor, weight_concat: torch.Tensor, out: torch.Tensor) -> None:
    tmp = _get_gate_up_tmp(out)
    x_2d = x.reshape(-1, x.shape[-1])
    weight_concat_2d = weight_concat.reshape(-1, weight_concat.shape[-1])
    tmp_2d = tmp.reshape(-1, tmp.shape[-1])
    torch.matmul(x_2d, weight_concat_2d, out=tmp_2d)
    
    seq_len = x_2d.shape[0]
    hidden = out.shape[-1]
    grid = ((seq_len * hidden + 1023) // 1024,)
    gelu_mul_concat_kernel[grid](tmp_2d, out.view(-1), seq_len, hidden)

def matmul_small_res_hybrid(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Hybrid: cuBLAS GEMM + fused Triton residual add."""
    tmp = _get_res_tmp(out)
    torch.matmul(x, weight, out=tmp)    # cuBLAS GEMM
    # Fused epilogue: out += tmp
    numel = out.numel()
    grid = ((numel + 4095) // 4096,)  # Updated to match new BLOCK_SIZE=4096
    add_inplace_kernel[grid](tmp, out, numel)

def matmul_512x1152x1152_twopart_bias_res_cublas(
    old: torch.Tensor,
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    cuBLAS alternative for the 512x1152x1152 two-part kernel:
    out = old + bias + inp @ weight

    NOTE: old and out may alias (same buffer), so we use addmm which
    handles this correctly via cuBLAS's native C = α·A·B + β·C.
    """
    inp_2d = inp.reshape(-1, 1152)
    out_2d = out.reshape(-1, 1152)
    if USE_FUSED_BIAS_ADD:
        # Fuse bias into GEMM via cuBLASLt epilogue: tmp = bias + inp @ weight
        # Then residual add: out += tmp
        tmp = _get_res_tmp(out)
        tmp_2d = tmp.reshape(-1, 1152)
        torch.addmm(bias, inp_2d, weight, out=tmp_2d)
        numel = out_2d.numel()
        grid = ((numel + 4095) // 4096,)  # Updated to match new BLOCK_SIZE=4096
        add_inplace_kernel[grid](tmp_2d, out_2d, numel)
    else:
        old_2d = old.reshape(-1, 1152)
        # addmm: out = 1.0 * old + 1.0 * (inp @ weight)
        torch.addmm(old_2d, inp_2d, weight, out=out_2d)
        out_2d.add_(bias)

def matmul_small_bias_res_mod_cublas(
    inp: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor,
    res: torch.Tensor,
    i_mod: int,
) -> None:
    """
    cuBLAS alternative for matmul_small_bias_res_mod:
    out[i] = inp[i] @ weight + bias + res[i % i_mod]
    """
    out_2d = out.reshape(-1, out.shape[-1])
    weight_2d = weight.reshape(-1, out_2d.shape[-1])
    # Match Triton kernel layout: inp is physically arranged as [seq_len, features]
    # where seq_len == 256 * nviews and features == 14 * 14 * 3.
    inp_2d = inp.reshape(out_2d.shape[0], weight_2d.shape[0])
    torch.matmul(inp_2d, weight_2d, out=out_2d)
    out_2d.add_(bias)

    res_2d = res.reshape(i_mod, out_2d.shape[-1])
    if out_2d.shape[0] % i_mod == 0:
        repeats = out_2d.shape[0] // i_mod
        out_2d.add_(res_2d.unsqueeze(0).expand(repeats, i_mod, out_2d.shape[-1]).reshape(out_2d.shape))
    else:
        row_idx = torch.arange(out_2d.shape[0], device=out_2d.device) % i_mod
        out_2d.add_(res_2d.index_select(0, row_idx))


@triton.jit
def matmul_small_bias_res(inp_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
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
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_bias_res_mod(inp_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                            i_mod : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] % i_mod) * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        # More aggressive, larger tiles for L40S
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def scaled_matmul_small_bias_res(inp_ptr, inp_norm_factor_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask = i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0
        )
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        ).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0
            )
            x = x * norm_factor[:, None]
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 16, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 16, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        # More aggressive, larger tiles for L40S
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def rms_fused_matmul_small_bias_res(inp_ptr, weight_ptr, out_ptr, bias_ptr, res_ptr,
                         seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    """Fused RMSNorm factor + matmul + bias + residual in a single kernel."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        # 1) Compute RMS factor for these rows
        sum_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features
                + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                       & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms = tl.rsqrt(sum_sq / features + 1e-6)

        # 2) Matmul over normalized x with bias + residual
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden
            + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                   & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0,
        ).to(tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0,
        ).to(tl.float32)

        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features
                + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                       & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other = 0.0,
            ).to(tl.float32)
            x = x * rms[:, None]
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden
                + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features)
                       & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other = 0.0,
            ).to(tl.float32)
            acc = tl.dot(x, w, acc)

        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden
            + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len)
                   & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
        )

@triton.jit
def matmul_small_bias(inp_ptr, weight_ptr, out_ptr, bias_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_bias_silu(inp_ptr, weight_ptr, out_ptr, bias_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
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
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 32}, num_warps=2, num_stages=1),
        # More aggressive larger tiles for wide encoder/decoder matmuls
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def matmul_small_res(inp_ptr, weight_ptr, out_ptr, res_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small(inp_ptr, weight_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_bias_gelu(inp_ptr, weight_ptr, out_ptr, bias_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask = ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other = 0.0
        )
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
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_split_k(inp_ptr, weight_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr,
                         BLOCK_SIZE_N : tl.constexpr, BLOCK_SIZE_M : tl.constexpr, BLOCK_SIZE_K : tl.constexpr, SPLIT_K : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    grid_k = SPLIT_K
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j*grid_k)) * BLOCK_SIZE_N
        j = (p //grid_k % grid_j) * BLOCK_SIZE_M
        k_s = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        k_beg = k_s * tl.cdiv(features, BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        k_end = (k_s + 1) * tl.cdiv(features, BLOCK_SIZE_K) // SPLIT_K * BLOCK_SIZE_K
        for k in range(k_beg, k_end, BLOCK_SIZE_K):
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
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :] + k_s * seq_len * hidden,
            acc,
            mask = ((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def merge_split_k_bias_res_512(out_ptr, bias_ptr, res_ptr, final_out_ptr, seq_len : tl.constexpr, hidden : tl.constexpr, SPLIT_K : tl.constexpr):
    """Original merge kernel with BLOCK=512 (baseline)."""
    BLOCK: tl.constexpr = 512
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    total : tl.constexpr = seq_len * hidden
    for i in range(pid * BLOCK, total, psize * BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < total
        acc = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(bias_ptr + offs % hidden, mask=mask, other=0.0).to(tl.float32)
        for k in range(SPLIT_K):
            acc += tl.load(out_ptr + k * total + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(final_out_ptr + offs, acc.to(tl.bfloat16), mask=mask)

@triton.jit
def merge_split_k_bias_res_4096(out_ptr, bias_ptr, res_ptr, final_out_ptr, seq_len : tl.constexpr, hidden : tl.constexpr, SPLIT_K : tl.constexpr):
    """Optimized merge kernel with BLOCK=4096 (8x larger blocks for better memory bandwidth)."""
    BLOCK: tl.constexpr = 4096
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    total : tl.constexpr = seq_len * hidden
    for i in range(pid * BLOCK, total, psize * BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < total
        acc = tl.load(res_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.load(bias_ptr + offs % hidden, mask=mask, other=0.0).to(tl.float32)
        for k in range(SPLIT_K):
            acc += tl.load(out_ptr + k * total + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(final_out_ptr + offs, acc.to(tl.bfloat16), mask=mask)

def conv2d_embed_n256_1152_res(images, patch_w, patch_b, pos_emb, out):
    nviews = images.shape[0]
    img_input = images.view(nviews, 16, 14, 16, 14, 3).permute(0, 1, 3, 2, 4, 5).contiguous()
    if USE_CUBLAS_BIAS_RES_MOD:
        matmul_small_bias_res_mod_cublas(
            inp=img_input,
            weight=patch_w,
            out=out,
            bias=patch_b,
            res=pos_emb,
            i_mod=256,
        )
        return
    matmul_small_bias_res_mod[(256 * nviews // 64) * (1152 // 64),](
        img_input,
        patch_w,
        out,
        patch_b,
        pos_emb,
        seq_len = 256 * nviews,
        features = 3 * 14 * 14,
        hidden = 1152,
        i_mod = 256,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )

@triton.jit
def layer_norm_small_kernel(x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, seq_len : tl.constexpr, features : tl.constexpr, eps : tl.constexpr = 1e-5):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    MAX_LEN : tl.constexpr = 2048

    for i in range(pid, seq_len, psize):
        x = tl.load(x_ptr + i * features + tl.arange(0, MAX_LEN), mask = tl.arange(0, MAX_LEN) < features, other = 0.0)
        mean = tl.sum(x) * (1.0 / features)
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered * (tl.arange(0, MAX_LEN) < features)) * (1.0 / features)
        inv_std = 1.0 / tl.sqrt(var+ eps)
        x = x_centered * inv_std
        x = x * tl.load(norm_w_ptr + tl.arange(0, MAX_LEN), mask = tl.arange(0, MAX_LEN) < features, other = 0.0)
        x = x + tl.load(norm_b_ptr + tl.arange(0, MAX_LEN), mask = tl.arange(0, MAX_LEN) < features, other = 0.0)
        tl.store(out_ptr + i * features + tl.arange(0, MAX_LEN), x.to(tl.bfloat16), mask = tl.arange(0, MAX_LEN) < features)

def _layer_norm_dispatch(x_ptr, out_ptr, norm_w, norm_b, seq_len, features, eps=1e-5):
    """Dispatch to fast or original layer_norm kernel based on flag."""
    if USE_FAST_LAYER_NORM:
        layer_norm_fast_kernel[seq_len,](x_ptr, out_ptr, norm_w, norm_b,
                                          seq_len=seq_len, features=features, eps=eps)
    else:
        layer_norm_small_kernel[seq_len,](x_ptr, out_ptr, norm_w, norm_b,
                                           seq_len=seq_len, features=features, eps=eps)

def layer_norm_n256_1152(x, norm_w, norm_b, out):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    _layer_norm_dispatch(x, out, norm_w, norm_b, seq_len=seq_len, features=1152)

USE_CUBLAS_ENCODER_QKV = os.getenv("USE_CUBLAS_ENCODER_QKV", "1") == "1"

def layer_norm_QKV_matmul_n256_1152_3456_bias(x, norm_w, norm_b, qkv_w, qkv_b, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    _layer_norm_dispatch(x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)

    if USE_CUBLAS_ENCODER_QKV:
        torch.addmm(qkv_b, x_norm.view(seq_len, 1152), qkv_w, out=out.view(seq_len, 3456))
        return
    matmul_small_bias[((seq_len + 63) // 64) * (3456 // 64),](
        x_norm,
        qkv_w,
        out,
        qkv_b,
        seq_len = seq_len,
        features = 1152,
        hidden = 3456,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )

@triton.jit
def matmul_512x1152x1152_twopart_bias_res(old_ptr, inp_ptr, weight_ptr, bias_ptr, out_ptr, out2_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    hidden1 : tl.constexpr = 1024

    BLOCK_SIZE_N1 : tl.constexpr = 64
    BLOCK_SIZE_M1 : tl.constexpr = 64
    BLOCK_SIZE_K1 : tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N1)
    grid_j = tl.cdiv(hidden1, BLOCK_SIZE_M1)

    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N1
        j = (p % grid_j) * BLOCK_SIZE_M1
        acc = tl.load(old_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(tl.float32)
        acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :]).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K1):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K1))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K1))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K1))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden),
                other = 0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N1))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M1))[None, :],
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N1))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M1))[None, :] < hidden)
        )

    BLOCK_SIZE_N2 : tl.constexpr = 32
    BLOCK_SIZE_M2 : tl.constexpr = 32
    BLOCK_SIZE_K2 : tl.constexpr = 32
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N2)
    grid_j = tl.cdiv(hidden - hidden1, BLOCK_SIZE_M2)
    grid_k = 4
    for p in range(pid, grid_i * grid_j * grid_k, psize):
        i = (p // (grid_j*grid_k)) * BLOCK_SIZE_N2
        j = ((p //grid_k) % grid_j) * BLOCK_SIZE_M2 + hidden1
        k0 = p % grid_k
        acc = tl.zeros((BLOCK_SIZE_N2, BLOCK_SIZE_M2), dtype=tl.float32)
        if k0 == 0:
            acc += tl.load(old_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(tl.float32)
            acc += tl.load(bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :]).to(tl.float32)
        for k in range(k0 * BLOCK_SIZE_K2, features, BLOCK_SIZE_K2 * grid_k):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K2))[None, :],
                mask = ((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K2))[None, :] < features),
                other = 0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K2))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M2))[None, :],
                mask = ((k + tl.arange(0, BLOCK_SIZE_K2))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden),
                other = 0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out2_ptr + (i + tl.arange(0, BLOCK_SIZE_N2))[:, None] * ((hidden - hidden1)*grid_k) + (j - hidden1 + tl.arange(0, BLOCK_SIZE_M2))[None, :] + k0 * (hidden - hidden1),
            acc.to(tl.bfloat16),
            mask = ((i + tl.arange(0, BLOCK_SIZE_N2))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M2))[None, :] < hidden)
        )

@triton.jit
def combine_1536_1152_twopart(out_ptr, inp_ptr, seq_len:tl.constexpr, hidden:tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid * 2, seq_len, psize * 2):
        inp = tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 + tl.arange(0, 128)).to(tl.float32)
        for j in range(1, 4):
            inp += tl.load(inp_ptr + (i + tl.arange(0, 2)[:, None]) * 128 * 4 + tl.arange(0, 128) + 128 * j).to(tl.float32)
        tl.store(out_ptr + (i + tl.arange(0, 2)[:, None]) * hidden + 1024 + tl.arange(0, 128), inp.to(tl.bfloat16))

def matmul_n256_1152_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    if num_views == 2:
        if USE_CUBLAS_TWOPART_RES:
            matmul_512x1152x1152_twopart_bias_res_cublas(
                old=res,
                inp=x,
                weight=weight,
                bias=bias,
                out=out,
            )
            return
        matmul_512x1152x1152_twopart_bias_res[(256,)](
            old_ptr = res,
            inp_ptr = x,
            weight_ptr = weight,
            bias_ptr = bias,
            out_ptr = out,
            out2_ptr = buf,
            seq_len = 512,
            features = 1152,
            hidden = 1152,
        )
        combine_1536_1152_twopart[(256,)](
            out_ptr = out,
            inp_ptr = buf,
            seq_len = 512,
            hidden = 1152,
        )
        return
    seq_len = 256 * num_views
    matmul_small_bias_res[((seq_len + 31) // 32) * (1152 // 64),](
        x,
        weight,
        out,
        bias,
        res,
        seq_len = seq_len,
        features = 1152,
        hidden = 1152,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 32
    )

USE_CUBLAS_ENCODER_FFN = os.getenv("USE_CUBLAS_ENCODER_FFN", "1") == "1"

def layer_norm_matmul_n256_1152_4304_bias_gelu(x, norm_w, norm_b, weight, bias, out, x_norm):
    num_views = x.shape[0]
    seq_len = 256 * num_views

    _layer_norm_dispatch(x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152)

    if USE_CUBLAS_ENCODER_FFN:
        out_2d = out.view(seq_len, 4304)
        torch.addmm(bias, x_norm.view(seq_len, 1152), weight, out=out_2d)
        numel = seq_len * 4304
        gelu_inplace_kernel[((numel + 1023) // 1024,)](out_2d, numel)
        return
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_K = 64
    matmul_small_bias_gelu[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * ((4304 + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M),](
        x_norm,
        weight,
        out,
        bias,
        seq_len = seq_len,
        features = 1152,
        hidden = 4304,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_K = BLOCK_SIZE_K
    )

def matmul_n256_4304_1152_bias_res(x, weight, bias, res, out, buf):
    num_views = x.shape[0]
    seq_len = 256 * num_views
    if USE_CUBLAS_4304_1152_RES:
        matmul_4304_1152_bias_res_cublas(x, weight, bias, out)
        return
    matmul_split_k[((seq_len + 64) // 64) * (1152 // 64) * 4,](
        x,
        weight,
        buf,
        seq_len = seq_len,
        features = 4304,
        hidden = 1152,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64,
        SPLIT_K = 4
    )
    if MERGE_BLOCK_SIZE == 512:
        merge_split_k_bias_res_512[(seq_len * 1152 + 511) // 512,](
            buf,
            bias,
            res,
            out,
            seq_len = seq_len,
            hidden = 1152,
            SPLIT_K = 4
        )
    else:
        merge_split_k_bias_res_4096[(seq_len * 1152 + 4095) // 4096,](
            buf,
            bias,
            res,
            out,
            seq_len = seq_len,
            hidden = 1152,
            SPLIT_K = 4
        )

@torch.compile
def AttnMultiKey(QKV):
    QKV = QKV.view(-1, 256, 3, 16, 72).permute(0, 2, 3, 1, 4)
    Q = QKV[:, 0]
    K = QKV[:, 1]
    V = QKV[:, 2]
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(Q.shape[0], 256, 1152)
    return attn

def vision_encoder(weights, buffers, num_views):
    conv2d_embed_n256_1152_res(
        buffers['observation_images_normalized'],
        weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],
        weights['vision_position_embedding'],
        buffers['vision_x']
    )

    for i in range(27):
        layer_norm_QKV_matmul_n256_1152_3456_bias(
            buffers['vision_x'],
            weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],
            weights['vision_attn_qkv_w'][i],
            weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'],
            buffers['vision_x_norm']
        )

        attn = AttnMultiKey(buffers['vision_QKV'])

        matmul_n256_1152_1152_bias_res(
            attn,
            weights['vision_attn_o_w'][i],
            weights['vision_attn_o_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf']
        )

        layer_norm_matmul_n256_1152_4304_bias_gelu(
            buffers['vision_x'],
            weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i],
            weights['vision_ffn_up_w'][i],
            weights['vision_ffn_up_b'][i],
            buffers['vision_hidden'],
            buffers['vision_x_norm']
        )

        matmul_n256_4304_1152_bias_res(
            buffers['vision_hidden'],
            weights['vision_ffn_down_w'][i],
            weights['vision_ffn_down_b'][i],
            buffers['vision_x'],
            buffers['vision_x'],
            buffers['vision_x_split_k_buf']
        )

@triton.jit
def rms_norm_kernel(inp_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    BLOCK_SIZE : tl.constexpr = 512
    for i in range(pid, seq_len, psize):
        sum_x = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + 1e-6)

        for j in range(0, features, BLOCK_SIZE):
            x = tl.load(inp_ptr + i * features + j + tl.arange(0, BLOCK_SIZE))
            x = x * factor
            tl.store(out_ptr + i * features + j + tl.arange(0, BLOCK_SIZE), x)
@triton.jit
def matmul_small_gate(inp_ptr, weight1_ptr, weight2_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr = 128,
    BLOCK_SIZE_M : tl.constexpr = 64,
    BLOCK_SIZE_K : tl.constexpr = 32):

    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, hidden, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(
                    inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K), 
                    mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                    other = 0.0
                )
                w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
                w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
                acc2 = tl.dot(x, w2, acc2)
            acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
            acc = (acc * acc2).to(tl.bfloat16)
            tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M), acc, mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)

def matmul_small_gate_dispatch(x, weight1, weight2, out, seq_len, features, hidden):
    if USE_HYBRID_GATE:
        matmul_small_gate_hybrid(x, weight1, weight2, out)
        return
    if USE_CUBLASLT_GATE:
        matmul_small_gate_cublas(x, weight1, weight2, out)
        return
    matmul_small_gate[( (seq_len + 127)//128, (hidden + 63)//64 )](
        x, weight1, weight2, out, seq_len,
        features, hidden
    )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        # Larger tiles for high-throughput FFN gates on L40S
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def scaled_matmul_small_gate(inp_ptr, inp_norm_factor_ptr, weight1_ptr, weight2_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr = 64,
    BLOCK_SIZE_M : tl.constexpr = 64,
    BLOCK_SIZE_K : tl.constexpr = 64):

    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        norm_factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask = i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=0
        )
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K), 
                mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other = 0.0
            )
            x = x * norm_factor[:, None]
            w = tl.load(weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc = tl.dot(x, w, acc)
            w2 = tl.load(weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M))
            acc2 = tl.dot(x, w2, acc2)
        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M), acc, mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        # Larger tiles for high-throughput FFN gates on L40S
        triton.Config({"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["seq_len", "features", "hidden"],
)
@triton.jit
def rms_fused_matmul_small_gate(inp_ptr, weight1_ptr, weight2_ptr, out_ptr, seq_len : tl.constexpr, features : tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N : tl.constexpr = 64,
    BLOCK_SIZE_M : tl.constexpr = 64,
    BLOCK_SIZE_K : tl.constexpr = 64):
    """Fused RMSNorm factor + gated matmul (gate & up) with GELU * mul in one kernel."""
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M

        # 1) Compute RMS factor for these rows
        sum_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features
                + k + tl.arange(0, BLOCK_SIZE_K),
                mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other = 0.0,
            ).to(tl.float32)
            sum_sq += tl.sum(x * x, axis=1)
        rms = tl.rsqrt(sum_sq / features + 1e-6)

        # 2) Matmul over normalized x for gate and up
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features
                + k + tl.arange(0, BLOCK_SIZE_K),
                mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
                other = 0.0,
            ).to(tl.float32)
            x = x * rms[:, None]
            w1 = tl.load(
                weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden
                + j + tl.arange(0, BLOCK_SIZE_M)
            ).to(tl.float32)
            acc = tl.dot(x, w1, acc)
            w2 = tl.load(
                weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * hidden
                + j + tl.arange(0, BLOCK_SIZE_M)
            ).to(tl.float32)
            acc2 = tl.dot(x, w2, acc2)

        acc = acc * tl.sigmoid(1.5957691216057308 * acc * (1 + 0.044715 * acc * acc))
        acc = (acc * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + j + tl.arange(0, BLOCK_SIZE_M),
            acc,
            mask = i + tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len,
        )

def rms_matmul_n_2048_16384_gate(x, weight1, weight2, weight_concat, out, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    if USE_GATED_CONCAT:
        matmul_small_gate_up_hybrid(x_norm, weight_concat, out)
    else:
        matmul_small_gate_dispatch(x_norm, weight1, weight2, out, seq_len, 2048, 16384)

def matmul_n_16384_2048_res(x, weight, out):
    seq_len = x.shape[0]
    if USE_HYBRID_RES:
        matmul_small_res_hybrid(x, weight, out)
        return
    # Use conservative grid size calculation (assuming minimum block sizes from autotuner)
    BLOCK_SIZE_N = 128
    if seq_len < 512:
        BLOCK_SIZE_N = 64
    matmul_small_res[((seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N) * (2048 // 64),](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 16384,
        hidden = 2048
    )

USE_CUBLAS_ENCODER_PROJ = os.getenv("USE_CUBLAS_ENCODER_PROJ", "1") == "1"

def layer_norm_matmul_n256_1152_2048_bias(x, norm_w, norm_b, proj_w, proj_b, out, x_norm):
    seq_len = x.shape[0] * 256
    _layer_norm_dispatch(x, x_norm, norm_w, norm_b, seq_len=seq_len, features=1152, eps=1e-5)
    if USE_CUBLAS_ENCODER_PROJ:
        torch.addmm(proj_b, x_norm.view(seq_len, 1152), proj_w, out=out[:seq_len])
        return
    matmul_small_bias[((seq_len + 63) // 64) * (2048 // 64),](
        x_norm,
        proj_w,
        out,
        proj_b,
        seq_len = seq_len,
        features = 1152,
        hidden = 2048,
        BLOCK_SIZE_N = 64,
        BLOCK_SIZE_M = 64,
        BLOCK_SIZE_K = 64
    )

@triton.jit
def matmul_n_2048_2560_qkv_rope(inp_ptr, weight_QKV_ptr, rope_weights_ptr, Q_ptr, K_ptr, V_ptr, seq_len : tl.constexpr, features : tl.constexpr, head_dim : tl.constexpr, num_head : tl.constexpr):
    BLOCK_SIZE_N : tl.constexpr = 64
    BLOCK_SIZE_M : tl.constexpr = 64
    BLOCK_SIZE_K : tl.constexpr = 64
    pid1 = tl.program_id(axis=0)
    psize1 = tl.num_programs(axis=0)
    pid2 = tl.program_id(axis=1)
    psize2 = tl.num_programs(axis=1)
    for i in range(pid1 * BLOCK_SIZE_N, seq_len, psize1 * BLOCK_SIZE_N):
        for j in range(pid2 * BLOCK_SIZE_M, (num_head + 2) * head_dim, psize2 * BLOCK_SIZE_M):
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
            for k in range(0, features, BLOCK_SIZE_K):
                x = tl.load(inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * features + k + tl.arange(0, BLOCK_SIZE_K), mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i, other = 0.0)
                w = tl.load(weight_QKV_ptr + (k + tl.arange(0, BLOCK_SIZE_K)[:, None]) * ((num_head + 2) * head_dim) + j + tl.arange(0, BLOCK_SIZE_M))
                acc = tl.dot(x, w, acc)
            acc = acc.to(tl.bfloat16)
            if j < (num_head + 1) * head_dim:
                x0, x1 = tl.split(acc.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x_cossin = tl.load(rope_weights_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * head_dim + j % head_dim + tl.arange(0, BLOCK_SIZE_M)[None, :], mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i, other = 0.0)
                x_cos, x_sin = tl.split(x_cossin.reshape(BLOCK_SIZE_N, BLOCK_SIZE_M // 2, 2))
                x0_ = x0 * x_cos - x1 * x_sin
                x1_ = x1 * x_cos + x0 * x_sin
                acc = tl.interleave(x0_, x1_)
            if j < num_head * head_dim:
                out_ptr = Q_ptr
                out_stride = num_head * head_dim
            elif j < (num_head + 1) * head_dim:
                out_ptr = K_ptr
                out_stride = head_dim
            else:
                out_ptr = V_ptr
                out_stride = head_dim
            tl.store(out_ptr + (i + tl.arange(0, BLOCK_SIZE_N)[:, None]) * out_stride + j % out_stride + tl.arange(0, BLOCK_SIZE_M), acc, mask = tl.arange(0, BLOCK_SIZE_N)[:, None] < seq_len - i)

def rms_matmul_n_2048_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm):
    seq_len = x.shape[0]
    rms_norm_kernel[(seq_len,)](x, x_norm, seq_len, 2048)
    if USE_CUBLAS_ROPE_QKV:
        qkv_tmp = _get_qkv_tmp(x_norm, 2560)
        torch.matmul(x_norm.view(seq_len, 2048), weight_qkv, out=qkv_tmp)
        rope_split_qkv_kernel[(128,)](qkv_tmp, rope_weight, Q, K, V, seq_len, 256, 8)
    else:
        matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64, 2560 // 64)](
            x_norm, weight_qkv, rope_weight, Q, K, V, seq_len, 2048, 256, 8
        )

def AttnSingleKey(Q, K, V, scale):
    logits = torch.matmul(Q, K.T) * scale
    logits = torch.nn.functional.softmax(logits, dim=-1)
    attn = torch.matmul(logits, V).view(-1, 2048)
    return attn

def matmul_n_2048_2048_res(x, weight, out):
    seq_len = x.shape[0]
    if USE_HYBRID_RES:
        matmul_small_res_hybrid(x, weight, out)
        return
    matmul_small_res[((seq_len + 127) // 128) * (2048 // 64),](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 2048,
        hidden = 2048
    )

def matmul_res_fused_rms(x, weight, residual, x_norm, features=2048):
    """Fused: residual += x @ weight; x_norm = rms_norm(residual).
    Combines cuBLAS GEMM + fused_add_rms_norm_kernel (1 fewer kernel launch)."""
    if USE_FUSED_ADD_RMS_NORM:
        tmp = _get_res_tmp(residual)
        torch.matmul(x, weight, out=tmp)
        seq_len = residual.shape[0]
        fused_add_rms_norm_kernel[(seq_len,)](tmp, residual, x_norm, seq_len, features)
    else:
        if USE_HYBRID_RES:
            matmul_small_res_hybrid(x, weight, residual)
        else:
            seq_len = x.shape[0]
            matmul_small_res[((seq_len + 127) // 128) * (features // 64),](
                x, weight, residual, residual, seq_len=seq_len, features=x.shape[-1], hidden=features
            )
        seq_len = residual.shape[0]
        rms_norm_kernel[(seq_len,)](residual, x_norm, seq_len, features)

def matmul_qkv_rope_only(x_norm, weight_qkv, rope_weight, Q, K, V, features=2048, head_dim=256, num_heads=8):
    """QKV matmul + RoPE, WITHOUT rms_norm (rms_norm done separately or fused)."""
    seq_len = x_norm.shape[0]
    if USE_CUBLAS_ROPE_QKV:
        qkv_tmp = _get_qkv_tmp(x_norm, (num_heads + 2) * head_dim)
        torch.matmul(x_norm.view(seq_len, features), weight_qkv, out=qkv_tmp)
        grid = (128,)
        rope_split_qkv_kernel[grid](qkv_tmp, rope_weight, Q, K, V,
                                     seq_len, head_dim, num_heads)
    else:
        total_cols = (num_heads + 2) * head_dim
        matmul_n_2048_2560_qkv_rope[((seq_len + 63) // 64, total_cols // 64)](
            x_norm, weight_qkv, rope_weight, Q, K, V, seq_len, features, head_dim, num_heads
        )

def transformer_encoder(weights, buffers, encoder_seq_len):
    layer_norm_matmul_n256_1152_2048_bias(
        buffers['vision_x'],
        weights['vision_final_norm_w'],
        weights['vision_final_norm_b'],
        weights['encoder_multi_modal_projector_w'],
        weights['encoder_multi_modal_projector_b'],
        buffers['encoder_x'],
        buffers['vision_x_norm']
    )
    if USE_FUSED_ADD_RMS_NORM:
        # ── Restructured encoder loop: fuse add_inplace + rms_norm ──
        seq_len = encoder_seq_len
        # First iteration: standalone rms_norm + QKV (no preceding residual add)
        rms_norm_kernel[(seq_len,)](buffers['encoder_x'], buffers['encoder_x_norm'], seq_len, 2048)
        matmul_qkv_rope_only(
            buffers['encoder_x_norm'], weights['encoder_attn_qkv_w'][0],
            buffers['encoder_rope_weights'], buffers['encoder_Q'],
            buffers['encoder_K'][0, :encoder_seq_len], buffers['encoder_V'][0, :encoder_seq_len],
        )
        for i in range(17):  # i = 0..16
            scale = 1.0 / (256 ** 0.5)
            attn = AttnSingleKey(buffers['encoder_Q'], buffers['encoder_K'][i, :encoder_seq_len], buffers['encoder_V'][i, :encoder_seq_len], scale)

            # Fused: encoder_x += attn @ attn_o_w; encoder_x_norm = rms_norm(encoder_x)
            matmul_res_fused_rms(attn, weights['encoder_attn_o_w'][i], buffers['encoder_x'], buffers['encoder_x_norm'])
            # FFN gate (rms_norm already done in fused kernel)
            if USE_GATED_CONCAT:
                matmul_small_gate_up_hybrid(
                    buffers['encoder_x_norm'], weights.get('encoder_ffn_gate_up_w', [None]*18)[i],
                    buffers['encoder_hidden']
                )
            else:
                matmul_small_gate_dispatch(buffers['encoder_x_norm'], weights['encoder_ffn_gate_w'][i], weights['encoder_ffn_up_w'][i], buffers['encoder_hidden'], seq_len, 2048, 16384)
            # Fused: encoder_x += hidden @ ffn_down_w; encoder_x_norm = rms_norm(encoder_x)
            matmul_res_fused_rms(buffers['encoder_hidden'], weights['encoder_ffn_down_w'][i], buffers['encoder_x'], buffers['encoder_x_norm'])
            # QKV + RoPE for next iteration (rms_norm already done)
            matmul_qkv_rope_only(
                buffers['encoder_x_norm'], weights['encoder_attn_qkv_w'][i + 1],
                buffers['encoder_rope_weights'], buffers['encoder_Q'],
                buffers['encoder_K'][i + 1, :encoder_seq_len], buffers['encoder_V'][i + 1, :encoder_seq_len],
            )
    else:
        # ── Original encoder loop ──
        for i in range(18):
            rms_matmul_n_2048_2560_qkv_rope(
                buffers['encoder_x'],
                weights['encoder_attn_qkv_w'][i],
                buffers['encoder_rope_weights'],
                buffers['encoder_Q'],
                buffers['encoder_K'][i, :encoder_seq_len],
                buffers['encoder_V'][i, :encoder_seq_len],
                buffers['encoder_x_norm']
            )
            if i != 17:
                scale = 1.0 / (256 ** 0.5)
                attn = AttnSingleKey(buffers['encoder_Q'], buffers['encoder_K'][i, :encoder_seq_len], buffers['encoder_V'][i, :encoder_seq_len], scale)
                matmul_n_2048_2048_res(attn, weights['encoder_attn_o_w'][i], buffers['encoder_x'])
                rms_matmul_n_2048_16384_gate(
                    buffers['encoder_x'], 
                    weights.get('encoder_ffn_gate_w', [None]*18)[i], 
                    weights.get('encoder_ffn_up_w', [None]*18)[i], 
                    weights.get('encoder_ffn_gate_up_w', [None]*18)[i], 
                    buffers['encoder_hidden'], buffers['encoder_x_norm']
                )
                matmul_n_16384_2048_res(buffers['encoder_hidden'], weights['encoder_ffn_down_w'][i], buffers['encoder_x'])

@triton.jit
def matvec_bias_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr, features : tl.constexpr, hidden : tl.constexpr):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)

    BLOCK_SIZE_N : tl.constexpr = 32
    BLOCK_SIZE_M : tl.constexpr = 8
    for j in range(pid * BLOCK_SIZE_M, hidden, psize * BLOCK_SIZE_M):
        acc = tl.load(bias_ptr + j + tl.arange(0, BLOCK_SIZE_M)).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_N):
            x = tl.load(x_ptr + k + tl.arange(0, BLOCK_SIZE_N), mask = (k + tl.arange(0, BLOCK_SIZE_N) < features), other = 0.0)
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_N)[:, None]) * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask = (k + tl.arange(0, BLOCK_SIZE_N)[:, None] < features) & (j + tl.arange(0, BLOCK_SIZE_M)[None, :] < hidden),
                other = 0.0
            )
            acc += tl.sum(x[:, None] * w, axis=0)
        tl.store(out_ptr + j + tl.arange(0, BLOCK_SIZE_M), acc.to(tl.bfloat16), mask = j + tl.arange(0, BLOCK_SIZE_M) < hidden)

def matmul_1_32_1024_bias(x, weight, bias, out):
    matvec_bias_kernel[((1024 + 7)//8, )](
        x,
        weight,
        bias,
        out,
        features = 32,
        hidden = 1024
    )

def matmul_k_32_1024_bias_silu(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 32,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 32
    )

def matmul_k_1024_1024_bias(x, weight, bias, out):
    seq_len = x.shape[0]
    matmul_small_bias[((seq_len + 31) // 32) * (1024 // 32),] (
        x, weight, out, bias,
        seq_len = seq_len,
        features = 1024,
        hidden = 1024,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64
    )

@triton.jit
def rmsnorm_factor_kernel(inp_ptr, factor_ptr, rows: tl.constexpr, features: tl.constexpr, eps: tl.constexpr = 1e-6, BLOCK_SIZE: tl.constexpr = 1024):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    for i in range(pid, rows, psize):
        sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            x = tl.load(inp_ptr + i * features + offs, mask=offs < features, other=0.0)
            sum_x += x * x
        factor = tl.rsqrt(tl.sum(sum_x) / features + eps)
        tl.store(factor_ptr + i, factor)

def rms_matmul_k_1024_32_bias_res(x, weight, bias, out, x_norm_factor):
    seq_len = x.shape[0]
    # Fused RMSNorm factor + matmul + bias + residual in a single Triton kernel.
    rms_fused_matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),](
        x,
        weight,
        out,
        bias,
        out,
        seq_len=seq_len,
        features=1024,
        hidden=32,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_K=256,
    )

@triton.jit
def scaled_matmul_rope_qkv(
    inp_ptr, inp_norm_factor_ptr, seq_len: tl.constexpr, features: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
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

        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_SIZE_M),
            mask = start_i + tl.arange(0, BLOCK_SIZE_M) < seq_len, other=0
        )
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask = offs_k[None, :] < features, other = 0
            )
            # x = x * norm_factor[:, None]
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * ((num_heads + 2) * head_dim) + offs_j,
                mask = offs_k[:, None] < features, other = 0
            )
            accumulator = tl.dot(x, w, accumulator)
        accumulator = accumulator * norm_factor[:, None]
        
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

def rms_matmul_k_1024_2560_qkv_rope(x, weight_qkv, rope_weight, Q, K, V, x_norm_factor):
    seq_len = x.shape[0]
    rmsnorm_factor_kernel[(128,)](x, x_norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
    scaled_matmul_rope_qkv[(128,)](
        x, x_norm_factor, seq_len, 1024, 256, 8,
        weight_qkv, rope_weight, Q, K, V,
    )

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=1),
        # Larger tiles for attention logits matmul on L40S
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_abT_scale(
    q_ptr, k_ptr, out_ptr, M : tl.constexpr, N : tl.constexpr, K : tl.constexpr,
    scale_factor: tl.constexpr,
    BLOCK_SIZE_M : tl.constexpr = 32, BLOCK_SIZE_N : tl.constexpr = 32, BLOCK_SIZE_K : tl.constexpr = 64,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    grid_m = triton.cdiv(M, BLOCK_SIZE_M)
    grid_n = triton.cdiv(N, BLOCK_SIZE_N)

    while pid < grid_m * grid_n:
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_i = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_j = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_SIZE_K):
            offs_k = k + tl.arange(0, BLOCK_SIZE_K)
            x = tl.load(q_ptr + offs_i[:, None] * K + offs_k[None, :], mask = offs_k[None, :] < K, other = 0)
            w = tl.load(k_ptr + offs_j[:, None] * K + offs_k[None, :], mask = offs_k[None, :] < K, other = 0)
            accumulator = tl.dot(x, tl.trans(w), accumulator)
        accumulator = accumulator * scale_factor
        tl.store(out_ptr + offs_i[:, None] * N + offs_j[None, :], accumulator.to(tl.bfloat16),
                 mask = (offs_i[:, None] < M) & (offs_j[None, :] < N))
        pid += psize


@triton.jit
def softmax_kernel_mask0(
    inp_ptr, queries : tl.constexpr, keys : tl.constexpr,
    num_heads : tl.constexpr, encoder_seq_len : tl.constexpr,
    out_ptr, BLOCK_SIZE_M: tl.constexpr = 4, BLOCK_SIZE: tl.constexpr = 1024):

    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)

    assert BLOCK_SIZE >= queries, f"BLOCK_SIZE must be >= N, got {BLOCK_SIZE} < {queries}"

    for i in range(pid * BLOCK_SIZE_M, queries, psize * BLOCK_SIZE_M):
        offs_i = i + tl.arange(0, BLOCK_SIZE_M)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]
        
        attn_mask = (offs_i < queries) & (offs_j < keys)
        attn_mask = attn_mask & ((offs_i >= num_heads) | (offs_j <= encoder_seq_len))
        vals = tl.load(inp_ptr + offs_i * keys + offs_j,
                       mask = attn_mask, other = -float('inf'))
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        vsum = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        vals = vals / vsum
        vals = vals.to(tl.bfloat16)
        tl.store(
            out_ptr + offs_i * keys + offs_j, vals,
            mask = (offs_i < queries) & (offs_j < keys)
        )

USE_CUBLAS_ABT_SCALE = os.getenv("USE_CUBLAS_ABT_SCALE", "1") == "1"

def matmul_abT_scale_dispatch(Q, K, out, scale):
    """Q @ K^T * scale — cuBLAS or Triton."""
    total_queries = Q.shape[0]
    total_keys = K.shape[0]
    head_dim = Q.shape[1]
    if USE_CUBLAS_ABT_SCALE:
        if USE_SCALED_MM and out.dtype == Q.dtype:
            # Fuse scale into GEMM via cuBLAS alpha: C = scale * Q @ K^T
            # Eliminates the separate out.mul_(scale) kernel
            torch.addmm(out, Q, K.t(), beta=0.0, alpha=scale, out=out)
        elif out.dtype == Q.dtype:
            torch.mm(Q, K.t(), out=out)
            out.mul_(scale)
        else:
            out.copy_(torch.mm(Q, K.t()))
            out.mul_(scale)
        return
    matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
        Q, K, out, total_queries, total_keys, head_dim, scale)

def matmul_k8_256_n_softmax_mask0(Q, K, out, encoder_seq_len):
    total_queries = Q.shape[0]
    total_keys = K.shape[0]
    matmul_abT_scale_dispatch(Q, K, out, 256 ** -0.5)
    softmax_kernel_mask0[((total_queries + 3) // 4,)](out,
        total_queries, total_keys, 8, encoder_seq_len, out,
        BLOCK_SIZE_M=4, BLOCK_SIZE=1024)

USE_CUBLAS_K8_N_256 = os.getenv("USE_CUBLAS_K8_N_256", "1") == "1"

def matmul_k8_n_256(x, V, out):
    total_queries = x.shape[0]
    total_keys = V.shape[0]
    head_dim = 256
    if USE_CUBLAS_K8_N_256:
        torch.matmul(x, V, out=out)
        return
    matmul_small[((total_keys + 31) // 32) * (head_dim // 32),](
        x, V, out,
        seq_len = total_queries,
        features = total_keys,
        hidden = head_dim,
        BLOCK_SIZE_N = 32,
        BLOCK_SIZE_M = 32,
        BLOCK_SIZE_K = 64,
    )

def matmul_k_2048_1024_res(x, weight, out):
    seq_len = x.shape[0]
    if USE_HYBRID_RES:
        matmul_small_res_hybrid(x, weight, out)
        return
    matmul_small_res[(128,)](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 2048,
        hidden = 1024
    )

def rms_matmul_k_1024_4096_gate(x, weight1, weight2, weight_concat, out, x_norm_factor):
    seq_len = x.shape[0]
    if USE_GATED_CONCAT:
        rms_norm_kernel[(seq_len,)](x, x_norm_factor, seq_len, 1024)
        matmul_small_gate_up_hybrid(x_norm_factor, weight_concat, out)
    else:
        # Fused RMSNorm factor + gated matmul (gate & up) with GELU * mul.
        rms_fused_matmul_small_gate[(128,)](
            x,
            weight1,
            weight2,
            out,
            seq_len = seq_len,
            features = 1024,
            hidden = 4096,
            BLOCK_SIZE_N = 64,
            BLOCK_SIZE_M = 64,
            BLOCK_SIZE_K = 64,
        )

def matmul_k_4096_1024_res(x, weight, out):
    seq_len = x.shape[0]
    if USE_HYBRID_RES:
        matmul_small_res_hybrid(x, weight, out)
        return
    matmul_small_res[(((seq_len + 15) // 16) * (1024 // 32),)](
        x,
        weight,
        out,
        out,
        seq_len = seq_len,
        features = 4096,
        hidden = 1024
    )

def transformer_decoder(weights, buffers, encoder_seq_len):
    matmul_1_32_1024_bias(
        buffers['observation_state_normalized'],
        weights['decoder_state_in_proj_w'],
        weights['decoder_state_in_proj_b'],
        buffers['decoder_state_buf']
    )
    for step in range(10):
        buffers['decoder_x'][:1].copy_(buffers['decoder_state_buf'])
        matmul_k_32_1024_bias_silu(
            buffers['diffusion_noise'],
            weights['decoder_action_fused_in_proj_w'],
            weights['decoder_action_fused_time_biases'][step%10],
            buffers['decoder_x_buf']
        )
        matmul_k_1024_1024_bias(
            buffers['decoder_x_buf'],
            weights['decoder_action_mlp_w'],
            weights['decoder_action_mlp_b'],
            buffers['decoder_x'][1:]
        )
        for i in range(18):
            rms_matmul_k_1024_2560_qkv_rope(
                buffers['decoder_x'], weights['decoder_attn_qkv_w'][i],
                buffers['decoder_rope_weights'],
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i][encoder_seq_len:],
                buffers['encoder_V'][i][encoder_seq_len:],
                buffers['decoder_norm_factor_buf']
            )
            matmul_k8_256_n_softmax_mask0(
                buffers['decoder_q_buf'],
                buffers['encoder_K'][i],
                buffers['decoder_attn_buf'],
                encoder_seq_len
            )
            matmul_k8_n_256(
                buffers['decoder_attn_buf'],
                buffers['encoder_V'][i],
                buffers['decoder_q_buf']
            )

            matmul_k_2048_1024_res(
                buffers['decoder_q_buf'].view(-1, 2048),
                weights['decoder_attn_o_w'][i],
                buffers['decoder_x']
            )
            rms_matmul_k_1024_4096_gate(
                buffers['decoder_x'],
                weights.get('decoder_ffn_gate_w', [None]*18)[i],
                weights.get('decoder_ffn_up_w', [None]*18)[i],
                weights.get('decoder_ffn_gate_up_w', [None]*18)[i],
                buffers['decoder_hidden'],
                buffers['decoder_x_norm'] if USE_GATED_CONCAT else buffers['decoder_norm_factor_buf']
            )
            matmul_k_4096_1024_res(
                buffers['decoder_hidden'],
                weights['decoder_ffn_down_w'][i],
                buffers['decoder_x']
            )
        rms_matmul_k_1024_32_bias_res(
            buffers['decoder_x'][1:],
            weights['decoder_action_fused_out_proj_w'],
            weights['decoder_action_fused_out_proj_b'],
            buffers['diffusion_noise'],
            buffers['decoder_norm_factor_buf'],
        )

def pi0_model(weights, buffers, num_views):
    encoder_seq_len = buffers['encoder_x'].shape[0]
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder(weights, buffers, encoder_seq_len)

class Pi0InferenceIMLE:
    def __init__(self, checkpoint, num_views, chunk_size):
        self.num_views = num_views
        self.chunk_size = chunk_size
        encoded_prompt = checkpoint['language_embeds']
        self.prompt_len = len(encoded_prompt)

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

            "decoder_state_in_proj_w":            torch.empty(32, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_state_in_proj_b":            torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_in_proj_w":     torch.empty(32, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_time_biases":   torch.empty(10, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_mlp_w":               torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_mlp_b":               torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_out_proj_w":    torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_fused_out_proj_b":    torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),

            "language_embeds": torch.empty(self.prompt_len, 2048,  dtype = torch.bfloat16, device = "cuda"),
        }

        if USE_GATED_CONCAT:
            self.weights.update({
                "encoder_ffn_gate_up_w":              torch.empty(18, 2048, 32768,        dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_gate_up_w":              torch.empty(18, 1024, 8192,         dtype = torch.bfloat16, device = "cuda"),
            })
        else:
            self.weights.update({
                "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
                "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
                "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            })

        encoder_seq_len = num_views * 256 + self.prompt_len
        decoder_seq_len = chunk_size + 1

        self.buffers = {
            'observation_images_normalized':      torch.empty(num_views, 224, 224,3,           dtype=torch.bfloat16,   device = "cuda"),
            'observation_state_normalized':       torch.empty(32,                              dtype = torch.bfloat16, device = "cuda"),
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
            'decoder_rope_weights':               torch.empty(decoder_seq_len, 256,            dtype = torch.bfloat16, device = "cuda"),
            'decoder_x':                          torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_norm':                     torch.empty((decoder_seq_len, 1024),         dtype = torch.bfloat16, device = "cuda"),
            'decoder_x_buf':                      torch.empty((chunk_size, 1024),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_state_buf':                  torch.empty((1, 1024),                       dtype = torch.bfloat16, device = "cuda"),
            'decoder_norm_factor_buf':            torch.empty((decoder_seq_len,),              dtype = torch.bfloat16, device = "cuda"),
            'decoder_q_buf':                      torch.empty((decoder_seq_len * 8, 256),      dtype = torch.bfloat16, device = "cuda"),
            'decoder_attn_buf':                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len),  dtype = torch.bfloat16, device = "cuda"),
            'decoder_hidden':                     torch.empty((decoder_seq_len, 4096),         dtype = torch.bfloat16, device = "cuda"),
            'decode_split_k_buf':                 torch.empty((2, decoder_seq_len, 1024),      dtype = torch.float32, device = "cuda"),
        }

        position_ids = torch.arange(encoder_seq_len, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self.buffers['encoder_rope_weights'].copy_(
            torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        )

        position_ids = torch.arange(decoder_seq_len, device="cuda") + encoder_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self.buffers['decoder_rope_weights'].copy_(
            torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        )

        for k, v in checkpoint.items():
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

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()
    
    def record_run(self):
        self.buffers['encoder_x'][self.num_views * 256:].copy_(self.weights['language_embeds'])
        pi0_model(self.weights, self.buffers, self.num_views)
    
    def record_infer_graph(self):
        for i in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(self, observation_images_normalized, observation_state_normalized, diffusion_noise):
        self.buffers['observation_images_normalized'].copy_(observation_images_normalized)
        self.buffers['observation_state_normalized'].copy_(observation_state_normalized)
        self.buffers['diffusion_noise'].copy_(diffusion_noise)
        self.infer_graph.replay()
        return self.buffers['diffusion_noise']