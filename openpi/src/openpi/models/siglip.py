# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding

# Module-level flag: replaced JAX matmuls with torch.matmul (cuBLAS) via pure_callback
USE_TORCH_MATMUL = False

# Module-level flag: when True AND USE_TORCH_MATMUL is True, casts inputs to
# float32 before torch.matmul so the entire matmul runs in f32 (not just the
# accumulator).  The output is then cast back to bfloat16.
FORCE_F32_MATMUL = True

def torch_matmul_hack(x, w, b=None):
    """Bridge JAX -> Torch(cuBLAS) -> JAX.

    When FORCE_F32_MATMUL is True the inputs are cast to float32 before the
    matmul giving full f32 precision (not just f32 accumulation of bf16
    products).  The result is cast back to bf16.
    """
    import torch
    import numpy as _np

    x_np = _np.array(x)
    w_np = _np.array(w)

    original_w_shape = w_np.shape
    if w_np.ndim > 2:
        w_np = w_np.reshape(w_np.shape[0], -1)

    # --- numpy → torch (handle bfloat16 which numpy doesn't support natively) ---
    def _to_torch(arr):
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.astype(_np.float32)).to(torch.bfloat16).cuda()
        return torch.from_numpy(arr).cuda()

    x_pt = _to_torch(x_np)
    w_pt = _to_torch(w_np)

    if FORCE_F32_MATMUL:
        y_pt = torch.matmul(x_pt.float(), w_pt.float())
    else:
        y_pt = torch.matmul(x_pt, w_pt)

    if b is not None:
        b_np = _np.array(b)
        if b_np.ndim > 1:
            b_np = b_np.flatten()
        b_pt = _to_torch(b_np)
        if FORCE_F32_MATMUL:
            y_pt = y_pt + b_pt.float()
        else:
            y_pt = y_pt + b_pt

    if FORCE_F32_MATMUL:
        y_pt = y_pt.to(torch.bfloat16)

    if len(original_w_shape) > 2:
        target_shape = y_pt.shape[:-1] + original_w_shape[1:]
        y_pt = y_pt.view(*target_shape)

    return jnp.array(y_pt.float().cpu().numpy()).astype(x.dtype)


def torch_batched_matmul_hack(x, y):
    """Bridge JAX batched matmul -> torch.matmul (cuBLAS) -> JAX.

    Handles arbitrary leading batch dimensions, e.g. the attention score
    computation ``(B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K)``.

    Respects FORCE_F32_MATMUL — when True the matmul runs entirely in f32 and
    the result is cast back to bfloat16.
    """
    import torch
    import numpy as _np

    x_np = _np.array(x)
    y_np = _np.array(y)

    def _to_torch(arr):
        if arr.dtype.name == 'bfloat16':
            return torch.from_numpy(arr.astype(_np.float32)).to(torch.bfloat16).cuda()
        return torch.from_numpy(arr).cuda()

    x_pt = _to_torch(x_np)
    y_pt = _to_torch(y_np)

    if FORCE_F32_MATMUL:
        z_pt = torch.matmul(x_pt.float(), y_pt.float()).to(torch.bfloat16)
    else:
        z_pt = torch.matmul(x_pt, y_pt)

    return jnp.array(z_pt.float().cpu().numpy()).astype(x.dtype)

# Module-level flag: when True, the ViT stem (patch embed + posemb) runs in
# dtype_mm (typically bfloat16) instead of float32.  Useful for precision-parity
# testing against the Triton inference path which always uses bfloat16.
FORCE_DTYPE_IN_STEM = False

# Module-level flag: when True, replaces nn.MultiHeadDotProductAttention with
# an explicit reimplementation that captures per-step intermediates (Q, K, V,
# attn_weights, context).  Uses identical parameter names so existing
# checkpoints load correctly.  Also captures MLP intermediates (up_proj, gelu).
CAPTURE_ATTN_INTERMEDIATES = False


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class CustomDense(nn.Module):
    """Drop-in replacement for nn.Dense that supports torch_matmul_hack."""
    features: int
    dtype: str = 'float32'
    kernel_init: any = nn.initializers.xavier_uniform()
    bias_init: any = nn.initializers.zeros_init()
    
    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features), self.dtype)
        bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
        
        if USE_TORCH_MATMUL:
            out_shape = inputs.shape[:-1] + (self.features,)
            return jax.pure_callback(
                torch_matmul_hack,
                jax.ShapeDtypeStruct(out_shape, inputs.dtype),
                inputs, kernel, bias
            )
        else:
            y = jnp.dot(inputs, kernel)
            y = y + bias
            return y

class CustomDenseGeneral(nn.Module):
    """Drop-in for nn.DenseGeneral that supports torch_matmul_hack."""
    features: any # (heads, head_dim) or int
    dtype: str = 'float32'
    kernel_init: any = nn.initializers.xavier_uniform()
    bias_init: any = nn.initializers.zeros_init()
    axis: any = -1 # dummy to match API
    
    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        
        if isinstance(self.features, int):
            # Case 2: Output projection (H, D) -> Out
            # We assume axis=(-2, -1) or similar contracting last 2 dims
            # Or axis=-1 if input is just (..., D).
            # In _DebugMHDA output proj, input is (..., H, D).
            
            out_features = self.features
            
            # Check input rank. If we are projecting last 2 dims:
            if self.axis == (-2, -1) or (isinstance(self.axis, tuple) and set(self.axis) == {-2, -1}):
                 in_H = inputs.shape[-2]
                 in_D = inputs.shape[-1]
                 kernel = self.param('kernel', self.kernel_init, (in_H, in_D, out_features), self.dtype)
                 bias = self.param('bias', self.bias_init, (out_features,), self.dtype)
                 
                 if USE_TORCH_MATMUL:
                     inputs_flat = inputs.reshape(inputs.shape[:-2] + (-1,))
                     kernel_flat = kernel.reshape(-1, out_features)
                     out_shape = inputs.shape[:-2] + (out_features,)
                     
                     return jax.pure_callback(
                         torch_matmul_hack,
                         jax.ShapeDtypeStruct(out_shape, inputs.dtype),
                         inputs_flat, kernel_flat, bias
                     )
                 else:
                     y = jnp.einsum('...hd,hdo->...o', inputs, kernel)
                     y = y + bias
                     return y
            else:
                 # Assume standard dense (last dim)
                 kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], out_features), self.dtype)
                 bias = self.param('bias', self.bias_init, (out_features,), self.dtype)
                 if USE_TORCH_MATMUL:
                    out_shape = inputs.shape[:-1] + (out_features,)
                    return jax.pure_callback(
                        torch_matmul_hack,
                        jax.ShapeDtypeStruct(out_shape, inputs.dtype),
                        inputs, kernel, bias
                    )
                 else:
                    return jnp.dot(inputs, kernel) + bias

        else:
            # Case 1: QKV projection (D) -> (H, HD)
            D = inputs.shape[-1]
            H, HD = self.features
            
            kernel = self.param('kernel', self.kernel_init, (D, H, HD), self.dtype)
            bias = self.param('bias', self.bias_init, (H, HD), self.dtype)
            
            if USE_TORCH_MATMUL:
                out_shape = inputs.shape[:-1] + (H, HD)
                return jax.pure_callback(
                    torch_matmul_hack,
                    jax.ShapeDtypeStruct(out_shape, inputs.dtype),
                    inputs, kernel, bias
                )
            else:
                # x: (..., D), k: (D, H, HD) -> (..., H, HD)
                y = jnp.einsum('...d,dhk->...hk', inputs, kernel)
                y = y + bias
                return y


class _DebugMHDA(nn.Module):
    """Explicit reimplementation of nn.MultiHeadDotProductAttention that
    captures all intermediate activations.  Uses the same internal parameter
    names (query, key, value, out) so existing checkpoints load via
    name='MultiHeadDotProductAttention_0'.

    When USE_TORCH_MATMUL is True, the attention score computation
    (Q @ K^T) and context computation (probs @ V) are ALSO routed through
    cuBLAS via ``torch_batched_matmul_hack``, giving full cuBLAS parity.
    Softmax is forced to f32 in this mode for numerical consistency with
    PyTorch's ``torch.softmax(x.float(), dim=-1).to(bf16)`` pattern.

    Reference: flax/linen/attention.py  MultiHeadDotProductAttention.__call__
    """
    num_heads: int
    dtype: str = "float32"

    @nn.compact
    def __call__(self, inputs_q, inputs_kv):
        features = inputs_q.shape[-1]
        head_dim = features // self.num_heads

        dense = lambda name: CustomDenseGeneral(       # noqa: E731
            axis=-1,
            dtype=self.dtype,
            features=(self.num_heads, head_dim),
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init(),  # Explicit zero init for bias
            name=name,
        )

        query = dense('query')(inputs_q)   # (B, L, H, D)
        key   = dense('key')(inputs_kv)    # (B, L, H, D)
        value = dense('value')(inputs_kv)  # (B, L, H, D)

        # Flax scales Q (not scores): query = query / sqrt(depth)
        depth = query.shape[-1]
        query = query / jnp.sqrt(depth).astype(self.dtype)

        if USE_TORCH_MATMUL:
            # ---- Route attention matmuls through cuBLAS ----
            # q: (B, Q, H, D) -> (B, H, Q, D)
            q_t = jnp.transpose(query, (0, 2, 1, 3))
            # k: (B, K, H, D) -> (B, H, D, K)   (transposed for Q @ K^T)
            k_t = jnp.transpose(key, (0, 2, 3, 1))

            # scores: (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K)
            scores_shape = q_t.shape[:-1] + (k_t.shape[-1],)
            attn_weights = jax.pure_callback(
                torch_batched_matmul_hack,
                jax.ShapeDtypeStruct(scores_shape, query.dtype),
                q_t, k_t,
            )

            # Softmax in f32 for consistency with PyTorch path
            attn_weights = jax.nn.softmax(
                attn_weights.astype(jnp.float32)
            ).astype(self.dtype)

            # v: (B, K, H, D) -> (B, H, K, D)
            v_t = jnp.transpose(value, (0, 2, 1, 3))

            # context: (B, H, Q, K) @ (B, H, K, D) -> (B, H, Q, D)
            ctx_shape = attn_weights.shape[:-1] + (v_t.shape[-1],)
            context_t = jax.pure_callback(
                torch_batched_matmul_hack,
                jax.ShapeDtypeStruct(ctx_shape, query.dtype),
                attn_weights, v_t,
            )

            # Back to (B, Q, H, D)
            context = jnp.transpose(context_t, (0, 2, 1, 3))
        else:
            # ---- Standard JAX path (XLA matmul) ----
            # Attention weights: (..., H, Q, K)
            attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)
            attn_weights = jax.nn.softmax(attn_weights)

            # Context: (..., Q, H, D)
            context = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

        # Output projection
        out = CustomDenseGeneral(
            features=features,
            axis=(-2, -1),
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init(),
            name='out',
        )(context)

        intermediates = {
            'query': query,         # (B, L, H, D). Note: Q is already scaled
            'key': key,             # (B, L, H, D)
            'value': value,         # (B, L, H, D)
            'attn_weights': attn_weights,  # (B, H, Q, K)
            'context': context,     # (B, Q, H, D)
        }
        return out, intermediates


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = CustomDense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, name='Dense_0', **inits)(x)
        up_proj = x
        # Use exact GELU (approximate=False)
        x = jax.nn.gelu(x, approximate=False)
        gelu_out = x
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = CustomDense(d, dtype=self.dtype_mm, name='Dense_1', **inits)(x)
        if CAPTURE_ATTN_INTERMEDIATES:
            return x, {"up_proj": up_proj, "gelu": gelu_out}
        return x


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        out["ln_sa"] = y

        if CAPTURE_ATTN_INTERMEDIATES:
            # Explicit attention — captures Q, K, V, softmax, context
            # name= forces the same param scope as nn.MultiHeadDotProductAttention
            y, attn_inter = _DebugMHDA(
                num_heads=self.num_heads,
                dtype=self.dtype_mm,
                name='MultiHeadDotProductAttention_0',
            )(y, y)
            out["sa"] = y
            out["attn_q"]       = attn_inter["query"]
            out["attn_k"]       = attn_inter["key"]
            out["attn_v"]       = attn_inter["value"]
            out["attn_weights"] = attn_inter["attn_weights"]
            out["attn_context"] = attn_inter["context"]
        else:
            y = out["sa"] = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=deterministic,
                dtype=self.dtype_mm,
            )(y, y)

        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        out["ln_mlp"] = y

        if CAPTURE_ATTN_INTERMEDIATES:
            mlp_out, mlp_inter = MlpBlock(
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(y, deterministic)
            y = out["mlp"] = mlp_out
            out["mlp_up_proj"] = mlp_inter["up_proj"]
            out["mlp_gelu"]    = mlp_inter["gelu"]
        else:
            y = out["mlp"] = MlpBlock(
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                dtype_mm=self.dtype_mm,
            )(y, deterministic)

        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        # FORCE_DTYPE_IN_STEM: when True, use dtype_mm (bfloat16) for parity
        # with the Triton inference path.
        _stem_dt = self.dtype_mm if FORCE_DTYPE_IN_STEM else jnp.float32
        image = jnp.asarray(image, _stem_dt)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=_stem_dt,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", _stem_dt)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
