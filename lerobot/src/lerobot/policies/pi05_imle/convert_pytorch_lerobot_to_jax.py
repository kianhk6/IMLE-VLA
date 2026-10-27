#!/usr/bin/env python3
"""
Convert a PyTorch LeRobot PI05 checkpoint to JAX (OpenPI) format.

This is the reverse of OpenPiMLE/examples/convert_jax_model_to_pytorch.py.

Usage:
    python convert_pytorch_lerobot_to_jax.py \
        --pytorch_path /path/to/lerobot_checkpoint/pretrained_model \
        --output_path /path/to/output_jax_checkpoint \
        --reference_jax_path /path/to/reference_openpi_jax_checkpoint/<step>
"""

import argparse
import os
import pathlib
import shutil

import ml_dtypes
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pytorch_safetensors(safetensors_path: str) -> dict[str, np.ndarray]:
    """Load a safetensors file as numpy arrays (works without PyTorch).

    ml_dtypes must be imported before safetensors so that numpy recognises
    bfloat16.
    """
    import ml_dtypes as _  # noqa: F811 – registers bfloat16 with numpy
    from safetensors.numpy import load_file
    return load_file(safetensors_path)


def to_f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def to_bf16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=ml_dtypes.bfloat16)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_pytorch_to_jax(
    pytorch_path: str,
    output_path: str,
    reference_jax_path: str | None = None,
):
    """Convert PyTorch LeRobot PI05 checkpoint → JAX/orbax checkpoint.

    The output checkpoint is saved at ``<output_path>/params/`` in the format
    expected by ``openpi.models.model.restore_params``.
    """
    import jax
    import orbax.checkpoint as ocp

    # ==================================================================
    # 1.  Load PyTorch safetensors
    # ==================================================================
    safetensors_path = os.path.join(pytorch_path, "model.safetensors")
    print(f"[1/5] Loading PyTorch checkpoint from {safetensors_path}")
    pt_state = load_pytorch_safetensors(safetensors_path)

    # Strip "model." prefix; drop EMA.
    # NOTE: embed_tokens is tied with lm_head in safetensors, so we keep
    # paligemma.lm_head.weight (it IS the embedding) but skip expert lm_head.
    state: dict[str, np.ndarray] = {}
    for k, v in pt_state.items():
        if k.startswith("model_ema."):
            continue
        key = k.removeprefix("model.")
        # Expert lm_head is not in JAX checkpoint (different arch)
        if "gemma_expert.lm_head" in key:
            continue
        state[key] = to_f32(v)

    print(f"    → {len(state)} parameter tensors (excl. EMA / expert lm_head)")

    # ==================================================================
    # 2.  Architecture constants
    # ==================================================================
    # PaliGemma (gemma_2b)
    PH, PL, PN, PD, PM = 2048, 18, 8, 256, 16384   # hidden, layers, heads, head_dim, mlp_dim
    # Vision tower (SigLIP)
    VH, VL, VN = 1152, 27, 16                        # hidden, layers, heads
    VD = VH // VN                                     # head_dim = 72
    # Expert (gemma_300m)
    EH, EL, EN, ED, EM = 1024, 18, 8, 256, 4096

    # shorthand prefixes
    VIS = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    LLM = "paligemma_with_expert.paligemma.model.language_model"
    EXP = "paligemma_with_expert.gemma_expert.model"

    # ==================================================================
    # 3.  Vision Tower  →  img/*
    # ==================================================================
    print("[2/5] Converting vision tower …")

    # --- patch embedding ---
    # PT (C_out, C_in, H, W)  →  JAX (H, W, C_in, C_out)
    img_emb_kernel = state[f"{VIS}.embeddings.patch_embedding.weight"].transpose(2, 3, 1, 0)
    img_emb_bias   = state[f"{VIS}.embeddings.patch_embedding.bias"]

    # --- position embedding ---
    # PT (num_patches, hidden)  →  JAX (1, num_patches, hidden)
    img_pos = state[f"{VIS}.embeddings.position_embedding.weight"].reshape(1, -1, VH)

    # --- encoder blocks (stacked across 27 layers) ---
    def _vis(layer, suffix):
        return state[f"{VIS}.encoder.layers.{layer}.{suffix}"]

    ln0_s  = np.stack([_vis(i, "layer_norm1.weight") for i in range(VL)])
    ln0_b  = np.stack([_vis(i, "layer_norm1.bias")   for i in range(VL)])
    ln1_s  = np.stack([_vis(i, "layer_norm2.weight") for i in range(VL)])
    ln1_b  = np.stack([_vis(i, "layer_norm2.bias")   for i in range(VL)])

    mlp0_k = np.stack([_vis(i, "mlp.fc1.weight").T for i in range(VL)])   # (in, out)
    mlp0_b = np.stack([_vis(i, "mlp.fc1.bias")     for i in range(VL)])
    mlp1_k = np.stack([_vis(i, "mlp.fc2.weight").T for i in range(VL)])
    mlp1_b = np.stack([_vis(i, "mlp.fc2.bias")     for i in range(VL)])

    # Attention – reverse .reshape(-1, VH).T  →  .T.reshape(VH, VN, VD)
    ak_k = np.stack([_vis(i, "self_attn.k_proj.weight").T.reshape(VH, VN, VD) for i in range(VL)])
    ak_b = np.stack([_vis(i, "self_attn.k_proj.bias").reshape(VN, VD)         for i in range(VL)])
    av_k = np.stack([_vis(i, "self_attn.v_proj.weight").T.reshape(VH, VN, VD) for i in range(VL)])
    av_b = np.stack([_vis(i, "self_attn.v_proj.bias").reshape(VN, VD)         for i in range(VL)])
    aq_k = np.stack([_vis(i, "self_attn.q_proj.weight").T.reshape(VH, VN, VD) for i in range(VL)])
    aq_b = np.stack([_vis(i, "self_attn.q_proj.bias").reshape(VN, VD)         for i in range(VL)])
    ao_k = np.stack([_vis(i, "self_attn.out_proj.weight").T.reshape(VN, VD, VH) for i in range(VL)])
    ao_b = np.stack([_vis(i, "self_attn.out_proj.bias")                          for i in range(VL)])

    # --- post layernorm ---
    enc_norm_s = state[f"{VIS}.post_layernorm.weight"]
    enc_norm_b = state[f"{VIS}.post_layernorm.bias"]

    # --- multimodal projector ---
    head_k = state["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"].T
    head_b = state["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"]

    # ==================================================================
    # 4.  PaliGemma LLM  →  llm/*  (suffix-less keys)
    # ==================================================================
    print("[3/5] Converting PaliGemma LLM …")

    def _llm(layer, suffix):
        return state[f"{LLM}.layers.{layer}.{suffix}"]

    # embed_tokens is tied with lm_head in HuggingFace PaliGemma;
    # safetensors stores tied weights once under lm_head.weight
    llm_embed = state["paligemma_with_expert.paligemma.lm_head.weight"]

    # q_einsum: PT (NH*HD, H) → reshape(NH, HD, H) → transpose(0,2,1) → (NH, H, HD)
    llm_q = np.stack([
        _llm(i, "self_attn.q_proj.weight").reshape(PN, PD, PH).transpose(0, 2, 1)
        for i in range(PL)
    ])  # (18, 8, 2048, 256)

    # kv_einsum: stack k & v  →  (layers, 2, 1, H, HD)
    llm_kv = np.stack([
        np.stack([
            _llm(i, "self_attn.k_proj.weight").T[np.newaxis],   # (1, H, HD)
            _llm(i, "self_attn.v_proj.weight").T[np.newaxis],
        ])
        for i in range(PL)
    ])  # (18, 2, 1, 2048, 256)

    # attn_vec_einsum (o_proj):
    #   forward: JAX (NH, HD, H) → transpose(2,0,1) → (H, NH, HD) → reshape(NH*HD, H)
    #   reverse: PT (NH*HD, H) → reshape(H, NH, HD) → transpose(1,2,0) → (NH, HD, H)
    llm_ov = np.stack([
        _llm(i, "self_attn.o_proj.weight").reshape(PH, PN, PD).transpose(1, 2, 0)
        for i in range(PL)
    ])  # (18, 8, 256, 2048)

    # MLP gating_einsum: (layers, 2, H, MLP)
    llm_ge = np.stack([
        np.stack([
            _llm(i, "mlp.gate_proj.weight").T,
            _llm(i, "mlp.up_proj.weight").T,
        ])
        for i in range(PL)
    ])  # (18, 2, 2048, 16384)

    llm_ml = np.stack([_llm(i, "mlp.down_proj.weight").T for i in range(PL)])  # (18, 16384, 2048)

    # Norms – standard RMSNorm (scale only)
    llm_pre_att  = np.stack([_llm(i, "input_layernorm.weight")          for i in range(PL)])
    llm_pre_ffw  = np.stack([_llm(i, "post_attention_layernorm.weight") for i in range(PL)])
    llm_fnorm    = state[f"{LLM}.norm.weight"]

    # ==================================================================
    # 5.  Expert (Gemma-300M)  →  llm/*_1  keys
    # ==================================================================
    print("[4/5] Converting expert model …")

    def _exp(layer, suffix):
        return state[f"{EXP}.layers.{layer}.{suffix}"]

    # q_einsum_1
    exp_q = np.stack([
        _exp(i, "self_attn.q_proj.weight").reshape(EN, ED, EH).transpose(0, 2, 1)
        for i in range(EL)
    ])  # (18, 8, 1024, 256)

    # kv_einsum_1
    exp_kv = np.stack([
        np.stack([
            _exp(i, "self_attn.k_proj.weight").T[np.newaxis],
            _exp(i, "self_attn.v_proj.weight").T[np.newaxis],
        ])
        for i in range(EL)
    ])  # (18, 2, 1, 1024, 256)

    # attn_vec_einsum_1 (o_proj)
    #   forward: JAX (NH, HD, H) → reshape(NH*HD, H) → transpose(1,0) → (H, NH*HD)
    #   reverse: PT (H, NH*HD) → .T → (NH*HD, H) → reshape(NH, HD, H)
    exp_ov = np.stack([
        _exp(i, "self_attn.o_proj.weight").T.reshape(EN, ED, EH)
        for i in range(EL)
    ])  # (18, 8, 256, 1024)

    # MLP
    exp_ge = np.stack([
        np.stack([
            _exp(i, "mlp.gate_proj.weight").T,
            _exp(i, "mlp.up_proj.weight").T,
        ])
        for i in range(EL)
    ])  # (18, 2, 1024, 4096)

    exp_ml = np.stack([_exp(i, "mlp.down_proj.weight").T for i in range(EL)])  # (18, 4096, 1024)

    # Adaptive norms (Dense layers)
    exp_pa_b = np.stack([_exp(i, "input_layernorm.dense.bias")            for i in range(EL)])
    exp_pa_k = np.stack([_exp(i, "input_layernorm.dense.weight").T        for i in range(EL)])
    exp_pf_b = np.stack([_exp(i, "post_attention_layernorm.dense.bias")   for i in range(EL)])
    exp_pf_k = np.stack([_exp(i, "post_attention_layernorm.dense.weight").T for i in range(EL)])

    exp_fn_b = state[f"{EXP}.norm.dense.bias"]
    exp_fn_k = state[f"{EXP}.norm.dense.weight"].T

    # ==================================================================
    # 6.  Projection layers  →  top-level keys
    # ==================================================================
    proj_keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    proj = {}
    for pk in proj_keys:
        proj[f"{pk}_kernel"] = state[f"{pk}.weight"].T
        proj[f"{pk}_bias"]   = state[f"{pk}.bias"]

    # ==================================================================
    # 7.  Apply target dtypes
    # ==================================================================
    # Keep ALL weights as float32 for lossless roundtrip conversion.
    # (Previously LLM weights were cast to bfloat16 here, but that can
    #  introduce precision loss during PyTorch→JAX→PyTorch roundtrips.)
    # Vision tower: float32   (already float32)
    # PaliGemma LLM: float32  (kept as float32 — no bf16 cast)
    # Expert: float32          (already float32)
    # Projections: float32     (already float32)

    # ==================================================================
    # 8.  Build nested param tree  (with "value" leaves for NNX compat)
    # ==================================================================
    print("[5/5] Building JAX parameter tree & saving …")

    V = lambda x: {"value": x}  # noqa: E731  — wrap leaf in {"value": ...}

    params_tree = {
        "params": {
            "PaliGemma": {
                "img": {
                    "embedding": {"kernel": V(img_emb_kernel), "bias": V(img_emb_bias)},
                    "pos_embedding": V(img_pos),
                    "Transformer": {
                        "encoderblock": {
                            "LayerNorm_0": {"scale": V(ln0_s), "bias": V(ln0_b)},
                            "LayerNorm_1": {"scale": V(ln1_s), "bias": V(ln1_b)},
                            "MlpBlock_0": {
                                "Dense_0": {"kernel": V(mlp0_k), "bias": V(mlp0_b)},
                                "Dense_1": {"kernel": V(mlp1_k), "bias": V(mlp1_b)},
                            },
                            "MultiHeadDotProductAttention_0": {
                                "key":   {"kernel": V(ak_k), "bias": V(ak_b)},
                                "value": {"kernel": V(av_k), "bias": V(av_b)},
                                "query": {"kernel": V(aq_k), "bias": V(aq_b)},
                                "out":   {"kernel": V(ao_k), "bias": V(ao_b)},
                            },
                        },
                        "encoder_norm": {"scale": V(enc_norm_s), "bias": V(enc_norm_b)},
                    },
                    "head": {"kernel": V(head_k), "bias": V(head_b)},
                },
                "llm": {
                    "embedder": {"input_embedding": V(llm_embed)},
                    "final_norm": {"scale": V(llm_fnorm)},
                    "final_norm_1": {
                        "Dense_0": {"kernel": V(exp_fn_k), "bias": V(exp_fn_b)},
                    },
                    "layers": {
                        "attn": {
                            "attn_vec_einsum":   {"w": V(llm_ov)},
                            "attn_vec_einsum_1": {"w": V(exp_ov)},
                            "kv_einsum":         {"w": V(llm_kv)},
                            "kv_einsum_1":       {"w": V(exp_kv)},
                            "q_einsum":          {"w": V(llm_q)},
                            "q_einsum_1":        {"w": V(exp_q)},
                        },
                        "mlp": {
                            "gating_einsum": V(llm_ge),
                            "linear":        V(llm_ml),
                        },
                        "mlp_1": {
                            "gating_einsum": V(exp_ge),
                            "linear":        V(exp_ml),
                        },
                        "pre_attention_norm": {"scale": V(llm_pre_att)},
                        "pre_attention_norm_1": {
                            "Dense_0": {"kernel": V(exp_pa_k), "bias": V(exp_pa_b)},
                        },
                        "pre_ffw_norm": {"scale": V(llm_pre_ffw)},
                        "pre_ffw_norm_1": {
                            "Dense_0": {"kernel": V(exp_pf_k), "bias": V(exp_pf_b)},
                        },
                    },
                },
            },
            "action_in_proj":  {"kernel": V(proj["action_in_proj_kernel"]),  "bias": V(proj["action_in_proj_bias"])},
            "action_out_proj": {"kernel": V(proj["action_out_proj_kernel"]), "bias": V(proj["action_out_proj_bias"])},
            "time_mlp_in":     {"kernel": V(proj["time_mlp_in_kernel"]),     "bias": V(proj["time_mlp_in_bias"])},
            "time_mlp_out":    {"kernel": V(proj["time_mlp_out_kernel"]),    "bias": V(proj["time_mlp_out_bias"])},
        }
    }

    # ==================================================================
    # 9.  Save with orbax
    # ==================================================================
    save_dir = pathlib.Path(output_path) / "params"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"    Saving to {save_dir} …")
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(save_dir), params_tree)
    print(f"    ✓ Saved JAX checkpoint to {save_dir}")

    # Copy assets if available alongside the pytorch checkpoint
    assets_src = pathlib.Path(pytorch_path).parent / "assets"
    if not assets_src.exists() and reference_jax_path:
        assets_src = pathlib.Path(reference_jax_path) / "assets"
    if assets_src.exists():
        assets_dst = pathlib.Path(output_path) / "assets"
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)
        print(f"    ✓ Copied assets from {assets_src}")

    # ==================================================================
    # 10.  Verify against reference (optional)
    # ==================================================================
    if reference_jax_path:
        verify_against_reference(str(save_dir), reference_jax_path)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_against_reference(saved_params_dir: str, reference_jax_path: str):
    """Load both checkpoints and compare shapes / keys."""
    import orbax.checkpoint as ocp
    from flax import traverse_util

    ref_params_dir = os.path.join(reference_jax_path, "params")
    print(f"\n[verify] Comparing shapes against reference {ref_params_dir}")

    ckptr = ocp.PyTreeCheckpointer()

    # ---------- reference ----------
    ref_meta = ckptr.metadata(ref_params_dir)
    ref_flat = {}

    def _flatten_meta(tree, prefix=""):
        if hasattr(tree, "items"):
            for k, v in tree.items():
                _flatten_meta(v, f"{prefix}/{k}" if prefix else k)
        else:
            # tree is an ArrayMetadata or similar
            ref_flat[prefix] = tree

    _flatten_meta(ref_meta)

    # ---------- converted ----------
    conv_meta = ckptr.metadata(saved_params_dir)
    conv_flat = {}
    _flatten_meta2 = lambda t, p="": (  # noqa: E731
        [_flatten_meta2(v, f"{p}/{k}" if p else k) for k, v in t.items()]
        if hasattr(t, "items")
        else conv_flat.update({p: t})
    )
    _flatten_meta2(conv_meta)

    # Compare
    ref_keys = set(ref_flat.keys())
    conv_keys = set(conv_flat.keys())

    missing = ref_keys - conv_keys
    extra = conv_keys - ref_keys
    common = ref_keys & conv_keys

    if missing:
        print(f"  ⚠  Missing keys ({len(missing)}):")
        for k in sorted(missing):
            print(f"      {k}: {ref_flat[k]}")

    if extra:
        print(f"  ⚠  Extra keys ({len(extra)}):")
        for k in sorted(extra):
            print(f"      {k}: {conv_flat[k]}")

    shape_mismatch = 0
    for k in sorted(common):
        r = ref_flat[k]
        c = conv_flat[k]
        r_shape = getattr(r, "shape", None)
        c_shape = getattr(c, "shape", None)
        if r_shape != c_shape:
            print(f"  ✗  Shape mismatch  {k}: ref {r_shape}  vs  conv {c_shape}")
            shape_mismatch += 1

    if not missing and not extra and shape_mismatch == 0:
        print(f"  ✓  All {len(common)} keys match in shape!")
    else:
        print(f"  Summary: {len(common)} common, {len(missing)} missing, "
              f"{len(extra)} extra, {shape_mismatch} shape mismatches")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch LeRobot PI05 checkpoint → JAX (OpenPI) format"
    )
    parser.add_argument(
        "--pytorch_path", required=True,
        help="Path to PyTorch checkpoint dir (contains model.safetensors)",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Path to save JAX checkpoint (params/ will be created inside)",
    )
    parser.add_argument(
        "--reference_jax_path", default=None,
        help="Optional: path to reference JAX checkpoint for shape verification",
    )
    args = parser.parse_args()
    convert_pytorch_to_jax(args.pytorch_path, args.output_path, args.reference_jax_path)


if __name__ == "__main__":
    main()
