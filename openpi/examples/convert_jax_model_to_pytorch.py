#!/usr/bin/env python3
"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
    # Just inspect keys:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --inspect_only

    # Convert to PyTorch:
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /path/to/checkpoint --output_path /path/to/output

Example:
    # pi0_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_droid_pytorch

    # pi0_aloha_sim
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim_pytorch

    # pi05_droid
    python examples/convert_jax_model_to_pytorch.py --checkpoint_dir /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid --output_path /home/$USER/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch
"""

import json
import os
import pathlib
import shutil
import tempfile
from typing import Literal

from flax.nnx import traversals
import jax
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils
import openpi.training.config as _config


def convert_jax_array_to_torch(value):
    """Convert JAX/numpy array to PyTorch tensor, preserving original dtype.
    
    This function preserves the exact dtype from the source array:
    - JAX bfloat16 → PyTorch bfloat16
    - JAX float32 → PyTorch float32
    - etc.
    
    Args:
        value: JAX array, numpy array, or PyTorch tensor
        
    Returns:
        PyTorch tensor with the same dtype as the source
    """
    if isinstance(value, torch.Tensor):
        return value
    
    # For JAX arrays, check dtype BEFORE converting to numpy
    # (np.array() may convert bfloat16 to float32)
    if hasattr(value, 'dtype'):
        jax_dtype_str = str(value.dtype).lower()
        if 'bfloat16' in jax_dtype_str:
            # JAX bfloat16 → convert through float32, then to PyTorch bfloat16
            arr = np.array(value, dtype=np.float32)
            return torch.from_numpy(arr).to(torch.bfloat16)
        elif 'float32' in jax_dtype_str or 'float' in jax_dtype_str:
            # JAX float32 → PyTorch float32
            arr = np.array(value)
            return torch.from_numpy(arr)
        else:
            # Other dtypes - standard conversion
            arr = np.array(value)
            return torch.from_numpy(arr)
    
    # For numpy arrays, check dtype directly
    elif isinstance(value, np.ndarray):
        dtype_str = str(value.dtype).lower()
        if 'bfloat16' in dtype_str:
            return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(value)
    
    # Fallback for other array-like objects
    elif hasattr(value, '__array__'):
        try:
            arr = np.array(value)
            dtype_str = str(arr.dtype).lower()
            if 'bfloat16' in dtype_str:
                return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
            else:
                return torch.from_numpy(arr)
        except (TypeError, ValueError):
            return value
    else:
        return value


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format.
    
    Each parameter's dtype is detected directly from the JAX/numpy array and preserved
    during conversion to PyTorch via convert_jax_array_to_torch().
    
    Args:
        state_dict: JAX parameter dictionary (values are jax.Array or np.ndarray)
        config: Model config
    
    Returns:
        (pytorch_state_dict, expert_dict)
    """
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    val = state_dict.pop(jax_key)
    state_dict[pytorch_key] = convert_jax_array_to_torch(val.transpose(3, 2, 0, 1) if len(val.shape) == 4 else val)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key))

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key).reshape(-1, config.vision_config.hidden_size))

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}")
    encoderblock_layernorm0_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}")
    encoderblock_layernorm1_scale = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}")
    encoderblock_layernorm1_bias = state_dict.pop(f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}")

    encoderblock_mlp_dense0_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}")
    encoderblock_mlp_dense0_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}")
    encoderblock_mlp_dense1_kernel = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}")
    encoderblock_mlp_dense1_bias = state_dict.pop(f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}")

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = convert_jax_array_to_torch(encoderblock_layernorm0_scale[i].transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = convert_jax_array_to_torch(encoderblock_layernorm0_bias[i])
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = convert_jax_array_to_torch(encoderblock_layernorm1_scale[i].transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = convert_jax_array_to_torch(encoderblock_layernorm1_bias[i])
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = convert_jax_array_to_torch(encoderblock_mlp_dense0_kernel[i].transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = convert_jax_array_to_torch(encoderblock_mlp_dense0_bias[i])
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = convert_jax_array_to_torch(encoderblock_mlp_dense1_kernel[i].transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = convert_jax_array_to_torch(encoderblock_mlp_dense1_bias[i])
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1))
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1))
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1))
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose())
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = convert_jax_array_to_torch(encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1))

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key).transpose())

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key))

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key).transpose())

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key))

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key))

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm/scale{suffix}")

    for i in range(config.text_config.num_hidden_layers):
        # Attention weights — dtype is detected from the jax array by convert_jax_array_to_torch
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"
        ] = convert_jax_array_to_torch(q_proj_weight_reshaped)

        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"
        ] = convert_jax_array_to_torch(llm_attention_kv_einsum[i, 0, 0].transpose())

        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"
        ] = convert_jax_array_to_torch(llm_attention_kv_einsum[i, 1, 0].transpose())

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"
        ] = convert_jax_array_to_torch(o_proj_weight_reshaped)

        # MLP weights
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"
        ] = convert_jax_array_to_torch(llm_mlp_gating_einsum[i, 0].transpose())

        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"
        ] = convert_jax_array_to_torch(llm_mlp_gating_einsum[i, 1].transpose())

        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"
        ] = convert_jax_array_to_torch(llm_mlp_linear[i].transpose())

        # Layer norm weights
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"
        ] = convert_jax_array_to_torch(llm_input_layernorm[i])

        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = convert_jax_array_to_torch(llm_post_attention_layernorm[i])

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = convert_jax_array_to_torch(state_dict.pop(jax_key))

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = convert_jax_array_to_torch(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = "/value" if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict else ""

    llm_attention_attn_vec_einsum = state_dict.pop(f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}")
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if pi05:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_post_attention_layernorm_bias = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}")
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}")
        llm_post_attention_layernorm = state_dict.pop(f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}")

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"] = (
            convert_jax_array_to_torch(q_proj_weight_reshaped)
        )

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"] = (
            convert_jax_array_to_torch(k_proj_weight_reshaped)
        )
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"] = (
            convert_jax_array_to_torch(v_proj_weight_reshaped)
        )

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"] = (
            convert_jax_array_to_torch(o_proj_weight_reshaped)
        )

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"] = (
            convert_jax_array_to_torch(gate_proj_weight.transpose())
        )
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"] = (
            convert_jax_array_to_torch(up_proj_weight.transpose())
        )
        state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"] = convert_jax_array_to_torch(
            llm_mlp_linear[i].transpose()
        )

        if pi05:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"] = (
                convert_jax_array_to_torch(llm_input_layernorm_bias[i])
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"] = (
                convert_jax_array_to_torch(llm_post_attention_layernorm_bias[i])
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"] = (
                convert_jax_array_to_torch(llm_input_layernorm_kernel[i].transpose())
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"] = (
                convert_jax_array_to_torch(llm_post_attention_layernorm_kernel[i].transpose())
            )
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"] = (
                convert_jax_array_to_torch(llm_input_layernorm[i])
            )
            state_dict[f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"] = (
                convert_jax_array_to_torch(llm_post_attention_layernorm[i])
            )

    # Handle final norm layer
    if pi05:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}")
        final_norm_kernel = state_dict.pop(f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}")
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = convert_jax_array_to_torch(final_norm_bias)
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = convert_jax_array_to_torch(final_norm_kernel.transpose())
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = convert_jax_array_to_torch(
            state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        final_state_dict[key] = convert_jax_array_to_torch(value)

    return final_state_dict


def slice_initial_orbax_checkpoint(checkpoint_dir: str, restore_precision: str | None = None, preserve_dtypes: bool = False):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    
    Args:
        restore_precision: If provided, convert all params to this dtype. If None and preserve_dtypes=True, keeps original dtypes.
        preserve_dtypes: If True, load as jax.Array to preserve original dtypes (bfloat16/float32) per-parameter.
    
    Returns:
        dict with "paligemma_params" and "projection_params"
    """
    if preserve_dtypes:
        # Load as jax.Array with dtype=None to preserve per-parameter dtypes exactly as stored.
        # JAX arrays carry their dtype (bfloat16, float32, etc.) through all operations
        # (transpose, reshape, indexing), so convert_jax_array_to_torch can read it at conversion time.
        params = openpi.models.model.restore_params(
            f"{checkpoint_dir}/params/", restore_type=jax.Array, dtype=None
        )
        print("Loaded JAX checkpoint with original per-parameter dtypes preserved.")
    else:
        # Normal loading - convert to specified precision or default
        dtype_to_restore = restore_precision if restore_precision else "float32"
        params = openpi.models.model.restore_params(
            f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=dtype_to_restore
        )

    return {"paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"), "projection_params": params}


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir) if not checkpoint_dir.startswith("gs://") else checkpoint_dir
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def load_and_convert_jax_norm_stats(checkpoint_dir: str) -> dict[str, dict[str, torch.Tensor]] | None:
    """
    Load normalization statistics from JAX checkpoint and convert to lerobot format.
    
    Args:
        checkpoint_dir: Path to JAX checkpoint directory
    
    Returns:
        Dictionary in lerobot format: {"observation.state": {"mean": ..., "std": ..., ...}, "action": {...}}
        Returns None if stats not found
    """
    # Exact path structure from JAX checkpoint
    norm_stats_path = os.path.join(checkpoint_dir, "assets", "physical-intelligence", "libero", "norm_stats.json")
    
    if not os.path.exists(norm_stats_path):
        print(f"Warning: norm_stats.json not found at {norm_stats_path}")
        return None
    
    print(f"Loading normalization stats from: {norm_stats_path}")
    
    # Load JAX normalization stats
    with open(norm_stats_path, 'r') as f:
        jax_data = json.load(f)
    
    # Extract norm_stats if nested
    if 'norm_stats' in jax_data:
        jax_stats = jax_data['norm_stats']
    else:
        jax_stats = jax_data
    
    # Convert to lerobot format
    lerobot_stats = {}
    
    # Convert state stats: "state" -> "observation.state"
    if "state" in jax_stats:
        state_stats = jax_stats["state"]
        lerobot_stats["observation.state"] = {
            "mean": torch.tensor(state_stats["mean"], dtype=torch.float32),
            "std": torch.tensor(state_stats["std"], dtype=torch.float32),
            "q01": torch.tensor(state_stats["q01"], dtype=torch.float32),
            "q99": torch.tensor(state_stats["q99"], dtype=torch.float32),
            "q10": torch.tensor(state_stats.get("q10", state_stats["q01"]), dtype=torch.float32),
            "q50": torch.tensor(state_stats.get("q50", state_stats["mean"]), dtype=torch.float32),
            "q90": torch.tensor(state_stats.get("q90", state_stats["q99"]), dtype=torch.float32),
            "min": torch.tensor(state_stats.get("min", state_stats["q01"]), dtype=torch.float32),
            "max": torch.tensor(state_stats.get("max", state_stats["q99"]), dtype=torch.float32),
            "count": torch.tensor([1.0], dtype=torch.float32),
        }
    
    # Convert action stats: "actions" -> "action"
    if "actions" in jax_stats:
        action_stats = jax_stats["actions"]
        lerobot_stats["action"] = {
            "mean": torch.tensor(action_stats["mean"], dtype=torch.float32),
            "std": torch.tensor(action_stats["std"], dtype=torch.float32),
            "q01": torch.tensor(action_stats["q01"], dtype=torch.float32),
            "q99": torch.tensor(action_stats["q99"], dtype=torch.float32),
            "q10": torch.tensor(action_stats.get("q10", action_stats["q01"]), dtype=torch.float32),
            "q50": torch.tensor(action_stats.get("q50", action_stats["mean"]), dtype=torch.float32),
            "q90": torch.tensor(action_stats.get("q90", action_stats["q99"]), dtype=torch.float32),
            "min": torch.tensor(action_stats.get("min", action_stats["q01"]), dtype=torch.float32),
            "max": torch.tensor(action_stats.get("max", action_stats["q99"]), dtype=torch.float32),
            "count": torch.tensor([1.0], dtype=torch.float32),
        }
    
    print(f"Converted normalization stats for {len(lerobot_stats)} features")
    return lerobot_stats


def convert_pi0_checkpoint(
    checkpoint_dir: str, precision: str, output_path: str, model_config: openpi.models.pi0_config.Pi0Config, preserve_jax_precision: bool = True
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16) - only used if preserve_jax_precision=False
        output_path: Path to save the converted PyTorch model
        model_config: Model config
        preserve_jax_precision: If True, preserve each parameter's dtype exactly as stored in the JAX checkpoint.
                                Each parameter's dtype is detected dynamically from the array itself.
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")
    print(f"Training config dtype: {model_config.dtype}")
    print(f"Preserve JAX precision: {preserve_jax_precision}")
    
    if preserve_jax_precision:
        print("Note: Each parameter's dtype will be detected directly from the JAX checkpoint.")
        print("      No assumptions are made — dtypes are read per-parameter from the arrays themselves.")
        print(f"\n⚠️  Training config dtype is '{model_config.dtype}', but checkpoint may have different dtypes.")
        print("      The conversion script will preserve whatever dtype is in the checkpoint.")
        print("      If you want to convert based on training config, set preserve_jax_precision=False.")

    # Load params — when preserve_jax_precision=True, arrays are jax.Array with original dtypes
    # This preserves what's actually in the checkpoint, not what the training config says
    initial_params = slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, 
        restore_precision=None if preserve_jax_precision else "float32",
        preserve_dtypes=preserve_jax_precision
    )

    # Process projection params — convert_jax_array_to_torch reads dtype from each array
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        # convert_jax_array_to_torch detects dtype (bfloat16/float32) from the array itself
        projection_params[f"{key}.weight"] = convert_jax_array_to_torch(weight).T
        projection_params[f"{key}.bias"] = convert_jax_array_to_torch(bias)

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # Process PaliGemma weights — dtypes are detected per-parameter from the jax arrays
    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params, action_expert_config, num_expert=1, checkpoint_dir=checkpoint_dir, pi05=model_config.pi05
    )

    # Instantiate model
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    # Combine all parameters — each tensor already has the correct dtype from JAX
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Print dtype summary before loading - this shows what we got from JAX
    dtype_counts = {}
    for name, tensor in all_params.items():
        dtype_str = str(tensor.dtype)
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
    print("Parameter dtype summary from JAX checkpoint (after conversion to PyTorch):")
    for dtype_str, count in sorted(dtype_counts.items()):
        print(f"  {dtype_str}: {count} parameters")
    
    # Verify: Check if we're preserving dtypes correctly
    # This should match what's in the JAX checkpoint
    if preserve_jax_precision:
        print("\n✓ Dtype preservation: Each parameter's dtype is preserved from JAX array")
        print("  (No assumptions - dtypes are read directly from each JAX array)")

    # Load state dict — this may cast to model's default dtype (float32)
    pi0_model.load_state_dict(all_params, strict=False)

    if preserve_jax_precision:
        # Reassign each parameter directly from all_params to preserve the per-parameter
        # dtypes that load_state_dict may have lost (it copies into existing float32 buffers).
        reassigned = 0
        for name, param in pi0_model.named_parameters():
            if name in all_params and param.dtype != all_params[name].dtype:
                param.data = all_params[name]
                reassigned += 1
        print(f"Reassigned {reassigned} parameters to restore original JAX dtypes.")
        
        # Print final dtype breakdown by component
        component_dtypes = {}
        for name, param in pi0_model.named_parameters():
            dtype_str = str(param.dtype)
            if "vision_tower" in name:
                component = "Vision Tower"
            elif "language_model" in name:
                component = "LLM Backbone"
            elif "gemma_expert" in name:
                component = "Action Expert"
            elif "multi_modal_projector" in name:
                component = "MM Projector"
            else:
                component = "Projections"
            key = (component, dtype_str)
            component_dtypes[key] = component_dtypes.get(key, 0) + 1
        
        print("Final dtype breakdown by component:")
        for (component, dtype_str), count in sorted(component_dtypes.items()):
            print(f"  {component:20s} | {dtype_str:15s} | {count} params")
        
        # VERIFICATION: Check actual dtypes right before saving
        print("\n=== VERIFICATION: Actual dtypes in model before save ===")
        llm_params = [(n, p) for n, p in pi0_model.named_parameters() if "language_model" in n]
        if llm_params:
            # Sample a few LLM parameters
            sample_dtypes = {}
            for name, param in llm_params[:5]:
                sample_dtypes[name] = str(param.dtype)
            print(f"Sample LLM parameter dtypes (first 5):")
            for name, dtype_str in sample_dtypes.items():
                print(f"  {name}: {dtype_str}")
            
            # Count by dtype for all LLM params
            llm_dtype_counts = {}
            for _, param in llm_params:
                dtype_str = str(param.dtype)
                llm_dtype_counts[dtype_str] = llm_dtype_counts.get(dtype_str, 0) + 1
            print(f"LLM dtype distribution in MODEL (all {len(llm_params)} params): {llm_dtype_counts}")
        
        # Also check what all_params had for LLM (original JAX dtypes)
        llm_all_params = {n: p for n, p in all_params.items() if "language_model" in n}
        if llm_all_params:
            llm_all_dtype_counts = {}
            for name, tensor in llm_all_params.items():
                dtype_str = str(tensor.dtype)
                llm_all_dtype_counts[dtype_str] = llm_all_dtype_counts.get(dtype_str, 0) + 1
            print(f"LLM dtype distribution in all_params (original JAX, before load_state_dict): {llm_all_dtype_counts}")
        
        # Check what will actually be saved (by getting state_dict from model)
        model_state_dict = pi0_model.state_dict()
        llm_state_dict = {n: t for n, t in model_state_dict.items() if "language_model" in n}
        if llm_state_dict:
            llm_saved_dtype_counts = {}
            for name, tensor in llm_state_dict.items():
                dtype_str = str(tensor.dtype)
                llm_saved_dtype_counts[dtype_str] = llm_saved_dtype_counts.get(dtype_str, 0) + 1
            print(f"LLM dtype distribution in model.state_dict() (what will be SAVED): {llm_saved_dtype_counts}")
    else:
        # Original behavior: convert entire model to single precision
        if precision == "float32":
            pi0_model = pi0_model.to(torch.float32)
        elif precision == "bfloat16":
            pi0_model = pi0_model.to(torch.bfloat16)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(pi0_model, os.path.join(output_path, "model.safetensors"))

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON - this will be loaded by PI05Config.from_pretrained
    # Save in the format expected by PI05Config
    config_dict = {
        "type": "pi05",
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "dtype": "bfloat16" if preserve_jax_precision else precision,
        "preserve_jax_precision": preserve_jax_precision,  # This flag tells PyTorch to skip its own precision conversion
        "chunk_size": model_config.action_horizon,
        "n_action_steps": model_config.action_horizon,
        "max_action_dim": model_config.action_dim,
        "max_state_dim": model_config.action_dim,  # Use same as action_dim for pi05
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",  # Pi0.5 uses quantiles for state
            "ACTION": "QUANTILES",  # Pi0.5 uses quantiles for action
        },
    }
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create and save processor pipelines for lerobot compatibility
    print("Creating processor pipelines...")
    try:
        # Import lerobot modules - these should be available if lerobot is installed
        import sys
        # Try to add the lerobot path if it's not in sys.path
        # Assuming the Open-PiMLE directory is at the same level or accessible
        possible_lerobot_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Open-PiMLE", "src"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Open-PiMLE", "src"),
        ]
        for lerobot_path in possible_lerobot_paths:
            if os.path.exists(lerobot_path) and lerobot_path not in sys.path:
                sys.path.insert(0, lerobot_path)
        
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
        import draccus
        
        # Create PI05Config from the config dict we just saved
        # Remove preserve_jax_precision and type before parsing (they're not part of PI05Config)
        config_for_parsing = config_dict.copy()
        config_for_parsing.pop("preserve_jax_precision", None)
        config_for_parsing.pop("type", None)
        
        # Parse config using draccus
        with draccus.config_type("json"):
            # Create a temporary config file for parsing
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump(config_for_parsing, f, indent=2)
                temp_config_path = f.name
            
            try:
                policy_config = draccus.parse(PI05Config, temp_config_path, args=[])
            finally:
                # Clean up temp file
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        
        # Validate and set up features (required for processors)
        policy_config.validate_features()
        
        # Load and convert JAX normalization statistics
        print("Loading normalization statistics from JAX checkpoint...")
        dataset_stats = load_and_convert_jax_norm_stats(checkpoint_dir)
        
        if dataset_stats:
            print(f"Successfully loaded normalization stats for {len(dataset_stats)} features")
        else:
            print("Warning: Could not load normalization stats from JAX checkpoint")
            print("  Processors will be created without normalization statistics")
        
        # Create processors with JAX normalization statistics
        preprocessor, postprocessor = make_pi05_pre_post_processors(
            config=policy_config,
            dataset_stats=dataset_stats,  # Use JAX normalization stats
        )
        
        # Save processors to output directory
        print(f"Saving preprocessor to {output_path}...")
        preprocessor.save_pretrained(output_path)
        
        print(f"Saving postprocessor to {output_path}...")
        postprocessor.save_pretrained(output_path)
        
        print("Processor pipelines created and saved successfully!")
    except ImportError as e:
        print(f"Warning: Could not import lerobot modules to create processors: {e}")
        print("You may need to run the migration script to create processor files:")
        print(f"  python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {output_path}")
    except Exception as e:
        print(f"Warning: Could not create processor pipelines: {e}")
        import traceback
        traceback.print_exc()
        print("You may need to run the migration script to create processor files:")
        print(f"  python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {output_path}")

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
    preserve_jax_precision: bool = True,
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion
        inspect_only: Only inspect parameter keys, don't convert
    """
    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print("Error: --output_path is required for conversion. Use --inspect_only to only view keys.")
            return
        convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config, preserve_jax_precision=preserve_jax_precision)


if __name__ == "__main__":
    tyro.cli(main)