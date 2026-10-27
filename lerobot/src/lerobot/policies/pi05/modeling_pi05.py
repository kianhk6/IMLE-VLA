#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import builtins
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

# Conditional imports for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from peft.tuners.lora import LoraLayer
    _peft_available = True
except ImportError:
    _peft_available = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    LoraLayer = None


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _lm(language_model):
    """
    Language-model accessor that follows your rule:
    - If LoRA is not active, nothing is wrapped and we use the object directly.
    - If ONLY language is PEFT-wrapped, the underlying module is at `.model`.

    We detect this by checking where `.layers` lives.
    """
    if hasattr(language_model, "layers"):
        return language_model
    if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
        return language_model.model
    return language_model


def _is_peft_wrapped(obj: object) -> bool:
    """
    True only for PEFT wrappers.

    Important: many HF models have a `.base_model` attribute even when NOT wrapped,
    so we must only treat it as PEFT when the chain `.base_model.model` exists.
    """
    if not _peft_available:
        return False
    try:
        from peft import PeftModel  # type: ignore
    except Exception:
        return False
    return isinstance(obj, PeftModel)


def _filter_lora_target_modules(requested: list[str], available: set[str]) -> list[str]:
    """Filter LoRA target module names to those present, preserving order and uniqueness."""
    seen: set[str] = set()
    filtered: list[str] = []
    for name in requested:
        if name in available and name not in seen:
            filtered.append(name)
            seen.add(name)
    return filtered


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert, use_lora: bool
):
    # Strictly flag-driven access (per your request):
    # - If LoRA is OFF: access exactly like the reference code (no probing / no unwrap checks).
    # - If LoRA is ON: support PEFT-wrapped paligemma and language-only LoRA.
    if not use_lora:
        paligemma_lm = paligemma.language_model
        paligemma_model_lm = paligemma.model.language_model
    else:
        if _is_peft_wrapped(paligemma):
            paligemma_base = paligemma.base_model.model
        else:
            paligemma_base = paligemma
        paligemma_lm = paligemma_base.language_model
        paligemma_model_lm = paligemma_base.model.language_model
        # If language_model is PEFT-wrapped (language-only LoRA), use `.model.*`
        paligemma_lm = _lm(paligemma_lm)
        paligemma_model_lm = _lm(paligemma_model_lm)
    
    # gemma_expert is not PEFT-wrapped, use original access pattern
    models = [paligemma_lm, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma_model_lm.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma_lm.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_lm.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma_lm.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        pi05_config: PI05Config | None = None,
        defer_lora: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.pi05_config = pi05_config
        self.use_lora = pi05_config.use_lora if pi05_config else False
        self._lora_applied = False

        # Convert dtypes
        self.to_bfloat16_for_selected_params(precision)

        # Apply LoRA if requested (unless deferred for loading pretrained weights first)
        if self.use_lora and not defer_lora:
            if _peft_available:
                self._apply_lora()
            else:
                raise ImportError(
                    "LoRA requires the 'peft' library. Install it with: pip install peft"
                )
        
        

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _get_available_modules(self) -> dict[str, set[str]]:
        """Get available linear module types by component."""
        available = {"vision_tower": set(), "language_model": set(), "expert": set()}
        
        for name, module in self.paligemma.named_modules():
            if isinstance(module, nn.Linear):
                module_type = name.split(".")[-1] if name else ""
                if "vision_tower" in name or "vision_model" in name:
                    available["vision_tower"].add(module_type)
                elif "language_model" in name:
                    available["language_model"].add(module_type)
        
        for name, module in self.gemma_expert.named_modules():
            if isinstance(module, nn.Linear):
                module_type = name.split(".")[-1] if name else ""
                available["expert"].add(module_type)
        
        return available

    def _validate_lora_targets(self, target_modules: list[str]) -> None:
        """Validate target modules exist in VLM and log available modules."""
        available = self._get_available_modules()
        # If LoRA is language-only, only validate against language model modules.
        if getattr(self.pi05_config, "lora_language_model_only", False):
            vlm_available = available["language_model"]
        else:
            vlm_available = available["vision_tower"] | available["language_model"]
        target_set = set(target_modules)
        
        # Log available modules
        logging.info("Available module types for LoRA:")
        logging.info(f"  Vision Tower: {sorted(available['vision_tower'])}")
        logging.info(f"  Language Model: {sorted(available['language_model'])}")
        logging.info(f"  Expert (no LoRA): {sorted(available['expert'])}")
        
        # Check for missing targets
        missing = target_set - vlm_available
        if missing:
            raise ValueError(
                f"Target modules not found in VLM: {sorted(missing)}\n"
                f"Available in VLM: {sorted(vlm_available)}"
            )

    def _apply_lora(self):
        """Apply LoRA to the VLM only. Expert remains fully trainable."""
        if self._lora_applied:
            logging.warning("LoRA already applied, skipping")
            return
            
        if not _peft_available:
            raise ImportError("LoRA requires 'peft' library. Install with: pip install peft")

        config = self.pi05_config
        # In language-only LoRA mode, target modules must be restricted to what exists in the LM.
        # This avoids failures like requesting `out_proj` when the LM uses `o_proj` naming.
        if getattr(config, "lora_language_model_only", False):
            available = self._get_available_modules()
            lm_available = available["language_model"]
            filtered_targets = _filter_lora_target_modules(config.lora_target_modules, lm_available)
            removed = [m for m in config.lora_target_modules if m not in set(filtered_targets)]
            if removed:
                logging.info(
                    "LoRA (language-only): filtering target modules not present in language_model: "
                    f"{sorted(set(removed))}"
                )
            if not filtered_targets:
                raise ValueError(
                    "LoRA (language-only): after filtering, no valid target modules remain. "
                    f"Requested={config.lora_target_modules} Available={sorted(lm_available)}"
                )
            target_modules = filtered_targets
            # Validate against LM-only available modules (should now always pass).
            self._validate_lora_targets(target_modules)
        else:
            target_modules = config.lora_target_modules
            self._validate_lora_targets(target_modules)

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        if getattr(config, "lora_language_model_only", False):
            # Language-only LoRA: keep vision tower vanilla/unwrapped.
            logging.info(
                "Applying LoRA to VLM language model only "
                f"(r={config.lora_r}, alpha={config.lora_alpha})"
            )
            # Strict safety rule:
            # Freeze *all* PaliGemma params first (vision tower + multimodal projector + LM base weights).
            # Then unfreeze only LoRA adapter params in the LM PEFT wrapper.
            #
            # This prevents accidental training of non-LoRA params if HF/PEFT changes defaults
            # or adds/renames components.
            for p in self.paligemma.parameters():
                p.requires_grad_(False)

            lm = get_peft_model(self.paligemma.language_model, lora_config)
            self.paligemma.language_model = lm
            # Keep HF's nested reference in sync for vanilla access patterns:
            # `paligemma.model.language_model.*`
            if hasattr(self.paligemma, "model") and hasattr(self.paligemma.model, "language_model"):
                self.paligemma.model.language_model = lm

            # # Freeze everything again (covers any newly-created adapter params),
            # # then unfreeze only LoRA adapter params inside the LM wrapper.
            # for n, p in self.paligemma.named_parameters():
            #     p.requires_grad_("lora" in n.lower())

            # Vision tower must remain vanilla/frozen in language-only LoRA mode.
            if hasattr(self.paligemma, "vision_tower"):
                for p in self.paligemma.vision_tower.parameters():
                    p.requires_grad_(False)
            if hasattr(self.paligemma, "model") and hasattr(self.paligemma.model, "vision_tower"):
                for p in self.paligemma.model.vision_tower.parameters():
                    p.requires_grad_(False)
        else:
            logging.info(f"Applying LoRA to full VLM (r={config.lora_r}, alpha={config.lora_alpha})")
            self.paligemma = get_peft_model(self.paligemma, lora_config)

        self._lora_applied = True
        
        # Verify and print summary
        self._print_lora_summary()
    
    def apply_lora_if_needed(self):
        """Apply LoRA if it was deferred during init. Called after loading pretrained weights."""
        if self.use_lora and not self._lora_applied:
            print("✓ Applying LoRA adapters to pretrained model...")
            self._apply_lora()

    def _print_lora_summary(self):
        """Verify LoRA application and print summary."""
        # Count LoRA modules by component
        vision_count = 0
        language_count = 0
        
        for name, module in self.paligemma.named_modules():
            if _peft_available and isinstance(module, LoraLayer):
                if "vision_tower" in name or "vision_model" in name:
                    vision_count += 1
                elif "language_model" in name:
                    language_count += 1
        
        total_lora = vision_count + language_count
        if total_lora == 0:
            raise RuntimeError("No LoRA modules applied to VLM!")
        
        # Count parameters
        vlm_total = sum(p.numel() for p in self.paligemma.parameters())
        vlm_trainable = sum(p.numel() for p in self.paligemma.parameters() if p.requires_grad)
        expert_total = sum(p.numel() for p in self.gemma_expert.parameters())
        expert_trainable = sum(p.numel() for p in self.gemma_expert.parameters() if p.requires_grad)

        logging.info(f"✓ LoRA applied: {vision_count} vision, {language_count} language modules")
        logging.info(f"  VLM: {vlm_trainable:,}/{vlm_total:,} trainable ({vlm_trainable/vlm_total*100:.1f}%)")
        logging.info(f"  Expert (fully trainable): {expert_trainable:,}/{expert_total:,} ({expert_trainable/expert_total*100:.1f}%)")

    def _get_paligemma_base(self):
        """Get base paligemma model, unwrapping PEFT wrapper if present."""
        # When LoRA is disabled, do not check/assume any wrapping.
        if not getattr(self, "use_lora", False):
            return self.paligemma
        if _is_peft_wrapped(self.paligemma):
            return self.paligemma.base_model.model
        return self.paligemma


    def embed_image(self, image: torch.Tensor):
        paligemma_base = self._get_paligemma_base()
        return paligemma_base.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        paligemma_base = self._get_paligemma_base()
        return _lm(paligemma_base.language_model).embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        
        # Get base models (unwrapped from PEFT if needed for VLM only)
        paligemma_base = self._get_paligemma_base()
        # Expert has no PEFT wrapper, use directly
        gemma_expert = self.gemma_expert
        
        if inputs_embeds[1] is None:
            prefix_output = _lm(paligemma_base.language_model).forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [_lm(paligemma_base.language_model), gemma_expert.model]
            num_layers = paligemma_base.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(gemma_expert.model, "gradient_checkpointing")
                and gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        use_lora=self.use_lora,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                        use_lora=self.use_lora,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """Update EMA parameters in place with ema_param <- alpha * ema_param + (1 - alpha) * param.
    
    Supports LoRA models where EMA may have merged weights (different structure than training model).
    Uses name-based matching to handle structural differences from PEFT wrappers.
    """
    # Check if structures match (same number of modules) - use fast path if so
    ema_modules = list(ema_net.modules())
    net_modules = list(net.modules())
    
    if len(ema_modules) == len(net_modules):
        # Fast path: structures match, use original strict matching
        for ema_module, module in zip(ema_modules, net_modules):
            for (n_p_ema, p_ema), (n_p, p) in zip(
                ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False), strict=True
            ):
                assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
                if isinstance(p, dict):
                    raise RuntimeError("Dict parameter not supported")
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                    # Copy BatchNorm parameters, and non-trainable parameters directly.
                    p_ema.copy_(p.to(dtype=p_ema.dtype).data)
                else:
                    with torch.no_grad():
                        p_ema.mul_(alpha)
                        p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)
    else:
        # Slow path: structures differ (LoRA with merged EMA), use name-based matching
        # Build a mapping from normalized parameter names to training model parameters
        # This handles PEFT wrapper name differences (e.g., "base_model.model.X" vs "X")
        
        def normalize_param_name(name: str) -> str:
            """Remove PEFT wrapper prefixes from parameter names."""
            # PEFT wraps models with prefixes like "base_model.model."
            prefixes_to_strip = ["base_model.model.", "base_layer."]
            for prefix in prefixes_to_strip:
                if prefix in name:
                    name = name.replace(prefix, "")
            return name
        
        def is_lora_param(name: str) -> bool:
            """Check if parameter is a LoRA adapter parameter."""
            lora_keywords = ["lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B", "lora_magnitude_vector"]
            return any(kw in name for kw in lora_keywords)
        
        # Get training model parameters (excluding LoRA adapters)
        net_params = {}
        for name, param in net.named_parameters():
            if not is_lora_param(name):
                normalized_name = normalize_param_name(name)
                net_params[normalized_name] = (param, param.requires_grad)
        
        # Update EMA parameters using name-based matching
        updated_count = 0
        for ema_name, ema_param in ema_net.named_parameters():
            normalized_ema_name = normalize_param_name(ema_name)
            
            if normalized_ema_name in net_params:
                param, requires_grad = net_params[normalized_ema_name]
                
                if not requires_grad:
                    # Copy non-trainable parameters directly
                    ema_param.data.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    with torch.no_grad():
                        ema_param.mul_(alpha)
                        ema_param.add_(param.to(dtype=ema_param.dtype).data, alpha=1 - alpha)
                updated_count += 1
        
        # Log if we're missing a significant number of parameters
        ema_param_count = sum(1 for _ in ema_net.parameters())
        if updated_count < ema_param_count * 0.9:  # Less than 90% matched
            import logging
            logging.warning(
                f"EMA update: only matched {updated_count}/{ema_param_count} parameters. "
                "This may indicate a model structure mismatch."
            )


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`

    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None, defer_lora: bool = False):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            pi05_config=config,
            defer_lora=defer_lora,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        # print("HERE!!!")
        # exit(0)
        # import transformers
        # assert(transformers.__version__ == "4.53.2" or transformers.__version__ == "4.53.3"), msg
        try:
            from transformers.models.siglip import check  # pyright: ignore[reportMissingImports]

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError as e:
            print("IMPORT ERROR:", e)
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        paligemma_base, expert_base = self._get_base_models()
        # If language-only LoRA, language_model may be a PEFT wrapper; flag is on the underlying `.model`.
        if hasattr(paligemma_base, "language_model"):
            if hasattr(paligemma_base.language_model, "gradient_checkpointing"):
                paligemma_base.language_model.gradient_checkpointing = True
            if hasattr(paligemma_base.language_model, "model") and hasattr(
                paligemma_base.language_model.model, "gradient_checkpointing"
            ):
                paligemma_base.language_model.model.gradient_checkpointing = True
        paligemma_base.vision_tower.gradient_checkpointing = True
        expert_base.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        paligemma_base, expert_base = self._get_base_models()
        if hasattr(paligemma_base, "language_model"):
            if hasattr(paligemma_base.language_model, "gradient_checkpointing"):
                paligemma_base.language_model.gradient_checkpointing = False
            if hasattr(paligemma_base.language_model, "model") and hasattr(
                paligemma_base.language_model.model, "gradient_checkpointing"
            ):
                paligemma_base.language_model.model.gradient_checkpointing = False
        paligemma_base.vision_tower.gradient_checkpointing = False
        expert_base.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _get_base_models(self):
        """Get base models, unwrapping PEFT wrappers if present (VLM only, Expert has no PEFT)."""
        paligemma = self.paligemma_with_expert.paligemma
        gemma_expert = self.paligemma_with_expert.gemma_expert
        
        # When LoRA is disabled, do not check/assume any wrapping.
        if not getattr(self.config, "use_lora", False):
            paligemma_base = paligemma
        else:
            # Unwrap PEFT if present (only VLM has LoRA/PEFT)
            if _is_peft_wrapped(paligemma):
                paligemma_base = paligemma.base_model.model
            else:
                paligemma_base = paligemma
            
        # Expert has no PEFT wrapper, use directly
        expert_base = gemma_expert
            
        return paligemma_base, expert_base

    def _get_paligemma_layers(self):
        """Get paligemma language model layers, handling PEFT wrapper."""
        paligemma_base, _ = self._get_base_models()
        return _lm(paligemma_base.language_model).layers

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        # Gradient checkpointing disabled - always call function directly
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    
    # Kian: where noise is sampled in training
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss.
        
        Dispatches to either flow matching or IMLE based on config.use_imle.
        """
        if self.config.use_imle:
            return self._forward_imle(images, img_masks, tokens, masks, actions, noise)
        else:
            return self._forward_flow_matching(images, img_masks, tokens, masks, actions, noise, time)

    def _forward_flow_matching(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Original flow matching forward pass."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self._get_paligemma_layers()[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def _forward_imle(self, images, img_masks, tokens, masks, actions, noise=None) -> Tensor:
        """IMLE (Implicit Maximum Likelihood Estimation) forward pass.
        
        Two-phase: no-grad for candidate selection, then grad for best sample only.
        """
        k = self.config.imle_k
        B = actions.shape[0]
        
        # Get original action dim from config (before padding)
        # This is needed for correct NN selection - padded dims have random outputs
        if hasattr(self.config, 'output_features') and ACTION in self.config.output_features:
            original_action_dim = self.config.output_features[ACTION].shape[0]
        else:
            original_action_dim = self.config.max_action_dim
        if noise is None:
            noise_shape = (B * k, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(noise_shape, actions.device)
        
        # ===== PHASE 1: No-grad candidate generation and selection (eval mode) =====
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
            prefix_embs = prefix_embs.repeat_interleave(k, dim=0)
            prefix_pad_masks = prefix_pad_masks.repeat_interleave(k, dim=0)
            prefix_att_masks = prefix_att_masks.repeat_interleave(k, dim=0)

            time = torch.ones(B * k, dtype=torch.float32, device=actions.device)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(noise, time)

            if self._get_paligemma_layers()[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
            generated_actions = self.action_out_proj(suffix_out)
            
            gen = generated_actions.view(B, k, *generated_actions.shape[1:])
            target = actions.unsqueeze(1)
            
            # Truncate to original_action_dim for distance computation to match logged loss
            gen_truncated = gen[:, :, :, :original_action_dim]
            target_truncated = target[:, :, :, :original_action_dim]
            
            dists = ((gen_truncated - target_truncated)**2).mean(dim=(-1, -2))
            best_idx = dists.argmin(dim=1)
            
            # Compute stats inside no_grad
            dist_mean = dists.mean().item()
            best_dists_nograd = dists[torch.arange(B, device=actions.device), best_idx]
            best_dist_mean_nograd = best_dists_nograd.mean().item()

        # ===== PHASE 2: Fresh forward pass with selected noise (train mode) =====
        if was_training:
            self.train()
        
        noise_reshaped = noise.view(B, k, *noise.shape[1:])
        best_noise = noise_reshaped[torch.arange(B, device=actions.device), best_idx]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        time = torch.ones(B, dtype=torch.float32, device=actions.device)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(best_noise, time)

        if self._get_paligemma_layers()[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        _, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        best_gen = self.action_out_proj(suffix_out)

        # Compute distance of regenerated best samples (with grad)
        # Truncate to original_action_dim for distance computation to match logged loss
        best_gen_truncated = best_gen[:, :, :original_action_dim]
        actions_truncated = actions[:, :, :original_action_dim]
        best_dists_grad = ((best_gen_truncated - actions_truncated)**2).mean(dim=(-1, -2))
        best_dist_mean_grad = best_dists_grad.mean().item()

        self._imle_dist_stats = {
            "dist_mean": dist_mean,
            "best_dist_mean_nograd": best_dist_mean_nograd,
            "best_dist_mean_grad": best_dist_mean_grad,
        }

        loss = ((best_gen - actions)**2)
        return loss

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions_flow(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        paligemma_base, _ = self._get_base_models()
        _lm(paligemma_base.language_model).config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions_imle(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        paligemma_base, _ = self._get_base_models()
        _lm(paligemma_base.language_model).config._attn_implementation = "eager"  # noqa: SLF001

        with torch.no_grad():
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        # One-step generation (IMLE)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        expanded_time = time.expand(bsize)

        actions = self.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noise,
            timestep=expanded_time,
        )

        return actions

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Dispatch to either flow matching or IMLE sampling based on config.use_imle."""
        if self.config.use_imle:
            return self.sample_actions_imle(
                images,
                img_masks,
                tokens,
                masks,
                noise,
                num_steps,
                **kwargs,
            )
        else:
            return self.sample_actions_flow(
                images,
                img_masks,
                tokens,
                masks,
                noise,
                num_steps,
                **kwargs,
            )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        # TODO: timestep should always be 1, get the noise embedding here
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        _, gemma_expert = self._get_base_models()
        gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # this is the expert model, fine tuning it with IMLE objective
        # as we are only inputting suffix_embs -> it (should) generate action
        # after the VLM path with prefix embedding, the kv's are cached for this run
        # The past_key_values
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        defer_lora: bool = False,
    ):
        """
        Args:
            config: Policy configuration class instance.
            defer_lora: If True, defer LoRA application until after weights are loaded.
                       Used by from_pretrained() when loading non-LoRA checkpoints.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor, defer_lora=defer_lora)
        
        # Cache tokenizer for truncation detection during evaluation
        self._tokenizer_cache = None

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        # Initialize EMA model if enabled
        self.model_ema = None
        self._ema_step_counter = 0
        if config.use_ema:
            import copy
            import logging
            logging.info("Creating EMA model for stable inference")
            self.model_ema = copy.deepcopy(self.model)
            self.model_ema.eval()
            # Disable gradients for EMA model
            for param in self.model_ema.parameters():
                param.requires_grad = False
            logging.info(f"EMA model created with decay={config.ema_decay}, update_every={config.ema_update_every}")
            
            # Merge LoRA weights in EMA model if requested (for inference optimization)
            if config.merge_ema_lora_for_inference and config.use_lora:
                logging.info("Merging LoRA weights in EMA model for inference...")
                self._merge_ema_lora()
                logging.info("✓ LoRA weights merged in EMA model")

        self.reset()



    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # First, load the state dict to check if it contains LoRA keys
        print(f"Loading model from: {pretrained_name_or_path}")
        try:
            from transformers.utils import cached_file

            # Try safetensors first
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                use_auth_token=kwargs.get("use_auth_token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            from safetensors.torch import load_file

            original_state_dict = load_file(resolved_file)
            print("✓ Loaded state dict from model.safetensors")
        except Exception as e:
            print(f"Could not load state dict from remote files: {e}")
            print("Creating model without loading pretrained weights")
            model = cls(config, **kwargs)
            return model

        # Check if checkpoint contains LoRA keys (e.g., lora_A, lora_B, base_layer)
        checkpoint_has_lora = any("lora_" in key or "base_layer" in key for key in original_state_dict.keys())
        use_lora = config.use_lora if hasattr(config, 'use_lora') else False
        
        # Count LoRA keys in checkpoint for logging
        lora_key_count = sum(1 for k in original_state_dict.keys() if "lora_" in k or "base_layer" in k)
        if checkpoint_has_lora:
            print(f"✓ LoRA checkpoint detected: found {lora_key_count} LoRA-related keys in checkpoint")
        
        # Heuristic: "thin" LoRA checkpoints (adapters-only) typically won't contain any large base weights.
        has_base_weights = any(
            k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight") or "vision_tower" in k
            for k in original_state_dict.keys()
        )
        thin_lora_checkpoint = checkpoint_has_lora and not has_base_weights
        
        if thin_lora_checkpoint:
            print(f"✓ Thin LoRA checkpoint detected (adapters-only, no base weights)")
        
        # Determine if we need to defer LoRA application:
        # - If use_lora=True but checkpoint has no LoRA keys → defer LoRA, load base weights, then apply LoRA
        # - If use_lora=True and checkpoint has LoRA keys → load directly with LoRA (resuming training or eval)
        defer_lora = use_lora and not checkpoint_has_lora
        
        if defer_lora:
            print("✓ Checkpoint is non-LoRA, will apply LoRA adapters after loading pretrained weights")
        elif use_lora and checkpoint_has_lora:
            print("✓ Checkpoint contains LoRA weights, loading directly with LoRA structure")
        
        # Initialize model.
        # If this is a thin LoRA checkpoint, we must first load the base model weights from `config.pretrained_path`
        # (e.g. `lerobot/pi05_base`) and then apply the adapter checkpoint on top.
        if thin_lora_checkpoint and use_lora and getattr(config, "pretrained_path", None):
            base_source = str(config.pretrained_path)
            if str(pretrained_name_or_path) != base_source:
                print(f"✓ Thin LoRA checkpoint detected; loading base weights from: {base_source}")
                model = cls.from_pretrained(base_source, config=config, strict=False, **kwargs)
                # Base is loaded and LoRA is applied during the base load; no need to defer.
                defer_lora = False
            else:
                model = cls(config, defer_lora=defer_lora)
        else:
            # Standard path (full checkpoint or non-LoRA checkpoint).
            model = cls(config, defer_lora=defer_lora)

        # Now manually load and remap the state dict
        # Initialize these variables before try block so they're available in except/finally
        remapped_state_dict = {}
        
        try:
            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            # Skip EMA keys - they're loaded from separate file (model_ema.safetensors)
            remap_count = 0

            for key, value in fixed_state_dict.items():
                # Skip EMA keys - loaded separately from model_ema.safetensors
                if key.startswith("model_ema.") or key == "_ema_step_counter":
                    continue
                    
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            # Load the remapped state dict into the model
            # Use strict=False for LoRA checkpoints (embed_tokens tied to lm_head)
            # Also use strict=False when deferring LoRA since we're loading base weights
            load_strict = strict and not use_lora
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=load_strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("✓ All keys loaded successfully!")
            
            # Load EMA from separate file if use_ema=True
            if config.use_ema:
                import os
                import copy
                import json
                # Try to load from model_ema.safetensors (only exists in local checkpoint directories)
                # HuggingFace base checkpoints never have EMA - EMA is only created during training
                ema_loaded = False
                if os.path.isdir(pretrained_name_or_path):
                    ema_path = os.path.join(pretrained_name_or_path, "model_ema.safetensors")
                    if os.path.exists(ema_path):
                        try:
                            from safetensors.torch import load_model as load_model_safetensor
                            # Initialize model_ema if it doesn't exist yet
                            if model.model_ema is None:
                                model.model_ema = copy.deepcopy(model.model)
                                model.model_ema.eval()
                                for param in model.model_ema.parameters():
                                    param.requires_grad = False
                            # Use load_model to properly handle tied weights
                            load_model_safetensor(model.model_ema, ema_path, strict=False)
                            
                            # Load step counter from metadata file
                            metadata_path = os.path.join(pretrained_name_or_path, "ema_metadata.json")
                            if os.path.exists(metadata_path):
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                                    model._ema_step_counter = metadata.get("_ema_step_counter", 0)
                            else:
                                model._ema_step_counter = 0
                            
                            print(f"✓ Loaded EMA from model_ema.safetensors (step counter: {model._ema_step_counter})")
                            ema_loaded = True
                        except Exception as e:
                            print(f"Warning: Failed to load EMA file: {e}")
                
                # If EMA file doesn't exist or failed to load, initialize EMA from current model
                if not ema_loaded:
                    if model.model_ema is None:
                        print("✓ No EMA file found, initializing EMA from loaded model...")
                        model.model_ema = copy.deepcopy(model.model)
                        model.model_ema.eval()
                        for param in model.model_ema.parameters():
                            param.requires_grad = False
                        model._ema_step_counter = 0
                        print(f"✓ EMA initialized with decay={config.ema_decay}") 
                                           
        except Exception as e:
            print(f"Warning: Could not remap state dict keys: {e}")

        # Apply LoRA after loading pretrained weights (if it was deferred)
        if defer_lora:
            model.model.paligemma_with_expert.apply_lora_if_needed()

        if getattr(model.config, "use_lora", False):
            lora_counts = model._count_lora_layers()
            print(
                f"✓ LoRA layers detected in VLM: total={lora_counts['total']}, "
                f"vision={lora_counts['vision']}, language={lora_counts['language']}"
            )
            
            # Get detailed LoRA parameter statistics
            paligemma = model.model.paligemma_with_expert.paligemma
            paligemma_total = sum(p.numel() for p in paligemma.parameters())
            paligemma_trainable = sum(p.numel() for p in paligemma.parameters() if p.requires_grad)
            paligemma_lora_percent = (paligemma_trainable / paligemma_total * 100) if paligemma_total > 0 else 0.0
            
            print(
                f"✓ Paligemma LoRA statistics: "
                f"{paligemma_trainable:,}/{paligemma_total:,} trainable parameters "
                f"({paligemma_lora_percent:.2f}% of paligemma is LoRA)"
            )

        # Optional: merge LoRA adapters into base weights for inference.
        if getattr(model.config, "merge_lora_for_inference", False) and getattr(model.config, "use_lora", False):
            print("✓ Merging LoRA adapters into base weights for inference...")
            model.merge_lora_for_inference()
        
        with torch.no_grad():
                # Access embed_tokens + lm_head robustly across:
                # - vanilla (no LoRA)
                # - full VLM PEFT wrap
                # - language-only LoRA wrap (paligemma not wrapped, only language_model is)
                paligemma_root = model.model.paligemma_with_expert.paligemma
                if _is_peft_wrapped(paligemma_root):
                    paligemma_root = paligemma_root.base_model.model

                paligemma_core = paligemma_root.model if hasattr(paligemma_root, "model") else paligemma_root
                lm_mod = (
                    paligemma_core.language_model
                    if hasattr(paligemma_core, "language_model")
                    else paligemma_root.language_model
                )
                embed = _lm(lm_mod).embed_tokens
                lm_head = paligemma_root.lm_head
                
                assert embed.weight.shape == (lm_head.out_features, lm_head.in_features), \
                    f"shape mismatch: embed={embed.weight.shape}, lm_head=({lm_head.out_features},{lm_head.in_features})"

                # 2) weight tying
                with torch.no_grad():
                    embed.weight = lm_head.weight  # make sure they share the same memory

                # 3) verification
                same_tensor = (lm_head.weight.data_ptr() == embed.weight.data_ptr())
                print(f"✓ Verified weight tying between embed_tokens and lm_head: {same_tensor}"   )

        # Freeze VLM parameters if LoRA is not enabled
        # With LoRA, only base parameters are frozen, LoRA parameters remain trainable
        if not model.config.use_lora:
            for p in model.model.paligemma_with_expert.paligemma.parameters():
                p.requires_grad_(False)
            print("✓ Frozen all Vision Language Model (VLM) parameters (LoRA disabled)")
        else:
            # With LoRA, freeze base model parameters but keep LoRA parameters trainable
            # PEFT automatically handles this, but we ensure base params are frozen
            for name, p in model.model.paligemma_with_expert.paligemma.named_parameters():
                if "lora" not in name.lower():
                    p.requires_grad_(False)
            print("✓ Frozen base VLM parameters, LoRA parameters remain trainable")

        # Count and print VLM parameter statistics
        vlm_stats = model.get_vlm_parameter_count()
        vlm_trainable_pct = (vlm_stats['trainable'] / vlm_stats['total'] * 100) if vlm_stats['total'] > 0 else 0.0
        print(f"✓ VLM parameters: {vlm_stats['total']:,} total, {vlm_stats['trainable']:,} trainable ({vlm_trainable_pct:.2f}%), {vlm_stats['frozen']:,} frozen")
        print(f"    Vision tower: {vlm_stats['vision_tower']:,}, Language embeddings: {vlm_stats['language_embeddings']:,}")

        return model

    def _count_lora_layers(self) -> dict[str, int]:
        """Best-effort count of LoRA layers in the VLM, for quick sanity-check logging."""
        try:
            from peft.tuners.lora import LoraLayer  # type: ignore
        except Exception:
            return {"vision": 0, "language": 0, "total": 0}

        paligemma = self.model.paligemma_with_expert.paligemma
        vision = 0
        language = 0
        total = 0
        for name, module in paligemma.named_modules():
            if isinstance(module, LoraLayer):
                total += 1
                if "vision_tower" in name or "vision_model" in name:
                    vision += 1
                elif "language_model" in name:
                    language += 1
        return {"vision": vision, "language": language, "total": total}

    def _merge_ema_lora(self) -> None:
        """
        Merge PEFT/LoRA adapters into the EMA model's base weights for inference.
        This is called during EMA model initialization if merge_ema_lora_for_inference=True.
        """
        try:
            from peft import PeftModel  # type: ignore
        except Exception:
            import logging
            logging.warning("_merge_ema_lora skipped: 'peft' is not installed")
            return

        pwe = self.model_ema.paligemma_with_expert

        # Case 1: full VLM wrapped (pwe.paligemma is a PeftModel)
        if isinstance(pwe.paligemma, PeftModel):
            pwe.paligemma = pwe.paligemma.merge_and_unload()
            return

        # Case 2: language-only LoRA (pwe.paligemma.language_model is a PeftModel)
        if hasattr(pwe.paligemma, "language_model") and isinstance(pwe.paligemma.language_model, PeftModel):
            merged_lm = pwe.paligemma.language_model.merge_and_unload()
            pwe.paligemma.language_model = merged_lm
            if hasattr(pwe.paligemma, "model") and hasattr(pwe.paligemma.model, "language_model"):
                pwe.paligemma.model.language_model = merged_lm
            return

        import logging
        logging.warning("_merge_ema_lora: no PEFT-wrapped modules found in EMA model to merge")

    def merge_lora_for_inference(self) -> None:
        """
        Merge PEFT/LoRA adapters into the base model weights for inference.
        This is an in-place operation and should not be used if you plan to keep training/resuming with LoRA.
        """
        try:
            from peft import PeftModel  # type: ignore
        except Exception:
            print("merge_lora_for_inference skipped: 'peft' is not installed")
            return

        pwe = self.model.paligemma_with_expert

        # Case 1: full VLM wrapped (pwe.paligemma is a PeftModel)
        if isinstance(pwe.paligemma, PeftModel):
            pwe.paligemma = pwe.paligemma.merge_and_unload()
            return

        # Case 2: language-only LoRA (pwe.paligemma.language_model is a PeftModel)
        if hasattr(pwe.paligemma, "language_model") and isinstance(pwe.paligemma.language_model, PeftModel):
            merged_lm = pwe.paligemma.language_model.merge_and_unload()
            pwe.paligemma.language_model = merged_lm
            if hasattr(pwe.paligemma, "model") and hasattr(pwe.paligemma.model, "language_model"):
                pwe.paligemma.model.language_model = merged_lm
            return

        print("merge_lora_for_inference: no PEFT-wrapped modules found to merge")

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                _, gemma_expert_base = self.model._get_base_models()
                expert_uses_adarms = getattr(
                    gemma_expert_base.model.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                _, gemma_expert_base = self.model._get_base_models()
                expert_uses_adarms = getattr(
                    gemma_expert_base.model.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        not_frozen = [p for p in self.parameters() if p.requires_grad]
        return not_frozen

    def get_vlm_parameter_count(self) -> dict:
        """Get the parameter count for the VLM (paligemma) used for prefix generation."""
        paligemma = self.model.paligemma_with_expert.paligemma
        
        # Unwrap PEFT if present
        if not getattr(self.config, "use_lora", False):
            paligemma_base = paligemma
        else:
            if _is_peft_wrapped(paligemma):
                paligemma_base = paligemma.base_model.model
            else:
                paligemma_base = paligemma
        
        total_params = sum(p.numel() for p in paligemma.parameters())
        trainable_params = sum(p.numel() for p in paligemma.parameters() if p.requires_grad)
        vision_tower_params = sum(p.numel() for p in paligemma_base.vision_tower.parameters())
        language_emb_params = sum(p.numel() for p in _lm(paligemma_base.language_model).embed_tokens.parameters())
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "vision_tower": vision_tower_params,
            "language_embeddings": language_emb_params,
        }

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        # Noise-locked reactive chunking state
        self._cached_noise: Tensor | None = None
        self._nlr_chunk_step: int = 0
        # Rotating cycle state
        self._unused_noise_buffer: Tensor | None = None
        self._window_index: int = 0
        # Continuity-based candidate selection state
        # Store full previous chunk (or at least up to n_action_steps*2) to access second window
        self._previous_chunk_actions: Tensor | None = None

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def update_ema(self):
        """Update the EMA model parameters."""
        if not self.config.use_ema or self.model_ema is None:
            return
        
        self._ema_step_counter += 1
        
        # Only update if we've passed the initial delay and it's time to update
        if (self._ema_step_counter >= self.config.ema_update_after_step and 
            self._ema_step_counter % self.config.ema_update_every == 0):
            update_ema_parameters(self.model_ema, self.model, self.config.ema_decay)

    def state_dict(self, *args, **kwargs):
        """Override state_dict - EMA is saved separately via _save_pretrained."""
        # Don't include EMA keys - they're saved to a separate file (model_ema.safetensors)
        return super().state_dict(*args, **kwargs)
    
    def _save_pretrained(self, save_directory: Path) -> None:
        """Override to save EMA weights to a separate file."""
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from safetensors.torch import save_model as save_model_as_safetensor
        
        # Save config
        self.config._save_pretrained(save_directory)
        
        # Save main model (without EMA)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
        print(f"✓ Saved main model to {SAFETENSORS_SINGLE_FILE}")
        
        # Save EMA to separate file if enabled
        # Use save_model (not save_file) to properly handle tied weights (embed_tokens <-> lm_head)
        if self.config.use_ema and self.model_ema is not None:
            ema_file = save_directory / "model_ema.safetensors"
            save_model_as_safetensor(self.model_ema, str(ema_file))
            print(f"✓ Saved EMA weights to model_ema.safetensors (step counter: {self._ema_step_counter})")
            
            # Save step counter separately as JSON (safetensors only stores tensors)
            import json
            counter_file = save_directory / "ema_metadata.json"
            with open(counter_file, "w") as f:
                json.dump({"_ema_step_counter": self._ema_step_counter}, f)
            print(f"✓ Saved EMA metadata (step counter: {self._ema_step_counter})")

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict - EMA is loaded separately via from_pretrained."""
        # Filter out any EMA keys that might be in old checkpoints (backwards compatibility)
        regular_state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.startswith("model_ema.") and k != "_ema_step_counter"
        }
        
        # Always use strict=False internally since model_ema is a registered submodule
        # but its weights are loaded separately from model_ema.safetensors
        result = super().load_state_dict(regular_state_dict, strict=False)
        
        # Filter out expected missing/unexpected keys
        missing_keys = [k for k in result.missing_keys if not k.startswith("model_ema.")]
        unexpected_keys = [k for k in result.unexpected_keys if not k.startswith("model_ema.")]
        
        # If strict=True was requested, raise error for real issues (non-EMA keys)
        if strict and (missing_keys or unexpected_keys):
            from torch.nn.modules.module import _IncompatibleKeys
            raise RuntimeError(
                f"Error(s) in loading state_dict for {self.__class__.__name__}:\n"
                f"\tMissing key(s): {missing_keys}\n"
                f"\tUnexpected key(s): {unexpected_keys}"
            )
        
        from torch.nn.modules.module import _IncompatibleKeys
        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    def _rtc_enabled(self) -> bool:


        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            # Ensure tensor is on the same device as the model
            if img.device != device:
                img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            
            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
        
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Predict a chunk of actions given environment observations.

        Args:
            batch: Environment observation dict.
            noise: Optional pre-sampled noise tensor.  When provided it is
                forwarded to ``model.sample_actions`` so that the same noise
                can be reused across calls (noise-locked reactive chunking).
        """
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Decode and log the prompt tokens (always), and warn if truncated
        max_token_len = self.config.tokenizer_max_length
        try:
            # Lazy load tokenizer (cache it for future use)
            if self._tokenizer_cache is None:
                from transformers import AutoTokenizer
                self._tokenizer_cache = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

            for batch_idx in range(tokens.shape[0]):
                batch_tokens = tokens[batch_idx]
                batch_mask = masks[batch_idx]

                # Count actual tokens (non-padding tokens)
                actual_token_count = int(batch_mask.sum().item())

                # Decode the tokens that are in the batch (post-tokenization)
                actual_tokens = batch_tokens[batch_mask.bool()].cpu()
                decoded_prompt = self._tokenizer_cache.decode(actual_tokens, skip_special_tokens=False)

                # Try to reconstruct the original (pre-truncation) prompt from batch
                original_prompt = None
                original_token_count = actual_token_count

                if "task" in batch:
                    task_text = batch["task"]
                    if isinstance(task_text, (list, tuple)) and batch_idx < len(task_text):
                        task_str = task_text[batch_idx] if isinstance(task_text[batch_idx], str) else str(task_text[batch_idx])
                    elif isinstance(task_text, str):
                        task_str = task_text
                    else:
                        task_str = None

                    if task_str:
                        state_tensor = None
                        if "observation.state" in batch:
                            state_batch = batch["observation.state"]
                            if batch_idx < state_batch.shape[0]:
                                state_tensor = state_batch[batch_idx:batch_idx+1]

                        if state_tensor is not None:
                            cleaned_text = task_str.strip().replace("_", " ").replace("\n", " ")
                            state_padded = pad_vector(state_tensor, self.config.max_state_dim)[0]
                            state_np = state_padded.cpu().numpy()
                            discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                            state_str = " ".join(map(str, discretized_state))
                            original_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "

                            original_tokens = self._tokenizer_cache(
                                original_prompt,
                                return_tensors="pt",
                                truncation=False,
                                padding=False,
                            )["input_ids"][0]
                            original_token_count = len(original_tokens)

                # Determine if truncation occurred
                was_truncated = actual_token_count >= max_token_len and (
                    original_prompt is None or original_token_count > max_token_len
                )

                if was_truncated and original_prompt:
                    logging.warning(
                        f"[Evaluation] Prompt TRUNCATED (batch {batch_idx}): "
                        f"Token length ({original_token_count}) exceeds max length ({max_token_len}).\n"
                        f"  Before truncation ({original_token_count} tokens): {original_prompt!r}\n"
                        f"  After  truncation ({actual_token_count} tokens): {decoded_prompt!r}"
                    )
                elif was_truncated:
                    logging.warning(
                        f"[Evaluation] Prompt likely TRUNCATED (batch {batch_idx}): "
                        f"Token count ({actual_token_count}) equals max length ({max_token_len}). "
                        f"Prompt: {decoded_prompt!r}"
                    )
        except Exception as e:
            logging.warning(f"[Evaluation] Failed to decode prompt for logging: {e}")

        # Use EMA model for inference if available, otherwise use training model
        model = self.model_ema if (self.config.use_ema and self.model_ema is not None) else self.model
        
        # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
        actions = model.sample_actions(images, img_masks, tokens, masks, noise=noise, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss = losses.mean()

        loss_dict = {
            "loss": loss.item(),
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        # Include IMLE distance stats if IMLE is enabled
        if self.config.use_imle and hasattr(self.model, '_imle_dist_stats'):
            loss_dict.update(self.model._imle_dist_stats)

        return loss, loss_dict
