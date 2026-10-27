#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05_imle.pi05_imle_infer import Pi05InferenceIMLE as Pi05Inference
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


class PI05IMLEPolicy(PreTrainedPolicy):
    """PI05 Policy using Triton inference for IMLE (Implicit Maximum Likelihood Estimation).
    
    This policy wraps the Triton-optimized Pi05Inference model for IMLE mode (single forward pass)
    and uses the same processor pipeline as PI05Policy, ensuring consistency with the PyTorch implementation.
    """

    config_class = PI05Config
    name = "pi05_imle_triton"

    def __init__(
        self,
        config: PI05Config,
        triton_checkpoint_path: str | Path,
        jax_checkpoint_dir: str | Path | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance.
            triton_checkpoint_path: Path to the .pkl Triton checkpoint file.
            jax_checkpoint_dir: Path to original JAX checkpoint (for norm_stats if needed).
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Load Triton model
        triton_checkpoint_path = Path(triton_checkpoint_path)
        if not triton_checkpoint_path.exists():
            raise FileNotFoundError(f"Triton checkpoint not found: {triton_checkpoint_path}")
        
        logging.info(f"Loading Triton model from {triton_checkpoint_path}...")
        with open(triton_checkpoint_path, "rb") as f:
            weights = pickle.load(f)

        # Determine num_views from config or environment
        # LIBERO uses 2 views, others use 3
        # Check if config has num_views attribute, otherwise infer from environment
        if hasattr(config, 'num_views'):
            num_views = config.num_views
        else:
            # Try to infer from image features
            num_views = len(config.image_features) if config.image_features else 2
            # LIBERO typically has 2 views
            if num_views == 0:
                num_views = 2  # Default for LIBERO

        # Determine if IMLE mode
        is_imle = getattr(config, 'use_imle', False)

        # Create Triton inference model
        self.triton_model = Pi05Inference(
            checkpoint=weights,
            num_views=num_views,
            chunk_size=config.chunk_size,
            tokenizer_path="google/paligemma-3b-pt-224",
            max_tokenize_len=config.tokenizer_max_length,
            discrete_state_input=True,  # pi05 always uses discrete state
            is_imle=is_imle,
        )

        logging.info(
            f"Triton model loaded: num_views={num_views}, chunk_size={config.chunk_size}, "
            f"is_imle={is_imle}"
        )

        self.reset()

    def reset(self):
        """Reset policy state (action queue)."""
        self._action_queue = deque()

    def sample_noise(self, shape, device):
        """Sample noise for diffusion/IMLE (same as PyTorch).
        
        Args:
            shape: Tuple of tensor dimensions (batch_size, chunk_size, max_action_dim)
            device: Device to place the noise tensor on
            
        Returns:
            Noise tensor with shape `shape`, dtype float32, on `device`.
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def _preprocess_images(self, batch: dict[str, Tensor]) -> torch.Tensor:
        """Preprocess images for Triton model using the exact same logic as PyTorch model.
        
        Uses the same preprocessing steps as PI05Policy._preprocess_images, then converts
        the output from [B, C, H, W] format to (num_views, H, W, C) format for Triton.
        
        Args:
            batch: Batch dictionary with image keys from config.image_features.
            
        Returns:
            Stacked images tensor of shape (num_views, H, W, C) in [-1, 1] range.
        """
        # Use exact same preprocessing logic as PyTorch model
        images = []
        img_masks = []

        # Get device (Triton always on CUDA)
        device = "cuda"

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch (exact same as PyTorch)
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

        # Convert from PyTorch format [B, C, H, W] to Triton format (num_views, H, W, C)
        # Take first batch item and convert to channels-last
        observation_images_list = []
        for img in images:
            # img is [B, C, H, W], take first batch item: [C, H, W]
            img_single = img[0]  # (C, H, W)
            # Convert to channels-last: (H, W, C)
            img_single = img_single.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            observation_images_list.append(img_single)

        # Stack images: (num_views, H, W, C)
        observation_images = torch.stack(observation_images_list, dim=0)

        return observation_images

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[dict[str, Any]]) -> Tensor:
        """Predict a chunk of actions using Triton inference.
        
        Args:
            batch: Batch dictionary with:
                - Image keys from config.image_features (format: (B, C, H, W) or (B, H, W, C))
                - "observation.language.tokens": Pre-tokenized token IDs (B, seq_len) - same as PyTorch
                - "observation.language.attention_mask": Attention mask (B, seq_len) - same as PyTorch
            **kwargs: Additional kwargs:
                - "noise": Optional noise tensor (B, chunk_size, action_dim)
                - "num_steps": Deprecated - IMLE always uses 1 step, kept for API compatibility
                
        Returns:
            Actions tensor of shape (B, chunk_size, action_dim).
        """
        self.eval()

        # Preprocess images: (num_views, H, W, C)
        observation_images = self._preprocess_images(batch)

        # Extract pre-tokenized tokens from batch (same as PyTorch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]  # (B, seq_len) - already tokenized by processor
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]  # (B, seq_len)

        # Check batch size - Triton only supports single batch
        bsize = tokens.shape[0]
        device = tokens.device  # Get device before indexing
        if bsize > 1:
            raise ValueError(
                f"Triton inference only supports batch_size=1, but got batch_size={bsize}. "
                f"Please process samples one at a time or use PyTorch model for batch processing."
            )

        # Take first batch item for Triton (processes one at a time)
        tokens = tokens[0]  # (seq_len,)
        masks = masks[0]  # (seq_len,)

        # Generate noise if not provided (same as PyTorch)
        if "noise" in kwargs:
            noise = kwargs["noise"]
            if noise.ndim == 2:
                noise = noise[None, ...]  # Add batch dim
        else:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)


        noise_batch = noise[0]  # (chunk_size, action_dim), float32

        # IMLE always uses num_steps=1 (single forward pass, direct action prediction)
        num_steps = kwargs.get("num_steps", 1)
        if num_steps != 1:
            raise ValueError(f"IMLE only supports num_steps=1, got num_steps={num_steps}")

        # Call Triton forward with pre-tokenized tokens (same as PyTorch)
        output_actions = self.triton_model.forward(
            observation_images_normalized=observation_images,
            diffusion_noise=noise_batch,
            language_tokens=tokens,  # Pre-tokenized token IDs
            language_attention_mask=masks,  # Attention mask
            num_steps=1,  # IMLE always uses 1 step
        )

        # Convert output to float32 and add batch dimension
        output_actions = output_actions.to(dtype=torch.float32)
        if output_actions.ndim == 2:  # (action_horizon, action_dim)
            output_actions = output_actions.unsqueeze(0)  # (1, action_horizon, action_dim)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        output_actions = output_actions[:, :, :original_action_dim]

        return output_actions

    def get_optim_params(self) -> dict:
        """Returns optimizer parameters.
        
        Note: Triton models are inference-only and do not support training.
        Returns an empty list since there are no trainable parameters.
        """
        return []

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.
        
        Note: Triton models are inference-only. This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "PI05IMLEPolicy is inference-only. Use predict_action_chunk() or select_action() for inference."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PI05Config | None = None,
        triton_checkpoint_path: str | Path | None = None,
        jax_checkpoint_dir: str | Path | None = None,
        **kwargs,
    ) -> "PI05IMLEPolicy":
        """Load a pretrained Triton IMLE policy.
        
        Args:
            pretrained_name_or_path: Path to .pkl file or directory containing .pkl file.
            config: Optional config. If None, will try to load from pretrained path.
            triton_checkpoint_path: Path to .pkl file. If None, pretrained_name_or_path must be a .pkl file.
            jax_checkpoint_dir: Path to original JAX checkpoint (for norm_stats if needed).
            **kwargs: Additional kwargs passed to config.from_pretrained().
            
        Returns:
            PI05IMLEPolicy instance.
        """
        pretrained_path = Path(pretrained_name_or_path)
        
        if triton_checkpoint_path is None:
            # Check if pretrained_name_or_path is a .pkl file
            if pretrained_path.suffix == ".pkl":
                triton_checkpoint_path = pretrained_path
            else:
                # Try to find .pkl file in directory
                pkl_files = list(pretrained_path.glob("*.pkl"))
                if len(pkl_files) == 0:
                    raise ValueError(f"No .pkl file found in {pretrained_path}")
                elif len(pkl_files) > 1:
                    raise ValueError(
                        f"Multiple .pkl files found in {pretrained_path}: {pkl_files}. "
                        f"Please specify the exact .pkl file path."
                    )
                triton_checkpoint_path = pkl_files[0]
        else:
            triton_checkpoint_path = Path(triton_checkpoint_path)

        if config is None:
            config = PI05Config.from_pretrained(pretrained_name_or_path, **kwargs)

        return cls(
            config=config,
            triton_checkpoint_path=triton_checkpoint_path,
            jax_checkpoint_dir=jax_checkpoint_dir,
        )
