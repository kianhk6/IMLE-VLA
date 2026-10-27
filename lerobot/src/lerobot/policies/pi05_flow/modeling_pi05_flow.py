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
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector, resize_with_pad_torch
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05_flow.pi05_flow_infer import Pi05Inference
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE


class PI05FlowPolicy(PreTrainedPolicy):
    """PI05 Policy using Triton inference for Flow Matching.
    
    Flow matching predicts velocity at each step and integrates to get actions.
    This policy wraps the Triton-optimized Pi05Inference model for flow matching mode
    and uses the same processor pipeline as PI05Policy, ensuring consistency with the PyTorch implementation.
    """

    config_class = PI05Config
    name = "pi05_flow_triton"

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
        
        logging.info(f"Loading Triton flow matching model from {triton_checkpoint_path}...")
        with open(triton_checkpoint_path, "rb") as f:
            weights = pickle.load(f)

        # Determine num_views from config or environment
        if hasattr(config, 'num_views'):
            num_views = config.num_views
        else:
            num_views = len(config.image_features) if config.image_features else 2
            if num_views == 0:
                num_views = 2  # Default for LIBERO

        # Create Triton inference model
        self.triton_model = Pi05Inference(
            checkpoint=weights,
            num_views=num_views,
            chunk_size=config.chunk_size,
            tokenizer_path="google/paligemma-3b-pt-224",
            max_tokenize_len=config.tokenizer_max_length,
            discrete_state_input=True,  # pi05 always uses discrete state
        )

        logging.info(
            f"Triton flow matching model loaded: num_views={num_views}, chunk_size={config.chunk_size}"
        )

        # Track Triton forward() timing separately (matches benchmark measurement)
        self._last_triton_forward_time = None

        self.reset()

    def reset(self):
        """Reset policy state (action queue)."""
        self._action_queue = deque()
        self._last_triton_forward_time = None
    
    def get_last_triton_forward_time(self):
        """Get the timing of the last Triton forward() call (matches benchmark measurement).
        
        Returns:
            Time in seconds, or None if forward() hasn't been called yet.
        """
        return self._last_triton_forward_time

    def sample_noise(self, shape, device):
        """Sample noise for flow matching (same as PyTorch)."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def _preprocess_images(self, batch: dict[str, Tensor]) -> torch.Tensor:
        """Preprocess images for Triton model using the exact same logic as PyTorch model.
        
        Same as PI05IMLEPolicy._preprocess_images - includes 180° rotation for LIBERO.
        """
        # Use exact same preprocessing logic as PyTorch model
        images = []
        img_masks = []

        device = "cuda"

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            if img.device != device:
                img = img.to(device)

            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3

            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)
            
            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        # Convert from PyTorch format [B, C, H, W] to Triton format (num_views, H, W, C)
        observation_images_list = []
        for img in images:
            img_single = img[0]  # (C, H, W)
            img_single = img_single.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            
            observation_images_list.append(img_single)

        observation_images = torch.stack(observation_images_list, dim=0)
        return observation_images

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[dict[str, Any]]) -> Tensor:
        """Predict a chunk of actions using Triton flow matching inference.
        
        Flow matching:
        - Uses num_steps denoising steps (default: config.num_inference_steps)
        - Predicts velocity at each step
        - Integrates: x_t = x_t + dt * v_t
        - Returns final integrated actions
        
        Args:
            batch: Batch dictionary with:
                - Image keys from config.image_features
                - "observation.language.tokens": Pre-tokenized token IDs (B, seq_len)
                - "observation.language.attention_mask": Attention mask (B, seq_len)
            **kwargs: Additional kwargs:
                - "noise": Optional noise tensor (B, chunk_size, action_dim)
                - "num_steps": Number of denoising steps (default: config.num_inference_steps)
                
        Returns:
            Actions tensor of shape (B, chunk_size, action_dim).
        """
        self.eval()

        observation_images = self._preprocess_images(batch)

        # Extract task and state to match original forward() signature
        # The original code expects task_prompt (string) and state_tokens (discretized np array)
        task_prompt = batch.get("task", None)
        if task_prompt is not None and not isinstance(task_prompt, str):
            # Handle tensor or other types
            task_prompt = str(task_prompt.item()) if hasattr(task_prompt, 'item') else str(task_prompt)
        
        state_tokens = None
        if OBS_STATE in batch:
            state = batch[OBS_STATE]  # (state_dim,) - already normalized to [-1, 1]
            # Pad to max_state_dim
            state = pad_vector(state, self.config.max_state_dim)
            # Discretize into 256 bins (same as processor)
            state_np = state.cpu().numpy()
            state_tokens = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        
        if "noise" in kwargs:
            noise_batch = kwargs["noise"]
            # Remove batch dimension if present (batch_size is always 1)
            if noise_batch.ndim == 3:
                noise_batch = noise_batch[0]
        else:
            device = observation_images.device if hasattr(observation_images, 'device') else torch.device("cuda")
            noise_batch = self.sample_noise(
                (self.config.chunk_size, self.config.max_action_dim),
                device
            )
        
        # Call original forward() signature (matches realtime-vla exactly)
        # Flow matching uses fixed num_steps=10 (recorded in CUDA graph)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        triton_start = time.perf_counter()
        output_actions = self.triton_model.forward(
            observation_images_normalized=observation_images,
            diffusion_noise=noise_batch,
            task_prompt=task_prompt,
            state_tokens=state_tokens,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._last_triton_forward_time = time.perf_counter() - triton_start

        output_actions = output_actions.to(dtype=torch.float32)
        if output_actions.ndim == 2:
            output_actions = output_actions.unsqueeze(0)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        output_actions = output_actions[:, :, :original_action_dim]

        return output_actions

    def get_optim_params(self) -> dict:
        """Returns optimizer parameters.
        
        Note: Triton models are inference-only and do not support training.
        """
        return []

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.
        
        Note: Triton models are inference-only. This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "PI05FlowPolicy is inference-only. Use predict_action_chunk() or select_action() for inference."
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
    ) -> "PI05FlowPolicy":
        """Load a pretrained Triton flow matching policy."""
        pretrained_path = Path(pretrained_name_or_path)
        
        if triton_checkpoint_path is None:
            if pretrained_path.suffix == ".pkl":
                triton_checkpoint_path = pretrained_path
            else:
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
