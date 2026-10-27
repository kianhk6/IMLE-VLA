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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("pi05")
@dataclass
class PI05Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # IMLE (Implicit Maximum Likelihood Estimation) parameters
    use_imle: bool = False  # Enable IMLE training instead of flow matching
    imle_k: int = 5  # Number of samples per batch item for IMLE
    imle_softmax_temp: float = 1.0  # Temperature for softmax weighting (lower = sharper, higher = smoother)

    # Noise-Locked Reactive Chunking (inference only, IMLE only)
    # When enabled, caches the noise vector for an entire chunk_size window and
    # re-runs the model every step with the latest observation but the same noise.
    # Only the action at the current chunk_step is executed, giving reactivity
    # without the variance of fresh noise each step.
    noise_locked_reactive: bool = False

    # Continuity-based candidate selection (inference only, when noise_locked_reactive=False)
    # When enabled, generates k candidate action chunks in parallel and selects the one
    # whose first n_action_steps are closest to the previous chunk's first n_action_steps.
    # k=1 means no selection (default behavior, single candidate).
    continuity_k_candidates: int = 1

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # LoRA (Low-Rank Adaptation) settings
    use_lora: bool = False  # Enable LoRA fine-tuning for VLM and expert models
    lora_r: int = 8  # LoRA rank (lower = fewer parameters, higher = more capacity)
    lora_alpha: int = 16  # LoRA alpha scaling factor (typically 2x lora_r)
    lora_dropout: float = 0.1  # LoRA dropout rate
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_language_model_only: bool = False  # If True, apply LoRA only to language model (vision tower frozen)
    
    # Target modules for LoRA (attention and MLP layers)
    lora_bias: str = "none"  # LoRA bias: "none", "all", or "lora_only"

    # Inference helpers for LoRA runs
    # - merge_lora_for_inference: merge adapters into base weights after loading (faster inference).
    #   Should generally be False for training/resume.
    # NOTE: This field is also present on the base `PreTrainedConfig` so it works for all policies;
    # we keep it here for discoverability in the PI05 config.
    merge_lora_for_inference: bool = False

    # Whether to also save a merged-weights artifact (in addition to the normal checkpoint) for inference.
    # This is handled by the training/checkpoint code.
    save_merged_lora_checkpoint: bool = False
    
    # Deprecated quantization fields (kept for backward compatibility with old checkpoints)
    # These are ignored - quantization is no longer supported, only LoRA
    use_quantization: bool = False  # Deprecated: quantization removed, kept for checkpoint compatibility
    quantization_bits: int = 4  # Deprecated: ignored
    quantization_type: str = "nf4"  # Deprecated: ignored

    # EMA (Exponential Moving Average) settings
    use_ema: bool = False  # Enable EMA for more stable inference
    ema_decay: float = 0.999  # EMA decay rate (alpha parameter)
    ema_update_every: int = 1  # Update EMA model every N training steps
    ema_update_after_step: int = 0  # Start EMA updates after this many steps
    merge_ema_lora_for_inference: bool = False  # Merge LoRA weights in EMA model for inference (only if use_ema=True and use_lora=True)

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Validate LoRA configuration
        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError(f"lora_r must be positive, got {self.lora_r}")
            if self.lora_alpha <= 0:
                raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
            if not (0.0 <= self.lora_dropout <= 1.0):
                raise ValueError(f"lora_dropout must be between 0 and 1, got {self.lora_dropout}")
            if self.lora_bias not in ["none", "all", "lora_only"]:
                raise ValueError(f"lora_bias must be 'none', 'all', or 'lora_only', got {self.lora_bias}")
            
            # Validate lora_target_modules
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules cannot be empty when use_lora=True")
            
            # Required attention layers for language models (Gemma-based)
            required_attention_layers = {"q_proj", "k_proj", "v_proj", "o_proj"}
            target_set = set(self.lora_target_modules)
            missing_attention = required_attention_layers - target_set
            
            if missing_attention:
                raise ValueError(
                    f"lora_target_modules is missing required attention layers: {sorted(missing_attention)}. "
                    f"Required: {sorted(required_attention_layers)}, Got: {sorted(target_set)}"
                )
            
            # Warn if MLP layers are missing (optional but recommended)
            recommended_mlp_layers = {"gate_proj", "up_proj", "down_proj"}
            missing_mlp = recommended_mlp_layers - target_set
            if missing_mlp:
                import warnings
                warnings.warn(
                    f"lora_target_modules is missing recommended MLP layers: {sorted(missing_mlp)}. "
                    f"Recommended for full language model fine-tuning: {sorted(recommended_mlp_layers)}",
                    UserWarning
                )

        # Validate EMA configuration
        if self.use_ema:
            if not (0.0 < self.ema_decay < 1.0):
                raise ValueError(f"ema_decay must be between 0 and 1, got {self.ema_decay}")
            if self.ema_update_every <= 0:
                raise ValueError(f"ema_update_every must be positive, got {self.ema_update_every}")
            if self.ema_update_after_step < 0:
                raise ValueError(f"ema_update_after_step must be non-negative, got {self.ema_update_after_step}")

        # Validate mutual exclusivity of noise_locked_reactive and continuity_k_candidates
        if self.noise_locked_reactive and self.continuity_k_candidates > 1:
            raise ValueError(
                "noise_locked_reactive and continuity_k_candidates > 1 cannot both be enabled. "
                "They are mutually exclusive inference strategies."
            )
        
        # Validate continuity_k_candidates
        if self.continuity_k_candidates < 1:
            raise ValueError(f"continuity_k_candidates must be >= 1, got {self.continuity_k_candidates}")


    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
