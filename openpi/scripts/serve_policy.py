import abc
import dataclasses
import enum
import json
import logging
import pathlib
import pickle
import socket
import time
from typing import Any

import jax
import numpy as np
import torch
import tyro
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
import openpi.transforms as transforms


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Sample kwargs to pass to policy as JSON string (e.g., '{"num_steps": 1}' for IMLE).
    sample_kwargs: str | None = None


@dataclasses.dataclass
class TritonCheckpoint:
    """Load a policy from a Triton .pkl checkpoint."""

    # Training config name (e.g., "pi05_libero").
    config: str
    # Path to the .pkl checkpoint file.
    pkl_path: str
    # Path to the original JAX checkpoint directory. The norm_stats are read from
    # <jax_checkpoint_dir>/assets/<asset_id>, so this must point at the checkpoint the .pkl was converted from.
    jax_checkpoint_dir: str | None = None
    # Sample kwargs to pass to policy as JSON string (e.g., '{"num_steps": 1}' for IMLE).
    sample_kwargs: str | None = None


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | TritonCheckpoint | Default = dataclasses.field(default_factory=Default)


# Import Triton inference classes - support both IMLE and Flow Matching
import sys
import dataclasses
realtime_vla_path = pathlib.Path(__file__).parent.parent / "realtime-vla"
realtime_vla_imle_path = pathlib.Path(__file__).parent.parent / "realtime-vla-imle"

# Add both paths to sys.path
if str(realtime_vla_path) not in sys.path:
    sys.path.insert(0, str(realtime_vla_path))
if str(realtime_vla_imle_path) not in sys.path:
    sys.path.insert(0, str(realtime_vla_imle_path))

# Import Flow Matching version (default, for non-IMLE models)
from pi05_infer import Pi05Inference as Pi05InferenceFlow


class TritonPolicy(_policy.Policy):
    """Custom Policy for Triton models that extracts prompt before transforms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store whether we need to extract prompt (for pi05 with discrete_state_input)
        self._needs_prompt = (
            isinstance(self._model, TritonModelWrapper)
            and self._model.discrete_state_input
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # Extract prompt before transforms if needed
        prompt = None
        if self._needs_prompt:
            prompt = obs.get("prompt")
            if prompt is not None and not isinstance(prompt, str):
                prompt = str(prompt)
            if prompt is None:
                # Try to get from default_prompt stored in model
                prompt = getattr(self._model, "default_prompt", None)

        # Use custom infer method that passes prompt through
        if self._needs_prompt and prompt is not None:
            return self._infer_with_prompt(obs, noise=noise, prompt=prompt)
        else:
            return super().infer(obs, noise=noise)

    def _infer_with_prompt(self, obs: dict, *, noise: np.ndarray | None = None, prompt: str | None = None) -> dict:
        """Internal method that allows passing prompt through."""
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # Convert inputs to PyTorch tensors and move to correct device
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
        sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device)
            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        
        # Pass prompt if available
        if prompt is not None:
            sample_kwargs["prompt"] = prompt

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs


class TritonModelWrapper(_model.BaseModel):
    """Wrapper for Triton inference models to implement BaseModel interface."""

    def __init__(
        self,
        triton_model: Any,  # Can be Pi05InferenceFlow (Flow Matching) or Pi05InferenceIMLE (IMLE)
        action_dim: int,
        action_horizon: int,
        max_token_len: int,
        discrete_state_input: bool = False,
        default_prompt: str | None = None,
        original_state_dim: int = 8,
    ):
        super().__init__(action_dim, action_horizon, max_token_len)
        self.triton_model = triton_model
        self.discrete_state_input = discrete_state_input
        self.default_prompt = default_prompt
        # Original state dimension BEFORE zero-padding by PadStatesAndActions.
        # TokenizePrompt in the JAX path runs before PadStatesAndActions, so it
        # digitizes the original (e.g. 8D) state.  We must replicate that here.
        self.original_state_dim = original_state_dim

    def to(self, device: str | torch.device) -> "TritonModelWrapper":
        """PyTorch-compatible device movement (no-op for Triton, always on CUDA)."""
        # Triton models are always on CUDA and don't need device movement
        return self

    def eval(self) -> "TritonModelWrapper":
        """PyTorch-compatible eval mode (no-op for Triton)."""
        # Triton models are always in inference mode
        return self

    def compute_loss(
        self,
        rng: Any,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> Any:
        """Not implemented for inference-only wrapper."""
        raise NotImplementedError("TritonModelWrapper is inference-only")

    def sample_actions(
        self,
        rng_or_device: Any,
        observation: _model.Observation,
        *,
        noise: torch.Tensor | None = None,
        prompt: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample actions using Triton model.

        Args:
            rng_or_device: Ignored for Triton (device is always CUDA).
            observation: Observation object with normalized images and state.
            noise: Optional noise tensor. If None, will be sampled.
            prompt: Optional prompt string. If None and discrete_state_input=True, uses default_prompt.
            **kwargs: Additional kwargs (ignored).

        Returns:
            Actions tensor of shape (batch_size, action_horizon, action_dim).
        """
        batch_size = observation.state.shape[0]
        device = "cuda"

        # Generate noise if not provided (same pattern as JAX/PyTorch models)
        # Policy.infer only passes noise if explicitly provided, otherwise sample_actions generates it
        if noise is None:
            noise = torch.normal(
                mean=0.0,
                std=1.0,
                size=(batch_size, self.action_horizon, self.action_dim),
                dtype=torch.float32,
                device=device,
            )

        # Convert images from Observation format to Triton format
        # Observation images are in (B, H, W, C) format, [-1, 1] range
        # Use image_mask to determine which views are valid (same as JAX/PyTorch)
        # This matches how JAX/PyTorch models determine which views to process
        
        # Get all possible image keys
        all_image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        
        # Filter to only include keys with valid masks (mask=True)
        # Extract mask value (handle both arrays and scalars)
        def get_mask_value(mask):
            if isinstance(mask, np.ndarray):
                return bool(mask[0] if mask.size > 0 else False)
            elif hasattr(mask, 'item'):
                return bool(mask.item())
            else:
                return bool(mask)
        
        image_keys = [
            key for key in all_image_keys
            if key in observation.images
            and key in observation.image_masks
            and get_mask_value(observation.image_masks[key])
        ]
        
        # Verify we have the expected number of views
        expected_num_views = self.triton_model.num_views
        if len(image_keys) != expected_num_views:
            raise ValueError(
                f"Expected {expected_num_views} valid image views, but found {len(image_keys)}. "
                f"Valid views: {image_keys}, "
                f"All masks: {[(k, get_mask_value(v)) for k, v in observation.image_masks.items() if k in all_image_keys]}"
            )
        
        images_list = []
        for key in image_keys:
            if key not in observation.images:
                raise ValueError(f"Missing image key: {key}")
            img = observation.images[key]  # Could be (B, H, W, C) or (B, C, H, W) - PyTorch tensor from Policy.infer()
            
            # Ensure it's on CUDA (keep float32 to match JAX/PyTorch)
            # Pi05Inference.forward will convert to bfloat16 when copying into buffer
            img = img.to(device=device)
            
            # Log original shape for debugging
            original_shape = img.shape
            
            # Take first batch item
            img = img[0]  # Now (H, W, C) or (C, H, W)
            
            # Handle both [C, H, W] and [H, W, C] formats (same logic as preprocess_observation_pytorch)
            # Check which dimension has 3 channels to determine format
            if len(img.shape) != 3:
                raise ValueError(f"Expected 3D image tensor after removing batch dim, got shape {img.shape} (original: {original_shape})")
            
            # Determine format: channels-first if dim 0 has 3, channels-last if dim 2 has 3
            if img.shape[0] == 3 and img.shape[2] != 3:
                # Channels-first: (C, H, W)
                img = img.permute(1, 2, 0)  # Convert to (H, W, C)
            elif img.shape[2] == 3 and img.shape[0] != 3:
                # Channels-last: (H, W, C) - already correct
                pass
            else:
                # Ambiguous case - log and try to infer
                logging.warning(
                    f"Ambiguous image format for {key}: shape {img.shape} (original: {original_shape}). "
                    f"Assuming channels-last format."
                )
                # If both dim 0 and dim 2 have 3, check which one makes sense (224x224 image)
                if img.shape[1] == 224:  # Height is 224, so likely (H, W, C)
                    pass  # Already channels-last
                else:
                    # Likely (C, H, W), convert
                    img = img.permute(1, 2, 0)
            
            # Verify final shape is (H, W, C)
            if img.shape[2] != 3:
                raise ValueError(
                    f"After format conversion, image {key} should have 3 channels in last dimension, "
                    f"but got shape {img.shape} (original: {original_shape})"
                )
            
            images_list.append(img)
        
        # Stack images: (num_views, H, W, C)
        observation_images = torch.stack(images_list, dim=0)
        
        # Verify final stacked shape matches Triton buffer expectation
        expected_shape = (len(image_keys), 224, 224, 3)
        if observation_images.shape != expected_shape:
            logging.warning(
                f"Stacked images shape {observation_images.shape} doesn't match expected {expected_shape}. "
                f"This may cause issues with Triton buffer."
            )

        # ── Images are already normalized to [-1, 1] float32 ──
        # Observation.from_dict() (called at line 156) already converts uint8 images
        # to float32 [-1, 1] range. Do NOT re-normalize here or you'll get double
        # normalization which collapses all pixel values to ≈ -1.0.
        # The Triton model expects float32 [-1, 1] images, same as JAX.
        assert observation_images.dtype == torch.float32, (
            f"Expected float32 images (already normalized by Observation.from_dict()), "
            f"got {observation_images.dtype}. This indicates a bug in the preprocessing pipeline."
        )
        # Verify images are in [-1, 1] range (with small tolerance for floating point)
        img_min, img_max = observation_images.min().item(), observation_images.max().item()
        if img_min < -1.1 or img_max > 1.1:
            logging.warning(
                f"Image values out of expected [-1, 1] range: min={img_min:.4f}, max={img_max:.4f}. "
                f"This may indicate incorrect normalization."
            )

        # Convert noise to bfloat16
        noise_batch = noise[0].to(dtype=torch.bfloat16, device=device)  # (action_horizon, action_dim)

        # Pi05 model only - always uses discrete_state_input=True
        # Need prompt string and state tokens
        prompt_str = prompt or self.default_prompt
        if prompt_str is None:
            raise ValueError("prompt is required for pi05 with discrete_state_input")
        
        # ── Digitize state — use only original dims (before PadStatesAndActions) ──
        # TokenizePrompt in the JAX path runs BEFORE PadStatesAndActions,
        # so it digitizes the original 8D state, not the 32D padded state.
        # Zero-padded dims normalise to ~0 → bin 128, which would corrupt the
        # prompt embedding.  Slice to original_state_dim to match JAX.
        state_normed = observation.state
        state_normed_np = state_normed[0].cpu().numpy()[:self.original_state_dim]  # (8,) not (32,)
        
        # Digitize using the same logic as PaligemmaTokenizer.tokenize()
        # Reference: openpi/models/tokenizer.py line 26
        discretized_state = np.digitize(state_normed_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_tokens = discretized_state.astype(np.int32)
        
        # Determine num_steps: IMLE always uses 1, Flow Matching doesn't use num_steps in forward()
        # Check if this is an IMLE model by checking the is_imle attribute
        is_imle_model = getattr(self.triton_model, 'is_imle', False)
        
        # Call Triton forward
        print(f"observation_images_normalized: {observation_images.shape}")
        print(f"diffusion_noise: {noise_batch.shape}")
        print(f"task_prompt: {prompt_str}")
        print(f"state_tokens: {state_tokens}")
        
        # Build forward kwargs - Flow Matching doesn't accept num_steps, IMLE does
        forward_kwargs = {
            "observation_images_normalized": observation_images,
            "diffusion_noise": noise_batch,
            "task_prompt": prompt_str,
            "state_tokens": state_tokens,
        }
        
        # Only pass num_steps for IMLE models (Flow Matching version doesn't accept it)
        if is_imle_model:
            forward_kwargs["num_steps"] = 1  # IMLE always uses single step
            print(f"num_steps: 1 (IMLE mode)")
        else:
            # Flow Matching: num_steps is handled internally by the model weights/configuration
            # The forward() method doesn't accept num_steps as a parameter
            print(f"num_steps: N/A (Flow Matching - handled by model configuration)")
        
        output_actions = self.triton_model.forward(**forward_kwargs)

        # Convert output to float32 and add batch dimension
        output_actions = output_actions.to(dtype=torch.float32)
        if output_actions.ndim == 2:  # (action_horizon, action_dim)
            output_actions = output_actions.unsqueeze(0)  # (1, action_horizon, action_dim)
        
        return output_actions


def create_triton_policy(
    train_config: _config.TrainConfig,
    pkl_path: str | pathlib.Path,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    jax_checkpoint_dir: str | pathlib.Path | None = None,
    env: EnvMode | None = None,
) -> _policy.Policy:
    """Create a policy from a Triton .pkl checkpoint using the same transform pipeline as JAX/PyTorch.

    Args:
        train_config: The training config to use to create the model.
        pkl_path: Path to the .pkl checkpoint file.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method.
            If `num_steps=1`, will use IMLE version from `realtime-vla-imle`.
        default_prompt: The default prompt to use for the policy.
        norm_stats: The norm stats to use for the policy. If not provided, will try to load from jax_checkpoint_dir.
        jax_checkpoint_dir: Path to the original JAX checkpoint directory (for loading norm_stats).
        env: Environment mode to determine num_views. LIBERO uses 2 views, DROID uses 3.

    Returns:
        A Policy object with the same transform pipeline as JAX/PyTorch models.
    """
    repack_transforms = repack_transforms or transforms.Group()
    pkl_path = pathlib.Path(pkl_path).resolve()
    sample_kwargs = sample_kwargs or {}

    # Pi05 only
    if train_config.model.model_type != _model.ModelType.PI05:
        raise ValueError("Triton wrapper only supports pi05 models")
    
    # Detect IMLE mode: num_steps=1 in sample_kwargs OR use_imle=True in config
    is_imle = sample_kwargs.get("num_steps") == 1
    if not is_imle:
        # Also check config for use_imle flag
        model_fields = {f.name for f in dataclasses.fields(train_config.model)}
        if 'use_imle' in model_fields:
            is_imle = getattr(train_config.model, 'use_imle', False)
    
    if is_imle:
        print("=" * 80)
        print("🚀 Detected IMLE mode - using realtime-vla-imle implementation (pi05_infer_imle.py)")
        print("=" * 80)
        logging.info("Detected IMLE mode - using realtime-vla-imle implementation (pi05_infer_imle.py)")
        # Import IMLE version with new naming convention
        try:
            from pi05_infer_imle import Pi05InferenceIMLE
            # Verify we're using the correct file from realtime-vla-imle folder
            import pi05_infer_imle
            imle_module_path = pathlib.Path(pi05_infer_imle.__file__).resolve()
            expected_imle_path = (realtime_vla_imle_path / "pi05_infer_imle.py").resolve()
            if imle_module_path == expected_imle_path:
                logging.info(f"✓ Verified: Using IMLE Triton from {imle_module_path}")
            else:
                logging.warning(
                    f"⚠ WARNING: Imported pi05_infer_imle from {imle_module_path}, "
                    f"but expected {expected_imle_path}. This may not be the correct IMLE implementation!"
                )
            # Alias for consistency
            Pi05Inference = Pi05InferenceIMLE
        except ImportError as e:
            raise ImportError(
                f"Failed to import IMLE version from pi05_infer_imle. "
                f"Make sure the file exists in {realtime_vla_imle_path}. "
                f"Original error: {e}"
            )
    else:
        print("=" * 80)
        print("🌊 Detected Flow Matching mode - using realtime-vla implementation (pi05_infer.py)")
        print("=" * 80)
        logging.info("Detected Flow Matching mode - using realtime-vla implementation (pi05_infer.py)")
        # Use Flow Matching version (already imported at top of file)
        Pi05Inference = Pi05InferenceFlow
    
    # Pi05 models always have discrete_state_input=True
    # Reference: openpi/models/pi0_config.py line 44-45
    # JAX/PyTorch set: discrete_state_input = pi05, which is True for pi05 models
    discrete_state_input = True

    # Load Triton model
    logging.info(f"Loading Triton model from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        weights = pickle.load(f)

    # Determine action_horizon and action_dim from config
    action_horizon = train_config.model.action_horizon
    action_dim = train_config.model.action_dim
    max_token_len = train_config.model.max_token_len

    # Get tokenizer path - use HuggingFace model identifier (same as pi05_infer.py expects)
    # AutoTokenizer.from_pretrained() will automatically download and cache it in HuggingFace cache
    # (typically ~/.cache/huggingface/), similar to how JAX downloads SentencePiece model
    # Reference: realtime-vla/pi05_infer.py line 549 uses AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_path = "google/paligemma-3b-pt-224"
    
    # Determine num_views from env flag
    # Only count real (unmasked) image views — masked-out zero-padded views are not passed to Triton
    if env == EnvMode.ALOHA:
        num_views = 3  # ALOHA: base + left_wrist + right_wrist (all real)
    else:
        num_views = 2  # DROID & LIBERO: base + wrist (right_wrist is zero-padded & masked out)
    
    logging.info(f"num_views: {num_views} (determined from env: {env})")
    # Create Triton inference model (pi05 only)
    # Use max_token_len from config (defaults to 200 for pi05, same as JAX/PyTorch)
    # For IMLE, pass is_imle=True; for Flow Matching, don't pass it (Flow Matching doesn't accept is_imle)
    triton_kwargs = {
        "checkpoint": weights,
        "num_views": num_views,
        "chunk_size": action_horizon,
        "tokenizer_path": tokenizer_path,
        "max_tokenize_len": max_token_len,
        "discrete_state_input": discrete_state_input,
    }
    # Only pass is_imle for IMLE models (Flow Matching version doesn't accept it)
    if is_imle:
        triton_kwargs["is_imle"] = True
    
    triton_model = Pi05Inference(**triton_kwargs)
    
    # Verify IMLE mode is correctly set
    if is_imle:
        if hasattr(triton_model, 'is_imle') and triton_model.is_imle:
            logging.info("✓ Verified: Triton model has is_imle=True - IMLE mode confirmed")
        else:
            logging.warning(
                f"⚠ WARNING: Expected is_imle=True but model has is_imle={getattr(triton_model, 'is_imle', 'NOT SET')}. "
                f"This may indicate the wrong implementation is being used!"
            )

    # Determine original state dimension (before zero-padding).
    # DROID: 7 joints + 1 gripper = 8;  LIBERO: 9 (eef_pos 3 + eef_quat 4 + gripper_qpos 2);
    # ALOHA: 14 joints + 2 grippers = 16.
    # This must match the state size produced by the data config's DroidInputs (or
    # equivalent) so that digitised state tokens match the JAX path exactly.
    if env == EnvMode.DROID:
        original_state_dim = 8
    elif env == EnvMode.LIBERO:
        original_state_dim = 9
    elif env == EnvMode.ALOHA:
        original_state_dim = 16
    else:
        original_state_dim = 8  # safe default

    # Create wrapper
    model = TritonModelWrapper(
        triton_model=triton_model,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        discrete_state_input=discrete_state_input,
        default_prompt=default_prompt,
        original_state_dim=original_state_dim,
    )

    # Load norm_stats (same as create_trained_policy)
    # We need data_config for loading norm_stats
    # Use jax_checkpoint_dir as assets_dirs to avoid downloading from GCS
    if jax_checkpoint_dir is None:
        raise ValueError(
            "--policy.jax-checkpoint-dir is required for Triton checkpoints: the norm_stats are loaded from "
            "<jax_checkpoint_dir>/assets/<asset_id>."
        )
    jax_checkpoint_dir = pathlib.Path(jax_checkpoint_dir)
    
    # Create data_config using local jax_checkpoint_dir to avoid GCS downloads
    # This will try to load norm_stats from local path, but we'll override it below
    data_config = train_config.data.create(jax_checkpoint_dir, train_config.model)
    
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("asset_id is required to load norm_stats from JAX checkpoint directory.")
        
        try:
            norm_stats = _checkpoints.load_norm_stats(jax_checkpoint_dir / "assets", data_config.asset_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load norm_stats from JAX checkpoint directory "
                f"{jax_checkpoint_dir / 'assets' / data_config.asset_id}. "
                f"This is required for proper normalization. Error: {e}"
            ) from e

    # Create policy with the SAME transform pipeline as JAX/PyTorch
    # This ensures DROID preprocessing/postprocessing matches exactly:
    # - Input: DroidInputs → Normalize → TokenizePrompt
    # - Output: Unnormalize → DroidOutputs
    # Use TritonPolicy to handle prompt extraction
    return TritonPolicy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,  # Includes DroidInputs for DROID
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,  # Includes TokenizePrompt for pi05
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,  # Includes DroidOutputs for DROID
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=True,  # Treat as PyTorch for device handling
        pytorch_device="cuda",
    )


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            # Parse sample_kwargs from JSON string if provided
            sample_kwargs = {}
            if args.policy.sample_kwargs:
                try:
                    sample_kwargs = json.loads(args.policy.sample_kwargs)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse sample_kwargs JSON: {e}. Using empty dict.")
            
            # Get config and potentially override use_imle for IMLE mode
            config = _config.get_config(args.policy.config)
            
            # If num_steps=1 in sample_kwargs, ensure use_imle=True in config for JAX models
            if sample_kwargs.get("num_steps") == 1:
                import dataclasses
                model_fields = {f.name for f in dataclasses.fields(config.model)}
                if 'use_imle' in model_fields:
                    current_use_imle = getattr(config.model, 'use_imle', None)
                    if current_use_imle != True:
                        logging.info(f"Overriding use_imle: {current_use_imle} -> True (for IMLE mode with num_steps=1)")
                        new_model = dataclasses.replace(config.model, use_imle=True)
                        config = dataclasses.replace(config, model=new_model)
            
            return _policy_config.create_trained_policy(
                config, args.policy.dir, 
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs
            )
        case TritonCheckpoint():
            # Parse sample_kwargs from JSON string if provided
            sample_kwargs = {}
            if args.policy.sample_kwargs:
                try:
                    sample_kwargs = json.loads(args.policy.sample_kwargs)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse sample_kwargs JSON: {e}. Using empty dict.")
            
            # Get config
            config = _config.get_config(args.policy.config)
            
            return create_triton_policy(
                config,
                args.policy.pkl_path,
                default_prompt=args.default_prompt,
                sample_kwargs=sample_kwargs,
                jax_checkpoint_dir=args.policy.jax_checkpoint_dir,
                env=args.env,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
