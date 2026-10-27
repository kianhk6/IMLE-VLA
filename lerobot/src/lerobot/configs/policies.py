# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import abc
import builtins
import json
import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import CONFIG_NAME
from huggingface_hub.errors import HfHubHTTPError

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import auto_select_torch_device, is_amp_available, is_torch_device_available

T = TypeVar("T", bound="PreTrainedConfig")
logger = getLogger(__name__)


@dataclass
class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC):  # type: ignore[misc,name-defined] #TODO: draccus issue
    """
    Base configuration class for policy models.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary with key representing the modality and the value specifies the
            normalization mode to apply.
        output_normalization_modes: Similar dictionary as `input_normalization_modes`, but to unnormalize to
            the original scale.
    """

    n_obs_steps: int = 1

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    device: str | None = None  # e.g. "cuda", "cuda:0", "cpu", or "mps"
    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    push_to_hub: bool = True  # type: ignore[assignment] # TODO: use a different name to avoid override
    repo_id: str | None = None

    # Upload on private repository on the Hugging Face hub.
    private: bool | None = None
    # Add tags to your policy on the hub.
    tags: list[str] | None = None
    # Add tags to your policy on the hub.
    license: str | None = None
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch.
    pretrained_path: Path | None = None

    # When True, checkpoints store the *full* model weights (base + any adapters / heads), even if
    # the model uses PEFT/LoRA (which often defaults to saving "adapters-only" weights).
    # This makes checkpoints self-contained for eval/inference, at the cost of larger checkpoints.
    save_full_weights: bool = True

    # If True, and the policy uses PEFT/LoRA, merge adapters into the base weights after loading for
    # inference (faster inference, smaller module tree). This should generally be False for training/resume.
    merge_lora_for_inference: bool = False

    # If True, and the policy uses PEFT/LoRA, also save an additional merged-weight artifact for inference.
    # (Implementation is policy-specific; policies that don't support it will ignore this flag.)
    save_merged_lora_checkpoint: bool = False

    def __post_init__(self) -> None:
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            logger.warning(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type

        # Automatically deactivate AMP if necessary
        if self.use_amp and not is_amp_available(self.device):
            logger.warning(
                f"Automatic Mixed Precision (amp) is not available on device '{self.device}'. Deactivating AMP."
            )
            self.use_amp = False

    @property
    def type(self) -> str:
        choice_name = self.get_choice_name(self.__class__)
        if not isinstance(choice_name, str):
            raise TypeError(f"Expected string from get_choice_name, got {type(choice_name)}")
        return choice_name

    @property
    @abc.abstractmethod
    def observation_delta_indices(self) -> list | None:  # type: ignore[type-arg] #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def action_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def reward_delta_indices(self) -> list | None:  # type: ignore[type-arg]    #TODO: No implementation
        raise NotImplementedError

    @abc.abstractmethod
    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError

    @abc.abstractmethod
    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_features(self) -> None:
        raise NotImplementedError

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[Any, Any] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **policy_kwargs: Any,
    ) -> T:
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                logger.error(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{CONFIG_NAME} not found on the HuggingFace Hub in {model_id}"
                ) from e

        if config_file is None:
            raise FileNotFoundError(f"{CONFIG_NAME} not found in {model_id}")

        # Read config file first to check if it has 'type' field
        with open(config_file) as f:
            config = json.load(f)

        # Filter out preserve_jax_precision field if present (not valid for PI05Config)
        # This field was used during JAX to PyTorch conversion but is not needed for evaluation
        had_preserve_jax_precision = "preserve_jax_precision" in config
        config.pop("preserve_jax_precision", None)

        # If 'type' field is missing, try to infer it from config fields
        if "type" not in config:
            inferred_type = None
            
            # Check for PI05/PI0 specific fields
            if "paligemma_variant" in config or "action_expert_variant" in config:
                # Distinguish between pi05 and pi0 based on normalization
                norm_mapping = config.get("normalization_mapping", {})
                if norm_mapping.get("STATE") == "QUANTILES":
                    inferred_type = "pi05"
                elif norm_mapping.get("STATE") == "MEAN_STD":
                    inferred_type = "pi0"
                else:
                    # Default to pi05 if we can't determine
                    inferred_type = "pi05"
            # Check for other policy types
            elif "horizon" in config and "n_obs_steps" in config:
                inferred_type = "diffusion"
            elif "action_chunk_size" in config:
                inferred_type = "vqbet"
            elif "temporal_ensemble_coeff" in config:
                inferred_type = "act"
            elif "num_categories" in config:
                inferred_type = "groot"
            
            if inferred_type:
                config["type"] = inferred_type
                logger.warning(
                    f"Config file {config_file} was missing 'type' field. "
                    f"Inferred type as '{inferred_type}' based on config fields. "
                    "Consider adding 'type' field to the config file for better reliability."
                )
                # Write the config with the type field to a temp file
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                    json.dump(config, f, indent=4)
                    config_file = f.name
            else:
                raise ValueError(
                    f"Config file {config_file} is missing 'type' field and cannot be inferred. "
                    "Please add 'type' field to config.json. Supported types: "
                    "pi05, pi0, diffusion, vqbet, act, groot, smolvla, tdmpc, sac"
                )
        elif had_preserve_jax_precision:
            # If type was present but we removed preserve_jax_precision, write cleaned config to temp file
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump(config, f, indent=4)
                config_file = f.name

        # HACK: Parse the original config to get the config subclass, so that we can
        # apply cli overrides.
        # This is very ugly, ideally we'd like to be able to do that natively with draccus
        # something like --policy.path (in addition to --policy.type)
        with draccus.config_type("json"):
            orig_config = draccus.parse(cls, config_file, args=[])

        # Remove type field before re-parsing with cli overrides
        with open(config_file) as f:
            config_for_override = json.load(f)
        
        config_for_override.pop("type", None)  # Use pop with default to avoid KeyError
        config_for_override.pop("preserve_jax_precision", None)  # Remove preserve_jax_precision if present
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(config_for_override, f, indent=4)
            config_file = f.name

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(orig_config.__class__, config_file, args=cli_overrides)
