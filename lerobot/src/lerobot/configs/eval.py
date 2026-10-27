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

import datetime as dt
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

from lerobot import envs, policies  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.policies import PreTrainedConfig

logger = getLogger(__name__)


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    env: envs.EnvConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000
    # Rename map for the observation to override the image and state keys
    rename_map: dict[str, str] = field(default_factory=dict)
    # For Triton models: path to original PyTorch checkpoint to load processor configs from
    # (processor JSONs are not in the Triton .pkl directory)
    original_torch_model_for_jsons: Path | None = None
    # Enable debug speed output (timing information)
    debug_speed: bool = False

    def __post_init__(self) -> None:
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            policy_path_obj = Path(policy_path)
            
            # Check if policy_path is a .pkl file (Triton model)
            is_triton = policy_path_obj.suffix == ".pkl"
            
            if is_triton:
                # Triton model: require original_torch_model_for_jsons
                if not self.original_torch_model_for_jsons:
                    raise ValueError(
                        f"Triton model detected (.pkl file) but 'original_torch_model_for_jsons' not provided. "
                        f"For Triton models, you must specify --original_torch_model_for_jsons "
                        f"pointing to the exact checkpoint directory (e.g., "
                        f"/path/to/checkpoints/010000) to load config.json and processor configs from."
                    )
                
                # Use the exact checkpoint path provided
                checkpoint_path = Path(self.original_torch_model_for_jsons)
                pretrained_model_dir = checkpoint_path / "pretrained_model"
                
                if pretrained_model_dir.exists():
                    config_path = str(pretrained_model_dir)
                else:
                    # Checkpoint path might already be pretrained_model directory or contain config.json directly
                    if (checkpoint_path / "config.json").exists():
                        config_path = str(checkpoint_path)
                    else:
                        raise ValueError(
                            f"Checkpoint path {checkpoint_path} does not contain 'pretrained_model' "
                            f"subdirectory or config.json. Expected structure: "
                            f"{checkpoint_path}/pretrained_model/config.json or {checkpoint_path}/config.json"
                        )
                
                logger.info(
                    f"Triton model (.pkl): loading config.json from checkpoint: {config_path}"
                )
            else:
                # Regular model: use policy_path directly
                config_path = policy_path
            
            self.policy = PreTrainedConfig.from_pretrained(config_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path_obj  # Keep original path (directory or .pkl file)

        else:
            logger.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type if self.policy is not None else 'scratch'}"
            else:
                self.job_name = (
                    f"{self.env.type}_{self.policy.type if self.policy is not None else 'scratch'}"
                )

            logger.warning(f"No job name provided, using '{self.job_name}' as job name.")

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
