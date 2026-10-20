import os
import json
import argparse
import pickle
import numpy as np
import torch
import cv2
from PIL import Image
import einops
from pi05_infer import Pi05Inference
from pi0_infer import Pi0Inference
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

class PiModelEvaluator:
    def __init__(self, task, model_type: str, triton_path: str, jax_path: str, norm_stats_dir:str, config_name: str, action_horizon:int = 50, state_len: int = 7, action_dim = 7, prompt: str = None, discrete_state_input: bool = True, tokenizer_path: str = None, model_version: str = "pi05"):
        self.triton_path = triton_path
        self.jax_path = jax_path
        self.task = task
        self.model_type = model_type
        self.norm_stats_dir = norm_stats_dir
        self.policy = None
        self.config_name = config_name
        self.action_horizon = action_horizon
        self.model_version = model_version
        self.norm_stats = None
        self.state_len = state_len
        self.action_dim = action_dim
        self.prompt = prompt
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.results = {
            'episode_results': []
        }
        if self.model_type == "triton":
            self.policy = self._load_model(model_type=self.model_type)
        elif self.model_type == "jax":
            self.policy = self._load_jax_model()

        self._digitize_bins = np.linspace(-1, 1, 256 + 1)[:-1]
        self._state_q01 = None
        self._state_q99 = None
        self._actions_q01 = None
        self._actions_q99 = None

    def _load_jax_model(self):
        config = _config.get_config(self.config_name)
        policy = _policy_config.create_trained_policy(config, self.jax_path)
        return policy

    def _load_model(self, model_type="triton"):
        print(f"load {model_type} model")
        with open(self.triton_path, 'rb') as f:
            weights = pickle.load(f)
        norm_stats = self._load_norm_stats(self.norm_stats_dir) if self.norm_stats_dir else None
        self.norm_stats = norm_stats
        if norm_stats is not None:
            if self.model_version == "pi05":
                self._state_q01 = np.array(norm_stats["state"]["q01"])
                self._state_q99 = np.array(norm_stats["state"]["q99"])
                self._actions_q01 = np.array(norm_stats["actions"]["q01"])
                self._actions_q99 = np.array(norm_stats["actions"]["q99"])
        
        if self.model_version == "pi05":
            policy = Pi05Inference(
                checkpoint = weights,
                num_views = 3,
                chunk_size = self.action_horizon,
                tokenizer_path=self.tokenizer_path,
                max_tokenize_len=200,
                max_prompt_text = self.prompt,
                discrete_state_input = self.discrete_state_input,
                state_dim_for_max_prompt = self.state_len
            )
        elif self.model_version == "pi0":
            policy = Pi0Inference(checkpoint = weights, num_views = 3, chunk_size = self.action_horizon)
        else:
            raise ValueError(f"Unknown model version: {self.model_version}")
        return policy

    def _load_norm_stats(self, norm_stats_dir: str) -> dict:
        norm_stats_path = os.path.join(norm_stats_dir, "norm_stats.json")
        if os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                return json.load(f)['norm_stats']
        return None

    def _parse_image(self, image) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = einops.rearrange(image, "c h w -> h w c")
        return image

    def _pad_to_dim(self, x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
        current_dim = x.shape[axis]
        if current_dim < target_dim:
            pad_width = [(0, 0)] * len(x.shape)
            pad_width[axis] = (0, target_dim - current_dim)
            return np.pad(x, pad_width)
        return x

    def _resize_with_pad(self, image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
        pil_image = Image.fromarray(image)
        cur_width, cur_height = pil_image.size
        if cur_width == width and cur_height == height:
            return image
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        zero_image = Image.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        return np.array(zero_image)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0 * 2.0 - 1.0

    def _normalize_state(self, state: np.ndarray, norm_stats: dict, target_dim: int = 32) -> np.ndarray:
        if norm_stats and "state" in norm_stats:
            if self.model_version == "pi05":
                q01 = self._pad_to_dim(self._state_q01 if self._state_q01 is not None else np.array(norm_stats["state"]["q01"]), target_dim)
                q99 = self._pad_to_dim(self._state_q99 if self._state_q99 is not None else np.array(norm_stats["state"]["q99"]), target_dim)
                return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            elif self.model_version == "pi0":
                state_mean = np.array(norm_stats["state"]["mean"])
                state_mean = self._pad_to_dim(state_mean, target_dim)
                state_std = np.array(norm_stats["state"]["std"])
                state_std = self._pad_to_dim(state_std, target_dim)
                return (state - state_mean) / (state_std + 1e-6)
        return None

    def _digitize_state(self, state_normed: np.ndarray) -> np.ndarray:
        return np.digitize(state_normed, bins=self._digitize_bins) - 1

    def _unnormalize_actions(self, actions: np.ndarray, norm_stats: dict, target_dim: int = 32) -> np.ndarray:
        if norm_stats and "actions" in norm_stats:
            if self.model_version == "pi05":
                q01 = self._pad_to_dim(self._actions_q01 if self._actions_q01 is not None else np.array(norm_stats["actions"]["q01"]), target_dim)
                q99 = self._pad_to_dim(self._actions_q99 if self._actions_q99 is not None else np.array(norm_stats["actions"]["q99"]), target_dim)
                return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            elif self.model_version == "pi0":
                actions_mean = np.array(norm_stats["actions"]["mean"])
                actions_mean = self._pad_to_dim(actions_mean, target_dim)
                actions_std = np.array(norm_stats["actions"]["std"])
                actions_std = self._pad_to_dim(actions_std, target_dim)
                return actions * (actions_std + 1e-6) + actions_mean
        return None

    def _apply_input_transforms(self, data: dict, action_dim: int = 32, norm_stats: dict = None) -> dict:
        state = self._pad_to_dim(data["state"], action_dim)
        state = self._normalize_state(state, norm_stats, action_dim)
        if self.model_version == "pi05":
            state = self._digitize_state(state)
        base_image = self._parse_image(data["base_0_rgb"])
        left_wrist_image = self._parse_image(data["left_wrist_0_rgb"])
        right_wrist_image = self._parse_image(data["right_wrist_0_rgb"])
        base_image = self._resize_with_pad(base_image, 224, 224)
        base_image = self._normalize_image(base_image)
        left_wrist_image = self._resize_with_pad(left_wrist_image, 224, 224)
        left_wrist_image = self._normalize_image(left_wrist_image)
        right_wrist_image = self._resize_with_pad(right_wrist_image, 224, 224)
        right_wrist_image = self._normalize_image(right_wrist_image)
        image_dict = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": left_wrist_image,
            "right_wrist_0_rgb": right_wrist_image,
        }
        inputs = {
            "state": state,
            "image": image_dict,
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

    def infer(self, inputs: dict, noise: np.ndarray) -> dict:
        if self.model_type == "triton":
            ori_state = np.asarray(inputs["state"]).copy()
            transformed_inputs = self._apply_input_transforms(inputs, action_dim = self.state_len, norm_stats = self.norm_stats)
            imgs = []
            for view in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
                img = transformed_inputs["image"][view]
                imgs.append(torch.from_numpy(img))
            observation_images = torch.stack(imgs, dim=0).to(torch.float32).cuda(non_blocking=True)
            observation_state = torch.from_numpy(transformed_inputs["state"].astype(np.float32)).unsqueeze(0).to(torch.float32).squeeze(0)
            diffusion_noise = torch.from_numpy(noise).to(torch.float32).cuda(non_blocking=True)
            if self.model_version == "pi05":
                forward_args = (observation_images, diffusion_noise, inputs["prompt"], transformed_inputs["state"])
            elif self.model_version == "pi0":
                forward_args = (observation_images, observation_state, diffusion_noise)

            actions = self.policy.forward(*forward_args)
            actions = actions.cpu().float().numpy()
            actions = self._unnormalize_actions(actions, self.norm_stats, 32)[:, :self.action_dim]
            actions[..., :self.action_dim] = actions[..., :self.action_dim] + ori_state[..., :self.action_dim]
            for i in range(self.action_dim // 7):
                actions[..., (i + 1) * 7 - 1] = actions[..., (i + 1) * 7 - 1] - ori_state[..., (i + 1) * 7 - 1]
            return {
                "actions": actions
            }
        elif self.model_type == "jax":
            actions = self.policy.infer(inputs, noise = noise)
            actions["actions"] = actions["actions"][:, :self.action_dim]
            return actions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton_path', type=str, default=None)
    parser.add_argument('--jax_path', type=str, default=None)
    parser.add_argument('--norm_stats_dir', type=str, default=None)
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='do something')
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--discrete_state_input', action='store_true')
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--model_version', type=str, choices=['pi0', 'pi05'], default='pi05')
    parser.add_argument('--model_type', type=str, choices=['triton', 'jax'], default='triton')
    args = parser.parse_args()

    example_image_global = cv2.imread("image1.png")
    example_image_left = cv2.imread("image2.png")
    example_image_right = cv2.imread("image3.png")
    noise = np.random.randn(50, 32).astype(np.float32)
    state = np.random.randn(14).astype(np.float32)

    pi_triton = PiModelEvaluator(
        task='check_consistency',
        model_type="triton",
        triton_path=args.triton_path,
        jax_path=args.jax_path,
        state_len=state.shape[0],
        action_dim=args.action_dim,
        norm_stats_dir=args.norm_stats_dir,
        config_name=args.config_name,
        prompt=args.prompt,
        discrete_state_input=args.discrete_state_input,
        tokenizer_path=args.tokenizer_path,
        model_version=args.model_version,
    )
    pi_jax = PiModelEvaluator(
        task='check_consistency',
        model_type="jax",
        triton_path=args.triton_path,
        jax_path=args.jax_path,
        state_len=state.shape[0],
        action_dim=args.action_dim,
        norm_stats_dir=args.norm_stats_dir,
        config_name=args.config_name,
    )
    triton_list = []
    jax_list = []
    for idx in range(10):
        state_idx = state * idx
        inputs_triton = {
            "base_0_rgb": example_image_global,
            "left_wrist_0_rgb": example_image_left,
            "right_wrist_0_rgb": example_image_right,
            "state": state_idx,
            "prompt": args.prompt
        }
        inputs_jax = {
            "observation/cam_high": example_image_global,
            "observation/cam_wrist_left": example_image_left,
            "observation/cam_wrist_right": example_image_right,
            "state": state_idx,
            "prompt": args.prompt
        }
        result_triton = pi_triton.infer(inputs_triton, noise)
        result_jax = pi_jax.infer(inputs_jax, noise)
        triton_list.append(result_triton['actions'])
        jax_list.append(result_jax['actions'])
        
    for idx in range(10):
        print("result_triton['actions'].shape:", triton_list[idx].shape)
        print("result_jax['actions'].shape:", jax_list[idx].shape)
        print(f"Triton actions range: [{triton_list[idx].min():.6f}, {triton_list[idx].max():.6f}]")
        print(f"JAX actions range: [{jax_list[idx].min():.6f}, {jax_list[idx].max():.6f}]")
        triton_jax_mae = np.mean(np.abs(triton_list[idx] - jax_list[idx]))
        print(f"Triton vs JAX MAE: {triton_jax_mae:.6f}")
        print("Per-dimension MAE:")
        for i in range(args.action_dim):
            dim_mae = np.mean(np.abs(triton_list[idx][:, i] - jax_list[idx][:, i]))
            print(f"{dim_mae:.6f}")

if __name__ == "__main__":
    main()
