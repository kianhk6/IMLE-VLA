import os
import json
import sys
import argparse
import pickle
import numpy as np # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
import cv2
from PIL import Image
import einops
import matplotlib # pyright: ignore[reportMissingImports]
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from matplotlib.gridspec import GridSpec # pyright: ignore[reportMissingImports]
from pi05_infer_imle import Pi05InferenceIMLE as Pi05Inference
from pi0_infer_imle import Pi0InferenceIMLE as Pi0Inference

# Import JAX for seeding (if available)
try:
    import jax # pyright: ignore[reportMissingImports]
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
# Add openpi src directory to Python path BEFORE importing openpi
openpi_src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(openpi_src_path))

from openpi.training import config as _config # pyright: ignore[reportMissingImports]
from openpi.policies import policy_config as _policy_config # pyright: ignore[reportMissingImports]

class PiModelEvaluator:
    def __init__(self, task, model_type: str, triton_path: str, jax_path: str, norm_stats_dir:str, config_name: str, action_horizon:int = 15, state_len: int = 8, action_dim = 8, prompt: str = None, discrete_state_input: bool = True, tokenizer_path: str = None, model_version: str = "pi05"):
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
        # Check config for AbsoluteActions to match JAX path behavior
        # JAX applies output transforms from config, including AbsoluteActions if present
        # We need to match this behavior in Triton path
        self._has_absolute_actions = False
        if self.config_name:
            try:
                config = _config.get_config(self.config_name)
                data_config = config.data.create(config.assets_dirs, config.model)
                from openpi import transforms as _transforms
                self._has_absolute_actions = any(
                    isinstance(transform, _transforms.AbsoluteActions)
                    for transform in data_config.data_transforms.outputs
                )
            except:
                # Default: assume no AbsoluteActions (matches pi05_droid)
                self._has_absolute_actions = False
        
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

    def _normalize_state(self, state: np.ndarray, norm_stats: dict) -> np.ndarray:
        """Normalize state using quantile (pi05) or z-score (pi0) normalization.
        
        Reference: openpi/src/openpi/transforms.py:137-145 (Normalize._normalize_quantile)
        Note: In JAX, norm_stats are SLICED to match state dim, not padded.
        """
        if norm_stats and "state" in norm_stats:
            state_dim = state.shape[-1]
            if self.model_version == "pi05":
                # Slice norm_stats to state dimension (same as JAX)
                q01 = np.array(norm_stats["state"]["q01"])[:state_dim]
                q99 = np.array(norm_stats["state"]["q99"])[:state_dim]
                return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            elif self.model_version == "pi0":
                state_mean = np.array(norm_stats["state"]["mean"])[:state_dim]
                state_std = np.array(norm_stats["state"]["std"])[:state_dim]
                return (state - state_mean) / (state_std + 1e-6)
        return state

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

    def _apply_input_transforms(self, data: dict, norm_stats: dict = None) -> dict:
        """
        Apply input transforms matching openpi's DROID preprocessing pipeline.
        
        Order matches JAX exactly:
        1. DroidInputs → 8D state
        2. Normalize → normalize 8D (slices norm_stats to match)
        3. TokenizePrompt → discretize 8D (for prompt, pi05 only)
        4. PadStatesAndActions → pad to 32D (for continuous state input, pi0 only)
        
        References:
        - State construction: openpi/src/openpi/policies/droid_policy.py:35-40 (DroidInputs.__call__)
        - Normalization order: openpi/src/openpi/policies/policy_config.py:77-82
        - Image mapping: openpi/src/openpi/policies/droid_policy.py:44-50 (DroidInputs.__call__)
        """
        # Step 1: Construct 8D state from joint_position (7D) + gripper_position (1D)
        # Reference: openpi/src/openpi/policies/droid_policy.py:36-40
        joint_pos = np.asarray(data["observation/joint_position"])
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([joint_pos, gripper_pos])  # 8D state for DROID
        
        # Step 2: Normalize 8D state (before padding, same as JAX)
        # Reference: openpi/src/openpi/transforms.py:137-145
        state = self._normalize_state(state, norm_stats)
        
        # Step 3: For pi05, discretize 8D state for prompt (before padding)
        # Reference: openpi/src/openpi/models/tokenizer.py:26
        if self.model_version == "pi05":
            state = self._digitize_state(state)
        else:
            # Step 4: For pi0, pad to 32D after normalization (for continuous state input)
            # Reference: openpi/src/openpi/transforms.py:334
            state = self._pad_to_dim(state, 32)
        
        # Map DROID images to model format (matching DroidInputs)
        # Reference: openpi/src/openpi/policies/droid_policy.py:44-50
        base_image = self._parse_image(data["observation/exterior_image_1_left"])
        wrist_image = self._parse_image(data["observation/wrist_image_left"])
        # Create right_wrist as zeros (DROID only has left wrist camera)
        # Reference: openpi/src/openpi/policies/droid_policy.py:50
        right_wrist_image = np.zeros_like(base_image)
        
        # Resize and normalize images
        base_image = self._resize_with_pad(base_image, 224, 224)
        base_image = self._normalize_image(base_image)
        wrist_image = self._resize_with_pad(wrist_image, 224, 224)
        wrist_image = self._normalize_image(wrist_image)
        right_wrist_image = self._resize_with_pad(right_wrist_image, 224, 224)
        right_wrist_image = self._normalize_image(right_wrist_image)
        
        image_dict = {
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
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
            # Extract original 8D state from DROID inputs (before transforms) for postprocessing
            # Needed if config uses AbsoluteActions transform to convert delta → absolute
            joint_pos = np.asarray(inputs["observation/joint_position"])
            gripper_pos = np.asarray(inputs["observation/gripper_position"])
            if gripper_pos.ndim == 0:
                gripper_pos = gripper_pos[np.newaxis]
            ori_state = np.concatenate([joint_pos, gripper_pos])  # 8D state
            
            transformed_inputs = self._apply_input_transforms(inputs, norm_stats = self.norm_stats)
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
            
            # Postprocessing: Match JAX path output transforms
            # If config uses AbsoluteActions, convert delta → absolute (only first 7 dims for joints)
            # Reference: openpi/src/openpi/transforms.py:234-242 (AbsoluteActions.__call__)
            # For pi05_droid: no AbsoluteActions, so actions stay as delta (consistent with JAX)
            if self._has_absolute_actions:
                # Only add state delta to first 7 dims (joints), not gripper (8th dim)
                actions[..., :7] = actions[..., :7] + ori_state[..., :7]
            
            return {
                "actions": actions
            }
        elif self.model_type == "jax":
            actions = self.policy.infer(inputs, noise = noise)
            actions["actions"] = actions["actions"][:, :self.action_dim]
            return actions

def plot_comparison(triton_actions, jax_actions, num_timesteps, action_dim, save_path="comparison_plot.png"):
    """
    Generate a comparison plot similar to the image shown.
    
    Args:
        triton_actions: numpy array of shape (num_timesteps, action_dim) - Triton model outputs
        jax_actions: numpy array of shape (num_timesteps, action_dim) - JAX model outputs
        num_timesteps: number of timesteps to plot
        action_dim: action dimension (typically 8 for DROID: 7 joints + 1 gripper)
        save_path: path to save the figure
    """
    # Calculate statistics
    mae_per_dim = np.mean(np.abs(triton_actions - jax_actions), axis=0)
    max_diff_per_dim = np.max(np.abs(triton_actions - jax_actions), axis=0)
    total_mae = np.mean(np.abs(triton_actions - jax_actions))
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3, 
                  left=0.08, right=0.95, top=0.95, bottom=0.05)
    
    # Dimension names
    dim_names = [f"Joint {i+1}" for i in range(7)] + ["Gripper Width"]
    
    # Plot each dimension
    timesteps = np.arange(num_timesteps)
    for i in range(action_dim):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot Triton (dashed red) and JAX (solid blue)
        ax.plot(timesteps, triton_actions[:, i], 'r--', linewidth=1.5, label='Triton', alpha=0.8)
        ax.plot(timesteps, jax_actions[:, i], 'b-', linewidth=1.5, label='JAX', alpha=0.8)
        
        # Fill area between lines to show difference
        ax.fill_between(timesteps, triton_actions[:, i], jax_actions[:, i], 
                       alpha=0.2, color='gray')
        
        # Formatting
        ax.set_xlabel('Timestep', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(f'{dim_names[i]}\nMAE: {mae_per_dim[i]:.6f}, Max Diff: {max_diff_per_dim[i]:.6f}', 
                    fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Add statistics table in the remaining space
    stats_ax = fig.add_subplot(gs[3, :])
    stats_ax.axis('off')
    
    # Overall statistics
    stats_text = "Overall Statistics:\n"
    stats_text += f"Total MAE: {total_mae:.6f}\n"
    stats_text += f"JAX Range: [{jax_actions.min():.4f}, {jax_actions.max():.4f}]\n"
    stats_text += f"Triton Range: [{triton_actions.min():.4f}, {triton_actions.max():.4f}]\n\n"
    
    # Per-dimension MAE
    stats_text += "Per-Dimension MAE:\n"
    for i in range(action_dim):
        stats_text += f"{dim_names[i]}: {mae_per_dim[i]:.6f}\n"
    
    stats_ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('Triton vs JAX Model Output Comparison', fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--triton_path', type=str, default=None)
    parser.add_argument('--jax_path', type=str, default=None)
    parser.add_argument('--norm_stats_dir', type=str, default=None)
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--prompt', type=str, default='do something')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='(DEPRECATED) Now uses same SentencePiece tokenizer as JAX automatically')
    parser.add_argument('--discrete_state_input', action='store_true')
    parser.add_argument('--action_dim', type=int, default=8, help='Action dimension (8 for DROID: 7 joints + 1 gripper)')
    parser.add_argument('--action_horizon', type=int, default=15)
    parser.add_argument('--model_version', type=str, choices=['pi0', 'pi05'], default='pi05')
    parser.add_argument('--model_type', type=str, choices=['triton', 'jax'], default='triton')
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of timesteps for visualization (use action_horizon for single inference)')
    parser.add_argument('--plot_comparison', action='store_true', help='Generate comparison plot')
    parser.add_argument('--plot_save_path', type=str, default='triton_vs_jax_comparison.png', help='Path to save comparison plot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_denoising_steps', type=int, default=10, help='Number of denoising steps (1 for IMLE, 10+ for flow matching)')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Make CUDA operations deterministic (may slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if HAS_JAX:
        # Set JAX random seed (global config)
        # Note: JAX uses explicit PRNG keys, but this helps with any global state
        import jax.random
    print(f"Using random seed: {args.seed}")

    # Three sample camera frames (image1.png, image2.png, image3.png in the working directory). The check only
    # needs identical inputs on both sides, so random frames are used when the files are absent.
    def _load_or_random(path):
        img = cv2.imread(path)
        if img is None:
            print(f"{path} not found; using a random 224x224 frame")
            img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        return img

    example_image_global = _load_or_random("image1.png")
    example_image_left = _load_or_random("image2.png")
    example_image_right = _load_or_random("image3.png")
    noise = np.random.randn(args.action_horizon, 32).astype(np.float32)
    # DROID uses 8D state: 7 joint positions + 1 gripper position
    # Reference: openpi/src/openpi/policies/droid_policy.py:36-40
    state = np.random.randn(8).astype(np.float32)  # Changed from 14 to 8 for DROID
    
    # Print input checksums for debugging reproducibility
    print(f"Noise checksum: {noise.sum():.6f}, State checksum: {state.sum():.6f}")

    pi_triton = PiModelEvaluator(
        task='check_consistency',
        model_type="triton",
        triton_path=args.triton_path,
        jax_path=args.jax_path,
        state_len=state.shape[0],
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
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
        action_horizon=args.action_horizon,
        norm_stats_dir=args.norm_stats_dir,
        config_name=args.config_name,
    )
    triton_list = []
    jax_list = []
    
    # Generate data for visualization
    # For single inference with action_horizon timesteps, or multiple inferences
    if args.plot_comparison and args.num_timesteps == args.action_horizon:
        # Single inference: use action_horizon as num_timesteps
        print(f"Generating comparison for single inference with {args.action_horizon} timesteps...")
        inputs_triton = {
            "observation/exterior_image_1_left": example_image_global,
            "observation/wrist_image_left": example_image_left,
            "observation/joint_position": state[:7],
            "observation/gripper_position": state[7:8] if state.shape[0] >= 8 else np.array([0.0]),
            "prompt": args.prompt
        }
        inputs_jax = {
            "observation/exterior_image_1_left": example_image_global,
            "observation/wrist_image_left": example_image_left,
            "observation/joint_position": state[:7],
            "observation/gripper_position": state[7:8] if state.shape[0] >= 8 else np.array([0.0]),
            "prompt": args.prompt
        }
        result_triton = pi_triton.infer(inputs_triton, noise)
        result_jax = pi_jax.infer(inputs_jax, noise)
        triton_actions = result_triton['actions']  # Shape: (action_horizon, action_dim)
        jax_actions = result_jax['actions']
        
        # Generate plot
        plot_comparison(triton_actions, jax_actions, args.action_horizon, args.action_dim, args.plot_save_path)
    else:
        # Multiple inferences: collect data over multiple timesteps
        num_iterations = max(10, args.num_timesteps // args.action_horizon) if args.plot_comparison else 10
        print(f"Running {num_iterations} iterations...")
        
        for idx in range(num_iterations):
            state_idx = state * (idx + 1)  # Vary state for each iteration
            # Use DROID input format for Triton (matching openpi's DroidInputs)
            # Reference: openpi/src/openpi/policies/droid_policy.py:35-74 (DroidInputs.__call__)
            # Reference: openpi/src/openpi/policies/droid_policy.py:10-18 (make_droid_example)
            inputs_triton = {
                "observation/exterior_image_1_left": example_image_global,
                "observation/wrist_image_left": example_image_left,
                "observation/joint_position": state_idx[:7],  # First 7 dims: joint positions
                "observation/gripper_position": state_idx[7:8] if state_idx.shape[0] >= 8 else np.array([0.0]),  # 8th dim: gripper
                "prompt": args.prompt
            }
            # JAX path uses openpi's Policy.infer which expects DROID format
            # Reference: openpi/src/openpi/policies/policy.py:68-106 (Policy.infer)
            inputs_jax = {
                "observation/exterior_image_1_left": example_image_global,
                "observation/wrist_image_left": example_image_left,
                "observation/joint_position": state_idx[:7],
                "observation/gripper_position": state_idx[7:8] if state_idx.shape[0] >= 8 else np.array([0.0]),
                "prompt": args.prompt
            }
            result_triton = pi_triton.infer(inputs_triton, noise)
            result_jax = pi_jax.infer(inputs_jax, noise)
            triton_list.append(result_triton['actions'])
            jax_list.append(result_jax['actions'])
        
        # Print text output
        for idx in range(min(10, len(triton_list))):
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
        
        # Generate plot if requested
        if args.plot_comparison and len(triton_list) > 0:
            # Concatenate all actions to create a longer time series
            triton_concatenated = np.concatenate(triton_list, axis=0)
            jax_concatenated = np.concatenate(jax_list, axis=0)
            num_plot_timesteps = min(args.num_timesteps, triton_concatenated.shape[0])
            plot_comparison(
                triton_concatenated[:num_plot_timesteps], 
                jax_concatenated[:num_plot_timesteps], 
                num_plot_timesteps, 
                args.action_dim, 
                args.plot_save_path
            )

if __name__ == "__main__":
    main()
