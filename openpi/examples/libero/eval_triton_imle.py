#!/usr/bin/env python3
"""
LIBERO evaluation with the single-step (IMLE) Triton engine from realtime-vla-imle.

This script evaluates a pi05 model on LIBERO tasks using the direct Triton inference
implementation instead of a websocket server.

Usage:
    python examples/libero/eval_triton_imle.py \
        --checkpoint_path /path/to/checkpoint.pkl \
        --task_suite_name libero_spatial \
        --num_trials_per_task 10 \
        --chunk_size 10
"""

import collections
import dataclasses
import logging
import math
import pathlib
import pickle
import sys

import imageio
import numpy as np
import torch
import tqdm
import tyro

from time import perf_counter

# Add realtime-vla-imle (single-step Triton kernels) to the Python path
script_path = pathlib.Path(__file__).resolve()
realtime_vla_path = script_path.parent.parent.parent / "realtime-vla-imle"
if realtime_vla_path.exists():
    sys.path.insert(0, str(realtime_vla_path))

# Add third_party/libero to path as fallback if libero is not installed
project_root = pathlib.Path(__file__).parent.parent.parent
libero_path = project_root / "third_party" / "libero"
if libero_path.exists():
    sys.path.insert(0, str(libero_path))

try:
    from pi05_infer_imle import Pi05InferenceIMLE as Pi05Inference
except ImportError as e:
    raise ImportError(f"Failed to import pi05_infer_imle from {realtime_vla_path}. Error: {e}")

try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    raise ImportError(
        f"Failed to import libero. Please install it by running:\n"
        f"  cd {project_root}\n"
        f"  pip install -e third_party/libero\n"
        f"Original error: {e}"
    )

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
IMAGE_SIZE = 224  # Model input size


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model checkpoint parameters
    #################################################################################################################
    checkpoint_path: str  # Path to checkpoint .pkl file
    chunk_size: int = 10  # Action chunk size (replan_steps)
    num_steps: int = 1  # Number of denoising steps (1 for IMLE, 10 for flow matching)
    is_imle: bool = True  # IMLE mode (single forward pass); set False with --num_steps 10 for flow matching
    discrete_state_input: bool = True  # Whether to use discrete state tokens
    max_tokenize_len: int = 200  # Maximum token length for prompts
    norm_stats_dir: str | None = None  # Directory containing norm_stats.json file (optional)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from pickle file."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    logging.info(f"✓ Checkpoint loaded successfully")
    return checkpoint


def load_norm_stats(norm_stats_dir: str | None) -> dict | None:
    """Load normalization statistics from JSON file.
    
    Reference: test_imle.py _load_norm_stats
    Expects norm_stats_dir to be a directory containing norm_stats.json
    """
    if norm_stats_dir is None:
        return None
    
    import json
    import os
    norm_stats_path = os.path.join(norm_stats_dir, "norm_stats.json")
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'r') as f:
            data = json.load(f)
            # Extract 'norm_stats' key if present (matches test_imle.py line 149)
            return data.get('norm_stats', data)
    else:
        logging.warning(f"Norm stats file not found: {norm_stats_path}, continuing without normalization")
        return None


def normalize_state(state: np.ndarray, norm_stats: dict | None, model_version: str = "pi05") -> np.ndarray:
    """Normalize state using quantile (pi05) or z-score (pi0) normalization.
    
    Reference: test_imle.py _normalize_state
    """
    if norm_stats and "state" in norm_stats:
        state_dim = state.shape[-1]
        if model_version == "pi05":
            # Slice norm_stats to state dimension (same as JAX)
            q01 = np.array(norm_stats["state"]["q01"])[:state_dim]
            q99 = np.array(norm_stats["state"]["q99"])[:state_dim]
            return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        elif model_version == "pi0":
            state_mean = np.array(norm_stats["state"]["mean"])[:state_dim]
            state_std = np.array(norm_stats["state"]["std"])[:state_dim]
            return (state - state_mean) / (state_std + 1e-6)
    return state


def digitize_state(state_normed: np.ndarray) -> np.ndarray:
    """Discretize normalized state to 256 bins.
    
    Reference: test_imle.py _digitize_state
    """
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    return np.digitize(state_normed, bins=bins) - 1


def unnormalize_actions(actions: np.ndarray, norm_stats: dict | None, target_dim: int = 32, model_version: str = "pi05") -> np.ndarray:
    """Unnormalize actions using quantile (pi05) or z-score (pi0) unnormalization.
    
    Reference: test_imle.py _unnormalize_actions
    """
    if norm_stats and "actions" in norm_stats:
        if model_version == "pi05":
            q01 = np.array(norm_stats["actions"]["q01"])
            q99 = np.array(norm_stats["actions"]["q99"])
            # Pad to target_dim if needed
            if len(q01) < target_dim:
                q01 = np.pad(q01, (0, target_dim - len(q01)), mode='constant', constant_values=0)
            if len(q99) < target_dim:
                q99 = np.pad(q99, (0, target_dim - len(q99)), mode='constant', constant_values=0)
            return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        elif model_version == "pi0":
            actions_mean = np.array(norm_stats["actions"]["mean"])
            actions_std = np.array(norm_stats["actions"]["std"])
            # Pad to target_dim if needed
            if len(actions_mean) < target_dim:
                actions_mean = np.pad(actions_mean, (0, target_dim - len(actions_mean)), mode='constant', constant_values=0)
            if len(actions_std) < target_dim:
                actions_std = np.pad(actions_std, (0, target_dim - len(actions_std)), mode='constant', constant_values=0)
            return actions * (actions_std + 1e-6) + actions_mean
    return actions


def preprocess_image(img: np.ndarray, target_size: int = IMAGE_SIZE) -> np.ndarray:
    """Preprocess image for model input.
    
    Matches test_imle.py preprocessing:
    1. Parse image (handle float/uint8, rearrange if needed)
    2. Resize with padding
    3. Normalize to [-1, 1] range
    
    Args:
        img: Input image as numpy array (H, W, 3) in uint8 [0, 255]
        target_size: Target size for resizing (default: 224)
    
    Returns:
        Preprocessed image as numpy array (H, W, 3) in float32 [-1, 1]
    """
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(img[::-1, ::-1])
    
    # Parse image (handle float/uint8, rearrange if needed)
    # Reference: test_imle.py _parse_image
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    
    # Resize with padding
    from PIL import Image
    pil_image = Image.fromarray(img)
    cur_width, cur_height = pil_image.size
    if cur_width == target_size and cur_height == target_size:
        img_resized = img
    else:
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
        zero_image = Image.new(resized_image.mode, (target_size, target_size), 0)
        pad_height = max(0, int((target_size - resized_height) / 2))
        pad_width = max(0, int((target_size - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        img_resized = np.array(zero_image)
    
    # Normalize to [-1, 1] range (matches test_imle.py _normalize_image)
    img_normalized = img_resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    
    return img_normalized


def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    logging.info(f"Initialized LIBERO environment with task: {task_description}, seed: {seed}")
    return env, task_description


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load checkpoint and normalization stats
    checkpoint = load_checkpoint(args.checkpoint_path)
    norm_stats = load_norm_stats(args.norm_stats_dir)

    # Initialize model
    logging.info("Initializing Pi05Inference model...")
    model = Pi05Inference(
        checkpoint=checkpoint,
        num_views=2,  # LIBERO uses 2 views: agentview and wrist
        chunk_size=args.chunk_size,
        discrete_state_input=args.discrete_state_input,
        max_tokenize_len=args.max_tokenize_len,
        is_imle=args.is_imle,
    )
    logging.info("✓ Model initialized")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Set max steps based on task suite
    if args.task_suite_name == "libero_spatial":
        max_steps = 280  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_inference_time, total_env_step_time, total_time = 0.0, 0.0, 0.0
    total_steps = 0
    total_inference_calls = 0

    time_start = perf_counter()
    task_avg_times = []
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        task_time_start = perf_counter()
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes in task", leave=False):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            pbar_episode = tqdm.tqdm(total=max_steps, desc="Episode progress", leave=False)
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed images
                    agentview_img = preprocess_image(obs["agentview_image"])
                    wrist_img = preprocess_image(obs["robot0_eye_in_hand_image"])

                    # Save original image for replay video (before preprocessing, so we see what robot actually saw)
                    # LIBERO images are already in uint8 [0, 255] format
                    replay_images.append(obs["agentview_image"].astype(np.uint8))

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare state tokens (discretize robot state)
                        # LIBERO state: 7D = eef_pos (3) + axis_angle (3) + gripper (1)
                        robot_state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)
                        
                        # Normalize state using quantile normalization (if norm_stats available)
                        # Reference: test_imle.py _normalize_state
                        state_normalized = normalize_state(robot_state, norm_stats, model_version="pi05")
                        
                        # Discretize state to tokens (0-255 range) using 256 bins
                        # For pi05, discretize 7D state (NOT padded to 32D)
                        # Reference: test_imle.py _digitize_state
                        state_tokens = digitize_state(state_normalized)
                        state_tokens = np.clip(state_tokens, 0, 255).astype(np.int32)

                        # Prepare images as torch tensors
                        # Model expects: (num_views, H, W, C) in float32, values in [-1, 1]
                        # Reference: test_imle.py line 317 - converts to float32, not bfloat16
                        agentview_tensor = torch.from_numpy(agentview_img).to(torch.float32).cuda()
                        wrist_tensor = torch.from_numpy(wrist_img).to(torch.float32).cuda()
                        # Stack: (num_views=2, H=224, W=224, C=3)
                        images_tensor = torch.stack([agentview_tensor, wrist_tensor], dim=0)

                        # Generate noise for diffusion/IMLE
                        noise = torch.randn(args.chunk_size, 32, dtype=torch.float32, device="cuda")

                        # Query model to get action
                        # Reference: test_imle.py line 322-323
                        inference_start = perf_counter()
                        action_output = model.forward(
                            observation_images_normalized=images_tensor,  # (2, 224, 224, 3) in float32, [-1, 1]
                            diffusion_noise=noise,
                            task_prompt=task_description,
                            state_tokens=state_tokens,  # Discretized 7D state tokens
                            num_steps=args.num_steps,
                        )
                        total_inference_time += perf_counter() - inference_start
                        total_inference_calls += 1

                        # Convert action output to numpy and unnormalize
                        # Reference: test_imle.py line 328-329
                        actions_normed = action_output.cpu().float().numpy()  # (chunk_size, 32)
                        actions_unnormed = unnormalize_actions(actions_normed, norm_stats, target_dim=32, model_version="pi05")
                        
                        # LIBERO expects 7D actions: [x, y, z, rx, ry, rz, gripper]
                        # Extract first 7 dimensions from unnormalized actions
                        action_chunk = actions_unnormed[:, :7]
                        
                        assert (
                            len(action_chunk) >= args.chunk_size
                        ), f"We want to replan every {args.chunk_size} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.chunk_size])

                    action = action_plan.popleft()

                    # Execute action in environment
                    env_step_start = perf_counter()
                    obs, reward, done, info = env.step(action.tolist())
                    total_env_step_time += perf_counter() - env_step_start

                    pbar_episode.update(1)
                    total_steps += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if total_inference_calls > 0:
                logging.info(f"# Avg inference per call: {total_inference_time / total_inference_calls:.4f} s")
            if total_steps > 0:
                logging.info(f"# Avg env step per step: {total_env_step_time / total_steps:.4f} s")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        avg_task_time = (perf_counter() - task_time_start) / args.num_trials_per_task
        task_avg_times.append(avg_task_time)

    total_time = perf_counter() - time_start
    logging.info("=" * 60)
    logging.info("FINAL RESULTS")
    logging.info("=" * 60)
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes) * 100:.1f}%")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    logging.info(f"Total time: {total_time:.2f} s")
    logging.info(f"Average time per episode: {total_time / total_episodes:.2f} s")
    if total_inference_calls > 0:
        logging.info(f"Average inference time: {total_inference_time / total_inference_calls * 1000:.2f} ms")
    logging.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    eval_libero(tyro.cli(Args))
