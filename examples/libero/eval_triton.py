#!/usr/bin/env python3
"""
LIBERO evaluation using Triton-based Pi05Inference.

This script evaluates a pi05 model on LIBERO tasks using the direct Triton inference
implementation instead of a websocket server.

Usage:
    python examples/libero/eval_triton.py \
        --checkpoint_path /path/to/pi05_libero_flow.pkl \
        --task_suite_name libero_spatial \
        --num_trials_per_task 10

    # To evaluate all LIBERO suites:
    for suite in libero_spatial libero_object libero_goal libero_10; do
        python examples/libero/eval_triton.py \
            --checkpoint_path /path/to/pi05_libero_flow.pkl \
            --task_suite_name $suite
    done

Note: norm_stats are automatically loaded from checkpoint_dir/assets/physical-intelligence/libero/norm_stats.json
      (consistent with serve_policy.py). No need to specify --norm_stats_dir.
"""

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import pickle
import sys

import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
import tyro

from PIL import Image
from time import perf_counter

# ---------------------------------------------------------------------------
# Add the realtime-vla directory to the Python path so we can import pi05_infer
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REALTIME_VLA_DIR = _SCRIPT_DIR.parent.parent / "realtime-vla"
if not _REALTIME_VLA_DIR.exists():
    # Fallback: try sibling of the OpenPiMLE repo
    _REALTIME_VLA_DIR = _SCRIPT_DIR.parent.parent.parent / "realtime-vla"
if _REALTIME_VLA_DIR.exists():
    sys.path.insert(0, str(_REALTIME_VLA_DIR))

# Add third_party/libero to path as fallback if libero is not installed as a package
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_LIBERO_THIRD_PARTY = _PROJECT_ROOT / "third_party" / "libero"
if _LIBERO_THIRD_PARTY.exists():
    sys.path.insert(0, str(_LIBERO_THIRD_PARTY))

try:
    from pi05_infer import Pi05Inference
except ImportError as e:
    raise ImportError(
        f"Failed to import pi05_infer. Make sure the realtime-vla directory exists at:\n"
        f"  {_REALTIME_VLA_DIR}\n"
        f"Error: {e}"
    )

try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    raise ImportError(
        f"Failed to import libero. Please install it, e.g.:\n"
        f"  pip install -e {_PROJECT_ROOT / 'third_party' / 'libero'}\n"
        f"Original error: {e}"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256   # resolution used to render training data
IMAGE_SIZE = 224               # model input resolution
LIBERO_STATE_DIM = 8           # eef_pos(3) + axis_angle(3) + gripper_qpos(2)
LIBERO_ACTION_DIM = 7          # x, y, z, rx, ry, rz, gripper

# Tokenizer for PaliGemma (used by Pi05 for discrete state input)
_DEFAULT_TOKENIZER_PATH = "google/paligemma-3b-pt-224"  # HuggingFace model id or a local snapshot directory


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Args:
    # --- Model checkpoint parameters ---
    checkpoint_path: str         # Converted .pkl checkpoint (see realtime-vla/convert_from_jax_pi05.py)
    tokenizer_path: str = _DEFAULT_TOKENIZER_PATH
    chunk_size: int = 10         # Must match config.model.action_horizon (pi05_libero = 10)
    replan_steps: int = 10       # How many predicted actions to execute before replanning
    num_views: int = 2           # Number of camera views (LIBERO: agentview + wrist = 2)
    discrete_state_input: bool = False  # Must match pi05_libero config (prompt via language_embeds)
    # norm_stats will be loaded from checkpoint_dir/assets/physical-intelligence/libero/norm_stats.json

    # --- LIBERO environment parameters ---
    task_suite_name: str = "libero_spatial"  # libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10     # Steps to wait for objects to settle in sim
    num_trials_per_task: int = 10  # Rollouts per task

    # --- Output ---
    video_out_path: str = "data/libero/videos"
    seed: int = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> dict:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logging.info(f"Loading checkpoint from {path} ...")
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    logging.info(f"  Checkpoint keys: {list(ckpt.keys())}")
    return ckpt


def load_norm_stats(checkpoint_path: str) -> dict | None:
    """Load normalization stats from checkpoint directory (consistent with serve_policy.py).
    
    Expects: checkpoint_dir/assets/physical-intelligence/libero/norm_stats.json
    This matches the structure used by serve_policy.py and the JAX checkpoint format.
    """
    checkpoint_dir = pathlib.Path(checkpoint_path).parent
    norm_stats_path = checkpoint_dir / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    
    if not norm_stats_path.exists():
        logging.warning(f"norm_stats.json not found at {norm_stats_path}; continuing without normalisation.")
        return None
    
    logging.info(f"Loading norm_stats from {norm_stats_path}")
    with open(norm_stats_path) as f:
        data = json.load(f)
    return data.get("norm_stats", data)


def normalize_state(state: np.ndarray, norm_stats: dict | None) -> np.ndarray:
    """Quantile-normalise state to [-1, 1] using q01 / q99."""
    if norm_stats is None or "state" not in norm_stats:
        return state
    dim = state.shape[-1]
    q01 = np.array(norm_stats["state"]["q01"])[:dim]
    q99 = np.array(norm_stats["state"]["q99"])[:dim]
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def digitize_state(state_normed: np.ndarray) -> np.ndarray:
    """Discretise normalised state into 256 bins (values 0..255)."""
    bins = np.linspace(-1, 1, 257)[:-1]
    tokens = np.digitize(state_normed, bins=bins) - 1
    return np.clip(tokens, 0, 255).astype(np.int32)


def unnormalize_actions(actions: np.ndarray, norm_stats: dict | None, target_dim: int = 32) -> np.ndarray:
    """Undo quantile normalisation of actions."""
    if norm_stats is None or "actions" not in norm_stats:
        return actions
    q01 = np.array(norm_stats["actions"]["q01"])
    q99 = np.array(norm_stats["actions"]["q99"])
    # Pad to target_dim so we can broadcast against the 32-D model output
    if len(q01) < target_dim:
        q01 = np.pad(q01, (0, target_dim - len(q01)))
    if len(q99) < target_dim:
        q99 = np.pad(q99, (0, target_dim - len(q99)))
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def preprocess_image(img: np.ndarray, target_size: int = IMAGE_SIZE, rotate180: bool = False) -> np.ndarray:
    """Preprocess a LIBERO image for the model.

    1. (Optional) 180° rotation – needed for live LIBERO simulation images
       because the simulator renders them upside-down (matches main.py / JAX eval).
    2. Resize with padding to ``target_size x target_size``
    3. Normalise to [-1, 1]

    Returns (H, W, 3) float32 array in [-1, 1].
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)

    if rotate180:
        img = np.ascontiguousarray(img[::-1, ::-1])  # exact same as main.py
    pil_img = Image.fromarray(img)
    cur_w, cur_h = pil_img.size
    if cur_w != target_size or cur_h != target_size:
        ratio = max(cur_w / target_size, cur_h / target_size)
        rh, rw = int(cur_h / ratio), int(cur_w / ratio)
        resized = pil_img.resize((rw, rh), resample=Image.BILINEAR)
        canvas = Image.new(resized.mode, (target_size, target_size), 0)
        pad_h = max(0, (target_size - rh) // 2)
        pad_w = max(0, (target_size - rw) // 2)
        canvas.paste(resized, (pad_w, pad_h))
        img = np.array(canvas)
    else:
        img = np.array(pil_img)

    return img.astype(np.float32) / 255.0 * 2.0 - 1.0


def _quat2axisangle(quat):
    """Quaternion → axis-angle (from robosuite)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def re_encode_prompt(checkpoint: dict, prompt_text: str, tokenizer_path: str) -> dict:
    """Re-encode a task prompt into language_embeds using the embedding weight.

    This replaces the baked-in language_embeds so the model uses the correct
    task description.  Matches the encoding from convert_from_jax_pi05.py.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    emb_w = checkpoint["embedding_weight"].to(dtype=torch.bfloat16, device="cuda")
    num_emb, emb_dim = emb_w.shape
    language_embedding = nn.Embedding(num_emb, emb_dim).bfloat16().cuda()
    with torch.no_grad():
        language_embedding.weight.copy_(emb_w)
    prompt_str = [prompt_text.strip().replace("_", " ") + "\n"]
    tokens = tokenizer(prompt_str, max_length=48, return_tensors="pt")["input_ids"]
    tokens = tokens.to(device="cuda").squeeze(0)
    language_embeds = language_embedding(tokens) * (emb_dim ** 0.5)
    checkpoint["language_embeds"] = language_embeds.cpu()
    return checkpoint


def pad_language_embeds(checkpoint: dict, min_len: int = 48) -> dict:
    """Pad language_embeds to at least ``min_len`` tokens so the encoder buffer
    is large enough for any LIBERO task prompt."""
    cur = checkpoint["language_embeds"]
    if cur.shape[0] >= min_len:
        return checkpoint
    pad = torch.zeros(min_len - cur.shape[0], cur.shape[1], dtype=cur.dtype)
    checkpoint["language_embeds"] = torch.cat([cur, pad], dim=0)
    return checkpoint


def update_model_prompt(
    model: "Pi05Inference",
    prompt_text: str,
    tokenizer_path: str,
) -> None:
    """Re-encode a new task prompt and hot-swap it into the model's weights.

    Works with discrete_state_input=False where forward() reads
    model.weights['language_embeds'] directly.

    Requires model._embedding_weight to be set (see eval_libero()).
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    emb_w = model._embedding_weight                               # (V, 2048) bf16 cuda
    num_emb, emb_dim = emb_w.shape
    language_embedding = nn.Embedding(num_emb, emb_dim).bfloat16().cuda()
    with torch.no_grad():
        language_embedding.weight.copy_(emb_w)
    prompt_str = [prompt_text.strip().replace("_", " ") + "\n"]
    tokens = tokenizer(prompt_str, max_length=48, return_tensors="pt")["input_ids"]
    tokens = tokens.to(device="cuda").squeeze(0)
    new_embeds = language_embedding(tokens) * (emb_dim ** 0.5)    # (N, 2048) bf16
    model.weights["language_embeds"] = new_embeds


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    logging.info(f"Env initialised – task: {task_description}, seed: {seed}")
    return env, task_description


def _max_steps_for_suite(name: str) -> int:
    return {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[name]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Load checkpoint & norm stats ---
    checkpoint = load_checkpoint(args.checkpoint_path)
    norm_stats = load_norm_stats(args.checkpoint_path)  # Load from checkpoint directory's assets folder
    if norm_stats is None:
        logging.warning("⚠️  norm_stats is None — actions will NOT be unnormalized!")
    else:
        logging.info(f"norm_stats loaded: keys={list(norm_stats.keys())}")
        if "actions" in norm_stats:
            q01 = np.array(norm_stats["actions"]["q01"])
            q99 = np.array(norm_stats["actions"]["q99"])
            logging.info(f"  actions q01[:7]={q01[:7]}")
            logging.info(f"  actions q99[:7]={q99[:7]}")

    # --- Build the Triton model ---
    logging.info("Initialising Pi05Inference …")
    # discrete_state_input=False → the model reads language_embeds from
    # model.weights directly.  We encode a dummy prompt and pad the buffer
    # so it's large enough for any LIBERO task description (up to 48 tokens).
    checkpoint = re_encode_prompt(
        checkpoint,
        "pick up the black bowl on the stove and place it on the table",
        args.tokenizer_path,
    )
    checkpoint = pad_language_embeds(checkpoint, min_len=48)

    # Save embedding_weight before Pi05Inference consumes the checkpoint
    # (it skips embedding_weight when copying into model.weights)
    embedding_weight = checkpoint["embedding_weight"].to(dtype=torch.bfloat16, device="cuda")

    model = Pi05Inference(
        checkpoint=checkpoint,
        num_views=args.num_views,
        chunk_size=args.chunk_size,
        discrete_state_input=args.discrete_state_input,
    )
    model._embedding_weight = embedding_weight  # stash for update_model_prompt
    logging.info("Model ready.")

    # --- LIBERO task suite ---
    bench = benchmark.get_benchmark_dict()
    suite = bench[args.task_suite_name]()
    n_tasks = suite.n_tasks
    max_steps = _max_steps_for_suite(args.task_suite_name)
    logging.info(f"Suite: {args.task_suite_name} ({n_tasks} tasks, max {max_steps} steps)")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # --- Evaluation bookkeeping ---
    total_episodes = 0
    total_successes = 0
    total_inference_time = 0.0
    total_env_step_time = 0.0
    total_steps = 0
    total_inference_calls = 0
    time_start = perf_counter()

    for task_id in tqdm.tqdm(range(n_tasks), desc="Tasks"):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Hot-swap the task prompt into the model (updates language_embeds)
        update_model_prompt(model, task_description, args.tokenizer_path)
        logging.info(f"  [DIAG] After update_model_prompt: language_embeds shape={model.weights['language_embeds'].shape}")

        task_episodes = 0
        task_successes = 0

        for ep_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            obs = env.set_init_state(init_states[ep_idx])

            action_plan: collections.deque = collections.deque()
            replay_images: list[np.ndarray] = []
            t = 0
            done = False

            pbar = tqdm.tqdm(total=max_steps, desc="Steps", leave=False)
            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to settle
                    if t < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # --- Preprocess images ---
                    # 180° rotation needed: LIBERO sim renders upside-down
                    # (matches main.py / JAX eval which also rotates 180°)
                    agentview_raw = np.ascontiguousarray(obs["agentview_image"])
                    wrist_raw = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
                    agentview = preprocess_image(agentview_raw, rotate180=True)
                    wrist = preprocess_image(wrist_raw, rotate180=True)

                    # Save raw image for video
                    replay_images.append(obs["agentview_image"].astype(np.uint8))

                    if not action_plan:
                        # Images → tensor (num_views, 224, 224, 3) float32
                        imgs = torch.stack([
                            torch.from_numpy(agentview),
                            torch.from_numpy(wrist),
                        ], dim=0).to(torch.float32).cuda()

                        # Noise for flow-matching denoising
                        noise = torch.randn(args.chunk_size, 32, dtype=torch.float32, device="cuda")

                        # --- Inference ---
                        # discrete_state_input=False → prompt is baked into
                        # language_embeds (set per task via update_model_prompt).
                        t_inf = perf_counter()
                        raw_actions = model.forward(
                            observation_images_normalized=imgs,
                            diffusion_noise=noise,
                        )
                        total_inference_time += perf_counter() - t_inf
                        total_inference_calls += 1

                        # Post-process: unnormalise, take first 7 dims
                        actions_np = raw_actions.cpu().float().numpy()  # (chunk_size, 32)
                        actions_unnorm = unnormalize_actions(actions_np, norm_stats, target_dim=32)
                        action_chunk = actions_unnorm[:, :LIBERO_ACTION_DIM]  # (chunk_size, 7)

                        # --- Diagnostic prints (first inference only) ---
                        if total_inference_calls == 1:
                            logging.info(f"  [DIAG] imgs range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}], shape={tuple(imgs.shape)}, dtype={imgs.dtype}")
                            logging.info(f"  [DIAG] raw_actions[0,:7]:  {actions_np[0,:7]}")
                            logging.info(f"  [DIAG] unnorm_actions[0,:7]: {action_chunk[0]}")
                            logging.info(f"  [DIAG] prompt_len (language_embeds): {model.weights['language_embeds'].shape}")
                            logging.info(f"  [DIAG] task: {task_description}")

                        # Only execute `replan_steps` actions from the chunk
                        n_use = min(args.replan_steps, len(action_chunk))
                        action_plan.extend(action_chunk[:n_use])

                    # --- Execute one action ---
                    action = action_plan.popleft()
                    t_env = perf_counter()
                    obs, reward, done, info = env.step(action.tolist())
                    total_env_step_time += perf_counter() - t_env

                    pbar.update(1)
                    total_steps += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception: {e}")
                    import traceback; traceback.print_exc()
                    break

            pbar.close()
            task_episodes += 1
            total_episodes += 1

            # Save video
            suffix = "success" if done else "failure"
            tag = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{tag}_ep{ep_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            logging.info(f"Done={done}  |  cumulative: {total_successes}/{total_episodes} "
                         f"({total_successes / total_episodes * 100:.1f}%)")
            if total_inference_calls:
                logging.info(f"  avg inference: {total_inference_time / total_inference_calls * 1000:.1f} ms")

        logging.info(f"Task success: {task_successes}/{task_episodes}")

    # --- Final summary ---
    elapsed = perf_counter() - time_start
    logging.info("=" * 60)
    logging.info("FINAL RESULTS")
    logging.info("=" * 60)
    logging.info(f"Suite           : {args.task_suite_name}")
    logging.info(f"Checkpoint      : {args.checkpoint_path}")
    logging.info(f"Success rate    : {total_successes}/{total_episodes} "
                 f"({total_successes / total_episodes * 100:.1f}%)")
    logging.info(f"Total time      : {elapsed:.1f} s")
    if total_inference_calls:
        logging.info(f"Avg inference   : {total_inference_time / total_inference_calls * 1000:.1f} ms")
    if total_steps:
        logging.info(f"Avg env step    : {total_env_step_time / total_steps * 1000:.1f} ms")
    logging.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eval_libero(tyro.cli(Args))
