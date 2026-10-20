"""
Validate Triton IMLE server against ground truth DROID RLDS data,
with optional side-by-side JAX model comparison.

Loads validation samples from the DROID RLDS dataset (with O(1) episode-level seeking),
sends each observation to the running Triton policy server via websocket, receives
predicted action chunks, and computes L2 / MSE loss against the ground truth actions.

If --jax-checkpoint-dir is provided, also loads the JAX model locally and runs
inference on the same samples for a side-by-side loss comparison.

Usage:
  1. Start the Triton server:
     python scripts/serve_policy.py --env=DROID policy:triton-checkpoint \
       --policy.config=16_horrizon_05_lr_2_factor_96_batch \
       --policy.pkl-path=/path/to/converted_droid_imle.pkl \
       --policy.jax-checkpoint-dir=/path/to/jax_checkpoint/<step> \
       --policy.sample-kwargs='{"num_steps": 1}'

  2. Run this validation script (connects to server):
     python scripts/validate_triton_droid.py \
       --host=localhost --port=8000 \
       --rlds-data-dir=/path/to/droid_rlds \
       --validation-start-sample=19750000 \
       --validation-samples=1000 \
       --action-chunk-size=16 \
       --jax-config=16_horrizon_05_lr_2_factor_96_batch \
       --jax-checkpoint-dir=/path/to/jax_checkpoint/<step>
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wsc
import tqdm as tqdm_module


# ─────────────────────────────────────────────────────────────────────────────
# RLDS validation dataset loader (self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def load_validation_dataset(
    data_dir: str,
    validation_samples: int = 1_000,
    validation_start_sample: int | None = 19_750_000,
    action_chunk_size: int = 16,
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
    action_space: str = "joint_position",
):
    """Load DROID RLDS validation data with O(1) episode-level seeking.

    Mirrors DroidRldsDataset.create_validation_dataset:
      - TFDS split slicing skips episodes at the file level (zero iteration).
      - Image decoding is deferred until AFTER the residual skip + take.
      - No shuffling, no repeat, no shuffle buffer.
      - Deterministic: always uses exterior_image_1_left, first language instruction.

    Returns:
        A tf.data.Dataset (unbatched) yielding individual numpy samples.
    """
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import openpi.shared.download as download

    tf.config.set_visible_devices([], "GPU")

    TOTAL_FRAMES_APPROX = 20_000_000
    ds_name, version = "droid", "1.0.1"

    builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
    num_episodes = builder.info.splits["train"].num_examples
    avg_frames_per_episode = TOTAL_FRAMES_APPROX / max(num_episodes, 1)

    # ── O(1) episode-level seek via TFDS split slicing ──
    residual_skip = 0
    if validation_start_sample is not None and validation_start_sample > 0:
        target_ep = int(validation_start_sample / avg_frames_per_episode)
        episodes_for_val = int(validation_samples / avg_frames_per_episode) + 1
        safety = max(episodes_for_val, 500)
        start_ep = max(0, target_ep - safety)
        split_str = f"train[{start_ep}:]"
        residual_skip = max(0, int((target_ep - start_ep) * avg_frames_per_episode))
        logging.info(
            f"Validation: TFDS split '{split_str}' — "
            f"seeking past {start_ep}/{num_episodes} episodes at file level, "
            f"~{residual_skip:,} residual frame skip, "
            f"avg {avg_frames_per_episode:.0f} frames/episode"
        )
    else:
        split_str = "train"
        logging.info(f"Validation: loading full split 'train' ({num_episodes} episodes)")

    dataset = dl.DLataset.from_rlds(builder, split=split_str, shuffle=False)

    # Filter out unsuccessful trajectories
    dataset = dataset.filter(
        lambda traj: tf.strings.regex_full_match(
            traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
        )
    )

    # ── Filter dict (same logic as training) ──
    num_parallel_calls = tf.data.AUTOTUNE
    if filter_dict_path is not None:
        cached = download.maybe_download(filter_dict_path)
        with Path(cached).open("r") as f:
            filter_dict = json.load(f)
        logging.info(f"Validation: filter dict with {len(filter_dict)} episodes")

        keys_tensor, values_tensor = [], []
        for episode_key, ranges in tqdm_module.tqdm(filter_dict.items(), desc="Building filter table"):
            for start, end in ranges:
                for t in range(start, end):
                    keys_tensor.append(f"{episode_key}--{t}")
                    values_tensor.append(True)
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
        )
        logging.info("Validation: filter hash table initialized")
    else:
        filter_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
        )

    action_key = "joint_position" if action_space == "joint_position" else "joint_velocity"
    logging.info(f"Validation: using action_dict/{action_key} as ground truth actions")

    def restructure(traj):
        """Reformat observation and action keys — deterministic variant for validation."""
        actions = tf.concat(
            (traj["action_dict"][action_key], traj["action_dict"]["gripper_position"]),
            axis=-1,
        )
        # Deterministic: always use exterior_image_1_left and first language instruction
        exterior_img = traj["observation"]["exterior_image_1_left"]
        wrist_img = traj["observation"]["wrist_image_left"]
        instruction = traj["language_instruction"]

        traj_len = tf.shape(traj["action"])[0]
        indices = tf.as_string(tf.range(traj_len))
        step_id = (
            traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
            + "--"
            + traj["traj_metadata"]["episode_metadata"]["file_path"]
            + "--"
            + indices
        )
        passes_filter = filter_table.lookup(step_id)

        return {
            "actions": actions,
            "observation": {
                "image": exterior_img,
                "wrist_image": wrist_img,
                "joint_position": traj["observation"]["joint_position"],
                "gripper_position": traj["observation"]["gripper_position"],
            },
            "prompt": instruction,
            "passes_filter": passes_filter,
        }

    dataset = dataset.traj_map(restructure, num_parallel_calls)

    def chunk_actions(traj):
        """Splits episode into action chunks."""
        traj_len = tf.shape(traj["actions"])[0]
        indices = tf.broadcast_to(
            tf.range(action_chunk_size)[None], [traj_len, action_chunk_size]
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None], [traj_len, action_chunk_size]
        )
        indices = tf.minimum(indices, traj_len - 1)
        traj["actions"] = tf.gather(traj["actions"], indices)
        return traj

    dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

    # Flatten to individual frames
    dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

    # Apply filter
    dataset = dataset.filter(lambda frame: frame["passes_filter"])

    # Remove filter key
    def remove_passes_filter(frame):
        frame.pop("passes_filter")
        return frame

    dataset = dataset.map(remove_passes_filter)

    # ── Residual skip (images still encoded — fast) ──
    if residual_skip > 0:
        logging.info(f"Validation: residual skip of ~{residual_skip:,} frames (images still encoded)")
        dataset = dataset.skip(residual_skip)

    # Take only the validation window
    logging.info(f"Validation: taking {validation_samples:,} samples")
    dataset = dataset.take(validation_samples)

    # NOW decode images (only on the validation subset)
    def decode_images(frame):
        frame = dict(frame)
        obs = dict(frame["observation"])
        obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
        obs["wrist_image"] = tf.io.decode_image(obs["wrist_image"], expand_animations=False, dtype=tf.uint8)
        frame["observation"] = obs
        return frame

    dataset = dataset.map(decode_images, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# JAX policy loader
# ─────────────────────────────────────────────────────────────────────────────

def load_jax_policy(config_name: str, checkpoint_dir: str, sample_kwargs: dict | None = None):
    """Load a JAX policy using the standard openpi config + checkpoint pipeline.

    Args:
        config_name: Name of the training config (e.g. '16_horrizon_05_lr_2_factor_96_batch').
        checkpoint_dir: Path to the JAX checkpoint directory.
        sample_kwargs: Optional kwargs for sample_actions (e.g. {"num_steps": 1}).

    Returns:
        A Policy object that can be called with policy.infer(obs).
    """
    import dataclasses
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    config = _config.get_config(config_name)

    # If num_steps=1, enable IMLE mode
    if sample_kwargs and sample_kwargs.get("num_steps") == 1:
        model_fields = {f.name for f in dataclasses.fields(config.model)}
        if "use_imle" in model_fields:
            current_use_imle = getattr(config.model, "use_imle", None)
            if current_use_imle is not True:
                logging.info(f"JAX: Overriding use_imle: {current_use_imle} -> True (IMLE mode with num_steps=1)")
                new_model = dataclasses.replace(config.model, use_imle=True)
                config = dataclasses.replace(config, model=new_model)

    logging.info(f"Loading JAX policy: config={config_name}, checkpoint={checkpoint_dir}")
    logging.info("This may take several minutes — loading large checkpoint from NAS...")
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        sample_kwargs=sample_kwargs,
    )
    logging.info("✓ JAX policy loaded successfully.")
    return policy


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

def validate(args):
    if args.flow_matching_mode:
        logging.info(
            "Flow matching mode enabled: forcing JAX num_steps=%d and "
            "reporting MSE_if_delta (delta->absolute diagnostic).",
            args.flow_num_steps,
        )

    def to_abs8(actions: np.ndarray, obs_joint: np.ndarray, *, assume_delta: bool) -> np.ndarray:
        """Convert predictions to absolute 8D DROID action space for fair MSE comparison."""
        out = np.asarray(actions, dtype=np.float64)
        if out.ndim != 2:
            raise ValueError(f"Expected 2D action array [horizon, action_dim], got shape {out.shape}")
        if out.shape[-1] < 8:
            raise ValueError(f"Expected at least 8 action dims, got {out.shape[-1]}")
        out = out[:, :8].copy()
        if assume_delta:
            out[:, :7] += np.asarray(obs_joint, dtype=np.float64)[None, :7]
        return out

    # ── Set random seed for reproducibility ──
    if args.seed is not None:
        np.random.seed(args.seed)
        logging.info(f"Random seed set to {args.seed} for noise generation")
    
    # ── Connect to Triton server ──
    logging.info(f"Connecting to Triton server at {args.host}:{args.port}...")
    client = _wsc.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"Server metadata: {client.get_server_metadata()}")

    # Warm up: send a dummy observation so Triton kernels are compiled
    logging.info("Warming up Triton server with dummy observation...")
    # Send uint8 images — DroidInputs._parse_image() passes uint8 through unchanged,
    # and serve_policy.py sample_actions() normalizes to [-1, 1] before Triton forward
    dummy_obs = {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7).astype(np.float32),
        "observation/gripper_position": np.random.rand(1).astype(np.float32),
        "prompt": "pick up the object",
    }
    dummy_noise = np.random.randn(args.action_chunk_size, 32).astype(np.float32)
    warmup_request = dict(dummy_obs)
    warmup_request["noise"] = dummy_noise
    _ = client.infer(warmup_request)
    logging.info("Triton warmup complete.")

    # ── Optionally load JAX policy ──
    jax_policy = None
    if args.jax_checkpoint_dir:
        # Seed JAX if seed is provided
        if args.seed is not None:
            try:
                import jax
                jax_key = jax.random.PRNGKey(args.seed)
                logging.info(f"JAX random key initialized with seed {args.seed}")
            except ImportError:
                logging.warning("JAX not available, skipping JAX seeding")
        
        sample_kwargs = None
        if args.jax_sample_kwargs:
            try:
                sample_kwargs = json.loads(args.jax_sample_kwargs)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse --jax-sample-kwargs: {e}. Using None.")
        if args.flow_matching_mode:
            sample_kwargs = sample_kwargs or {}
            sample_kwargs["num_steps"] = args.flow_num_steps
            logging.info("Flow matching mode: forcing JAX sample_kwargs['num_steps']=%d", args.flow_num_steps)

        jax_policy = load_jax_policy(
            config_name=args.jax_config,
            checkpoint_dir=args.jax_checkpoint_dir,
            sample_kwargs=sample_kwargs,
        )

        # Warm up JAX model (JIT compilation) with same noise path
        logging.info("Warming up JAX model (JIT compilation)...")
        _ = jax_policy.infer(dummy_obs, noise=dummy_noise)
        logging.info("JAX warmup complete.")

    # ── Load validation data ──
    logging.info("Loading DROID RLDS validation data...")
    dataset = load_validation_dataset(
        data_dir=args.rlds_data_dir,
        validation_samples=args.validation_samples,
        validation_start_sample=args.validation_start_sample,
        action_chunk_size=args.action_chunk_size,
        filter_dict_path=args.filter_dict_path,
        action_space=args.action_space,
    )

    # ── Metrics accumulators ──
    # Triton
    triton_mse_losses = []
    triton_mse_per_step = []
    triton_mse_if_delta_losses = []
    triton_times = []
    # JAX (optional)
    jax_mse_losses = []
    jax_mse_per_step = []
    jax_mse_if_delta_losses = []
    jax_times = []
    # Triton vs JAX action difference (MSE)
    triton_jax_mse_diffs = []

    num_processed = 0

    logging.info("Starting validation loop...")
    pbar = tqdm_module.tqdm(
        dataset.as_numpy_iterator(),
        total=args.validation_samples,
        desc="Validating",
    )

    for sample in pbar:
        # ── Extract raw RLDS fields ──
        obs_image = sample["observation"]["image"]           # (H, W, 3) uint8
        obs_wrist = sample["observation"]["wrist_image"]     # (H, W, 3) uint8
        obs_joint = sample["observation"]["joint_position"]  # (7,)
        obs_grip = sample["observation"]["gripper_position"] # (1,)
        prompt = sample["prompt"]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        # Some DROID episodes have empty language instructions — skip them
        if not prompt or not prompt.strip():
            continue

        gt_actions = sample["actions"]  # (action_chunk_size, 8) — ground truth

        # ── Resize images to 224×224 (matching DROID client) ──
        # Send as uint8 — DroidInputs._parse_image() passes uint8 through unchanged,
        # and serve_policy.py sample_actions() normalizes to [-1, 1] before Triton forward.
        # Do NOT pre-normalize to [-1,1] here; _parse_image would garble it:
        #   (255 * [-1,1]).astype(uint8) wraps negative values.
        obs_image_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs_image, 224, 224)
        )
        obs_wrist_resized = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs_wrist, 224, 224)
        )

        # ── Generate shared noise so Triton & JAX see the same randomness ──
        noise = np.random.randn(args.action_chunk_size, 32).astype(np.float32)

        # ── Format for client / policy ──
        request_data = {
            "observation/exterior_image_1_left": obs_image_resized,
            "observation/wrist_image_left": obs_wrist_resized,
            "observation/joint_position": obs_joint.astype(np.float64),
            "observation/gripper_position": obs_grip.astype(np.float64),
            "prompt": prompt,
        }

        # ── Triton inference (via server) — include noise in request ──
        triton_request = dict(request_data)
        triton_request["noise"] = noise  # server extracts and passes to Policy.infer()
        t0 = time.perf_counter()
        triton_result = client.infer(triton_request)
        triton_dt = time.perf_counter() - t0
        triton_times.append(triton_dt)
        triton_pred = triton_result["actions"]  # (action_horizon, 8)

        # ── JAX inference (local, optional) — same noise for fair comparison ──
        jax_pred = None
        jax_dt = 0.0
        if jax_policy is not None:
            t1 = time.perf_counter()
            jax_result = jax_policy.infer(request_data, noise=noise)
            jax_dt = time.perf_counter() - t1
            jax_times.append(jax_dt)
            jax_pred = jax_result["actions"]  # (action_horizon, 8)

        # ── Compute Triton losses vs ground truth (MSE only, matching training validation) ──
        min_h = min(triton_pred.shape[0], gt_actions.shape[0])
        gt = gt_actions[:min_h, :8].astype(np.float64)
        t_pred_abs = to_abs8(triton_pred[:min_h], obs_joint, assume_delta=False)
        t_pred_if_delta = to_abs8(triton_pred[:min_h], obs_joint, assume_delta=True)

        # Per-step MSE: mean((pred - gt)^2) per timestep
        t_per_step_mse = np.mean((t_pred_abs - gt) ** 2, axis=-1)
        triton_mse_per_step.append(t_per_step_mse)
        # Overall MSE for this sample
        t_mse = np.mean((t_pred_abs - gt) ** 2)
        triton_mse_losses.append(t_mse)
        t_mse_if_delta = np.mean((t_pred_if_delta - gt) ** 2)
        triton_mse_if_delta_losses.append(t_mse_if_delta)

        # ── Compute JAX losses vs ground truth (if available) ──
        if jax_pred is not None:
            min_h_j = min(jax_pred.shape[0], gt_actions.shape[0])
            gt_j = gt_actions[:min_h_j, :8].astype(np.float64)
            j_pred_abs = to_abs8(jax_pred[:min_h_j], obs_joint, assume_delta=False)
            j_pred_if_delta = to_abs8(jax_pred[:min_h_j], obs_joint, assume_delta=True)

            # Per-step MSE: mean((pred - gt)^2) per timestep
            j_per_step_mse = np.mean((j_pred_abs - gt_j) ** 2, axis=-1)
            jax_mse_per_step.append(j_per_step_mse)
            # Overall MSE for this sample
            j_mse = np.mean((j_pred_abs - gt_j) ** 2)
            jax_mse_losses.append(j_mse)
            j_mse_if_delta = np.mean((j_pred_if_delta - gt_j) ** 2)
            jax_mse_if_delta_losses.append(j_mse_if_delta)

            # Triton vs JAX action difference (MSE)
            min_h_tj = min(triton_pred.shape[0], jax_pred.shape[0])
            triton_cmp = to_abs8(triton_pred[:min_h_tj], obs_joint, assume_delta=False)
            jax_cmp = to_abs8(jax_pred[:min_h_tj], obs_joint, assume_delta=False)
            diff_mse = np.mean((triton_cmp - jax_cmp) ** 2)
            triton_jax_mse_diffs.append(diff_mse)

        num_processed += 1

        # ── Update progress bar ──
        running_triton_mse = np.mean(triton_mse_losses)
        postfix = {
            "T_MSE": f"{running_triton_mse:.6f}",
            "T_ms": f"{triton_dt * 1000:.1f}",
        }
        if triton_mse_if_delta_losses:
            postfix["T_ifΔ"] = f"{np.mean(triton_mse_if_delta_losses):.6f}"
        if jax_policy is not None and jax_mse_losses:
            running_jax_mse = np.mean(jax_mse_losses)
            running_diff = np.mean(triton_jax_mse_diffs)
            postfix["J_MSE"] = f"{running_jax_mse:.6f}"
            postfix["T-J"] = f"{running_diff:.6f}"
            postfix["J_ms"] = f"{jax_dt * 1000:.1f}"
            if jax_mse_if_delta_losses:
                postfix["J_ifΔ"] = f"{np.mean(jax_mse_if_delta_losses):.6f}"
        pbar.set_postfix(postfix)

        if num_processed % 5 == 0:
            msg = (
                f"[{num_processed}] "
                f"Triton: MSE={running_triton_mse:.6f} "
                f"sample_MSE={t_mse:.6f} infer={triton_dt*1000:.1f}ms"
            )
            msg += f" sample_MSE_if_delta={t_mse_if_delta:.6f}"
            if jax_policy is not None and jax_mse_losses:
                msg += (
                    f"  |  JAX: MSE={running_jax_mse:.6f} "
                    f"sample_MSE={j_mse:.6f} infer={jax_dt*1000:.1f}ms"
                    f"  |  T-J diff={running_diff:.6f}"
                )
                msg += f" JAX_sample_MSE_if_delta={j_mse_if_delta:.6f}"
            logging.info(msg)

    # ── Final report ──
    if num_processed == 0:
        logging.error("No samples were processed!")
        return

    triton_mse_arr = np.array(triton_mse_losses)
    triton_per_step = np.array(triton_mse_per_step)
    triton_mse_if_delta_arr = np.array(triton_mse_if_delta_losses) if triton_mse_if_delta_losses else None
    triton_inf_arr = np.array(triton_times)

    print("\n" + "=" * 80)
    print(f"VALIDATION RESULTS — {num_processed} samples")
    print(f"  Validation start sample: {args.validation_start_sample}")
    print(f"  Action chunk size:       {args.action_chunk_size}")
    print(f"  Action space (GT):       {args.action_space}")
    print("=" * 80)

    # ── Triton results ──
    print("\n  ┌─── TRITON ────────────────────────────────────────────────────┐")
    print(f"  │  Mean MSE:           {np.mean(triton_mse_arr):.6f}")
    if triton_mse_if_delta_arr is not None:
        print(f"  │  Mean MSE if delta:  {np.mean(triton_mse_if_delta_arr):.6f}")
    print(f"  │  Median MSE:         {np.median(triton_mse_arr):.6f}")
    print(f"  │  Std MSE:            {np.std(triton_mse_arr):.6f}")
    print(f"  │  Min MSE:             {np.min(triton_mse_arr):.6f}")
    print(f"  │  Max MSE:             {np.max(triton_mse_arr):.6f}")
    print(f"  │  Per-step mean MSE (step 0..{triton_per_step.shape[1]-1}):")
    triton_step_means = np.mean(triton_per_step, axis=0)
    for i, v in enumerate(triton_step_means):
        print(f"  │    step {i:2d}: {v:.6f}")
    print(f"  │  Inference latency:")
    print(f"  │    mean:   {np.mean(triton_inf_arr)*1000:.1f} ms")
    print(f"  │    median: {np.median(triton_inf_arr)*1000:.1f} ms")
    print(f"  │    p95:    {np.percentile(triton_inf_arr, 95)*1000:.1f} ms")
    print(f"  │    p99:    {np.percentile(triton_inf_arr, 99)*1000:.1f} ms")
    print(f"  └─────────────────────────────────────────────────────────────────┘")

    # ── JAX results (if available) ──
    results = {
        "num_samples": num_processed,
        "validation_start_sample": args.validation_start_sample,
        "validation_samples_requested": args.validation_samples,
        "action_chunk_size": args.action_chunk_size,
        "action_space": args.action_space,
        "flow_matching_mode": bool(args.flow_matching_mode),
        "flow_num_steps": int(args.flow_num_steps),
        "triton": {
            "mean_mse": float(np.mean(triton_mse_arr)),
            "mean_mse_if_delta": (
                float(np.mean(triton_mse_if_delta_arr)) if triton_mse_if_delta_arr is not None else None
            ),
            "median_mse": float(np.median(triton_mse_arr)),
            "std_mse": float(np.std(triton_mse_arr)),
            "min_mse": float(np.min(triton_mse_arr)),
            "max_mse": float(np.max(triton_mse_arr)),
            "per_step_mean_mse": triton_step_means.tolist(),
            "inference_mean_ms": float(np.mean(triton_inf_arr) * 1000),
            "inference_median_ms": float(np.median(triton_inf_arr) * 1000),
            "inference_p95_ms": float(np.percentile(triton_inf_arr, 95) * 1000),
        },
    }

    if jax_policy is not None and jax_mse_losses:
        jax_mse_arr = np.array(jax_mse_losses)
        jax_per_step = np.array(jax_mse_per_step)
        jax_mse_if_delta_arr = np.array(jax_mse_if_delta_losses) if jax_mse_if_delta_losses else None
        jax_inf_arr = np.array(jax_times)
        diff_arr = np.array(triton_jax_mse_diffs)

        print(f"\n  ┌─── JAX ──────────────────────────────────────────────────────┐")
        print(f"  │  Mean MSE:           {np.mean(jax_mse_arr):.6f}")
        if jax_mse_if_delta_arr is not None:
            print(f"  │  Mean MSE if delta:  {np.mean(jax_mse_if_delta_arr):.6f}")
        print(f"  │  Median MSE:         {np.median(jax_mse_arr):.6f}")
        print(f"  │  Std MSE:            {np.std(jax_mse_arr):.6f}")
        print(f"  │  Min MSE:             {np.min(jax_mse_arr):.6f}")
        print(f"  │  Max MSE:             {np.max(jax_mse_arr):.6f}")
        print(f"  │  Per-step mean MSE (step 0..{jax_per_step.shape[1]-1}):")
        jax_step_means = np.mean(jax_per_step, axis=0)
        for i, v in enumerate(jax_step_means):
            print(f"  │    step {i:2d}: {v:.6f}")
        print(f"  │  Inference latency:")
        print(f"  │    mean:   {np.mean(jax_inf_arr)*1000:.1f} ms")
        print(f"  │    median: {np.median(jax_inf_arr)*1000:.1f} ms")
        print(f"  │    p95:    {np.percentile(jax_inf_arr, 95)*1000:.1f} ms")
        print(f"  │    p99:    {np.percentile(jax_inf_arr, 99)*1000:.1f} ms")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

        print(f"\n  ┌─── TRITON vs JAX (action difference) ──────────────────────┐")
        print(f"  │  Mean MSE diff:      {np.mean(diff_arr):.6f}")
        print(f"  │  Median MSE diff:    {np.median(diff_arr):.6f}")
        print(f"  │  Max MSE diff:        {np.max(diff_arr):.6f}")
        print(f"  │  Triton mean MSE:    {np.mean(triton_mse_arr):.6f}")
        print(f"  │  JAX mean MSE:       {np.mean(jax_mse_arr):.6f}")
        print(f"  │  Δ mean MSE (T-J):    {np.mean(triton_mse_arr) - np.mean(jax_mse_arr):.6f}")
        print(f"  └─────────────────────────────────────────────────────────────────┘")

        results["jax"] = {
            "mean_mse": float(np.mean(jax_mse_arr)),
            "mean_mse_if_delta": (
                float(np.mean(jax_mse_if_delta_arr)) if jax_mse_if_delta_arr is not None else None
            ),
            "median_mse": float(np.median(jax_mse_arr)),
            "std_mse": float(np.std(jax_mse_arr)),
            "min_mse": float(np.min(jax_mse_arr)),
            "max_mse": float(np.max(jax_mse_arr)),
            "per_step_mean_mse": jax_step_means.tolist(),
            "inference_mean_ms": float(np.mean(jax_inf_arr) * 1000),
            "inference_median_ms": float(np.median(jax_inf_arr) * 1000),
            "inference_p95_ms": float(np.percentile(jax_inf_arr, 95) * 1000),
        }
        results["triton_vs_jax"] = {
            "mean_mse_diff": float(np.mean(diff_arr)),
            "median_mse_diff": float(np.median(diff_arr)),
            "max_mse_diff": float(np.max(diff_arr)),
            "delta_mean_mse": float(np.mean(triton_mse_arr) - np.mean(jax_mse_arr)),
        }

    print("=" * 80)

    # ── Save results ──
    if args.output_path:
        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser(
        description="Validate Triton IMLE server against ground truth DROID RLDS data, "
                    "with optional JAX model side-by-side comparison."
    )

    # ── Triton server connection ──
    parser.add_argument(
        "--host", default="localhost",
        help="Triton server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Triton server port (default: 8000)",
    )

    # ── Data ──
    parser.add_argument(
        "--rlds-data-dir", required=True,
        help="Parent directory of the DROID RLDS dataset (the directory that contains `droid/`)",
    )
    parser.add_argument(
        "--validation-samples", type=int, default=1000,
        help="Number of validation samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--validation-start-sample", type=int, default=19_750_000,
        help="Starting sample index — uses O(1) episode-level seeking (default: 19750000)",
    )
    parser.add_argument(
        "--action-chunk-size", type=int, default=16,
        help="Action chunk size / horizon (default: 16)",
    )
    parser.add_argument(
        "--filter-dict-path",
        default="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        help="Path to DROID filter dict (default: gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json)",
    )
    parser.add_argument(
        "--action-space",
        choices=["joint_position", "joint_velocity"],
        default="joint_position",
        help="Which DROID action space to use as ground truth (default: joint_position). "
             "Use joint_velocity for pi05_droid baseline (trained on velocities).",
    )

    # ── JAX comparison (optional) ──
    parser.add_argument(
        "--jax-config", default="16_horrizon_05_lr_2_factor_96_batch",
        help="Training config name for JAX model (default: 16_horrizon_05_lr_2_factor_96_batch)",
    )
    parser.add_argument(
        "--jax-checkpoint-dir", default=None,
        help="JAX checkpoint directory. If provided, loads JAX model for side-by-side comparison.",
    )
    parser.add_argument(
        "--jax-sample-kwargs", default='{"num_steps": 1}',
        help='JSON string of sample_kwargs for JAX model (default: \'{"num_steps": 1}\')',
    )
    parser.add_argument(
        "--flow-matching-mode",
        action="store_true",
        help=(
            "Enable flow matching eval mode. Forces JAX sample num_steps to --flow-num-steps "
            "and logs MSE_if_delta (convert predicted joint deltas to absolute before MSE)."
        ),
    )
    parser.add_argument(
        "--flow-num-steps", type=int, default=10,
        help="Number of denoising/integration steps for flow matching mode (default: 10).",
    )

    # ── Output ──
    parser.add_argument(
        "--output-path", default=None,
        help="Path to save JSON results (optional). Example: data/validation_results.json",
    )
    
    # ── Reproducibility ──
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for noise generation (for reproducibility). If not provided, uses random noise each run.",
    )

    args = parser.parse_args()
    validate(args)
