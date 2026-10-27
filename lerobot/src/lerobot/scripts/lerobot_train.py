#!/usr/bin/env python

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
import sys
from pathlib import Path

# Ensure we use the local source code instead of any installed package
# Get the directory containing this script and add the src directory to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # Go up from scripts/lerobot/src/lerobot/scripts/
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.utils.constants import ACTION
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def compute_validation_l2_loss(
    policy: PreTrainedPolicy,
    val_dataloader: torch.utils.data.DataLoader,
    preprocessor,
    accelerator: Accelerator,
    max_batches: int = 50,
) -> dict:
    """
    Compute L2 loss between predicted and ground truth actions on validation set.
    
    Args:
        policy: The policy model.
        val_dataloader: Validation dataloader.
        preprocessor: Preprocessor for the batch.
        accelerator: The Accelerator instance.
        max_batches: Maximum number of batches to validate on (for efficiency).
    
    Returns:
        Dictionary with validation metrics.
    """
    policy.eval()
    total_l2_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= max_batches:
                break
            
            batch = preprocessor(batch)
            
            # Get ground truth actions
            gt_actions = batch[ACTION]  # Shape: (B, chunk_size, action_dim)
            
            # Get predicted actions using inference path
            with accelerator.autocast():
                unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
                pred_actions = unwrapped_policy.predict_action_chunk(batch)
            
            # Ensure same shape for comparison
            if gt_actions.dim() == 2:
                pred_actions = pred_actions[:, 0, :]
            elif gt_actions.dim() == 3:
                min_chunk_size = min(pred_actions.shape[1], gt_actions.shape[1])
                pred_actions = pred_actions[:, :min_chunk_size, :].contiguous()
                gt_actions = gt_actions[:, :min_chunk_size, :].contiguous()
            
            # Compute L2 loss (MSE)
            l2_loss = torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='sum')
            total_l2_loss += l2_loss.item()
            total_samples += gt_actions.numel()
            
            # Clear memory periodically to prevent accumulation
            if (i + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    policy.train()
    
    mean_l2_loss = total_l2_loss / total_samples if total_samples > 0 else 0.0
    
    return {
        "val_l2_loss": mean_l2_loss,
        "val_samples": total_samples,
    }


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    # Update EMA model if enabled
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update_ema"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update_ema()

    train_metrics.loss = loss.item()

    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    if is_main_process:
        logging.info("[DEBUG] About to initialize wandb...")

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if is_main_process:
        logging.info("[DEBUG] Wandb init done, setting seed...")

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    if is_main_process:
        logging.info("[DEBUG] Seed set, about to create dataset...")

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)
        logging.info("[DEBUG] Dataset created on main, waiting for others...")

    accelerator.wait_for_everyone()
    if is_main_process:
        logging.info("[DEBUG] All processes synced after main dataset creation")

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)
    
    if is_main_process:
        logging.info("[DEBUG] All datasets loaded, continuing...")

    # Create validation split if val_freq > 0 and val_split > 0
    val_dataloader = None
    val_episode_indices = None
    train_episode_indices = None
    if cfg.val_freq > 0 and cfg.val_split > 0:
        if is_main_process:
            logging.info(f"Setting up validation with {cfg.val_split*100:.1f}% of episodes")
        
        # Split episodes: use last val_split fraction for validation
        total_episodes = dataset.num_episodes
        num_val_episodes = max(1, int(total_episodes * cfg.val_split))
        
        # Get all episode indices
        all_episode_indices = list(range(total_episodes))
        val_episode_indices = all_episode_indices[-num_val_episodes:]
        train_episode_indices = all_episode_indices[:-num_val_episodes]
        
        if is_main_process:
            logging.info(f"Train episodes: {len(train_episode_indices)}, Val episodes: {len(val_episode_indices)}")

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    # Only create eval_env on main process to avoid MuJoCo rendering conflicts in multi-GPU training
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
            eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
    
    # Wait for main process to finish creating eval_env before continuing
    accelerator.wait_for_everyone()
    if is_main_process:
        logging.info("[DEBUG] Synced after eval_env creation")

    if is_main_process:
        logging.info("Creating policy")
    # Update config.device to use accelerator's device to respect CUDA_VISIBLE_DEVICES
    cfg.policy.device = str(device)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )
    if is_main_process:
        logging.info("[DEBUG] Policy created, waiting for all processes...")

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()
    if is_main_process:
        logging.info("[DEBUG] All processes have policy")

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # Initialize environment processors (only used on main process during eval)
    env_preprocessor, env_postprocessor = None, None
    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    # Use train_episode_indices if validation split is enabled
    train_episodes_for_sampler = train_episode_indices if train_episode_indices is not None else dataset.episodes
    
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=train_episodes_for_sampler,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        # Use EpisodeAwareSampler even without drop_n_last_frames when we have a validation split
        if train_episode_indices is not None:
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.meta.episodes["dataset_from_index"],
                dataset.meta.episodes["dataset_to_index"],
                episode_indices_to_use=train_episode_indices,
                shuffle=True,
            )
        else:
            shuffle = True
            sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Create validation dataloader using EpisodeAwareSampler for validation episodes
    if val_episode_indices is not None:
        val_batch_size = cfg.val_batch_size if cfg.val_batch_size is not None else cfg.batch_size
        
        # Create sampler for validation episodes (no shuffle, no frame dropping)
        val_sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_episode_indices,
            shuffle=False,  # No shuffling for validation
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            dataset,  # Use the same dataset, sampler handles episode filtering
            num_workers=cfg.num_workers,
            batch_size=val_batch_size,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    # Prepare everything with accelerator
    if is_main_process:
        logging.info("[DEBUG] About to accelerator.prepare()...")
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    if is_main_process:
        logging.info("[DEBUG] accelerator.prepare() done!")
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),

        # IMLE distance stats
        "dist_mean": AverageMeter("dist_mean", ":.4f"),
        "best_dist_mean_nograd": AverageMeter("best_nograd", ":.4f"),
        "best_dist_mean_grad": AverageMeter("best_grad", ":.4f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        if output_dict:
            for k, v in output_dict.items():
                # Set IMLE stats (dist_*, weight_*, softmax_temp)
                if k.startswith("dist_") or k.startswith("weight_") or k == "softmax_temp":
                    setattr(train_tracker, k, float(v))

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        is_val_step = cfg.val_freq > 0 and step % cfg.val_freq == 0 and val_dataloader is not None

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                    # Log IMLE stats keys to verify they're included
                    imle_keys = [k for k in wandb_log_dict.keys() if k.startswith('dist_') or k.startswith('best_dist_') or k.startswith('weight_') or k == 'softmax_temp']
                    if imle_keys:
                        logging.info(f"IMLE stats being logged: {imle_keys}")
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        # Validation step: compute L2 loss between predicted and ground truth actions
        if is_val_step:
            if is_main_process:
                logging.info(f"Validation at step {step}")
                val_start_time = time.perf_counter()
                
                val_metrics = compute_validation_l2_loss(
                    policy=policy,
                    val_dataloader=val_dataloader,
                    preprocessor=preprocessor,
                    accelerator=accelerator,
                    max_batches=50,  # Limit validation batches for efficiency
                )
                val_time = time.perf_counter() - val_start_time
                val_metrics["val_time_s"] = val_time
                
                logging.info(
                    f"Validation L2 Loss: {val_metrics['val_l2_loss']:.6f} "
                    f"(samples: {val_metrics['val_samples']}, time: {val_time:.2f}s)"
                )
                
                if wandb_logger:
                    wandb_logger.log_dict(val_metrics, step, mode="val")

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                
                # Get policy for evaluation
                eval_policy = accelerator.unwrap_model(policy)
                
                # Temporarily merge LoRA in EMA model for faster evaluation
                # Using PEFT's reversible merge (merge_adapter/unmerge_adapter) instead of
                # permanent merge_and_unload. This allows:
                # 1. Fast inference during eval (no LoRA forward pass overhead)
                # 2. Unmerge after eval so EMA updates continue to work
                # 3. Checkpoints save unmerged state (with LoRA adapters)
                ema_was_merged = False
                if hasattr(eval_policy, 'model_ema') and eval_policy.model_ema is not None:
                    use_lora = getattr(eval_policy.config, 'use_lora', False)
                    if use_lora:
                        try:
                            from peft import PeftModel
                            pwe = eval_policy.model_ema.paligemma_with_expert
                            # Check if paligemma is a PeftModel (full VLM LoRA)
                            if isinstance(pwe.paligemma, PeftModel):
                                pwe.paligemma.merge_adapter()
                                ema_was_merged = True
                                logging.info("✓ Temporarily merged LoRA in EMA model for faster eval")
                            # Check if language_model is a PeftModel (language-only LoRA)
                            elif hasattr(pwe.paligemma, 'language_model') and isinstance(pwe.paligemma.language_model, PeftModel):
                                pwe.paligemma.language_model.merge_adapter()
                                ema_was_merged = True
                                logging.info("✓ Temporarily merged LoRA in EMA language model for faster eval")
                        except (ImportError, AttributeError) as e:
                            logging.debug(f"Could not merge LoRA for eval: {e}")
                    
                    logging.info("Using EMA model for evaluation")
                
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=eval_policy,
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")
                
                # Unmerge LoRA after evaluation to restore EMA model for continued training
                if ema_was_merged:
                    try:
                        from peft import PeftModel
                        pwe = eval_policy.model_ema.paligemma_with_expert
                        if isinstance(pwe.paligemma, PeftModel):
                            pwe.paligemma.unmerge_adapter()
                            logging.info("✓ Unmerged LoRA in EMA model, ready for continued training")
                        elif hasattr(pwe.paligemma, 'language_model') and isinstance(pwe.paligemma.language_model, PeftModel):
                            pwe.paligemma.language_model.unmerge_adapter()
                            logging.info("✓ Unmerged LoRA in EMA language model, ready for continued training")
                    except Exception as e:
                        logging.warning(f"Could not unmerge LoRA after eval: {e}")

            accelerator.wait_for_everyone()

    if is_main_process:
        if eval_env:
            close_envs(eval_env)
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()
