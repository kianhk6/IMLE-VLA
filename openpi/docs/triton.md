# Fast inference with Triton (55 Hz)

Run all commands from the `openpi/` folder.

The JAX server runs π₀.₅ flow matching at about 15 Hz. The 55 Hz number in the paper comes from running the IMLE head
with hand-written Triton kernels and CUDA graphs (`realtime-vla-imle/`). The weights are
exported once from the JAX checkpoint to a `.pkl`, then served by the same `scripts/serve_policy.py` so the clients do not
change. We ship each exported model as a self-contained **Triton package**:

```
<package>/
├── <name>.pkl                       # all weights for the realtime-vla-imle engine (≈6.7 GB, bfloat16)
├── assets/<asset_id>/norm_stats.json   # normalization statistics of the training run (asset_id: droid, physical-intelligence/libero)
├── paligemma_tokenizer.model        # optional: lets the server start offline
└── engine/                          # optional: provenance copy of realtime-vla-imle/*.py
```

### 1. (LoRA fine-tunes only) merge the LoRA factors

The engine runs plain fused einsums and the converter reads only the base weights, so LoRA factors must be folded
first. The paper's DROID and LIBERO configs train the full action expert and skip this step.

```bash
JAX_PLATFORMS=cpu python scripts/merge_all_lora.py /path/to/ckpt/<step>/params /path/to/ckpt/<step>_merged/params
cp -r /path/to/ckpt/<step>/assets /path/to/ckpt/<step>_merged/     # norm stats travel with the checkpoint
```

The script folds both LoRA families (VLM rank 16, action expert rank 32) and runs a functional check on every folded
site. It handles the head-sum trap of `attn_vec_einsum`, where the LoRA B factor must be summed over heads; a naive
per-head fold is silently wrong there.

### 2. Convert the JAX checkpoint to a `.pkl`

```bash
PKG=/path/to/imle_droid_triton
CKPT=/path/to/checkpoints/16_horrizon_05_lr_2_factor_96_batch/16_horrizon_05_lr_2_factor_96_batch/<step>
mkdir -p $PKG/assets/droid $PKG/engine

CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu python realtime-vla/convert_from_jax_pi05.py \
    --jax_path $CKPT \
    --output $PKG/imle_droid_triton.pkl \
    --prompt "<the task prompt>" \
    --tokenizer_path google/paligemma-3b-pt-224

cp $CKPT/assets/droid/norm_stats.json $PKG/assets/droid/
cp realtime-vla-imle/*.py $PKG/engine/
cp ~/.cache/openpi/big_vision/paligemma_tokenizer.model $PKG/
```

- `--jax_path` is the step directory that contains `params/`.
- The converter needs a GPU (`torch` tensors are built on CUDA). `JAX_PLATFORMS=cpu` keeps JAX off the GPU during the export.
- `--prompt` is stored in the `.pkl` as pre-computed language embeddings. π₀.₅ tokenizes the real prompt and the
  discretized state at every call (`discrete_state_input=True`), so the stored prompt is only a fallback; pass the task prompt.
- `--tokenizer_path` is a Hugging Face id or a local snapshot. `google/paligemma-3b-pt-224` is gated: accept the terms and
  run `huggingface-cli login` first.
- For LIBERO use the LIBERO checkpoint, `assets/physical-intelligence/libero/norm_stats.json`, and `--env=LIBERO` below.

### 3. Serve the package

```bash
scripts/serve_triton.sh $PKG 16_horrizon_05_lr_2_factor_96_batch 0 8000 DROID
# or the released package: scripts/serve_triton.sh ./ckpt/droid/triton 16_horrizon_05_lr_2_factor_96_batch 0 8000 DROID
```

This is equivalent to:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/serve_policy.py --env=DROID --port=8000 policy:triton-checkpoint \
    --policy.config=16_horrizon_05_lr_2_factor_96_batch \
    --policy.pkl-path=$PKG/imle_droid_triton.pkl \
    --policy.jax-checkpoint-dir=$PKG \
    --policy.sample-kwargs='{"num_steps": 1}'
```

- `--policy.config` supplies only the model geometry (`action_horizon`, `max_token_len`, `discrete_state_input`) and the
  data transforms. It must match the checkpoint the `.pkl` came from.
- `--policy.jax-checkpoint-dir` is where the server reads `assets/<asset_id>/norm_stats.json`. Wrong statistics degrade the
  policy silently, so keep the package's own copy.
- `--env` sets the camera count and state layout (`DROID`: 8-D state, `LIBERO`: 9-D state, both 2 views).
- `--default-prompt "<prompt>"` (before `policy:`) supplies a prompt when the client does not send one.
- The Triton engine needs a visible GPU even to import (Triton autotuning binds to the CUDA driver), so the server, the
  converter and the checks below all run on the GPU machine.
- The first request captures the CUDA graphs and takes a few seconds. Then the engine runs at about 18 ms per forward pass
  on an L40S for two 224×224 views and a short prompt. The clients above connect unchanged.

### 4. Validate the served policy against ground truth

Before a policy goes on the robot, check it offline against recorded data. `scripts/validate_triton_droid.py` is the tool
we used to accept every package in the paper. It works against **any** running policy server, JAX or Triton:

1. It loads validation samples from the DROID RLDS dataset (episode-level seeking, so it starts in seconds even deep into
   the dataset). Each sample is the real camera images, the proprioceptive state, the prompt, and the ground-truth action chunk.
2. It sends each observation to the server exactly like the robot client does and receives the predicted action chunk.
3. It reports the MSE between predicted and ground-truth actions (overall, per chunk step) and the server-side latency.
4. With `--jax-checkpoint-dir` it also loads the JAX model in-process and runs it on the same samples with the same noise
   (`--seed`), then prints the Triton-vs-JAX action MSE next to both ground-truth numbers.

Terminal 1, the policy under test. Either the Triton package:

```bash
scripts/serve_triton.sh $PKG 16_horrizon_05_lr_2_factor_96_batch 0 8000 DROID
```

or the JAX checkpoint itself:

```bash
python scripts/serve_policy.py --env=DROID --port=8000 policy:checkpoint \
    --policy.config=16_horrizon_05_lr_2_factor_96_batch --policy.dir=$CKPT --policy.sample-kwargs='{"num_steps": 1}'
```

Terminal 2, the validator:

```bash
PYTHONPATH=src python scripts/validate_triton_droid.py \
    --host=localhost --port=8000 \
    --rlds-data-dir=/path/to/droid_rlds \
    --validation-start-sample=19750000 --validation-samples=1000 \
    --action-chunk-size=16 --action-space=joint_position \
    --seed=0 \
    --jax-config=16_horrizon_05_lr_2_factor_96_batch --jax-checkpoint-dir=$CKPT \
    --output-path=data/validation_results.json
```

- `--validation-start-sample 19750000` starts after the samples the paper configs train on, so these are held-out frames.
- `--action-chunk-size` must equal the config's `action_horizon` (16 for the DROID configs).
- `--action-space joint_position` matches the IMLE-VLA DROID configs. Use `joint_velocity` and `--flow-matching-mode` to
  score the stock `pi05_droid` flow-matching baseline (it is trained on velocities and needs 10 denoising steps).
- Drop the two `--jax-*` flags to score a served policy on its own, for example the JAX server from Terminal 1.

How to read the result block:

- **Triton MSE ≈ JAX MSE** (and a small Triton-vs-JAX MSE): the conversion is correct. We accepted packages at Triton/JAX
  ratios within a few percent; the difference is bfloat16 kernel noise.
- **Both MSEs ≈ the run's training-time validation loss**: the checkpoint is healthy and the serving pipeline (transforms,
  prompt, norm stats) matches training. A large gap with a good JAX number usually means wrong `norm_stats.json` on the
  Triton side.
- The latency lines are the server-side forward times (mean, median, p95, p99).

Two more checks:

```bash
# Kernel-level check: identical synthetic inputs and noise through the Triton engine and the JAX model, prints the MAE per action dimension
PYTHONPATH=src python realtime-vla-imle/test.py --triton_path $PKG/imle_droid_triton.pkl \
    --jax_path $CKPT --norm_stats_dir $CKPT/assets/droid --config_name 16_horrizon_05_lr_2_factor_96_batch \
    --prompt "<the task prompt>" --action_dim 8 --action_horizon 16 --discrete_state_input --num_denoising_steps 1 --plot_comparison

# LIBERO success rate without a server (the engine is called directly)
python examples/libero/eval_triton_imle.py --checkpoint_path $PKG/imle_libero_triton.pkl \
    --norm_stats_dir $PKG/assets/physical-intelligence/libero --task_suite_name libero_10 --chunk_size 10 --num_trials_per_task 50
```
