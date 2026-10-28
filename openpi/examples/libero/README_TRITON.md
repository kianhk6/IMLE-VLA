# LIBERO Evaluation with Triton Inference

This directory contains two evaluation scripts that call the Triton inference engines directly instead of going through a websocket server:

- `eval_triton_imle.py`: single-step IMLE-VLA engine (`realtime-vla-imle/pi05_infer_imle.py`). This is the path that produces the 55 Hz number in the paper.
- `eval_triton.py`: the flow-matching pi0.5 engine from Dexmal's realtime-vla (`realtime-vla/pi05_infer.py`), used as the "Triton" baseline row.

## Overview

The `eval_triton_imle.py` script evaluates an IMLE-VLA checkpoint on LIBERO tasks using the optimized Triton kernels from `realtime-vla-imle/pi05_infer_imle.py`. This provides faster inference compared to the websocket-based approach.

## Prerequisites

1. **Checkpoint**: You need a converted checkpoint in pickle format (`.pkl`). The checkpoint should be a dictionary containing model weights compatible with `Pi05Inference`.

2. **Dependencies**: 
   - The in-repo `realtime-vla-imle/` directory (the script adds it to the Python path)
   - All dependencies from `realtime-vla-imle/pi05_infer_imle.py` (triton, torch, sentencepiece, etc.)
   - LIBERO environment setup

## Usage

### Basic Usage

```bash
python examples/libero/eval_triton_imle.py \
    --checkpoint_path /path/to/checkpoint.pkl \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --chunk_size 10
```

### Full Example

```bash
python examples/libero/eval_triton_imle.py \
    --checkpoint_path /path/to/converted_imle_checkpoint.pkl \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --chunk_size 10 \
    --num_steps 1 \
    --is_imle True \
    --discrete_state_input True \
    --max_tokenize_len 200 \
    --num_steps_wait 10 \
    --video_out_path data/libero/videos \
    --seed 7
```

### Parameters

#### Model Parameters
- `--checkpoint_path` (required): Path to the checkpoint `.pkl` file
- `--chunk_size`: Action chunk size (replan_steps), default: 10
- `--num_steps`: Number of denoising steps (1 for IMLE, 10+ for flow matching), default: 1
- `--is_imle`: Whether using IMLE mode (single forward pass), default: True
- `--discrete_state_input`: Whether to use discrete state tokens, default: True
- `--max_tokenize_len`: Maximum token length for prompts, default: 200

#### LIBERO Parameters
- `--task_suite_name`: Task suite to evaluate on. Options:
  - `libero_spatial`
  - `libero_object`
  - `libero_goal`
  - `libero_10`
  - `libero_90`
- `--num_trials_per_task`: Number of rollouts per task, default: 10
- `--num_steps_wait`: Number of steps to wait for objects to stabilize in sim, default: 10

#### Output Parameters
- `--video_out_path`: Path to save videos, default: `data/libero/videos`
- `--seed`: Random seed for reproducibility, default: 7

## Checkpoint Format

The checkpoint should be a pickle file containing a dictionary with the following structure:

```python
checkpoint = {
    "vision_patch_embedding_w": ...,
    "vision_patch_embedding_b": ...,
    "vision_position_embedding": ...,
    # ... other vision weights ...
    "encoder_attn_qkv_w": ...,
    "encoder_attn_o_w": ...,
    # ... other encoder weights ...
    "decoder_time_embeds": ...,
    "decoder_action_in_proj_w": ...,
    # ... other decoder weights ...
    "language_embeds": ...,  # or None if using discrete_state_input
    "embedding_weight": ...,  # Required if discrete_state_input=True
}
```

All weights should be torch tensors in bfloat16 format on CUDA device.

## Image Preprocessing

The script automatically:
1. Rotates images 180 degrees (to match training preprocessing)
2. Resizes with padding to 224x224
3. Normalizes to [0, 1] range
4. Converts to bfloat16

## State Tokenization

The script:
1. Concatenates robot state: `[eef_pos (3), axis_angle (3), gripper (1)]` = 7D
2. Pads to 32D if needed
3. Normalizes to [-1, 1] range
4. Discretizes into 256 bins (0-255 tokens)

## Action Format

The model outputs 32D actions, but LIBERO expects 7D actions:
- First 7 dimensions are extracted: `[x, y, z, rx, ry, rz, gripper]`
- Actions are executed directly in the environment

## Output

The script outputs:
- Success rate per task and overall
- Inference timing statistics
- Episode videos saved to `--video_out_path`
- Console logs with progress and results

## Differences from WebSocket Version

1. **Direct inference**: No websocket server needed
2. **Faster**: Uses optimized Triton kernels
3. **Lower latency**: No network overhead
4. **Simpler setup**: Just load checkpoint and run

## Troubleshooting

### Import Errors
If you get import errors for `pi05_infer`, make sure:
- The in-repo `realtime-vla-imle/` directory exists
- All dependencies are installed (triton, torch, sentencepiece, etc.)

### CUDA Out of Memory
- Reduce `--chunk_size`
- Use fewer parallel environments
- Check GPU memory usage

### State Normalization Issues
- Adjust state normalization ranges in the script if your robot has different state ranges
- Consider using dataset statistics for proper normalization

### Action Scaling
- The model outputs actions that may need scaling/denormalization
- Check if actions are in the correct range for LIBERO environment
- You may need to add action postprocessing based on your training normalization

## Notes

- The script assumes images are in RGB format (H, W, 3)
- State tokenization uses simple normalization - you may want to use dataset statistics
- Action extraction takes first 7 dimensions - verify this matches your model's action space
- For flow matching (num_steps > 1), set `--is_imle False` and `--num_steps 10` (or higher)
