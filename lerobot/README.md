# IMLE-VLA in LeRobot

This folder is the [LeRobot](https://github.com/huggingface/lerobot) (v0.4.2) implementation of IMLE-VLA used for the
LIBERO experiments in the paper. It adds a conditional-IMLE single-step action head to LeRobot's π₀.₅ policy, plus EMA
weights, LoRA fine-tuning and timing in the LIBERO evaluation. The JAX/openpi implementation and the Triton engine are in
[`../openpi/`](../openpi/README.md).

It is a separate Python package (`lerobot`) with its own dependencies. Install it in its own environment.

## Installation

```bash
cd lerobot
conda create -n imle-vla-lerobot python=3.10 -y
conda activate imle-vla-lerobot
pip install -e ".[pi,libero]"
```

## What is added to LeRobot

| Path | What |
|---|---|
| `src/lerobot/policies/pi05/configuration_pi05.py` | `use_imle`, `imle_k` (the cIMLE sample factor *m*), `use_ema` / `ema_decay`, `use_lora` / `lora_r` / `lora_alpha` |
| `src/lerobot/policies/pi05/modeling_pi05.py` | cIMLE training loss and single-step sampling (`sample_actions_imle`); EMA weights saved to and loaded from `model_ema.safetensors` and used for inference |
| `src/lerobot/scripts/lerobot_train.py` | IMLE statistics in the training log |
| `src/lerobot/scripts/lerobot_eval.py` | per-call inference timing in `eval_info.json` |
| `src/lerobot/policies/pi05_imle/convert_pytorch_lerobot_to_jax.py` | converts a checkpoint from this folder to an openpi JAX checkpoint |
| `src/lerobot/policies/pi05_imle/`, `pi05_flow/` | Triton inference engines for IMLE and flow matching inside LeRobot |
| `examples/imle_vla/` | `train_libero.sh` and `eval_libero.sh` |

## Checkpoints

The LIBERO checkpoints are on the Hugging Face Hub in [kinster/IMLE-VLA](https://huggingface.co/kinster/IMLE-VLA) under
`libero/`. For each execution horizon there is a **per-suite** checkpoint (the best one for that suite) and an
**overall** checkpoint (one model for all four suites). Each folder holds a `pretrained_model/` directory.

| Folders | Action chunk | Evaluate at |
|---|---|---|
| `libero/imle_vla_libero_h10_{spatial,object,goal,long,overall}` | 50 | H = 10 |
| `libero/imle_vla_libero_h30_{spatial,object,goal,long,overall}` | 100 | H = 30 |

```bash
huggingface-cli download kinster/IMLE-VLA --include "libero/imle_vla_libero_h10_overall/*" --local-dir ./ckpt
```

## Training

All LIBERO models fine-tune `lerobot/pi05_libero_finetuned` on `HuggingFaceVLA/libero` (all four suites) for 10k steps,
batch 35, EMA 0.999, cosine schedule with 1k warmup steps.

```bash
examples/imle_vla/train_libero.sh h50  ./outputs/imle_libero_h50     # 50-step chunks, imle_k 2, LR 1.25e-5
examples/imle_vla/train_libero.sh h100 ./outputs/imle_libero_h100    # 100-step chunks, imle_k 5, LR 1.25e-4
```

The script is a thin wrapper around `src/lerobot/scripts/lerobot_train.py`. Extra flags are passed through, for example
`--wandb.enable=true --wandb.project=<project>`. Two of the H = 30 checkpoints come from a LoRA variant of the h100
recipe:

```bash
examples/imle_vla/train_libero.sh h100 ./outputs/imle_libero_h100_lora \
    --policy.use_lora=true --policy.lora_r=16 --policy.lora_alpha=32 --policy.use_ema=false --batch_size=20
```

Checkpoints land in `<output_dir>/checkpoints/<step>/pretrained_model/`.

The released overall checkpoints are these recipes at step 4000 (`h50`) and step 5000 (`h100`), not the final step.
Checkpoints are saved every 1000 steps.

## Evaluation on LIBERO

```bash
examples/imle_vla/eval_libero.sh ./ckpt/libero/imle_vla_libero_h10_overall/pretrained_model 10 ./eval/h10_overall
examples/imle_vla/eval_libero.sh ./ckpt/libero/imle_vla_libero_h30_overall/pretrained_model 30 ./eval/h30_overall
```

For each suite this runs `src/lerobot/scripts/lerobot_eval.py` with:

- `--policy.n_action_steps=<H>`: the execution horizon, the number of actions executed open-loop per forward pass;
- `--policy.use_ema=true`: evaluate the EMA weights when the checkpoint has them;
- `--eval.n_episodes=50` per task and `--seed=1` (override with the 4th and 5th arguments).

Results are written to `<output_dir>/<suite>/eval_info.json`: `overall.pc_success` is the success rate, `per_task` the
per-task results, and `inference_s_total` / `avg_inference_s_per_call` the policy inference time.

## Converting to openpi / Triton

```bash
python src/lerobot/policies/pi05_imle/convert_pytorch_lerobot_to_jax.py \
    --pytorch_path ./ckpt/libero/imle_vla_libero_h10_overall/pretrained_model \
    --output_path ./ckpt_jax/imle_vla_libero_h10_overall
```

The result is a JAX checkpoint (`params/`) for the code in [`../openpi/`](../openpi/README.md). From there,
`openpi/realtime-vla/convert_from_jax_pi05.py` produces a Triton package, as described in
[`../openpi/docs/triton.md`](../openpi/docs/triton.md).

## Citation and license

See the [root README](../README.md) for the citation. This folder is a fork of
[huggingface/lerobot](https://github.com/huggingface/lerobot) and keeps its Apache 2.0 license (`LICENSE`).
