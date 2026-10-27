#!/bin/bash
# Fine-tune pi0.5 on LIBERO (all four suites) with the cIMLE single-step action head.
#
#   usage: examples/imle_vla/train_libero.sh <h50|h100> <output_dir>
#
#   h50  : 50-step action chunks, the recipe behind the imle_vla_libero_h10_* checkpoints (evaluated at H = 10)
#   h100 : 100-step action chunks, the recipe behind the imle_vla_libero_h30_* checkpoints (evaluated at H = 30)
#
# --env.task is only used when --eval_freq > 0; training always uses the full HuggingFaceVLA/libero dataset.
set -euo pipefail
RECIPE=${1:?h50 or h100}; OUT=${2:?output dir}
cd "$(dirname "$0")/../.."

case "$RECIPE" in
  h50)  EXTRA="--policy.chunk_size=50 --policy.n_action_steps=50 --policy.imle_k=2 \
               --policy.optimizer_lr=1.25e-5 --policy.scheduler_decay_lr=1.25e-7" ;;
  h100) EXTRA="--policy.chunk_size=100 --policy.n_action_steps=100 --policy.imle_k=5 \
               --policy.optimizer_lr=1.25e-4 --policy.scheduler_decay_lr=2.5e-6" ;;
  *) echo "unknown recipe $RECIPE"; exit 1 ;;
esac

python src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero --env.task=libero_spatial --eval_freq=0 \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_finetuned \
  --policy.use_imle=true \
  --policy.use_ema=true --policy.ema_decay=0.999 \
  --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 \
  --policy.compile_model=true --policy.gradient_checkpointing=true --policy.dtype=bfloat16 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=35 --steps=10_000 --save_freq=1000 --log_freq=10 --seed=1000 \
  --output_dir="$OUT" --job_name="imle_vla_libero_$RECIPE" \
  $EXTRA "${@:3}"
