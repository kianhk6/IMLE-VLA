#!/bin/bash
# Evaluate an IMLE-VLA LeRobot checkpoint on the four LIBERO suites.
#
#   usage: examples/imle_vla/eval_libero.sh <pretrained_model_dir> <H> <output_dir> [n_episodes=50] [seed=1]
#
#   H is the execution horizon: actions executed open-loop per forward pass (--policy.n_action_steps).
#   Use H = 10 for the imle_vla_libero_h10_* checkpoints and H = 30 for the imle_vla_libero_h30_* ones.
#   With use_ema=true the EMA weights (model_ema.safetensors) are used when the checkpoint has them.
#   Results: <output_dir>/<suite>/eval_info.json (overall pc_success, per-task results, inference timing).
set -euo pipefail
CKPT=${1:?pretrained_model dir}; H=${2:?execution horizon}; OUT=${3:?output dir}; N=${4:-50}; SEED=${5:-1}
cd "$(dirname "$0")/../.."
export MUJOCO_GL=${MUJOCO_GL:-egl}

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  python src/lerobot/scripts/lerobot_eval.py \
    --policy.path="$CKPT" --policy.use_ema=true \
    --policy.n_action_steps="$H" \
    --env.type=libero --env.task="$SUITE" --env.max_parallel_tasks=1 \
    --eval.n_episodes="$N" --eval.batch_size=5 --seed="$SEED" \
    --output_dir="$OUT/$SUITE"
done
