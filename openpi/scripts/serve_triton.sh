#!/bin/bash
# Serve an IMLE-VLA Triton package (converted .pkl) with scripts/serve_policy.py.
#
#   usage: scripts/serve_triton.sh <package_dir> <config_name> [gpu=0] [port=8000] [env=DROID]
#
# Package layout (see README.md, "Fast inference with Triton"):
#   <package_dir>/<name>.pkl                     weights for the realtime-vla-imle engine
#   <package_dir>/assets/<asset_id>/norm_stats.json   normalization stats of the training run
#   <package_dir>/paligemma_tokenizer.model      optional; seeds the openpi cache so no download is needed
#   <package_dir>/engine/                        optional provenance copy of realtime-vla-imle/
#
# The config name supplies only the model geometry (action_horizon, max_token_len, discrete_state_input)
# and the data transforms; weights come from the .pkl and norm stats from the package.
set -euo pipefail
PKG=$(cd "$1" && pwd); CFG=$2; GPU=${3:-0}; PORT=${4:-8000}; ENV=${5:-DROID}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PKL=$(ls "$PKG"/*.pkl | head -1)
[ -f "$PKL" ] || { echo "no .pkl in $PKG"; exit 1; }

# Seed the tokenizer cache from the package copy (same path openpi uses) so serving works offline.
CACHE="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}/big_vision"
if [ -f "$PKG/paligemma_tokenizer.model" ]; then
  mkdir -p "$CACHE"
  [ -f "$CACHE/paligemma_tokenizer.model" ] || cp "$PKG/paligemma_tokenizer.model" "$CACHE/"
fi

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "serving $PKL  config=$CFG  env=$ENV  gpu=$GPU  port=$PORT"
exec env CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python}" "$REPO/scripts/serve_policy.py" \
  --env="$ENV" --port="$PORT" policy:triton-checkpoint \
  --policy.config="$CFG" \
  --policy.pkl-path="$PKL" \
  --policy.jax-checkpoint-dir="$PKG" \
  --policy.sample-kwargs='{"num_steps": 1}'
