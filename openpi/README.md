# IMLE-VLA, openpi (JAX) implementation

This folder is a fork of Physical Intelligence's [openpi](https://github.com/Physical-Intelligence/openpi) (upstream
commit `86b6dee`) with the IMLE-VLA single-step action head. It is the code behind the DROID / real-robot experiments and
the Triton engine. Everything openpi can do (π₀, π₀-FAST, π₀.₅, ALOHA / DROID / LIBERO examples, PyTorch port) still
works; this README covers only what IMLE-VLA adds. Run all commands below from this `openpi/` folder.

## What is added to openpi

| Path | What |
|---|---|
| `src/openpi/models/pi0.py` | `_compute_loss_imle`: the cIMLE objective (Algorithm 1 of the paper). `_sample_actions_imle`: single-step sampling that reuses the prefix KV cache. |
| `src/openpi/models/pi0_config.py` | Two new fields: `use_imle` and `imle_k` (the cIMLE sample factor *m*). |
| `scripts/train.py` | IMLE metrics (`imle_first_dist`, `imle_min_dist_first5`), learning-rate logging, config dump, fast resume for RLDS data. |
| `src/openpi/training/config.py` | The paper configs `16_horrizon_05_lr_2_factor_96_batch` (DROID) and `libero_50_action_05_lr_005_min_2_factor_correct` (LIBERO), plus the exploratory `*imle*` configs used during development. |
| `scripts/serve_policy.py` | `policy:checkpoint` serves an IMLE checkpoint with `--policy.sample-kwargs='{"num_steps": 1}'`. New `policy:triton-checkpoint` serves a converted `.pkl` through the Triton engine with the same transform pipeline as JAX. |
| `realtime-vla-imle/` | The single-step Triton inference engine (`Pi05InferenceIMLE`), CUDA-graph captured. This is the 55 Hz path. Derived from Dexmal's [realtime-vla](https://github.com/dexmal/realtime-vla). |
| `realtime-vla/` | Vendored flow-matching Triton engine (the "Triton" baseline row) and the JAX → `.pkl` converter `convert_from_jax_pi05.py`. |
| `scripts/serve_triton.sh` | One-command server for a Triton package. |
| `scripts/merge_all_lora.py` | Folds LoRA factors into the base weights. Required before converting a LoRA fine-tune to Triton. |
| `scripts/validate_triton_droid.py` | Triton-vs-JAX parity check on DROID samples. |
| `examples/libero/main.py` | `--n-action-steps` is the execution horizon *H*. Reports per-call inference time (VLA-only wall-clock). |
| `examples/libero/eval_triton_imle.py`, `eval_triton.py` | Direct LIBERO evaluation with the Triton engines, no server. See `examples/libero/README_TRITON.md`. |
| `examples/droid/main.py`, `examples/jerk_metric.py` | DROID robot client with the proprioceptive-jerk metric used in the paper. |
| `environment.yml` | Exact conda export of the environment used for the paper's experiments. |


## Installation

Tested on Ubuntu 22.04 with Python 3.11, CUDA 12, JAX 0.5.3, PyTorch 2.7.1 and Triton 3.3.1 (NVIDIA L40S, A6000, H100).

```bash
git clone --recurse-submodules https://github.com/kianhk6/IMLE-VLA.git
cd IMLE-VLA/openpi

conda create -n imle-vla python=3.11 -y
conda activate imle-vla
pip install -e packages/openpi-client
pip install -e .              # JAX + PyTorch + Triton stack (pins in pyproject.toml)
pip install -e ".[rlds]"      # only for DROID RLDS training (tensorflow-cpu, dlimp)
```

`environment.yml` is the exact export of the environment we used. `conda env create -f environment.yml` reproduces it.

Checkpoints and assets from `gs://openpi-assets` download automatically on first use and are cached in `~/.cache/openpi`
(override with `OPENPI_DATA_HOME`). The PaliGemma SentencePiece tokenizer is cached in the same place.

For the LIBERO simulator follow [`examples/libero/README.md`](examples/libero/README.md). It needs the `third_party/libero`
submodule and its own Python environment.

## Checkpoints

The real-robot DROID model is on the Hugging Face Hub: **[kinster/IMLE-VLA](https://huggingface.co/kinster/IMLE-VLA)**,
as a JAX checkpoint (`droid/jax/57000/`, config `16_horrizon_05_lr_2_factor_96_batch`) and as a Triton package with the
same weights (`droid/triton/`).

```bash
pip install -U huggingface_hub
huggingface-cli download kinster/IMLE-VLA --include "droid/jax/*"    --local-dir ./ckpt   # JAX checkpoint
huggingface-cli download kinster/IMLE-VLA --include "droid/triton/*" --local-dir ./ckpt   # Triton package
```

`./ckpt/droid/jax/57000` is a valid `--policy.dir`, and `./ckpt/droid/triton` a valid package for `scripts/serve_triton.sh`.
The LIBERO checkpoints are LeRobot checkpoints; see [`../lerobot/README.md`](../lerobot/README.md).

## Training with cIMLE

cIMLE needs two fields on `Pi0Config`:

```python
pi0_config.Pi0Config(pi05=True, use_imle=True, imle_k=2, ...)
```

- `use_imle=True` switches the loss to cIMLE and the sampler to one forward pass.
- `imle_k` is the sample factor *m*. For each training example the model draws *m* noise vectors, generates *m* candidate
  chunks without gradient, keeps the candidate closest (L2) to the ground truth, and re-runs only that one with gradient.
  `imle_k=1` is plain L2 regression. The paper uses `imle_k=2`; `imle_k=5` gives the same success at higher cost.
- The VLM prefix is computed once per example and repeated *m* times, so the extra cost is *m* passes of the small action
  expert. Memory grows with *m*.
- The VLM is frozen (`frozen=["paligemma"]`). Only the action expert is trained, starting from a π₀.₅ checkpoint.

### DROID (the real-robot model)

The config `16_horrizon_05_lr_2_factor_96_batch` is the run behind the real-robot results: frozen VLM, `imle_k=2`,
16-step chunks, joint-position action space, batch 96, cosine schedule with peak LR 1.75e-5, EMA 0.99. It starts from
`gs://openpi-assets/checkpoints/pi05_droid` and reuses that checkpoint's DROID normalization statistics, so
`compute_norm_stats.py` is not needed.

1. Get the DROID RLDS dataset (see [`examples/droid/README_train.md`](examples/droid/README_train.md)).
2. Set `rlds_data_dir` in the config to the parent directory of your `droid` RLDS directory.
3. Launch:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py 16_horrizon_05_lr_2_factor_96_batch \
    --exp-name=16_horrizon_05_lr_2_factor_96_batch \
    --checkpoint-base-dir=/path/to/checkpoints \
    --resume \
    --fsdp_devices 1
```

- Checkpoints land in `/path/to/checkpoints/<config>/<exp-name>/<step>/` with `params/` and `assets/` inside.
- `--resume` lets the same command restart from the latest checkpoint after a preemption. On the first launch it starts fresh.
- `--fsdp_devices 1` is pure data parallelism over the visible GPUs. The paper run used 4 GPUs (24 samples each).
  Raise it to shard parameters when memory is tight.
- The training log and Weights & Biases show `loss` (the cIMLE nearest-neighbour L2), `imle_first_dist` (the distance a
  plain regression head would see) and `imle_min_dist_first5` (the distance after the assignment step).

### LIBERO

The paper's LIBERO models were trained with the LeRobot implementation; see [`../lerobot/README.md`](../lerobot/README.md).

### Your own dataset

Follow openpi's recipe (convert to LeRobot, define a `DataConfigFactory`, run `scripts/compute_norm_stats.py`) and set
`use_imle=True, imle_k=2` in the model config. The `pi05_libero_imle_*` configs are small worked examples.

## Serving a policy (JAX)

The policy server and the clients are unchanged from openpi; only two flags matter for IMLE-VLA.

```bash
# DROID / real robot
python scripts/serve_policy.py --env=DROID --port=8000 policy:checkpoint \
    --policy.config=16_horrizon_05_lr_2_factor_96_batch \
    --policy.dir=./ckpt/droid/jax/57000 \
    --policy.sample-kwargs='{"num_steps": 1}'

# LIBERO
python scripts/serve_policy.py --env=LIBERO --port=8000 policy:checkpoint \
    --policy.config=libero_50_action_05_lr_005_min_2_factor_correct \
    --policy.dir=/path/to/checkpoints/libero_50_action_05_lr_005_min_2_factor_correct/imle_libero/10000 \
    --policy.sample-kwargs='{"num_steps": 1}'
```

`num_steps=1` selects the single-step sampler. The configs above already set `use_imle=True`, so the server uses the IMLE
sampler for them in any case.

**Robot client.** Follow [`examples/droid/README.md`](examples/droid/README.md). The client in this repo adds the jerk
metric from the paper (per-episode and running averages, printed at every replan). The paper ran IMLE-VLA at
`--open_loop_horizon 12` (π₀.₅ used its default 8):

```bash
python examples/droid/main.py --remote_host=<server_ip> --remote_port=8000 --external_camera=left --open_loop_horizon 12
```

**LIBERO client.** This repo's `examples/libero/main.py` parses its arguments directly, so flags have no `--args.` prefix.
`--n-action-steps` is the execution horizon *H* (paper: 10 and 30; 50 trials per task):

```bash
python examples/libero/main.py --task-suite-name libero_10 --n-action-steps 10 --num-trials-per-task 50 --host 127.0.0.1 --port 8000
```

At the end the client prints the success rate per task and overall, the number of policy calls, and the total
inference time. That total is the VLA-only wall-clock (VLA-WC) metric of the paper.

## Fast inference with Triton (55 Hz)

The 55 Hz numbers come from running the single-step head with hand-written Triton kernels and CUDA graphs
(`realtime-vla-imle/`). A JAX checkpoint is exported once to a `.pkl` and served by the same `scripts/serve_policy.py`, so
the robot client does not change. The released DROID Triton package serves with one command:

```bash
scripts/serve_triton.sh ./ckpt/droid/triton 16_horrizon_05_lr_2_factor_96_batch 0 8000 DROID
```

Converting your own checkpoint, the package layout, and validating a served policy against ground-truth DROID actions
are in **[docs/triton.md](docs/triton.md)**.

## Citation and license

See the [root README](../README.md) for the citation. Apache 2.0, see [`../LICENSE`](../LICENSE) and [`../NOTICE`](../NOTICE).
