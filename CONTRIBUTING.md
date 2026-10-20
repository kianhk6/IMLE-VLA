# Contributing

Thank you for your interest in IMLE-VLA.

## Issues

Please open a GitHub issue for bugs in the IMLE-specific code (the cIMLE loss, single-step sampling,
the Triton engines, the training configs, or the evaluation scripts). Include:

- your OS, Python, JAX / PyTorch / Triton versions (see `environment.yml` for the tested set),
- the exact command you ran and the full traceback,
- the config name and checkpoint you used.

Problems with the base openpi code paths (data loading, ALOHA/UR5 examples, Docker) are usually best
reported upstream at [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi);
this repository tracks that project only up to the fork point.

## Pull requests

Small, focused pull requests are welcome. Run `pre-commit run --all-files` (ruff lint + format) before
opening one. Please do not reformat unrelated upstream files in the same PR.

## Checkpoints and data

We cannot host or redistribute third-party datasets. Checkpoint requests should go through issues.
