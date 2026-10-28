<div align="center">

# IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies

[![arXiv](https://img.shields.io/badge/arXiv-2609.10915-b31b1b.svg)](https://arxiv.org/abs/2609.10915)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://kianhk6.github.io/IMLE-VLA/)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-yellow.svg)](https://huggingface.co/kinster/IMLE-VLA)
[![IROS 2026](https://img.shields.io/badge/IROS-2026-2ea44f.svg)](https://kianhk6.github.io/IMLE-VLA/)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)

**Kian Hosseinkhani**<sup>1*</sup>, Qinhe Peng<sup>2</sup>, George Shramko<sup>1</sup>, Mehran Aghabozorgi<sup>1</sup>, Jianing Qian<sup>2</sup>,
Tristan Engst<sup>1</sup>, Alireza Moazeni<sup>1</sup>, Dinesh Jayaraman<sup>2</sup>, Ke Li<sup>1,3,4</sup>

<sup>1</sup>Simon Fraser University &nbsp;·&nbsp; <sup>2</sup>University of Pennsylvania &nbsp;·&nbsp; <sup>3</sup>Amii &nbsp;·&nbsp; <sup>4</sup>Canada CIFAR AI Chair

<img src="media/overview.png" width="100%" alt="IMLE-VLA overview: one forward pass per action chunk instead of ten">

</div>

IMLE-VLA replaces the iterative flow-matching action head of a vision-language-action (VLA) policy with a **single-step
conditional generator** trained with **conditional Implicit Maximum Likelihood Estimation (cIMLE)**. The pretrained VLM
backbone is unchanged; one forward pass of the action expert produces a whole action chunk instead of the 10 denoising
steps π₀.₅ needs, and the cIMLE objective keeps the action distribution multimodal. Results and real-robot videos are in
the [paper](https://arxiv.org/abs/2609.10915) and on the [project page](https://kianhk6.github.io/IMLE-VLA/).

## Which code to use

The repository holds two implementations of the same IMLE head. Each folder has its own installation and README.

| You want to | Use | Guide |
|---|---|---|
| Run or fine-tune the DROID / real-robot policy (JAX, openpi) | [`openpi/`](openpi/) | [openpi/README.md](openpi/README.md) |
| Serve a policy at 55 Hz with the Triton engine | [`openpi/`](openpi/) | [openpi/docs/triton.md](openpi/docs/triton.md) |
| Train or evaluate on LIBERO (PyTorch, LeRobot) | [`lerobot/`](lerobot/) | [lerobot/README.md](lerobot/README.md) |

## Checkpoints

All checkpoints are on the Hugging Face Hub at **[kinster/IMLE-VLA](https://huggingface.co/kinster/IMLE-VLA)**.

| Hub folder | What | Run with |
|---|---|---|
| `droid/jax/57000/` | DROID real-robot model, JAX checkpoint | [openpi](openpi/README.md#checkpoints) |
| `droid/triton/` | the same model as a Triton package | [openpi/docs/triton.md](openpi/docs/triton.md) |
| `libero/imle_vla_libero_h10_*` | LIBERO, per suite and overall, execution horizon 10 | [lerobot](lerobot/README.md#checkpoints) |
| `libero/imle_vla_libero_h30_*` | LIBERO, per suite and overall, execution horizon 30 | [lerobot](lerobot/README.md#checkpoints) |

## Training compute

Both models were trained on 4× NVIDIA L40S. Times are pure training time, without cluster requeue stalls.

| Model | Code | Batch | Time per step | Training time |
|---|---|---|---|---|
| LIBERO | [`lerobot/`](lerobot/README.md) | 35 per GPU | about 6 s | about 8 hours to the released checkpoint (step 5000) |
| DROID (real robot) | [`openpi/`](openpi/README.md) | 96 total (24 per GPU) | about 4.7 s | about 74 hours (3 days) to the released checkpoint (step 57000) |

## Citation

```bibtex
@inproceedings{Hosseinkhani2026IMLEVLAFS,
  title     = {{IMLE-VLA}: Fast Single-Step Action Generation for Vision-Language-Action Policies},
  author    = {Hosseinkhani, Kian and Peng, Qinhe and Shramko, George and Aghabozorgi, Mehran and
               Qian, Jianing and Engst, Tristan and Moazeni, Alireza and Jayaraman, Dinesh and Li, Ke},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026},
  eprint    = {2609.10915},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2609.10915}
}
```

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence),
[LeRobot](https://github.com/huggingface/lerobot) (Hugging Face) and [realtime-vla](https://github.com/dexmal/realtime-vla)
(Dexmal), with [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and [DROID](https://droid-dataset.github.io/).
This research was enabled in part by NSERC, the Canada Foundation for Innovation, the Canada CIFAR AI Chairs program, the
BC DRI Group and the Digital Research Alliance of Canada.

## License

Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). The Triton engines in `openpi/realtime-vla*/` are MIT (Dexmal).
Gemma / PaliGemma weights are subject to [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt).
