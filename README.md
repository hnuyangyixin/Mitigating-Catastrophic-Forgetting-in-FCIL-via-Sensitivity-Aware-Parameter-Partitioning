# Mitigating Catastrophic Forgetting in Federated Class-Incremental Learning via Sensitivity-Aware Parameter Partitioning

Research code accompanying the FedSAP paper by Yixin Yang, Xuefeng Liu, Jianwei Niu, Xinghao Wu, Wei Chen, and Shaojie Tang (ACM Multimedia 2026).

FedSAP partitions classifier parameters at the class level and combines Absent-Class Freezing (ACF) with Inter-Class Separation Regularization (ICSR).

## Methods

| Directory | Method |
| --- | --- |
| `FedSAP/` | Sensitivity-aware parameter partitioning (proposed) |
| `FedAvg/` | Federated averaging baseline |
| `FedProx/` | Federated proximal optimization baseline |
| `PackNet/` | Parameter partitioning baseline adapted to FCIL |
| `GLFC/` | Federated class-incremental learning baseline |
| `FCIL_LD/` | Low-dimensional projection baseline |
| `FedGP/` | Gradient projection baseline |

Each method retains its own `main.py`, `option.py`, `client.py`, `Server.py`, `ResNet.py`, and data processing modules. These are the implementations used in this local project, not a claim that they are the upstream authors' original repositories.

## Installation

Create and activate a Python virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU training, install a mutually compatible PyTorch/torchvision build appropriate for your CUDA environment. Dependency versions are not pinned because the original experiment environment was not supplied. Full training has not been rerun as part of packaging.

## Running experiments

Run from inside the selected method directory because data and output paths are relative to the working directory.

```bash
cd FedSAP
python main.py --dataset CIFAR100 --num_clients 3 --tasks_global 5 --epochs_local 30 --learning_rate 0.001 --batch_size 32 --img_size 32 --seed 2024 --device cuda
```

For the other six methods, change into the corresponding directory and use the same command. Method-specific options are described by `python main.py --help` and `option.py`.

TinyImageNet example using the source's default image size (the paper does not specify this size):

```bash
python main.py --dataset TinyImageNet --num_clients 3 --tasks_global 5 --epochs_local 30 --learning_rate 0.001 --batch_size 32 --img_size 32 --seed 2024 --device cuda
```

For the 10-task setting, use `--tasks_global 10`. Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before comparing numerical results with the paper.

## Data and outputs

CIFAR-100 is downloaded by torchvision into `local_datasets/` beneath the selected method directory. TinyImageNet's loader also provides download handling and uses `local_datasets/tiny-imagenet-200/`. Data downloads require network access. Datasets are not redistributed in this repository.

Training logs and generated models belong in each method's `logs/` and `model_data/` directories and are excluded from Git. Existing datasets, checkpoints, caches, logs, and scratch files were excluded from the release. The main training entry points do not actively load the supplied `pretrain_models` checkpoint. FedSAP analysis scripts require separately generated model files and may need path configuration.

## Citation

```bibtex
@inproceedings{yang2026fedsap,
  title = {Mitigating Catastrophic Forgetting in Federated Class-Incremental Learning via Sensitivity-Aware Parameter Partitioning},
  author = {Yang, Yixin and Liu, Xuefeng and Niu, Jianwei and Wu, Xinghao and Chen, Wei and Tang, Shaojie},
  booktitle = {Proceedings of the 35th ACM International Conference on Multimedia},
  year = {2026},
  doi = {10.1145/3767308.3535563}
}
```

Citation metadata is transcribed from the supplied camera-ready manuscript.

## License and acknowledgments

No code license has been selected yet. The manuscript's CC BY 4.0 notice is not applied to this code. Upstream source repositories and any required third-party notices still need to be confirmed before selecting a repository-wide license.

Baselines correspond to McMahan et al. (FedAvg, AISTATS 2017), Li et al. (FedProx, MLSys 2020), Mallya and Lazebnik (PackNet, CVPR 2018), Dong et al. (GLFC, CVPR 2022), Khazaei and Hatzinakos (FCIL_LD, CVPR 2025), and Dai et al. (FedGP, FLSys 2023).
