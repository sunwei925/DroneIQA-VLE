<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/https://github.com/sunwei925/DroneIQA-VLE)
[![GitHub stars](https://img.shields.io/github/stars/sunwei925/DroneIQA-VLE)](https://github.com/sunwei925/DroneIQA-VLE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-brightgreen?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/DroneIQA-VLE)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00416-red?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2607.00416)

**🥈 2nd Place Solution for the [ICME 2026 Drone-IQA Grand Challenge on Target-aware Image Quality Assessment for Low-altitude UAV Images](https://chengyanjiang.github.io/icme26-droneiqa/)**

*Official Implementation of "DroneIQA-VLE: Multi-Task Drone Image Quality Assessment via Vision-Language Ensemble"*

[📖 Report](https://arxiv.org/abs/2607.00416) | [📊 Challenge](https://chengyanjiang.github.io/icme26-droneiqa/)

</div>

---

## 📋 Table of Contents

- [🎯 Introduction](#-introduction)
- [🏗️ Model Architecture](#-model-architecture)
- [🏆 Challenge Results](#-challenge-results)
- [📦 Installation](#-installation)
- [📊 Dataset](#-dataset)
- [🏋️ Training](#-training)
- [🧪 Inference](#-inference)
- [📁 Directory Structure](#-directory-structure)
- [📚 Citation](#-citation)

---

## 🎯 Introduction

Unmanned Aerial Vehicle (UAV) imagery has become increasingly prevalent in applications such as surveillance, traffic monitoring, and emergency response. However, UAV images exhibit distinct quality characteristics compared to conventional natural images due to diverse viewpoints, small target regions, complex backgrounds, and spatially nonuniform degradations. These factors make standard **Image Quality Assessment (IQA)** methods less suitable for UAV scenarios. Traditional full-reference metrics such as PSNR and SSIM are impractical since pristine references are unavailable in real-world UAV deployments, while most no-reference IQA methods focus solely on global perceptual quality without considering target-region usability and background interference.

To promote research on UAV-oriented quality modeling, the **Drone-IQA GC 2026 Grand Challenge** introduces a target-aware benchmark comprising approximately 6,000 UAV images collected from the VisDrone and UAVDT datasets, annotated by 18 human raters along three perceptual dimensions:

- **Global Quality** — the overall visual quality of the image
- **Target Quality** — the visual quality of the target object framed in the image
- **Background Quality** — the visual quality of the background region

The challenge requires participants to predict the **global quality** score of UAV images, while target and background quality annotations serve as auxiliary supervision. Submissions are evaluated using the average of the **Pearson Linear Correlation Coefficient (PLCC)** and the **Spearman Rank Correlation Coefficient (SRCC)**.

This repository presents **DroneIQA-VLE**, our solution to the challenge. The framework jointly predicts the three quality scores by ensembling two complementary pipelines and arithmetically averaging their global quality outputs.

### 🏆 Key Achievements

- **🥈 2nd Place** in the ICME 2026 Drone-IQA Grand Challenge on Target-aware Image Quality Assessment for Low-altitude UAV Images.

---

## 🏗️ Model Architecture

DroneIQA-VLE ensembles two complementary modeling paradigms:

**1) SigLIP2 Multi-task Models** — Two SigLIP2 visual backbones spanning different architectural scales, each equipped with three independent regression heads (`global` / `target` / `background`):

| Backbone | Patch | Input Size | Feature Dim | Regression Heads |
|----------|:-----:|:----------:|:-----------:|------------------|
| SigLIP2 ViT-L/16 | 16 | 384×384 | 1024 | `1024→128→1` × 3 |
| SigLIP2 ViT-SO400M/14 | 14 | 378×378 | 1152 | `1152→128→1` × 3 |

**2) Qwen3.5-9B Multimodal LLM** — A LoRA-adapted (`rank=64`, `α=128`) Qwen3.5-9B, configured as sequence classification with 3 regression labels. The hidden representations are directly used as high-level features to regress the three continuous quality scores.

**Ensemble Strategy** — The final global quality prediction is the arithmetic mean of the two pipelines:

```
Q = (Q_SigLIP2 + Q_Qwen) / 2
```

where `Q_SigLIP2` is the averaged prediction across all SigLIP2 checkpoints (2 architectures × 3 splits) and `Q_Qwen` is the Qwen3.5-9B prediction, each mapped onto the quality scale by a four-parameter logistic function.

---

## 🏆 Challenge Results

Final results on the held-out test set of the ICME 2026 Drone-IQA Grand Challenge (metric: average of PLCC and SRCC on global quality). **Bold** indicates our result.

| Rank | Team | PLCC | SRCC | Score |
|:----:|------|:----:|:----:|:-----:|
| 1 | cmsr | 0.9512 | 0.9450 | 0.9481 |
| **2** | **VQA (DroneIQA-VLE, Ours)** | **0.9484** | **0.9420** | **0.9452** |
| 3 | Echo | 0.9394 | 0.9332 | 0.9363 |
| 4 | TASEAI | 0.9293 | 0.9244 | 0.9268 |
| 5 | Watrix | 0.9262 | 0.9226 | 0.9244 |

---

## 📦 Installation

### Requirements

- **Python** >= 3.12
- **PyTorch** >= 2.8
- **CUDA** >= 12.8 (for GPU inference)

### Hardware

- **GPU**: NVIDIA GPU with CUDA support. The submitted models were trained and tested on a **single NVIDIA H20 (96 GB)**; `CUDA_VISIBLE_DEVICES=0` is set by default.
- **VRAM**: >= 80 GB recommended (inference requires > 70 GB for Qwen3.5-9B with the LoRA adapter + SigLIP2 models).

### Environment Setup

```bash
conda create -n droneiqa python=3.12
conda activate droneiqa

pip install -r requirements.txt
```

Core dependencies:

| Package | Version | Role |
|---------|---------|------|
| `torch` | 2.8.0 | Deep learning framework |
| `torchvision` | 0.23.0 | Image transforms & data loading |
| `transformers` | 5.5.0 | Qwen3.5-9B model loading |
| `peft` | 0.18.1 | LoRA adapter support |
| `accelerate` | 1.13.0 | Model acceleration / device placement |
| `flash-attn` | 2.8.3 | Flash Attention for efficient inference |
| `open_clip_torch` | 3.3.0 | SigLIP2 ViT backbone creation |
| `timm` | 1.0.26 | Vision model utilities |
| `scipy` | 1.17.1 | Logistic curve fitting & `.mat` file I/O |
| `numpy` | 2.3.2 | Numerical computing |
| `pandas` | 2.3.3 | DataFrame / CSV processing |
| `Pillow` | 11.3.0 | Image loading |

---

## 📊 Dataset

DroneIQA-VLE is trained and evaluated on the **Drone-IQA GC 2026** benchmark, which contains approximately 6,000 UAV images collected from the VisDrone and UAVDT datasets, annotated by 18 human raters along three perceptual dimensions (global / target / background quality). The challenge provides **3,600** annotated training images, **1,200** validation images, and a held-out test set for final evaluation.

Please refer to the [challenge website](https://chengyanjiang.github.io/icme26-droneiqa/) for data download and format details.

Prepare the training set as follows:

- **Images** — place all training images in a directory (default: `/root/autodl-tmp/DroneIQA/train`).
- **Qwen3.5-9B** — a JSONL file `DroneIQA.jsonl` in the project root, each line containing `messages`, `images`, and `label` fields.
- **SigLIP2** — a CSV file with columns `filename`, `global_quality_mean`, `target_quality_mean`, `background_quality_mean` (default: `train.csv`).

---

## 🏋️ Training

Training consists of two independent stages: fine-tuning **Qwen3.5-9B** (multimodal LLM) and training **SigLIP2** (vision encoder). They can be run in any order.

### Stage 1 — Train Qwen3.5-9B (LoRA Fine-tuning)

Uses ms-swift (`swift/cli/sft.py`) to fine-tune Qwen3.5-9B with LoRA for 3-label regression (global / target / background quality).

```bash
bash train_Qwen3.5_9B.sh
```

Key training parameters:

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3.5-9B` |
| Task type | `seq_cls` (regression, 3 labels) |
| Tuner | LoRA (`rank=64`, `alpha=128`) |
| Target modules | `all-linear` (vision encoder + aligner + LLM) |
| Loss | PLCC + fidelity (`mixed`) |
| Precision | `bfloat16` |
| Learning rate | `1e-4` (cosine decay) |
| Batch size | `16` |
| Epochs | `3` |
| Max sequence length | `4096` |
| Image max tokens | `2048` |
| Output directory | `output_droneIQA/` |

The script also sets `HF_ENDPOINT=https://hf-mirror.com` for Chinese mainland access. Checkpoints are saved to `output_droneIQA/`.

### Stage 2 — Train SigLIP2 (Multi-task Regression)

Trains two SigLIP2 architectures with 3 quality prediction heads each, using 3 random cross-validation splits.

```bash
bash train_SigLip2.sh
```

This runs two sequential training jobs:

**1) `SigLIP2_ViTL_384_DroneIQA_MT`** (ViT-L/16-SigLIP2, 384px):

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_SigLip2.py \
    --model_name SigLIP2_ViTL_384_DroneIQA_MT \
    --multi_gpu \
    --epochs 10 \
    --train_batch_size 32 \
    --n_exp 3 \
    --lr 1e-5
```

**2) `SigLIP2_So400m_378_DroneIQA_MT`** (ViT-SO400M/14-SigLIP2, 378px):

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_SigLip2.py \
    --model_name SigLIP2_So400m_378_DroneIQA_MT \
    --multi_gpu \
    --epochs 10 \
    --train_batch_size 32 \
    --n_exp 3 \
    --lr 1e-5
```

Key training parameters:

| Parameter | Value |
|-----------|-------|
| Preprocessing | Resize shorter side to 432, random crop (384 / 378), normalize mean/std = 0.5 |
| Data split | 80% train / 20% val (random, per seed) |
| Loss | PLCC loss (sum over 3 tasks) |
| Optimizer | Adam (`lr=1e-5`, `weight_decay=1e-7`) |
| LR scheduler | StepLR (`decay_ratio=0.95`, `step=2`) |
| Epochs | 10 |
| Splits (`--n_exp`) | 3 |
| Best model selection | `global_quality_mean` (SRCC+PLCC)/2 |
| Checkpoint output | `--ckpt_path` (default: `/root/autodl-tmp/IQAModels/ckpts_drone_iqa_mt`) |

Each split saves a `.pth` (model weights) and `.mat` (predictions + labels for logistic mapping) to the checkpoint directory. For inference, copy the generated `.pth` and `.mat` files to `output_droneIQA/ckpts_SigCLIP/`.

---

## 🧪 Inference

### Model Weights

The trained model weights (`DroneIQA_VLE`) are available on [Baidu Yun](https://pan.baidu.com/s/1bMVUeFwtZWlIP9v8YkyNSw) (code: `qahw`). After downloading, extract them into the repository root so the layout matches the [Directory Structure](#-directory-structure).

The following pre-trained weights are required before running inference:

| Path | Description |
|------|-------------|
| `output_droneIQA/Qwen3.5-9B/` | Qwen3.5-9B base model |
| `output_droneIQA/v12-20260408-234616/checkpoint-675/` | Qwen LoRA adapter (trained on a single H20) |
| `popt.mat` | Qwen logistic mapping parameters |
| `output_droneIQA/ckpts_SigCLIP/*.pth` | SigLIP2 model weights (6 files: 2 architectures × 3 splits) |
| `output_droneIQA/ckpts_SigCLIP/*.mat` | SigLIP2 logistic mapping parameters (6 files) |

### Test Images

Place the test images (`.jpg` / `.jpeg` / `.png`) in a single directory, then pass the path via `--val_dir`. The output CSV is sorted by filename in ascending order.

> **Tip**: Before running on the test set, we recommend first verifying that inference works correctly on the validation set. Our performance on the val split:
>
> | PLCC | SRCC | Score (PLCC+SRCC)/2 |
> |:----:|:----:|:-------------------:|
> | 0.9483 | 0.9389 | 0.9436 |

This repository provides three inference scripts.

| Script | Usage | Input | Output |
|--------|-------|-------|--------|
| `test_combined.py` | Full ensemble | Image directory | CSV of `global_quality_mean` (Qwen + SigLIP2 averaged) |
| `test_qwen_one_image.py` | Single-image, Qwen3.5-9B only | One image | Global / target / background scores printed to terminal |
| `test_siglip_one_image.py` | Single-image, one SigLIP2 checkpoint (no ensemble) | One image + one `.pth` | Global / target / background scores printed to terminal |

### 1. Combined Ensemble (`test_combined.py`)

Runs both pipelines over a directory, averages their per-image `global_quality_mean` predictions, and writes results to a CSV.

```bash
python test_combined.py
```

Or with custom arguments:

```bash
python test_combined.py \
    --val_dir /path/to/validation/images \
    --ckpt_dir output_droneIQA/ckpts_SigCLIP \
    --output_csv my_result.csv \
    --batch_size 16 \
    --num_workers 4 \
    --round 4
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--val_dir` | str | `/root/autodl-tmp/DroneIQA/val` | Directory containing validation/test images |
| `--ckpt_dir` | str | `output_droneIQA/ckpts_SigCLIP` | SigLIP2 checkpoint directory |
| `--output_csv` | str | `droneIQA_combined_result.csv` | Output CSV file path |
| `--batch_size` | int | `16` | Batch size for SigLIP2 inference |
| `--num_workers` | int | `4` | DataLoader worker count |
| `--round` | int | `4` | Decimal places for rounding (`None` to disable) |

The output CSV contains two columns:

```csv
filename,global_quality_mean
img_0001.jpg,3.2145
img_0002.jpg,2.8763
...
```

### 2. Single-Image Qwen3.5-9B Test (`test_qwen_one_image.py`)

Runs **only** the LoRA-adapted Qwen3.5-9B pipeline on **one** image and prints the three quality scores. The global quality score is logistic-mapped via `popt.mat`; target and background scores are the raw model outputs.

```bash
python test_qwen_one_image.py --image_path /path/to/img.jpg

# With custom adapter / mapping paths
python test_qwen_one_image.py \
    --image_path /path/to/img.jpg \
    --adapter_path output_droneIQA/v12-20260408-234616/checkpoint-675 \
    --popt_path popt.mat
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_path` | str | *(required)* | Path to a single image |
| `--base_model` | str | `output_droneIQA/Qwen3.5-9B` | Qwen3.5-9B base model directory |
| `--adapter_path` | str | `output_droneIQA/v12-20260408-234616/checkpoint-675` | Qwen LoRA adapter directory |
| `--popt_path` | str | `popt.mat` | Qwen logistic mapping parameters |
| `--round` | int | `4` | Decimal places for rounding (`None` to disable) |

### 3. Single-Image SigLIP2 Test (`test_siglip_one_image.py`)

Loads **one** trained SigLIP2 checkpoint (no ensemble) and scores **one** image, printing the three quality scores. All three dimensions are logistic-mapped via the checkpoint's `.mat` file.

```bash
python test_siglip_one_image.py \
    --image_path /path/to/img.jpg \
    --model SigLIP2_ViTL_384_DroneIQA_MT \
    --ckpt_path output_droneIQA/ckpts_SigCLIP/SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v0_ep8_SRCC0.9271.pth

# Explicitly specify the .mat (defaults to the .pth stem with .mat)
python test_siglip_one_image.py \
    --image_path /path/to/img.jpg \
    --model SigLIP2_So400m_378_DroneIQA_MT \
    --ckpt_path path/to/xxx.pth \
    --mat_path path/to/xxx.mat
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_path` | str | *(required)* | Path to a single image |
| `--model` | str | `SigLIP2_ViTL_384_DroneIQA_MT` | Architecture: `SigLIP2_ViTL_384_DroneIQA_MT` (384) / `SigLIP2_So400m_378_DroneIQA_MT` (378) |
| `--ckpt_path` | str | *(required)* | Path to a single trained `.pth` checkpoint |
| `--mat_path` | str | `.pth` stem + `.mat` | Path to the `.mat` for logistic mapping |
| `--round` | int | `4` | Decimal places for rounding (`None` to disable) |

---

## 📁 Directory Structure

```
DroneIQA-VLE/
├── README.md
├── requirements.txt
├── train_Qwen3.5_9B.sh          # Training script: Qwen3.5-9B LoRA fine-tuning
├── train_SigLip2.sh             # Training script: SigLIP2 multi-task training
├── train_SigLip2.py             # SigLIP2 multi-task training entry
├── test_combined.py             # Inference: Qwen + SigLIP2 ensemble (directory)
├── test_qwen_one_image.py       # Inference: Qwen3.5-9B on a single image
├── test_siglip_one_image.py     # Inference: one SigLIP2 checkpoint on a single image
├── swift/cli/sft.py             # ms-swift SFT script (Qwen training)
├── swift/cli/infer.py           # ms-swift inference script (Qwen inference)
├── DroneIQA.jsonl               # Qwen training data (messages / images / label)
├── train.csv                    # SigLIP2 training labels
├── popt.mat                     # Logistic mapping parameters for Qwen
└── output_droneIQA/
    ├── Qwen3.5-9B/              # Qwen3.5-9B base model weights
    ├── v12-20260408-234616/
    │   └── checkpoint-675/      # LoRA adapter (training output)
    └── ckpts_SigCLIP/           # SigLIP2 checkpoints (.pth + .mat × 6)
```

---

## 📚 Citation

If you find this work useful for your research, please cite our report:

```bibtex
@article{sun2026droneiqa,
  title={DroneIQA-VLE: Multi-Task Drone Image Quality Assessment via Vision-Language Ensemble},
  author={Sun, Wei and Zhang, Weixia and Zhan, Hongjian and Lu, Mingkai and Gao, Yixuan and Zhai, Guangtao},
  journal={arXiv preprint arXiv:2607.00416},
  year={2026}
}
```

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
