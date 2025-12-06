

# RF-DETR Pig Detection Project

> End-to-end pig detection system based on the Roboflow Detection Transformer (RF-DETR)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Usage Details](#usage-details)
* [Project Structure](#project-structure)
* [Training Results](#training-results)
* [FAQ](#faq)
* [License](#license)

---

## 🎯 Project Overview

This project implements a pig detection system based on the latest Detection Transformer architecture, designed for farm automation and animal monitoring. It uses the RF-DETR (Roboflow Detection Transformer) model and supports multi-GPU distributed training for efficient and accurate object detection.

### Core Technologies

* **Model Architecture**: RF-DETR Small with DINOv2 backbone
* **Training Strategy**: 4-GPU Distributed Data Parallel (PyTorch DDP)
* **Input Resolution**: 512×512
* **Inference Speed**: ~40 images/second
* **Dataset Format**: COCO standard format

---

## ✨ Key Features

* ✅ **End-to-end pipeline**: from data processing to training and inference
* ✅ **Distributed training support** (tested with 4×RTX 2080 Ti)
* ✅ **Automation tools**: dataset creation, training, evaluation, inference scripts
* ✅ **Simplified inference** with a one-command interface
* ✅ **Visualization tools** for detection results
* ✅ **Comprehensive documentation**

---

## 💻 Requirements

### Hardware Requirements

* **GPU**: NVIDIA GPU with CUDA support (recommended ≥ 8GB VRAM)

  * Training: multi-GPU recommended (tested with 4×11GB)
  * Inference: single GPU (≥8GB) is sufficient
* **CPU**: multi-core processor
* **RAM**: 32GB recommended
* **Storage**: ~10GB free space

### Software Requirements

* Python 3.13 (or 3.8+)
* CUDA 11.8+ (12.x recommended)
* PyTorch 2.0+
* Linux (Ubuntu 20.04+ tested)

---

## 🚀 Installation

### 1. Clone the Project

```bash
cd /DATA1/yunzhu/DETR+1024/
git clone <repository-url> CV_HW1
cd CV_HW1
```

### 2. Install RF-DETR

```bash
# Clone RF-DETR repository
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# Switch to develop branch
git checkout develop

# Install dependencies
cd ..
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset

Ensure your file structure looks like this:

```
CV_HW1/
├── train/
│   ├── img/          # training images
│   └── gt.txt        # annotation file
└── test/
    └── img/          # test images
```

---

## 🎓 Quick Start

### 1. Create COCO-format Dataset

```bash
python create_rfdetr_dataset.py
```

This generates:

* `rfdetr_dataset/` (COCO-format dataset)
* Train/Val split (80/20)
* COCO JSON annotation files

### 2. Train the Model (Multi-GPU)

```bash
# Train with 4 GPUs
torchrun --nproc_per_node=4 train_rfdetr_multigpu2.py

# Train with 2 GPUs
torchrun --nproc_per_node=2 train_rfdetr_multigpu2.py

# Single GPU training
python train_rfdetr_multigpu2.py
```

Output will be saved in:
`runs/train/rfdetr_small_pig_detection_multigpu_<timestamp>/`

### 3. Run Inference

```bash
# Use latest checkpoint
python simple_inference.py

# Use a specific checkpoint
python simple_inference.py --checkpoint runs/train/rfdetr_small_*/checkpoint_best_regular.pth

# Adjust confidence threshold
python simple_inference.py --threshold 0.3

# Specify test image directory
python simple_inference.py --test_dir ./test/img
```

Output:

* `predictions.json`
* `submission.txt`

### 4. Visualize Results

```bash
python visualize_predictions.py \
    --predictions runs/train/rfdetr_small_*/predictions.json \
    --image_dir test/img \
    --output_dir visualizations \
    --num_samples 20
```

---

## 📚 Usage Details

### Training Config

Edit `train_rfdetr_multigpu2.py`:

```python
config = {
    "model_size": "small",
    "dataset_root": "rfdetr_dataset",
    "epochs": 50,
    "batch_size": 4,
    "lr": 1e-4,
    "resolution": 512,
    "use_pretrained": True,
    "save_dir": "runs/train",
    "num_gpus": 4
}
```

### Inference Options

```bash
python simple_inference.py \
    --checkpoint <checkpoint_path> \
    --test_dir <test_directory> \
    --output_dir <output_directory> \
    --threshold 0.5
```

### Monitor Training

```bash
tensorboard --logdir runs/train/rfdetr_small_pig_detection_multigpu_<timestamp>/tensorboard
```

---

## 📁 Project Structure


```
CV_HW1/
├── README.md
├── PROJECT_REPORT.md
├── requirements.txt
│
├── train/
│   ├── img/
│   └── gt.txt
│
├── test/
│   └── img/
│
├── rfdetr_dataset/
│   ├── train/
│   ├── val/
│   ├── annotations/
│   │   ├── instances_train.json
│   │   └── instances_val.json
│   └── README.md
│
├── rfdetr_dataset_roboflow/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/
│   └── train/
│       └── rfdetr_small_pig_detection_multigpu_<timestamp>/
│           ├── checkpoint_best_regular.pth
│           ├── train_config.json
│           ├── predictions.json
│           ├── submission.txt
│           └── tensorboard/
│
├── rf-detr/
│   └── rfdetr/
│
├── train_rfdetr_multigpu2.py
├── simple_inference.py
├── visualize_predictions.py
├── create_rfdetr_dataset.py
│
└── docs/
    ├── DATASET_SUMMARY.md
    └── ...
```

---

## 📊 Training Results

### Best Model Performance

| Metric               | Value                      |
| -------------------- | -------------------------- |
| Model                | RF-DETR Small              |
| Input Resolution     | 512×512                    |
| Epochs               | 50                         |
| Training Time        | ~2–3 hours (4×RTX 2080 Ti) |
| Test Images          | 1,864                      |
| Total Detections     | 51,547                     |
| Avg Detections/Image | 27.7 pigs                  |
| Inference Speed      | ~40 images/s               |

### Checkpoint Info

* **Path**: `runs/train/rfdetr_small_pig_detection_multigpu_20251010_122140/checkpoint_best_regular.pth`
* **Size**: ~200MB
* **Includes**:

  * Model weights
  * Optimizer state
  * LR scheduler
  * Config
  * Epoch index

---

## 🔧 FAQ


### Q1: CUDA Out of Memory

**Solution**:

* Reduce batch_size
* Lower resolution
* Increase GPU count
* Use gradient accumulation

### Q2: Distributed Training Not Working

**Solution**:

* Use torchrun
* Check environment variables
* Ensure all GPUs visible

### Q3: Checkpoint Load Failure

Handled automatically in `simple_inference.py`.

### Q4: Inference Too Slow

**Solution**:

* `model.optimize_for_inference`
* Batch inference
* Export ONNX
* Use TensorRT

### Q5: Dataset Format Error

Recreate dataset:

```bash
rm -rf rfdetr_dataset/ rfdetr_dataset_roboflow/
python create_rfdetr_dataset.py
```

---

## 🛠️ Advanced Usage

### Model Size Options

```python
config = {"model_size": "nano", "resolution": 384}
config = {"model_size": "small", "resolution": 512}
config = {"model_size": "medium", "resolution": 576}
```

### Export to ONNX

```python
model.export(
    output_path="model.onnx",
    opset_version=17,
    dynamic_batch=True
)
```

---

## 🤝 Contribution

Same process as typical GitHub workflow.

---

## 📝 Changelog

### v1.0.0 (2025-10-10)

* Initial release
* Full multi-GPU pipeline
* Simple inference script
* Visualization tools
* Documentation

---

## 📖 References

* RF-DETR GitHub
* RF-DETR Docs
* PyTorch DDP Tutorial
* COCO Format
* Supervision Library

---

## 📄 License

This project is under MIT License – see [LICENSE](LICENSE).

---

## 👨‍💻 Author

**Yunzhu**

* Project: RF-DETR Pig Detection System
* Date: October 2025

---

## 🙏 Acknowledgments

* RF-DETR by Roboflow
* PyTorch team
* COCO dataset community

---
