# RF-DETR 豬隻檢測專案

> 基於 Roboflow Detection Transformer (RF-DETR) 的端到端豬隻檢測系統

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目錄

- [專案簡介](#專案簡介)
- [主要特點](#主要特點)
- [環境需求](#環境需求)
- [安裝步驟](#安裝步驟)
- [快速開始](#快速開始)
- [詳細使用](#詳細使用)
- [專案結構](#專案結構)
- [訓練結果](#訓練結果)
- [常見問題](#常見問題)
- [授權協議](#授權協議)

---

## 🎯 專案簡介

本專案實現了基於最新 Detection Transformer 架構的豬隻檢測系統，用於農場自動化管理和動物監測。專案採用 RF-DETR (Roboflow Detection Transformer) 模型，支援多GPU分散式訓練，實現高效準確的目標檢測。

### 核心技術

- **模型架構**：RF-DETR Small with DINOv2 backbone
- **訓練策略**：4-GPU 分散式訓練 (PyTorch DDP)
- **輸入解析度**：512×512
- **檢測速度**：約40張/秒
- **資料格式**：COCO標準格式

---

## ✨ 主要特點

- ✅ **端到端流程**：從數據處理到訓練推理的完整pipeline
- ✅ **分散式訓練**：支援多GPU加速訓練（已測試4×RTX 2080 Ti）
- ✅ **自動化工具**：數據集構建、模型訓練、推理評估一站式腳本
- ✅ **簡化推理**：一鍵載入checkpoint並進行批次推理
- ✅ **視覺化支援**：檢測結果可視化工具
- ✅ **完整文檔**：詳細的使用說明和技術報告

---

## 💻 環境需求

### 硬體需求

- **GPU**：NVIDIA GPU with CUDA support (建議 ≥ 8GB VRAM)
  - 訓練：建議使用多GPU（4×11GB已測試）
  - 推理：單GPU即可（≥8GB）
- **CPU**：多核心處理器
- **RAM**：建議 ≥ 32GB
- **儲存**：約 10GB 可用空間

### 軟體需求

- Python 3.13 (或 3.8+)
- CUDA 11.8+ (建議 12.x)
- PyTorch 2.0+
- Linux 系統 (Ubuntu 20.04+ 已測試)

---

## 🚀 安裝步驟

### 1. Clone 專案

```bash
cd /DATA1/yunzhu/DETR+1024/
git clone <repository-url> CV_HW1
cd CV_HW1
```

### 2. 安裝 RF-DETR

```bash
# Clone RF-DETR repository
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# 切換到 develop 分支
git checkout develop

# 安裝依賴
cd ..
```

### 3. 安裝 Python 依賴

```bash
pip install -r requirements.txt
```

### 4. 準備數據集

確保你的數據結構如下：

```
CV_HW1/
├── train/
│   ├── img/          # 訓練圖片
│   └── gt.txt        # 標註文件
└── test/
    └── img/          # 測試圖片
```

---

## 🎓 快速開始

### 1. 創建 COCO 格式數據集

```bash
python create_rfdetr_dataset.py
```

這會生成：
- `rfdetr_dataset/`：COCO格式數據集
- 訓練/驗證分割（8:2比例）
- COCO JSON 標註文件

### 2. 訓練模型（多GPU）

```bash
# 使用 4 個 GPU 訓練
torchrun --nproc_per_node=4 train_rfdetr_multigpu2.py

# 使用 2 個 GPU 訓練
torchrun --nproc_per_node=2 train_rfdetr_multigpu2.py

# 單GPU訓練
python train_rfdetr_multigpu2.py
```

訓練輸出將保存在 `runs/train/rfdetr_small_pig_detection_multigpu_<timestamp>/`

### 3. 運行推理

```bash
# 使用最新的 checkpoint 進行推理
python simple_inference.py

# 指定特定的 checkpoint
python simple_inference.py --checkpoint runs/train/rfdetr_small_*/checkpoint_best_regular.pth

# 調整置信度閾值
python simple_inference.py --threshold 0.3

# 指定測試圖片目錄
python simple_inference.py --test_dir ./test/img
```

推理結果將保存為：
- `predictions.json`：詳細的JSON格式結果
- `submission.txt`：競賽提交格式

### 4. 視覺化結果

```bash
python visualize_predictions.py \
    --predictions runs/train/rfdetr_small_*/predictions.json \
    --image_dir test/img \
    --output_dir visualizations \
    --num_samples 20
```

---

## 📚 詳細使用

### 訓練配置調整

編輯 `train_rfdetr_multigpu2.py` 中的配置：

```python
config = {
    "model_size": "small",        # nano, small, medium
    "dataset_root": "rfdetr_dataset",
    "epochs": 50,                 # 訓練輪數
    "batch_size": 4,              # 總 batch size
    "lr": 1e-4,                   # 學習率
    "resolution": 512,            # 輸入解析度
    "use_pretrained": True,       # 使用預訓練權重
    "save_dir": "runs/train",
    "num_gpus": 4                 # GPU 數量（參考用）
}
```

### 推理選項

`simple_inference.py` 支援以下參數：

```bash
python simple_inference.py \
    --checkpoint <checkpoint_path>     # checkpoint 路徑
    --test_dir <test_directory>        # 測試圖片目錄
    --output_dir <output_directory>    # 輸出目錄
    --threshold 0.5                    # 置信度閾值 (0.0-1.0)
```

### 監控訓練進度

使用 TensorBoard 查看訓練曲線：

```bash
tensorboard --logdir runs/train/rfdetr_small_pig_detection_multigpu_<timestamp>/tensorboard
```

或使用 `watch nvidia-smi` 監控 GPU 使用狀況：

```bash
watch -n 1 nvidia-smi
```

---

## 📁 專案結構

```
CV_HW1/
├── README.md                        # 本文件
├── PROJECT_REPORT.md                # 詳細技術報告
├── requirements.txt                 # Python 依賴
│
├── train/                           # 原始訓練數據
│   ├── img/                         # 1,266 張訓練圖片
│   └── gt.txt                       # 原始標註文件
│
├── test/                            # 測試數據
│   └── img/                         # 1,864 張測試圖片
│
├── rfdetr_dataset/                  # COCO 格式數據集
│   ├── train/                       # 1,012 張訓練圖片
│   ├── val/                         # 254 張驗證圖片
│   ├── annotations/
│   │   ├── instances_train.json     # 訓練集標註
│   │   └── instances_val.json       # 驗證集標註
│   └── README.md
│
├── rfdetr_dataset_roboflow/         # Roboflow 格式（自動生成）
│   ├── train/
│   ├── valid/
│   └── test/
│
├── runs/                            # 訓練輸出
│   └── train/
│       └── rfdetr_small_pig_detection_multigpu_<timestamp>/
│           ├── checkpoint_best_regular.pth   # 最佳模型
│           ├── train_config.json             # 訓練配置
│           ├── predictions.json              # 預測結果
│           ├── submission.txt                # 提交文件
│           └── tensorboard/                  # TensorBoard 日誌
│
├── rf-detr/                         # RF-DETR 源碼
│   └── rfdetr/
│
├── train_rfdetr_multigpu2.py        # 🔥 主訓練腳本
├── simple_inference.py              # 🔥 推理腳本
├── visualize_predictions.py         # 視覺化工具
├── create_rfdetr_dataset.py         # 數據集構建
│
└── docs/                            # 文檔 (可選)
    ├── DATASET_SUMMARY.md
    └── ...
```

---

## 📊 訓練結果

### 最佳模型性能

| 指標 | 數值 |
|------|------|
| 模型架構 | RF-DETR Small |
| 輸入解析度 | 512×512 |
| 訓練輪數 | 50 epochs |
| 訓練時長 | ~2-3 小時 (4×RTX 2080 Ti) |
| 測試圖片數 | 1,864 張 |
| 總檢測數 | 51,547 個目標 |
| 平均檢測/圖 | 27.7 個豬隻 |
| 推理速度 | ~40 張/秒 |

### Checkpoint 資訊

- **路徑**：`runs/train/rfdetr_small_pig_detection_multigpu_20251010_122140/checkpoint_best_regular.pth`
- **大小**：約 200MB
- **包含內容**：
  - 模型權重 (`model`)
  - 優化器狀態 (`optimizer`)
  - 學習率調度器 (`lr_scheduler`)
  - 訓練配置 (`args`)
  - 當前 epoch

---

## 🔧 常見問題

### Q1: CUDA Out of Memory 錯誤

**問題**：訓練時出現 `RuntimeError: CUDA out of memory`

**解決方案**：
1. 減小 batch_size（在 `train_rfdetr_multigpu2.py` 中修改）
2. 降低輸入解析度（512 → 384）
3. 使用更多GPU分散負載
4. 啟用梯度累積

```python
config = {
    "batch_size": 2,  # 從 4 降至 2
    "resolution": 384,  # 從 512 降至 384
}
```

### Q2: 分散式訓練不工作

**問題**：torchrun 啟動後只使用一個GPU

**解決方案**：
1. 確認使用 `torchrun` 而不是 `python`
2. 檢查環境變數：`echo $RANK $WORLD_SIZE`
3. 確認所有GPU可見：`nvidia-smi`

```bash
# 正確的啟動方式
torchrun --nproc_per_node=4 train_rfdetr_multigpu2.py

# 錯誤的啟動方式
python train_rfdetr_multigpu2.py  # 只會使用單GPU
```

### Q3: Checkpoint 載入失敗

**問題**：推理時出現 shape mismatch 錯誤

**解決方案**：
已在 `simple_inference.py` 中處理：
- 自動從 checkpoint 提取類別數
- 呼叫 `reinitialize_detection_head()` 重建分類頭
- 處理 DDP 的 "module." 前綴

如果仍有問題，確認使用最新版本的 `simple_inference.py`。

### Q4: 推理速度太慢

**問題**：推理速度遠低於預期

**解決方案**：
1. 使用模型優化：

```python
model.optimize_for_inference(compile=True, batch_size=1)
```

2. 使用批次推理（修改 `simple_inference.py`）
3. 導出為 ONNX 格式
4. 使用 TensorRT 加速

### Q5: 數據集格式錯誤

**問題**：訓練時出現 COCO 格式相關錯誤

**解決方案**：
重新運行數據集創建腳本：

```bash
# 刪除舊數據集
rm -rf rfdetr_dataset/ rfdetr_dataset_roboflow/

# 重新創建
python create_rfdetr_dataset.py
```

確認 category_id 從 0 開始（不是 1）。

---

## 🛠️ 進階使用

### 使用不同模型大小

RF-DETR 提供三種模型大小：

```python
# Nano: 384×384, 較快但精度稍低
config = {"model_size": "nano", "resolution": 384}

# Small: 512×512, 平衡速度和精度 (推薦)
config = {"model_size": "small", "resolution": 512}

# Medium: 576×576, 最高精度但較慢
config = {"model_size": "medium", "resolution": 576}
```

### 自定義數據增強

編輯 `train_rfdetr_multigpu2.py` 中的 `create_training_config()`：

```python
config = TrainConfig(
    # ... 其他配置 ...
    multi_scale=True,        # 多尺度訓練
    use_augmentation=True,   # 數據增強
    horizontal_flip=True,    # 水平翻轉
    # 添加更多增強選項...
)
```

### 導出為 ONNX

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="path/to/checkpoint.pth")
model.export(
    output_path="model.onnx",
    opset_version=17,
    dynamic_batch=True
)
```

### 整合到生產環境

```python
# 載入優化的模型
from simple_inference import load_model_from_checkpoint

model = load_model_from_checkpoint("checkpoint_best_regular.pth")
model.optimize_for_inference(compile=True, batch_size=1)

# 批次推理
images = [...]  # PIL Images 列表
results = model.predict(images, threshold=0.5)
```

---

## 🤝 貢獻指南

歡迎提交 Issue 和 Pull Request！

1. Fork 本專案
2. 創建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改動 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟一個 Pull Request

---

## 📝 更新日誌

### v1.0.0 (2025-10-10)

- ✅ 初始版本發布
- ✅ 完整的多GPU訓練pipeline
- ✅ 簡化的推理腳本
- ✅ 視覺化工具
- ✅ 完整文檔

---

## 📖 參考資源

- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [RF-DETR 文檔](https://rfdetr.roboflow.com/)
- [PyTorch DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [COCO 數據格式](https://cocodataset.org/#format-data)
- [Supervision 庫](https://supervision.roboflow.com/)

---

## 📄 授權協議

本專案採用 MIT 授權協議 - 詳見 [LICENSE](LICENSE) 文件

---

## 👨‍💻 作者

**Yunzhu**

- 專案：RF-DETR 豬隻檢測系統
- 日期：2025年10月

---

## 🙏 致謝

- [Roboflow](https://roboflow.com/) 提供的 RF-DETR 模型
- PyTorch 團隊的優秀深度學習框架
- COCO 數據集格式標準

---

**如有問題或建議，歡迎開啟 Issue 討論！** 📮
