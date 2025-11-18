# 安裝指南

本文件提供 RF-DETR 豬隻檢測專案的詳細安裝說明。

---

## 📋 前置需求檢查

在開始安裝前，請確認以下項目：

### 1. 系統需求

```bash
# 檢查作業系統
uname -a  # 建議 Ubuntu 20.04+

# 檢查 GPU
nvidia-smi  # 應該顯示 GPU 資訊

# 檢查 CUDA 版本
nvcc --version  # 建議 CUDA 11.8 或 12.x
```

### 2. Python 版本

```bash
python --version  # Python 3.8+ (已測試 3.13)
```

---

## 🚀 快速安裝 (推薦)

### Step 1: 創建虛擬環境 (可選但推薦)

```bash
# 使用 conda
conda create -n rfdetr python=3.13
conda activate rfdetr

# 或使用 venv
python -m venv rfdetr_env
source rfdetr_env/bin/activate  # Linux/Mac
# 或 rfdetr_env\Scripts\activate  # Windows
```

### Step 2: 安裝 PyTorch

根據您的 CUDA 版本選擇：

```bash
# CUDA 12.1 (推薦)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本 (不建議，僅用於測試)
pip install torch torchvision torchaudio
```

驗證安裝：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: 安裝專案依賴

```bash
cd /DATA1/yunzhu/DETR+1024/CV_HW1

# 安裝最小依賴集
pip install -r requirements-minimal.txt

# 或安裝完整依賴（包含開發工具）
pip install -r requirements.txt
```

### Step 4: 設置 RF-DETR

```bash
# Clone RF-DETR repository
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# 切換到 develop 分支
git checkout develop

# 返回專案目錄
cd ..
```

### Step 5: 驗證安裝

```bash
# 測試所有關鍵導入
python -c "
import torch
import torchvision
import cv2
import supervision as sv
import transformers
import timm
print('✓ 所有依賴安裝成功！')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU 數量: {torch.cuda.device_count()}')
"
```

---

## 🔧 詳細安裝 (逐步說明)

### 1. 系統依賴安裝 (Ubuntu/Debian)

```bash
# 更新套件列表
sudo apt update

# 安裝基礎開發工具
sudo apt install -y build-essential git wget curl

# 安裝 Python 開發套件
sudo apt install -y python3-dev python3-pip

# 安裝 OpenCV 系統依賴
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# 安裝 COCO API 依賴
sudo apt install -y gcc g++
```

### 2. NVIDIA 驅動和 CUDA 安裝

如果尚未安裝 NVIDIA 驅動：

```bash
# 檢查推薦的驅動版本
ubuntu-drivers devices

# 安裝推薦的驅動
sudo ubuntu-drivers autoinstall

# 重啟系統
sudo reboot
```

CUDA Toolkit 安裝 (如果需要)：

```bash
# 下載 CUDA 12.1 (範例)
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# 安裝
sudo sh cuda_12.1.0_530.30.02_linux.run

# 設置環境變數
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Conda 環境設置 (推薦方式)

```bash
# 安裝 Miniconda (如果還沒有)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 創建專案環境
conda create -n rfdetr python=3.13
conda activate rfdetr

# 使用 conda 安裝 PyTorch (自動處理 CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安裝其他依賴
conda install -c conda-forge opencv pycocotools
pip install supervision transformers timm tensorboard pyyaml tqdm
```

### 4. 驗證 GPU 設置

```bash
# 檢查 GPU 資訊
nvidia-smi

# 檢查 CUDA 版本
nvcc --version

# 在 Python 中測試
python << EOF
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 數量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
EOF
```

---

## 🐛 常見安裝問題

### 問題 1: PyTorch 找不到 CUDA

**症狀**：`torch.cuda.is_available()` 返回 `False`

**解決方案**：
1. 確認 NVIDIA 驅動已安裝：`nvidia-smi`
2. 重新安裝匹配的 PyTorch 版本
3. 檢查 CUDA 版本匹配

```bash
# 卸載舊版本
pip uninstall torch torchvision torchaudio

# 重新安裝正確版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 問題 2: pycocotools 編譯失敗

**症狀**：安裝 pycocotools 時出現編譯錯誤

**解決方案**：

```bash
# 安裝編譯依賴
sudo apt install -y gcc g++ python3-dev

# 或使用預編譯版本
pip install pycocotools --no-build-isolation
```

### 問題 3: OpenCV 導入錯誤

**症狀**：`ImportError: libGL.so.1: cannot open shared object file`

**解決方案**：

```bash
# 安裝 OpenGL 依賴
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

### 問題 4: supervision 版本過舊

**症狀**：`AttributeError: 'Detections' object has no attribute 'confidence'`

**解決方案**：

```bash
# 升級到最新版本
pip install --upgrade supervision
```

### 問題 5: RF-DETR 導入失敗

**症狀**：`ModuleNotFoundError: No module named 'rfdetr'`

**解決方案**：

```bash
# 確認 rf-detr 目錄存在
ls -la rf-detr/

# 確認分支正確
cd rf-detr && git status && cd ..

# 如果需要，重新 clone
rm -rf rf-detr
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr && git checkout develop && cd ..
```

---

## 📦 離線安裝

如果在沒有網路的環境中：

### 1. 下載所有套件

```bash
# 在有網路的機器上
pip download -r requirements.txt -d ./packages

# 複製 packages 目錄到目標機器
```

### 2. 離線安裝

```bash
# 在目標機器上
pip install --no-index --find-links=./packages -r requirements.txt
```

---

## 🧪 安裝後測試

運行完整測試以確認所有功能正常：

```bash
# 1. 測試數據集創建
python create_rfdetr_dataset.py

# 2. 測試推理腳本 (需要先有 checkpoint)
python simple_inference.py --help

# 3. 檢查訓練腳本語法
python -m py_compile train_rfdetr_multigpu2.py

# 4. 測試 torchrun
torchrun --nproc_per_node=1 -m torch.distributed.launch --help
```

---

## 📝 環境匯出

保存當前環境以便復現：

```bash
# 使用 pip
pip freeze > installed_packages.txt

# 使用 conda
conda env export > environment.yml
```

復原環境：

```bash
# 使用 pip
pip install -r installed_packages.txt

# 使用 conda
conda env create -f environment.yml
```

---

## 🆘 獲取幫助

如果遇到安裝問題：

1. 查看本文件的「常見問題」部分
2. 檢查 GitHub Issues
3. 查閱官方文檔：
   - [PyTorch 安裝指南](https://pytorch.org/get-started/locally/)
   - [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
4. 提供以下資訊以便診斷：
   - 作業系統和版本
   - Python 版本
   - CUDA 版本
   - 完整錯誤訊息

---

**安裝完成後，請參閱 README.md 開始使用專案！** 🚀
