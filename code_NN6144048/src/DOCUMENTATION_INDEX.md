# 📚 專案文檔總覽

本目錄包含 RF-DETR 豬隻檢測專案的完整文檔。以下是各文件的說明和閱讀順序建議。

---

## 📋 文件清單

### 🎯 核心文檔（必讀）

1. **[README.md](README.md)** 📖
   - **用途**：專案主要說明文件
   - **內容**：專案簡介、快速開始、使用指南
   - **建議閱讀順序**：⭐ 第一個閱讀
   - **適合對象**：所有用戶

2. **[INSTALLATION.md](INSTALLATION.md)** 🔧
   - **用途**：詳細安裝指南
   - **內容**：環境設置、依賴安裝、常見問題
   - **建議閱讀順序**：第二個閱讀
   - **適合對象**：首次安裝的用戶

3. **[PROJECT_REPORT.md](PROJECT_REPORT.md)** 📊
   - **用途**：完整的技術報告（3-5頁）
   - **內容**：專案概述、數據集、模型訓練、結果分析
   - **建議閱讀順序**：深入了解時閱讀
   - **適合對象**：研究人員、技術評審

### 📦 配置文件

4. **[requirements.txt](requirements.txt)** 📝
   - **用途**：完整的 Python 依賴列表
   - **內容**：所有必需和可選的套件
   - **使用方式**：`pip install -r requirements.txt`

---

## 🗺️ 閱讀路線圖

完整學習 📚（2-3小時）

適合想要深入了解專案的研究人員：

1. **README.md** → 完整閱讀
2. **INSTALLATION.md** → 詳細安裝
3. **PROJECT_REPORT.md** → 技術細節
4. **DATASET_SUMMARY.md** → 數據集理解
5. **TRAINING_GUIDE.md** → 訓練流程
6. 實際操作：訓練模型

```bash
# 創建數據集
python create_rfdetr_dataset.py

# 訓練模型
torchrun --nproc_per_node=4 train_rfdetr_multigpu2.py

# 運行推理
python simple_inference.py
```

### 路線 C：問題排查 🔍（按需）

遇到問題時的查找順序：

1. **README.md** → 常見問題部分
2. **INSTALLATION.md** → 安裝問題
3. **TRAINING_GUIDE.md** → 訓練問題
4. GitHub Issues 或聯繫作者

---

## 📊 文件關係圖

```
README.md (主入口)
    ├── INSTALLATION.md (安裝)
    │   ├── requirements.txt
    │   └── requirements-minimal.txt
    │
    ├── PROJECT_REPORT.md (技術報告)
    │   ├── DATASET_SUMMARY.md
    │   ├── TRAINING_GUIDE.md
    │   └── RFDETR_DATASET_README.md
    │
    └── SUBMISSION_FORMAT_EXPLAINED.md (提交格式)
```

---

## 🎯 根據角色的推薦閱讀

### 👨‍💻 開發者

**必讀**：
- README.md
- INSTALLATION.md
- 程式碼註解

**選讀**：
- PROJECT_REPORT.md（技術實作細節）
- TRAINING_GUIDE.md（訓練參數調整）

### 🔬 研究人員

**必讀**：
- PROJECT_REPORT.md（完整技術報告）
- DATASET_SUMMARY.md（數據集分析）
- TRAINING_GUIDE.md（實驗設計）

**選讀**：
- README.md（快速參考）
- INSTALLATION.md（環境設置）

### 👨‍🏫 教學用途

**必讀**：
- README.md（專案概覽）
- PROJECT_REPORT.md（教學材料）
- INSTALLATION.md（學生環境設置）

**選讀**：
- 所有文檔（作為參考資料）

### 🏢 產品經理/專案管理

**必讀**：
- README.md（專案概述、成果）
- PROJECT_REPORT.md（第一、六、七章）


## 📝 文件更新記錄

| 文件 | 最後更新 | 版本 | 主要變更 |
|------|----------|------|----------|
| README.md | 2025-10-10 | v1.0.0 | 初始版本 |
| PROJECT_REPORT.md | 2025-10-10 | v1.0.0 | 初始版本 |
| INSTALLATION.md | 2025-10-10 | v1.0.0 | 初始版本 |
| requirements.txt | 2025-10-10 | v1.0.0 | 初始版本 |
| requirements-minimal.txt | 2025-10-10 | v1.0.0 | 初始版本 |

---

## 🔄 文件維護

### 更新流程

1. **定期更新**：每次重大變更後更新相關文檔
2. **版本管理**：使用 Git 追蹤文檔變更
3. **交叉檢查**：確保不同文檔間資訊一致

### 貢獻文檔

如果您想改進文檔：

1. 找出需要改進的部分
2. 提交 Issue 或 Pull Request
3. 描述具體的改進建議


